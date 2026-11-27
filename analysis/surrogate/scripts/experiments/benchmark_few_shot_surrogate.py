"""Replay hardware-aware, budgeted configuration ranking on held-out GPUs."""

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analysis.surrogate.scripts.experiments.audit_timing_protocol_transfer import measure
from analysis.surrogate.scripts.experiments.compare_gemm_resource_residual import aligned_grid, neighbor_ranks
from analysis.surrogate.scripts.experiments.compare_portable_surrogates import DEVICES, load_matched
from analysis.surrogate.scripts.models.compare_surrogates import FAMILIES, save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.features.kernel_resource_features import features_for_family
from analysis.surrogate.scripts.models.portable_rank_prior import NEIGHBORS, score_task, shape_distances

MODES = ('graph', 'individual')
TOP_BUDGETS = (1, 4, 8)
STRENGTHS = (0.5, 1., 2., 4.)
METHODS = ('rank_prior', 'uniform_portfolio', 'hardware_portfolio',
           'uniform_adaptive', 'hardware_adaptive', 'gemm_geometry_adaptive',
           'gemm_resource_adaptive', 'all_geometry_adaptive',
           'all_resource_adaptive', 'all_neutral_adaptive',
           'source_gate_adaptive', 'source_gate_blind',
           'counterfactual_adaptive', 'counterfactual_blind', 'counterfactual_frozen',
           'counterfactual_probe', 'counterfactual_probe_blind')
ANCHOR_METHODS = {
    'gemm_geometry_adaptive': 'geometry_matched',
    'gemm_resource_adaptive': 'resource_residual',
    'all_geometry_adaptive': 'geometry_matched',
    'all_resource_adaptive': 'resource_residual',
    'all_neutral_adaptive': 'neutral_resources',
    'source_gate_adaptive': 'resource_residual',
    'source_gate_blind': 'geometry_matched',
    'counterfactual_adaptive': 'resource_residual',
    'counterfactual_blind': 'geometry_matched',
    'counterfactual_frozen': 'resource_residual',
    'counterfactual_probe': 'resource_residual',
    'counterfactual_probe_blind': 'geometry_matched',
}
NEAR = np.log1p(.05)
PRESSURE_GROUPS = (
    ('ctas_per_sm',),
    ('thread_limited_wave_fill_proxy',),
    ('working_set_over_l2_proxy', 'input_bytes_over_l2'),
    ('panel_bytes_over_l2_proxy', 'grouped_panel_bytes_over_l2'),
    ('staged_bytes_over_shared_per_sm_proxy', 'staged_panel_bytes_over_sm_shared_limit_proxy'),
)


def hardware_distance(rows, dims):
    _, resources, names = features_for_family(rows)
    names = names['resources']
    indices = [names.index(name) for group in PRESSURE_GROUPS for name in group if name in names]
    if len(indices) < 4:
        raise ValueError('Missing launch and memory pressure features')
    pressure = np.log1p(resources[:, indices]).reshape((*dims, len(indices)))
    return np.abs(pressure[:, None] - pressure[None, :]).mean(axis=(3, 4))


def source_scenarios(timings, shapes, sources, train, queries):
    ref_names = [shapes[i] for i in train]
    distances = shape_distances(ref_names, [shapes[i] for i in queries])
    close = np.argsort(distances, axis=1, kind='stable')[:, :NEIGHBORS]
    source_times = timings[sources][:, np.asarray(train)[close], :]
    relative = np.log(source_times / source_times.min(axis=-1, keepdims=True))
    return relative.transpose(1, 0, 2, 3)


def source_weights(distances, strength):
    logits = -strength * np.asarray(distances)
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    # Retain a uniform component when all source GPUs differ from the target.
    return .8 * probability + .2 / len(probability)


class ShortlistSurrogate:
    """One task's source scenarios and hardware prior, with optional online updates."""

    def __init__(self, scenarios, device_weights, adaptive):
        neighbors, candidates = scenarios.shape[1:]
        self.scenarios = scenarios.reshape(-1, candidates)
        self.utility = (.65 * (self.scenarios <= NEAR)
                        + .35 * np.exp(-self.scenarios / .20))
        self.prior = np.repeat(device_weights / neighbors, neighbors)
        self.adaptive = adaptive
        self.chosen = []
        self.log_timings = []

    def weights_for_observations(self, candidates, log_timings):
        """Reweight scenarios from real or batched hypothetical log timings."""
        if not self.adaptive or len(candidates) < 2:
            return self.prior
        errors = self.scenarios[:, candidates] - np.asarray(log_timings)[..., None, :]
        errors -= np.median(errors, axis=-1, keepdims=True)
        # The shared offset cancels the unknown target latency scale.
        loss = np.log1p(np.square(errors / .14)).sum(axis=-1)
        likelihood = np.exp(loss.min(axis=-1, keepdims=True) - loss)
        posterior = self.prior * likelihood
        return .25 * self.prior + .75 * posterior / posterior.sum(axis=-1, keepdims=True)

    def next_candidate(self):
        if len(self.chosen) == self.scenarios.shape[1]:
            raise ValueError('Every candidate was already tried')
        weights = self.weights_for_observations(self.chosen, self.log_timings)
        incumbent = (self.utility[:, self.chosen].max(axis=1, keepdims=True)
                     if self.chosen else 0.)
        gains = weights @ np.maximum(self.utility - incumbent, 0.)
        gains += .08 * (weights @ self.utility)
        gains[self.chosen] = -np.inf
        return int(np.argmax(gains))

    def record(self, candidate, latency_ms):
        if (candidate < 0 or candidate >= self.scenarios.shape[1] or candidate in self.chosen
                or not np.isfinite(latency_ms) or latency_ms <= 0):
            raise ValueError('Expected a fresh candidate and finite positive target latency')
        self.chosen.append(candidate)
        self.log_timings.append(float(np.log(latency_ms)))


def replay(scenarios, weights, labels, adaptive=False, first_candidate=None, second_candidate=None):
    surrogate = ShortlistSurrogate(scenarios, weights, adaptive)
    for attempt in range(min(max(TOP_BUDGETS), len(labels))):
        proposal = first_candidate if attempt == 0 else second_candidate if attempt == 1 else None
        candidate = (proposal if proposal is not None and proposal not in surrogate.chosen
                     else surrogate.next_candidate())
        surrogate.record(candidate, labels[candidate])
    return surrogate.chosen


def metric_for_order(order, labels):
    rankings = np.ones(len(labels)) * len(labels)
    rankings[order] = np.arange(len(order))
    return score_task(labels, rankings)


def counterfactual_gate(audit):
    if not audit:
        return False
    observed = np.asarray(audit['resource'])
    for key in ('geometry_matched', 'neutral_resources', 'rank_prior'):
        gain = np.asarray(audit[key]) - observed
        if gain.mean() <= 0 or np.count_nonzero(gain > 0) < 2:
            return False
    return True


def choose_strength(source_times, eval_times, distances, shapes, sources, train, validation):
    """Tune hardware similarity only on an inner unseen source GPU and unseen shapes."""
    scores = defaultdict(list)
    for hidden in sources:
        inner = [d for d in sources if d != hidden]
        scenario_bank = source_scenarios(source_times, shapes, inner, train, validation)
        for strength in STRENGTHS:
            fold_losses = []
            for shape_id, scenarios in zip(validation, scenario_bank):
                weights = source_weights(distances[hidden, inner, shape_id], strength)
                actual = eval_times[hidden, shape_id]
                metrics = metric_for_order(replay(scenarios, weights, actual), actual)
                fold_losses.append(np.log1p(metrics['regret_b1']) + .5 * np.log1p(metrics['regret_b4']))
            scores[strength].append(float(np.mean(fold_losses)))
    return min(STRENGTHS, key=lambda strength: np.mean(scores[strength])), scores


def evaluate_fold(source_times, eval_times, distances, shapes, assignment, target, anchors):
    sources = [d for d in range(len(DEVICES)) if d != target]
    train, val, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                        for role in ('train', 'validation', 'test'))
    strength, losses = choose_strength(source_times, eval_times, distances, shapes, sources, train, val)
    scenario_bank = source_scenarios(source_times, shapes, sources, train, test)
    priors = neighbor_ranks(rankdata(source_times, axis=-1), shapes, sources, train, test)
    predictions = defaultdict(list)
    for shape_id, cases, prior in zip(test, scenario_bank, priors):
        weights = source_weights(distances[target, sources, shape_id], strength)
        uniform = np.full(len(sources), 1 / len(sources))
        actual = eval_times[target, shape_id]
        exact_random = {}
        close = int(np.sum(actual <= 1.05 * min(actual)))
        for budget in TOP_BUDGETS:
            size = min(budget, len(actual))
            exact_random[f'success5_b{budget}'] = 1 - comb(len(actual) - close, size) / comb(len(actual), size)
        predictions['rank_prior'].append(score_task(actual, prior))
        predictions['random'].append(exact_random)
        for method, use_specs, use_observations in (
                ('uniform_portfolio', False, False), ('hardware_portfolio', True, False),
                ('uniform_adaptive', False, True), ('hardware_adaptive', True, True)):
            order = replay(cases, weights if use_specs else uniform, actual, use_observations)
            predictions[method].append(metric_for_order(order, actual))
        for method in ANCHOR_METHODS:
            if method in anchors:
                anchor = anchors[method][shapes[shape_id]]
                settings = ({'second_candidate': anchor} if method.startswith('counterfactual_probe')
                            else {'first_candidate': anchor})
                order = replay(cases, uniform, actual, method != 'counterfactual_frozen', **settings)
                predictions[method].append(metric_for_order(order, actual))
            else:
                baseline = 'uniform_portfolio' if method == 'counterfactual_frozen' else 'uniform_adaptive'
                predictions[method].append(predictions[baseline][-1].copy())
    for method, records in predictions.items():
        for row, shape_id in zip(records, test):
            row['shape'] = shapes[shape_id]
    return dict(strength=strength, source_validation_losses=losses,
                test_shapes=len(test), candidates_per_shape=eval_times.shape[-1],
                used_anchors=sorted(anchors),
                task_scores=predictions,
                metrics={method: {key: float(np.mean([r[key] for r in records]))
                                  for key in records[0] if key != 'shape'}
                         for method, records in predictions.items()})


def summarize(families):
    methods = ('random', *METHODS)
    metrics = {method: {metric: float(np.mean([f[d]['metrics'][method][metric]
                                               for f in families.values() for d in DEVICES]))
                        for metric in next(iter(families.values()))[DEVICES[0]]['metrics'][method]}
               for method in methods}
    paired = {}
    rng = np.random.default_rng(92714)
    comparisons = (
        ('hardware_portfolio', 'rank_prior'),
        ('hardware_portfolio', 'uniform_portfolio'),
        ('hardware_adaptive', 'hardware_portfolio'),
        ('hardware_adaptive', 'rank_prior'),
        ('gemm_resource_adaptive', 'rank_prior'),
        ('gemm_resource_adaptive', 'gemm_geometry_adaptive'),
        ('gemm_resource_adaptive', 'uniform_adaptive'),
        ('all_resource_adaptive', 'all_geometry_adaptive'),
        ('all_resource_adaptive', 'all_neutral_adaptive'),
        ('all_resource_adaptive', 'rank_prior'),
        ('source_gate_adaptive', 'source_gate_blind'),
        ('source_gate_adaptive', 'rank_prior'),
        ('source_gate_adaptive', 'gemm_resource_adaptive'),
        ('counterfactual_adaptive', 'counterfactual_blind'),
        ('counterfactual_adaptive', 'counterfactual_frozen'),
        ('counterfactual_adaptive', 'rank_prior'),
        ('counterfactual_probe', 'counterfactual_probe_blind'),
        ('counterfactual_probe', 'counterfactual_adaptive'),
        ('counterfactual_probe', 'uniform_adaptive'),
    )
    for method, baseline in comparisons:
        paired[f'{method}_minus_{baseline}'] = {}
        for key in ('success5_b1', 'success5_b4', 'success5_b8', 'regret_b4'):
            differences = [np.mean([[a[key] - b[key] for a, b in zip(
                f[d]['task_scores'][method], f[d]['task_scores'][baseline])] for d in DEVICES], axis=0)
                for f in families.values()]
            draws = [v[rng.integers(len(v), size=(5000, len(v)))].mean(axis=1) for v in differences]
            paired[f'{method}_minus_{baseline}'][key] = {
                'difference': float(np.mean([v.mean() for v in differences])),
                'ci95': np.quantile(np.mean(draws, axis=0), (.025, .975)).tolist(),
            }
    return dict(macro=metrics, paired=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--splits', type=Path, nargs='+', required=True)
    parser.add_argument('--graph-ablation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, _, hashes = load_matched(args.inputs, args.t4, 'graph')
    assignments = {p.stem: json.loads(p.read_text())['assignment'] for p in args.splits}
    ablation = json.loads(args.graph_ablation.read_text())
    if (ablation['mode'] != 'graph' or not ablation['complete']
            or ablation['archive_sha256'] != hashes):
        raise ValueError('Require a verified graph ablation on the same archives')
    for p in args.splits:
        if ablation['splits'][p.stem]['split_sha256'] != sha(p.read_bytes()):
            raise ValueError(f'Graph ablation split does not match {p}')
    result = dict(verified_candidates=len(rows), devices=DEVICES, dtype='float16',
                  training_mode='graph', extra_target_calibration_rows=0,
                  budget='Every target benchmark is charged; only completed observations inform the next pick.',
                  bootstrap='Paired by workload shape and stratified by family, fixed four-device set',
                  gemm_anchor='GEMM gets the first pick from the source-only graph residual; other families use the uniform adaptive portfolio.',
                  source_gate='Apply the resource correction across any family only when it has the lowest source-held-GPU and shape validation loss.',
                  counterfactual_gate='Require mean and at least two pseudo-GPU wins over matched geometry, neutralized target specs, and rank prior. Compare hardware prediction as first pick or as the second paid probe.',
                  graph_ablation_sha256=sha(args.graph_ablation.read_bytes()),
                  split_sha256={p.stem: sha(p.read_bytes()) for p in args.splits},
                  archive_sha256=hashes, splits={})
    for split in assignments:
        result['splits'][split] = {mode: {'families': {}} for mode in MODES}
    for family in FAMILIES:
        shapes, configs, ordered = aligned_grid([r for r in rows if r['kernel_family'] == family])
        dims = (len(DEVICES), len(shapes), len(configs))
        distances = hardware_distance(ordered, dims)
        timings = {mode: np.asarray([measure(r, mode) for r in ordered]).reshape(dims) for mode in MODES}
        for split, assignment in assignments.items():
            anchors = {}
            for d, fold in ablation['splits'][split]['families'][family]['folds'].items():
                validated = counterfactual_gate(fold.get('counterfactual_source_validation'))
                anchors[d] = {
                    method: {task['shape']: task['shortlist'][0] for task in fold['task_scores'][record]}
                    for method, record in ANCHOR_METHODS.items()
                    if (not method.startswith('gemm_') or family == 'gemm')
                    and (not method.startswith('source_gate_') or fold['source_choice'] == 'resource_residual')
                    and (not method.startswith('counterfactual_') or validated)
                }
            for mode in MODES:
                folds = {device: evaluate_fold(timings['graph'], timings[mode], distances,
                                                shapes, assignment, target, anchors[device])
                         for target, device in enumerate(DEVICES)}
                result['splits'][split][mode]['families'][family] = folds
        print(f'Finished {family}', flush=True)
    for modes in result['splits'].values():
        for evaluated in modes.values():
            evaluated.update(summarize(evaluated['families']))
    save_json(args.output, result)


if __name__ == '__main__':
    main()
