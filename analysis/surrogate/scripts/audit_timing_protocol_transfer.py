"""Quantify shortlist transfer between timing modes and independent repeats."""

import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from compare_gemm_resource_residual import aligned_grid, neighbor_ranks
from compare_portable_surrogates import DEVICES, load_matched
from compare_surrogates import FAMILIES, save_json

MODES = ('graph', 'individual')
BUDGETS = (1, 4)
REPEATED_DEVICES = ('H100', 'RTX3070-WSL')
METHODS = ('rank_prior', 'geometry_matched', 'resource_residual', 'source_selected')


def measure(row, mode, pass_id=None):
    if pass_id is not None:
        return float(row[f'{mode}_pass{pass_id}_ms'])
    if f'{mode}_median_ms' in row:
        return float(row[f'{mode}_median_ms'])
    return float(row[mode]['median_ms'])


def results_at_budget(choices, labels, budget):
    best = np.take_along_axis(labels, choices[..., :budget], axis=-1).min(axis=-1)
    relative = best / labels.min(axis=-1)
    return {'success5': relative <= 1.05, 'regret': relative - 1.}


def repeated_modes(ordered, shape_count, config_count):
    """Compare transfer over different passes with and without changing timing mode."""
    repeated = {}
    for device in REPEATED_DEVICES:
        rows = [r for r in ordered if r['_device'] == device]
        repeated[device] = {
            mode: np.asarray([[measure(r, mode, p) for r in rows] for p in range(3)])
            .reshape((3, shape_count, config_count)) for mode in MODES}
    result, shape_differences = {}, {}
    for dst, src in (MODES, MODES[::-1]):
        direction = f'{src}_to_{dst}'
        result[direction], shape_differences[direction] = {}, {}
        for b in BUDGETS:
            by_gpu = {}
            for device, values in repeated.items():
                scores = {'same_mode': [], 'switched_mode': []}
                for source_pass, target_pass in permutations(range(3), 2):
                    for label, source_mode in (('same_mode', dst), ('switched_mode', src)):
                        chosen = np.argsort(values[source_mode][source_pass], axis=1, kind='stable')
                        scores[label].append(results_at_budget(chosen, values[dst][target_pass], b))
                by_gpu[device] = {
                    label: {metric: np.mean([r[metric] for r in trials], axis=0)
                            for metric in ('success5', 'regret')}
                    for label, trials in scores.items()}
            shape_differences[direction][str(b)] = np.mean([
                scores['same_mode']['success5'] - scores['switched_mode']['success5']
                for scores in by_gpu.values()], axis=0)
            result[direction][str(b)] = {
                gpu: {label: {metric: float(v.mean()) for metric, v in scores.items()}
                      for label, scores in modes.items()} for gpu, modes in by_gpu.items()}
    return result, shape_differences


def heldout_shortlists(family, shapes, labels, reports, assignments):
    results = {}
    shape_index = {s: i for i, s in enumerate(shapes)}
    pooled_ranks = np.mean([rankdata(labels[mode], axis=2) for mode in MODES], axis=0)
    for split in reports['graph']['splits']:
        train = [i for i, shape in enumerate(shapes) if assignments[split][shape] == 'train']
        results[split] = {}
        for dst, src in (MODES, MODES[::-1]):
            direction = f'{src}_to_{dst}'
            results[split][direction] = {}
            for d, device in enumerate(DEVICES):
                relevant = {mode: reports[mode]['splits'][split]['families'][family]['folds'][device]
                            for mode in MODES}
                device_results = {}
                for method in METHODS:
                    in_mode = relevant[dst]['task_scores'][method]
                    cross_mode = relevant[src]['task_scores'][method]
                    task_shapes = [t['shape'] for t in in_mode]
                    if task_shapes != [t['shape'] for t in cross_mode]:
                        raise ValueError(f'Shortlist shape mismatch: {device}, {family}')
                    test = [shape_index[s] for s in task_shapes]
                    observed = labels[dst][d, test]
                    orders = {label: np.asarray([t['shortlist'] for t in tasks])
                              for label, tasks in (('same_mode', in_mode), ('switched_mode', cross_mode))}
                    if method == 'rank_prior':
                        sources = [i for i in range(len(DEVICES)) if i != d]
                        pooled = neighbor_ranks(pooled_ranks, shapes, sources, train, test)
                        orders['pooled_modes'] = np.argsort(pooled, axis=1, kind='stable')
                    device_results[method] = {}
                    for b in BUDGETS:
                        values = {key: results_at_budget(order, observed, b)
                                  for key, order in orders.items()}
                        if not np.isclose(values['same_mode']['success5'].mean(),
                                          np.mean([t[f'success5_b{b}'] for t in in_mode])):
                            raise ValueError(f'Shortlist scoring mismatch: {device}, {family}')
                        device_results[method][str(b)] = {
                            key: dict(success5=float(score['success5'].mean()),
                                      regret=float(score['regret'].mean()),
                                      success5_by_shape=score['success5'].tolist())
                            for key, score in values.items()}
                results[split][direction][device] = device_results
    return results


def summarize_repeats(all_families, differences):
    repeated = {}
    rng = np.random.default_rng(31929)
    for dst, src in (MODES, MODES[::-1]):
        direction = f'{src}_to_{dst}'
        repeated[direction] = {}
        for b in map(str, BUDGETS):
            repeated[direction][b] = {
                label: {metric: float(np.mean([family['repeats'][direction][b][d][label][metric]
                                              for family in all_families.values()
                                              for d in REPEATED_DEVICES]))
                        for metric in ('success5', 'regret')} for label in ('same_mode', 'switched_mode')}
            draws = []
            for f in FAMILIES:
                vector = differences[f][direction][b]
                draws.append(vector[rng.integers(len(vector), size=(5000, len(vector)))].mean(axis=1))
            repeated[direction][b]['success_loss_ci95'] = np.quantile(
                np.mean(draws, axis=0), (.025, .975)).tolist()
    return repeated


def paired_success_difference(vectors):
    """Share resampled workload shapes across the fixed devices."""
    rng = np.random.default_rng(49278)
    draws = [v[rng.integers(len(v), size=(5000, len(v)))].mean(axis=1) for v in vectors]
    return dict(difference=float(np.mean([v.mean() for v in vectors])),
                ci95=np.quantile(np.mean(draws, axis=0), (.025, .975)).tolist())


def summarize_hardware(reports):
    results = {}
    for mode, report in reports.items():
        results[mode] = {}
        for split, entry in report['splits'].items():
            fs = entry['families']
            metrics = ('success5_b1', 'success5_b4', 'regret_b1', 'regret_b4')
            methods = next(iter(fs.values()))['macro']
            average = {method: {key: float(np.mean([v['macro'][method][key]
                                                    for v in fs.values()]))
                                for key in metrics} for method in methods}
            contrasts = {}
            for baseline in ('rank_prior', 'geometry_matched', 'neutral_resources'):
                contrasts[baseline] = {}
                for b in BUDGETS:
                    vectors = [np.mean([
                        [a[f'success5_b{b}'] - q[f'success5_b{b}'] for a, q in zip(
                            fold['task_scores']['resource_residual'], fold['task_scores'][baseline])]
                        for fold in f['folds'].values()], axis=0) for f in fs.values()]
                    contrasts[baseline][str(b)] = paired_success_difference(vectors)
            results[mode][split] = dict(macro=average, resource_minus=contrasts)
    return results


def summarize_heldout(all_families):
    results = {}
    for split in next(iter(all_families.values()))['shortlist_transfer']:
        results[split] = {}
        for dst, src in (MODES, MODES[::-1]):
            direction = f'{src}_to_{dst}'
            results[split][direction] = {}
            for method in METHODS:
                results[split][direction][method] = {}
                for b in map(str, BUDGETS):
                    families = [[device[method][b] for device in
                                 f['shortlist_transfer'][split][direction].values()]
                                for f in all_families.values()]
                    metrics = {label: {metric: float(np.mean([r[label][metric]
                                                             for f in families for r in f]))
                                       for metric in ('success5', 'regret')}
                               for label in families[0][0]}
                    for label in set(metrics) - {'same_mode'}:
                        vectors = [np.mean([
                            np.asarray(r['same_mode']['success5_by_shape'], dtype=float)
                            - r[label]['success5_by_shape'] for r in f], axis=0) for f in families]
                        metrics[label]['same_mode_minus'] = paired_success_difference(vectors)
                    results[split][direction][method][b] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--ablation-root', type=Path, required=True)
    parser.add_argument('--splits', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {mode: json.loads((args.ablation_root / f'{mode}.json').read_text()) for mode in MODES}
    if not all(r.get('complete') for r in reports.values()):
        raise ValueError('Require both completed family ablations')
    assignments = {p.stem: json.loads(p.read_text())['assignment'] for p in args.splits}
    rows, _, hashes = load_matched(args.inputs, args.t4, 'graph')
    all_families, differences = {}, {}
    for family in FAMILIES:
        shapes, configs, ordered = aligned_grid([r for r in rows if r['kernel_family'] == family])
        dims = (len(DEVICES), len(shapes), len(configs))
        labels = {mode: np.asarray([measure(r, mode) for r in ordered]).reshape(dims) for mode in MODES}
        if not all(np.isfinite(v).all() and (v > 0).all() for v in labels.values()):
            raise ValueError(f'Invalid timings for {family}')
        repeats, differences[family] = repeated_modes(ordered, len(shapes), len(configs))
        all_families[family] = dict(
            shapes=len(shapes), candidates_per_shape=len(configs), repeats=repeats,
            shortlist_transfer=heldout_shortlists(family, shapes, labels, reports, assignments))
    save_json(args.output, dict(
        repeats=summarize_repeats(all_families, differences),
        heldout=summarize_heldout(all_families), hardware=summarize_hardware(reports),
        families=all_families, archive_sha256=hashes,
        repeat_protocol='Off-diagonal pass pairs on H100 and RTX3070; paired by shape across both fixed devices',
        learned_protocol='GPU and shape holdouts; switch source training timing mode and re-score the fixed shortlist',
    ))


if __name__ == '__main__':
    main()
