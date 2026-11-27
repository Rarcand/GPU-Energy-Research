"""Compare static rankers and adaptive surrogate architectures on the same GPU folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from analysis.surrogate.scripts.experiments.audit_final_choice import EXPERIMENTS, load_precision_inputs
from analysis.surrogate.scripts.experiments.audit_timing_protocol_transfer import measure
from analysis.surrogate.scripts.experiments.benchmark_candidate_response import conditional_order, feature_distances, rbf
from analysis.surrogate.scripts.experiments.benchmark_few_shot_surrogate import ShortlistSurrogate, metric_for_order, replay, source_scenarios
from analysis.surrogate.scripts.experiments.capacity_bfloat16_transfer import verify_matched_grid
from analysis.surrogate.scripts.experiments.capacity_transport import centered, physical_phases, shift_bank
from analysis.surrogate.scripts.experiments.compare_gemm_resource_residual import aligned_grid, make_model
from analysis.surrogate.scripts.models.compare_surrogates import FAMILIES, save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.features.kernel_resource_features import features_for_family
from analysis.surrogate.scripts.models.portable_rank_prior import score_task


METRICS = ('regret_b1', 'success5_b1', 'regret_b4', 'success5_b4', 'regret_b8', 'success5_b8')
MODELS = ('current', 'frozen_scenarios', 'source_rank', 'ridge', 'boosted_tree',
          'ridge_geometry', 'boosted_tree_geometry', 'empirical_gp', 'geometry_gp',
          'empirical_gp_open_loop', 'geometry_gp_open_loop', 'ridge_then_empirical_gp',
          'boosted_tree_then_empirical_gp', 'ridge_then_latent_gp_2', 'ridge_then_latent_gp_4')
PAIRS = tuple((name, 'current') for name in MODELS[1:]) + (
    ('empirical_gp', 'empirical_gp_open_loop'),
    ('geometry_gp', 'geometry_gp_open_loop'),
    ('ridge', 'ridge_geometry'),
    ('boosted_tree', 'boosted_tree_geometry'),
    ('ridge_then_empirical_gp', 'ridge'),
    ('boosted_tree_then_empirical_gp', 'boosted_tree'),
    ('ridge_then_latent_gp_2', 'ridge_then_empirical_gp'),
    ('ridge_then_latent_gp_4', 'ridge_then_empirical_gp'),
)


def prepare_families(rows, dtype, devices, mode='graph'):
    align = aligned_grid if dtype == 'float16' else verify_matched_grid
    for family in FAMILIES:
        shapes, configs, ordered = align([r for r in rows if r['kernel_family'] == family])
        dims = len(devices), len(shapes), len(configs)
        times = np.asarray([measure(r, mode) for r in ordered]).reshape(dims)
        geometry_rows = [{**r, 'dtype': 'float16'} for r in ordered]
        geometry, ratios, _ = features_for_family(geometry_rows)
        physical = centered(physical_phases(geometry_rows, dims))
        g, r = (np.log1p(x).reshape((*dims, -1)) for x in (geometry, ratios))
        predictors = np.concatenate((g, r, physical), axis=-1)
        yield family, shapes, times, physical, g, predictors


def direct_rankers(times, predictors, sources, target, train, tests):
    matrix = predictors[sources][:, train]
    x = matrix.reshape(-1, matrix.shape[-1])
    actual = times[sources][:, train]
    y = np.log(actual / actual.min(axis=-1, keepdims=True)).ravel()
    dest = predictors[target, tests]
    return {name: make_model(kind).fit(x, y).predict(dest.reshape(-1, dest.shape[-1])).reshape(dest.shape[:2])
            for name, kind in (('ridge', 'ridge'), ('boosted_tree', 'tree'))}


def empirical_gp_prior(bank):
    scenario_logs = bank.reshape(-1, bank.shape[-1])
    scenario_logs = scenario_logs - scenario_logs.mean(axis=-1, keepdims=True)
    prior = scenario_logs.mean(axis=0)
    deviation = scenario_logs - prior
    empirical = deviation.T @ deviation / len(deviation)
    amplitude = np.clip(np.sqrt(empirical.diagonal().mean()), .10, .45)
    nugget = (.25 * amplitude) ** 2 * np.eye(len(prior))
    return prior, empirical + nugget, amplitude, nugget


def latent_covariances(covariance, nugget, ranks=(2, 4)):
    """Compress cross-candidate correlations while preserving marginal uncertainty."""
    values, vectors = np.linalg.eigh(covariance - nugget)
    for rank in ranks:
        factors = vectors[:, -rank:] * np.sqrt(np.maximum(values[-rank:], 0.0))
        compressed = factors @ factors.T
        missing_variance = np.maximum(np.diag(covariance - compressed), 0.0)
        yield rank, compressed + np.diag(missing_variance)


def conditional_policies(bank, weights, timings, geometry, source_geometry, alternative_anchors):
    first = ShortlistSurrogate(bank, weights, adaptive=False).next_candidate()
    prior, empirical, amplitude, nugget = empirical_gp_prior(bank)
    covariances = {
        'empirical_gp': empirical,
        'geometry_gp': rbf(feature_distances(source_geometry, geometry), amplitude) + nugget,
    }
    for model, covariance in covariances.items():
        for feedback in (True, False):
            name = model if feedback else f'{model}_open_loop'
            order = conditional_order(prior, covariance, timings, first, feedback=feedback)
            yield name, metric_for_order(order, timings)
    for ranker, anchor in alternative_anchors.items():
        order = conditional_order(prior, covariances['empirical_gp'], timings, anchor)
        yield f'{ranker}_then_empirical_gp', metric_for_order(order, timings)
    for rank, covariance in latent_covariances(empirical, nugget):
        order = conditional_order(prior, covariance, timings, alternative_anchors['ridge'])
        yield f'ridge_then_latent_gp_{rank}', metric_for_order(order, timings)


def evaluate_fold(data, assignment, target, archived):
    _, shapes, times, physical, geometry_features, predictors = data
    sources = [idx for idx in range(len(times)) if idx != target]
    train, tests = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                    for role in ('train', 'test'))
    geometry = centered(geometry_features)
    source_geometry = geometry[sources][:, train]
    blind = source_scenarios(times, shapes, sources, train, tests)
    hardware = shift_bank(blind, physical, sources, tests, target,
                          archived['coefficient'], archived['strength'])
    weights = np.full(len(sources), 1. / len(sources))
    direct = direct_rankers(times, predictors, sources, target, train, tests)
    geometry_only = direct_rankers(times, geometry_features, sources, target, train, tests)
    records = []
    for index, i in enumerate(tests):
        labels = times[target, i]
        cases = hardware[index]
        current = metric_for_order(replay(cases, weights, labels, adaptive=True), labels)
        old = archived['scores']['hardware'][index]
        if shapes[i] != old['shape'] or any(not np.isclose(current[k], old[k], atol=1e-12, rtol=0) for k in METRICS):
            raise ValueError(f'Original replay changed for {shapes[i]}')
        anchors = {model: int(values[index].argmin()) for model, values in direct.items()}
        results = {
            'current': current,
            'frozen_scenarios': metric_for_order(replay(cases, weights, labels, adaptive=False), labels),
            'source_rank': score_task(labels, rankdata(blind[index], axis=-1).mean(axis=(0, 1))),
            **{name: score_task(labels, v[index]) for name, v in direct.items()},
            **{f'{name}_geometry': score_task(labels, v[index]) for name, v in geometry_only.items()},
            **dict(conditional_policies(cases, weights, labels, geometry[target, i], source_geometry, anchors)),
        }
        records.append([[results[m][metric] for metric in METRICS] for m in MODELS])
    return np.asarray(records)


def report(families, grids):
    rng = np.random.default_rng(1729)
    result = {'means': {m: {} for m in MODELS}, 'paired': {f'{a}_minus_{b}': {} for a, b in PAIRS}}
    for k, metric in enumerate(METRICS):
        family_scores = [rows[..., k] for family, rows in families.items()
                         if not (metric.endswith('b8') and grids[family] <= 8)]
        draw = [rng.integers(len(a), size=(10000, len(a))) for a in family_scores]
        macro = np.mean([rows.mean(axis=0) for rows in family_scores], axis=0)
        for m, average in zip(MODELS, macro):
            result['means'][m][metric] = float(average)
        for method, base in PAIRS:
            a, b = MODELS.index(method), MODELS.index(base)
            pairs = [(rows[:, a] - rows[:, b])[d].mean(axis=1) for rows, d in zip(family_scores, draw)]
            interval = np.quantile(np.mean(pairs, axis=0), (.025, .975)).tolist()
            result['paired'][f'{method}_minus_{base}'][metric] = {
                'difference': float(macro[a] - macro[b]), 'ci95': interval,
            }
    result['by_family'] = {f: {m: dict(zip(METRICS, row.tolist()))
                              for m, row in zip(MODELS, rows.mean(axis=0))} for f, rows in families.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 't4', 'reports', 'output'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    args = parser.parse_args()
    result = {
        'protocol': ('Graph replay. Reuse the original hardware correction and its source-only validation. '
                     'Every competitor uses the same held-out GPU and workload shapes, candidate grid and probe budget. '
                     'The two GPs share the current method’s hardware-corrected scenario mean and first pick. '
                     'They differ in covariance: empirical from source timing scenarios, or smooth over configuration geometry. '
                     'The GP controls follow the same rule, replacing real feedback with predictions. '
                     'Ridge and boosted trees train on source shapes and predict relative log slowdown, '
                     'using static candidate geometry and hardware pressure. They and the nearest-source '
                     'rank prior try candidates in fixed rank order. The frozen scenario tuner keeps '
                     'the current portfolio rule but updates no scenario weights from measured outcomes. '
                     'The direct rankers have geometry-only controls. The two hybrid methods reuse each '
                     'direct ranker’s first pick and the empirical GP for the rest of the budget. They were '
                     'added after seeing the direct rankers’ results and are exploratory on these splits. '
                     'The two latent GP variants share the ridge anchor, prior mean, marginal predictive '
                     'variance, and policy with ridge plus empirical GP. Only cross-candidate correlations '
                     'are compressed to the top two or four components of the source covariance. '
                     'Four-probe mean regret is primary. Eight-probe means exclude families with exhausted grids. '
                     'Paired intervals resample shapes within families across fixed device folds.'),
        'experiments': {},
    }
    with threadpool_limits(limits=1):
        for dtype, rows, devices in load_precision_inputs(args.inputs, args.t4):
            groups = list(prepare_families(rows, dtype, devices))
            grids = {data[0]: data[2].shape[-1] for data in groups}
            for size in ('largest', 'next_largest'):
                split_name, report_name = EXPERIMENTS[dtype, size]
                split_path, report_path = args.reports / split_name, args.reports / report_name
                assignment = json.loads(split_path.read_bytes())['assignment']
                archived = json.loads(report_path.read_bytes())
                if tuple(archived['devices']) != tuple(devices):
                    raise ValueError('Archived device order differs from replay')
                if 'splits' in archived:
                    archived = archived['splits'][split_path.stem]
                scores = {}
                for data in groups:
                    family = data[0]
                    folds = [evaluate_fold(data, assignment, i, archived['families'][family][d])
                             for i, d in enumerate(devices)]
                    scores[family] = np.asarray(folds).mean(axis=0)
                result['experiments'][f'{dtype}_{size}'] = {
                    'devices': devices, 'split_sha256': sha(split_path.read_bytes()),
                    'original_report_sha256': sha(report_path.read_bytes()),
                    'device_shape_cases': len(devices) * sum(len(v) for v in scores.values()),
                    'candidates_by_family': grids, **report(scores, grids),
                }
                print(f'Completed {dtype}, {size}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == '__main__':
    main()
