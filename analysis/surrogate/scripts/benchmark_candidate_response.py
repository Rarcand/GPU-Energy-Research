"""Replay relative-latency kernel updates under matched, held-out-device probe budgets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.spatial.distance import cdist

from audit_timing_protocol_transfer import measure
from benchmark_few_shot_surrogate import (
    TOP_BUDGETS,
    ShortlistSurrogate,
    metric_for_order,
    replay,
    source_scenarios,
)
from compare_gemm_resource_residual import aligned_grid
from compare_portable_surrogates import DEVICES, load_matched
from compare_surrogates import FAMILIES, save_json
from gather_timing import sha
from kernel_resource_features import features_for_family

METHODS = ('scenario_static', 'scenario_online', 'gp_static', 'gp_geometry_open_loop',
           'gp_hardware_open_loop', 'gp_geometry_online', 'gp_neutral_online',
           'gp_hardware_online', 'gp_empirical_online')
METRICS = ('success5_b1', 'success5_b4', 'success5_b8', 'regret_b1', 'regret_b4', 'regret_b8')
COMPARISONS = (
    ('gp_hardware_online', 'gp_geometry_online'),
    ('gp_hardware_online', 'gp_neutral_online'),
    ('gp_hardware_online', 'scenario_online'),
    ('gp_hardware_online', 'gp_static'),
    ('gp_hardware_online', 'gp_hardware_open_loop'),
    ('gp_geometry_online', 'gp_static'),
    ('gp_geometry_online', 'gp_geometry_open_loop'),
    ('gp_geometry_online', 'scenario_online'),
    ('gp_empirical_online', 'scenario_online'),
    ('scenario_online', 'scenario_static'),
)
DENSITY_FACTOR = np.sqrt(2. * np.pi)
NOISE = .05


def center(features):
    return features - features.mean(axis=-2, keepdims=True)


def feature_distances(training, prediction):
    """Scale candidate features using only the source GPUs and training shapes."""
    scaled = np.sqrt(np.mean(np.square(training), axis=(0, 1, 2)))
    selected = scaled > 1e-8
    if not selected.any():
        return np.zeros((len(prediction), len(prediction)))
    normalized = prediction[:, selected] / np.maximum(scaled[selected], .07)
    return cdist(normalized, normalized, metric='sqeuclidean') / selected.sum()


def rbf(distances, amplitude):
    return amplitude ** 2 * np.exp(-distances / 2.)


def contrast_covariance(covariance, anchor):
    return covariance - covariance[[anchor], :] - covariance[:, [anchor]] + covariance[anchor, anchor]


def contrast_posterior(prior, covariance, indices, contrasts):
    """Condition log-latency contrasts on measurements sharing the same anchor."""
    variance = covariance.diagonal().copy()
    if not len(indices):
        return prior, variance
    cross = covariance[:, indices]
    probe_noise = NOISE ** 2 * (np.eye(len(indices)) + 1.)
    observed_covariance = covariance[np.ix_(indices, indices)] + probe_noise
    precision = np.linalg.solve(observed_covariance, cross.T)
    mean = prior + precision.T @ (contrasts - prior[indices])
    variance -= np.sum(cross * precision.T, axis=1)
    return mean, variance


def conditional_order(prior, covariance, latency, anchor, *, active=True, feedback=True,
                      budget=max(TOP_BUDGETS)):
    """Choose each next probe without accessing any unobserved target timing."""
    if not active:
        source_order = np.argsort(prior, kind='stable').tolist()
        source_order.remove(anchor)
        return [anchor, *source_order[:budget - 1]]

    selected = [anchor]
    # The open-loop control updates its uncertainty with the same acquisition
    # policy while substituting the prior mean for each observed timing.
    observed = [np.log(latency[anchor]) if feedback else prior[anchor]]
    covariance_anchored = contrast_covariance(covariance, anchor)
    prior_delta = prior - prior[anchor]
    for _ in range(1, min(budget, len(prior))):
        indices = selected[1:]
        contrasts = np.asarray(observed[1:]) - observed[0]
        mean, variance = contrast_posterior(prior_delta, covariance_anchored, indices, contrasts)
        incumbent = min(float(contrasts.min()), 0.) if len(contrasts) else 0.
        deviation = np.sqrt(np.maximum(variance, 1e-12))
        signed_gain = incumbent - mean
        offset = signed_gain / deviation
        expected_gain = signed_gain * ndtr(offset) + deviation * np.exp(-.5 * offset ** 2) / DENSITY_FACTOR
        expected_gain[selected] = -np.inf
        candidate = int(np.argmax(expected_gain))
        selected.append(candidate)
        observed.append(np.log(latency[candidate]) if feedback else prior[candidate])
    return selected


def paired_summary(metrics, families=FAMILIES):
    rng = np.random.default_rng(92714)
    output = {}
    for treatment, baseline in COMPARISONS:
        contrasts = {}
        for metric in METRICS:
            values = []
            for family in families:
                if metric.endswith('b8') and metrics[(family, DEVICES[0])]['candidates'] <= 8:
                    continue
                differences = []
                for device in DEVICES:
                    records = metrics[(family, device)]['metrics']
                    differences.append([
                        a[metric] - b[metric]
                        for a, b in zip(records[treatment], records[baseline])
                    ])
                values.append(np.mean(differences, axis=0))
            if not values:
                continue
            draws = [v[rng.integers(len(v), size=(5000, len(v)))].mean(axis=1) for v in values]
            contrasts[metric] = {
                'difference': float(np.mean([v.mean() for v in values])),
                'ci95': np.quantile(np.mean(draws, axis=0), (.025, .975)).tolist(),
            }
        output[f'{treatment}_minus_{baseline}'] = contrasts
    return output


def average_metrics(selected, non_exhausted=False):
    return {method: {metric: float(np.mean([
        np.mean([task[metric] for task in r['metrics'][method]]) for r in selected
        if not (non_exhausted and metric.endswith('b8') and r['candidates'] <= 8)
    ])) for metric in METRICS} for method in METHODS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--splits', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-task-scores', action='store_true',
                        help='Include each configuration probe sequence and its metrics')
    args = parser.parse_args()
    rows, _, hashes = load_matched(args.inputs, args.t4, 'graph')
    split_map = {path.stem: json.loads(path.read_text())['assignment'] for path in args.splits}
    result = {'verified_candidates': len(rows), 'devices': DEVICES, 'archive_sha256': hashes,
              'split_sha256': {path.stem: sha(path.read_bytes()) for path in args.splits},
              'protocol': 'FP16 graph replay with a common source-only scenario anchor',
              'details': ('Outer independent holdout per hardware and shape. Source-only feature scaling. '
                          'GP prior relative to anchor; acquisition via expected relative improvement. '
                          'Neutral resource interactions evaluated at source-mean hardware. Settings fixed a priori. '
                          'Open-loop GP controls substitute prior means for observed timings. '
                          'Every adaptive observation consumes one target configuration measurement. '
                          'Paired intervals stratified by shape family and average over the four fixed devices.'),
              'splits': {}}
    for split, assignment in split_map.items():
        scores = {}
        for family in FAMILIES:
            shapes, configs, ordered = aligned_grid([r for r in rows if r['kernel_family'] == family])
            dims = (len(DEVICES), len(shapes), len(configs))
            times = np.asarray([measure(r, 'graph') for r in ordered]).reshape(dims)
            train, val, test = (
                np.array([i for i, shape in enumerate(shapes) if assignment[shape] == role])
                for role in ('train', 'validation', 'test')
            )
            if not all(map(len, (train, val, test))) or len(train) + len(val) + len(test) != len(shapes):
                raise ValueError(f'{split} does not partition {family} into three nonempty shape sets')
            geom, resource, _ = features_for_family(ordered)
            geom, resource = (center(np.log1p(x).reshape((*dims, -1))) for x in (geom, resource))
            for target_idx, device in enumerate(DEVICES):
                sources = [i for i in range(len(DEVICES)) if i != target_idx]
                uniform = np.full(len(sources), 1. / len(sources))
                bank = source_scenarios(times, shapes, sources, train, test)
                train_g = geom[sources][:, train]
                train_r = resource[sources][:, train]
                target_g = geom[target_idx, test]
                target_r = resource[target_idx, test]
                neutral_r = resource[sources][:, test].mean(axis=0)
                records = defaultdict(list)
                for shape_idx, cases, geometry, resources, neutral in zip(
                    test, bank, target_g, target_r, neutral_r
                ):
                    labels = times[target_idx, shape_idx]
                    anchor = ShortlistSurrogate(cases, uniform, adaptive=False).next_candidate()
                    logs = center(cases.reshape(-1, len(configs), 1)).squeeze(-1)
                    prior = logs.mean(axis=0)
                    variation = logs - prior
                    empirical_covariance = variation.T @ variation / len(variation)
                    amplitude = np.clip(np.sqrt(empirical_covariance.diagonal().mean()), .10, .45)
                    nugget = (.25 * amplitude) ** 2 * np.eye(len(prior))
                    distances_g = feature_distances(train_g, geometry)
                    distances_r = feature_distances(train_r, resources)
                    distances_neutral = feature_distances(train_r, neutral)
                    cov_g = rbf(distances_g, amplitude) + nugget
                    cov_r = rbf(.5 * (distances_g + distances_r), amplitude) + nugget
                    cov_neutral = rbf(.5 * (distances_g + distances_neutral), amplitude) + nugget
                    cov_empirical = .7 * empirical_covariance + .3 * cov_g + nugget
                    orders = {
                        'scenario_static': replay(cases, uniform, labels, False, anchor),
                        'scenario_online': replay(cases, uniform, labels, True, anchor),
                        'gp_static': conditional_order(prior, cov_g, labels, anchor, active=False),
                        'gp_geometry_open_loop': conditional_order(
                            prior, cov_g, labels, anchor, feedback=False),
                        'gp_hardware_open_loop': conditional_order(
                            prior, cov_r, labels, anchor, feedback=False),
                        'gp_geometry_online': conditional_order(prior, cov_g, labels, anchor),
                        'gp_neutral_online': conditional_order(prior, cov_neutral, labels, anchor),
                        'gp_hardware_online': conditional_order(prior, cov_r, labels, anchor),
                        'gp_empirical_online': conditional_order(prior, cov_empirical, labels, anchor),
                    }
                    for policy, order in orders.items():
                        score = metric_for_order(order, labels)
                        score['shape'] = shapes[shape_idx]
                        score['order'] = order
                        records[policy].append(score)
                scores[family, device] = dict(candidates=len(configs), test_shapes=len(test),
                                              metrics=records)
            print(split, family, flush=True)
        result['splits'][split] = {
            'macro': average_metrics(list(scores.values()), non_exhausted=True),
            'macro_all': average_metrics(list(scores.values())),
            'family_counts': {
                family: {key: scores[family, DEVICES[0]][key] for key in ('candidates', 'test_shapes')}
                for family in FAMILIES
            },
            'families': {family: average_metrics([scores[family, d] for d in DEVICES])
                         for family in FAMILIES},
            'device_metrics': {device: average_metrics([scores[f, device] for f in FAMILIES],
                                                       non_exhausted=True)
                               for device in DEVICES},
            'paired': paired_summary(scores),
            'paired_families': {family: paired_summary(scores, (family,)) for family in FAMILIES},
        }
        if args.include_task_scores:
            result['splits'][split]['scores'] = {
                f'{family}:{device}': record for (family, device), record in scores.items()
            }
    save_json(args.output, result)


if __name__ == '__main__':
    main()
