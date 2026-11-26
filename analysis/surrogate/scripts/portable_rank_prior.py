"""Evaluate source-only shape-local configuration ranks on a matched FP16 grid."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import rankdata

from compare_ranker import BUDGETS, DEVICES as SOURCE_DEVICES, task_groups
from compare_surrogates import FAMILIES, save_json, split_groups
from gather_timing import sha


NEIGHBORS = 5
LOCALITY_PENALTY = 2.0
REDUCTIONS = frozenset(('softmax', 'cross_entropy'))
METHODS = ('family_rank_prior', 'shape_local_rank_prior', 'nearest_shape_slowdown_prior')
DEVICES = (*SOURCE_DEVICES, 'T4')


def config_key(row):
    if row['kernel_family'] in REDUCTIONS:
        # Absolute block size derives from reduction width, which varies by workload.
        return (row['block_multiplier'], row['num_warps'], row['num_stages'])
    return row['config_id']


def measurements_by_task(rows):
    measured = defaultdict(dict)
    shapes_by_family = defaultdict(set)
    for row in rows:
        family, group = row['kernel_family'], row['_group']
        values = measured[row['_device'], group]
        key = config_key(row)
        label = row['_y']
        if key in values or not math.isfinite(label) or label <= 0:
            raise ValueError(f'Invalid or duplicate label for {row["_device"]}, {group}, {key}')
        values[key] = label
        shapes_by_family[family].add(group)
    return measured, shapes_by_family


def shape_distances(train, test):
    def descriptors(group):
        return json.loads(group.split(':', 1)[1])

    train_specs = [descriptors(shape) for shape in train]
    test_specs = [descriptors(shape) for shape in test]
    fields = sorted(set().union(*(d.keys() for d in train_specs)) - {'locality'})

    def numeric(specs):
        return np.asarray([[math.log1p(float(d.get(f, 0))) for f in fields] for d in specs])

    train_x, test_x = numeric(train_specs), numeric(test_specs)
    spread = train_x.std(axis=0)
    scale = np.where(spread < .1, 1., spread)
    distances = np.linalg.norm((test_x[:, None] - train_x[None, :]) / scale, axis=2)
    locality_match = np.array([[source.get('locality') == target.get('locality')
                                for source in train_specs] for target in test_specs])
    return distances + LOCALITY_PENALTY * (~locality_match)


def aligned_labels(values, keys):
    if values.keys() != set(keys):
        raise ValueError('A portable configuration key was missing from the matched registry')
    return np.asarray([values[key] for key in keys])


def score_task(labels, rankings):
    order = np.argsort(rankings, kind='stable')
    oracle = labels.min()
    result = {}
    for b in BUDGETS:
        best = float(labels[order[:b]].min())
        result[f'success5_b{b}'] = float(best <= 1.05 * oracle)
        result[f'regret_b{b}'] = best / float(oracle) - 1.
    return result


def family_rankings(measured, sources, train_shapes, test_shapes, keys):
    source_labels = np.asarray([
        [aligned_labels(measured[device, g], keys) for device in sources]
        for g in train_shapes
    ])
    ranks = rankdata(source_labels, axis=2).mean(axis=1)
    slowdown = (source_labels / source_labels.min(axis=2, keepdims=True)).mean(axis=1)
    nearest = np.argsort(shape_distances(train_shapes, test_shapes), axis=1,
                         kind='stable')[:, :NEIGHBORS]
    return {
        'family_rank_prior': np.broadcast_to(ranks.mean(axis=0), (len(test_shapes), len(keys))),
        'shape_local_rank_prior': ranks[nearest].mean(axis=1),
        'nearest_shape_slowdown_prior': slowdown[nearest[:, 0]],
    }


def predict_rank_prior(rows, train_ids, predict_ids):
    """Return deterministic, lower-is-better shape-local scores in prediction order."""
    training = [rows[i] for i in train_ids]
    measured, shapes = measurements_by_task(training)
    devices = sorted({r['_device'] for r in training})
    destinations = defaultdict(lambda: defaultdict(list))
    for j, i in enumerate(predict_ids):
        row = rows[i]
        destinations[row['kernel_family']][row['_group']].append((j, config_key(row)))
    predictions = np.empty(len(predict_ids), dtype=float)
    for family, groups in destinations.items():
        train_shapes, test_shapes = sorted(shapes[family]), sorted(groups)
        if len(train_shapes) < NEIGHBORS or (set(train_shapes) & set(test_shapes)):
            raise ValueError(f'Expected a held-out shape split for {family}')
        keys = sorted(measured[devices[0], train_shapes[0]])
        key_to_index = {key: index for index, key in enumerate(keys)}
        ranks = family_rankings(measured, devices, train_shapes, test_shapes, keys)
        for group, scores in zip(test_shapes, ranks['shape_local_rank_prior']):
            for index, key in groups[group]:
                predictions[index] = scores[key_to_index[key]]
    return predictions


def evaluate_fold(rows, measured, shapes_by_family, split, target):
    from compare_portable_surrogates import random_baseline

    sources = [device for device in DEVICES if device != target]
    families = {method: {} for method in METHODS}
    non_exhausted = {method: {} for method in METHODS}
    task_scores = {method: [] for method in METHODS}
    tasks = {}
    for family in FAMILIES:
        train_shapes = sorted(g for g in shapes_by_family[family] if split[g] == 'train')
        test_shapes = sorted(g for g in shapes_by_family[family] if split[g] == 'test')
        if len(train_shapes) < NEIGHBORS or not test_shapes:
            raise ValueError(f'Insufficient source train or target test shapes for {family}')
        keys = sorted(measured[sources[0], train_shapes[0]])
        rankings = family_rankings(measured, sources, train_shapes, test_shapes, keys)
        tasks[family] = {'training_shapes_per_source': len(train_shapes),
                         'test_shapes': len(test_shapes), 'candidates_per_shape': len(keys)}
        actual = [aligned_labels(measured[target, g], keys) for g in test_shapes]
        for method in METHODS:
            scores = [score_task(y, p) for y, p in zip(actual, rankings[method])]
            task_scores[method].extend({'group': group, 'family': family, 'attempts': len(keys),
                                        **score} for group, score in zip(test_shapes, scores))
            metrics = {key: mean(score[key] for score in scores) for key in scores[0]}
            families[method][family] = metrics
            non_exhausted[method][family] = {
                f'{metric}_b{b}': metrics[f'{metric}_b{b}']
                for b in BUDGETS if len(keys) > b for metric in ('success5', 'regret')
            }

    test_rows = [r for r in rows if r['_device'] == target and split[r['_group']] == 'test']
    random = random_baseline(np.asarray([r['_y'] for r in test_rows]),
                             task_groups(test_rows, np.arange(len(test_rows))))
    methods = {}
    for method, by_family in families.items():
        keys = next(iter(by_family.values())).keys()
        subset = {key: [metrics[key] for metrics in non_exhausted[method].values() if key in metrics]
                  for key in keys}
        methods[method] = {'macro': {key: mean(f[key] for f in by_family.values()) for key in keys},
                           'non_exhausted_macro': {key: mean(values) for key, values in subset.items()
                                                   if values},
                           'by_family': by_family, 'task_scores': task_scores[method]}
    return {'target': target, 'sources': sources, 'test_rows': len(test_rows), 'tasks': tasks,
            'methods': methods, 'random': random}


def evaluate_split(rows, measured, shapes_by_family, split):
    if set(split) != set().union(*shapes_by_family.values()):
        raise ValueError('Shape assignment does not match the verified grid')
    folds = {device: evaluate_fold(rows, measured, shapes_by_family, split, device)
             for device in DEVICES}
    example = next(iter(folds.values()))
    equal_gpu = {'random': {key: mean(fold['random'][key] for fold in folds.values())
                            for key in example['random']}}
    for method in METHODS:
        equal_gpu[method] = {
            metric_set: {key: mean(fold['methods'][method][metric_set][key] for fold in folds.values())
                         for key in example['methods'][method][metric_set]}
            for metric_set in ('macro', 'non_exhausted_macro')
        }
    return {'folds': folds, 'equal_gpu_mean': equal_gpu}


def main():
    from compare_portable_surrogates import load_matched

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sensitivity-seed', type=int)
    args = parser.parse_args()
    rows, _, hashes = load_matched(args.inputs, args.t4, args.mode)
    measured, shapes_by_family = measurements_by_task(rows)
    split = json.loads(args.split.read_text())['assignment']
    results = {'registered': evaluate_split(rows, measured, shapes_by_family, split)}
    if args.sensitivity_seed is not None:
        alternate = split_groups(rows, seed=args.sensitivity_seed)
        results[f'seed_{args.sensitivity_seed}'] = evaluate_split(rows, measured, shapes_by_family,
                                                                 alternate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {
        'protocol': {'timing_mode': args.mode, 'dtype': 'float16', 'number_of_neighbors': NEIGHBORS,
                     'locality_mismatch_penalty': LOCALITY_PENALTY, 'target_calibration_rows': 0,
                     'budgets': BUDGETS, 'registered_split_sha256': sha(args.split.read_bytes()),
                     'sensitivity_seed': args.sensitivity_seed, 'archive_sha256': hashes,
                     'selection': 'Mean of source-device latency ranks across same-family train shapes. '
                                  'Shape-local variant retains five nearest shapes using source-scaled '
                                  'log dimensions and a penalty for mismatched gather locality. '
                                  'The nearest-shape slowdown variant uses the nearest training shape '
                                  'and averages latency normalized by each source device\'s optimum.'},
        'splits': results,
    })
    for name, result in results.items():
        values = result['equal_gpu_mean']['shape_local_rank_prior']['macro']
        print(args.mode, name, [round(100 * values[f'success5_b{b}'], 2) for b in BUDGETS])


if __name__ == '__main__':
    main()
