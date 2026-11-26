"""Matched FP16 leave-one-GPU-and-shape-out evaluation for portable surrogates."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from compare_ranker import BUDGETS, ranker_fit, summarize, task_groups, validation_regret
from compare_surrogates import fit, matrices, save_json, scores
from compare_t4_transfer import read_t4
from gather_timing import checked_rows, sha
from test_unseen_a5000 import load_a5000
from train_matched import training_rows, tree_fit

DEVICES = ('H100', 'RTX3070-WSL', 'RTXA5000', 'T4')
BASELINES = ('three', 'boosted_tree', 'flat_ranker', 'flat')
EXPERTS = ('three', 'boosted_tree', 'rank_tree')
CALIBRATION_ROWS = 56


def load_matched(inputs, t4, mode):
    metadata = json.loads((inputs / 'hardware.json').read_text())
    source, hashes = training_rows(inputs, mode, metadata)
    h100 = inputs / 'h100.zip'
    a5000 = inputs / 'a5000-pass00-raw.tar.gz'
    a5000_audit = inputs / 'a5000-pass00-verified.json'
    a5000_rows, a5000_arch = load_a5000(a5000, a5000_audit, h100, mode=mode)
    t4_rows, t4_arch = read_t4(t4, checked_rows(h100), mode)
    metadata['RTXA5000'] = {'architecture': a5000_arch, 'torch': '2.8.0+cu128',
                            'triton': '3.4.0'}
    metadata['T4'] = {'architecture': t4_arch, 'torch': '2.6.0+cu124', 'triton': '3.2.0'}
    rows = [r for r in source + a5000_rows if r['dtype'] == 'float16'] + t4_rows
    expected = {r['_key'] for r in t4_rows}
    for device in DEVICES:
        keys = [r['_key'] for r in rows if r['_device'] == device]
        if len(keys) != 15_000 or set(keys) != expected:
            raise ValueError(f'Mismatched FP16 candidates for {device}')
    hashes.update({str(p): sha(p.read_bytes()) for p in (a5000, a5000_audit, t4)})
    return rows, metadata, hashes


def percentile_ranks(predictions, groups):
    ranks = np.empty(len(predictions), dtype=float)
    for ids in groups.values():
        ranks[ids] = (rankdata(predictions[ids], method='average') - 1) / max(len(ids) - 1, 1)
    return ranks


def save_run(folder, method, seed, info, test_rows, predictions):
    if not np.isfinite(predictions).all():
        raise ValueError(f'Nonfinite predictions for {method}')
    tasks = scores(test_rows, np.arange(len(test_rows)), predictions, budgets=BUDGETS)
    if method not in ('three', 'boosted_tree', 'flat'):
        for task in tasks:
            task.pop('mape_pct')
            task.pop('mae_us')
    run = dict(seed=seed, method=method, target=test_rows[0]['_device'], training=info, tasks=tasks)
    save_json(folder / f'{method}-{seed}.metrics.json', run)
    return run


def calibration_weights(ranks, y, groups):
    """Calibrate one set of weights from 56 distinct target-side observations."""
    by_family = {}
    for (_, family, _), ids in groups.items():
        if len(ids) >= 4:
            by_family.setdefault(family, []).append(ids)
    rng = np.random.default_rng(57463)
    differences = []
    labels = 0
    for family in sorted(by_family):
        choices = by_family[family]
        for task in rng.choice(len(choices), size=2, replace=False):
            ids = choices[task]
            local = ranks[:, ids]
            preferred = np.argsort(local.mean(axis=0), kind='stable')[:2]
            disputed = np.argsort(-np.ptp(local, axis=0), kind='stable')
            selected = ids[list(dict.fromkeys(np.concatenate((preferred, disputed))))[:4]]
            measured = y[selected]
            labels += len(selected)
            good, bad = np.nonzero(measured[None, :] > 1.01 * measured[:, None])
            if len(good):
                differences.append((ranks[:, selected[good]] - ranks[:, selected[bad]]).T)
    if labels != CALIBRATION_ROWS:
        raise ValueError(f'Expected {CALIBRATION_ROWS} target calibration measurements, got {labels}')
    uniform = np.full(3, 1 / 3)
    if not differences:
        return uniform
    grid = [np.asarray((a, b, 3 - a - b)) / 3 for a in range(4) for b in range(4 - a)]

    def loss(weight):
        ranking = np.mean([np.logaddexp(0, 4 * (diff @ weight)).mean() for diff in differences])
        return ranking + .12 * np.square(weight - uniform).sum()

    return min(grid, key=loss)


def random_baseline(y, groups):
    from math import comb
    from statistics import mean
    by_family = {}
    for (_, family, _), ids in groups.items():
        actual = y[ids]
        good = int(np.sum(actual <= 1.05 * actual.min()))
        total = len(actual)
        by_family.setdefault(family, []).append([
            1 - comb(total - good, min(b, total)) / comb(total, min(b, total)) for b in BUDGETS
        ])
    return {f'success5_b{b}': mean(mean(v[i] for v in values) for values in by_family.values())
            for i, b in enumerate(BUDGETS)}


def fit_prior_blend(rows, arrays, y, indices, args):
    from portable_rank_prior import predict_rank_prior
    from portable_rank_tree import fit_rank_tree

    train, test = indices['train'], indices['test']
    prediction, tree_info, validation = fit_rank_tree(
        rows, arrays, y, indices, args.seeds[0], args, include_validation=True)
    weights = (0., .25, .5, .75, 1.)
    losses = {w: {} for w in weights}
    for device, (held_out, tree) in validation.items():
        available = np.asarray([i for i in train if rows[i]['_device'] != device])
        groups = task_groups([rows[i] for i in held_out], np.arange(len(held_out)))
        prior = percentile_ranks(predict_rank_prior(rows, available, held_out), groups)
        relative = percentile_ranks(tree, groups)
        for w in weights:
            losses[w][device] = validation_regret(y[held_out], w * prior + (1 - w) * relative,
                                                  groups)

    def risk(weight):
        values = tuple(losses[weight].values())
        return np.mean(values) + .5 * max(values), abs(weight - .5)

    chosen = min(weights, key=risk)
    groups = task_groups([rows[i] for i in test], np.arange(len(test)))
    prior = percentile_ranks(predict_rank_prior(rows, train, test), groups)
    relative = percentile_ranks(prediction[test], groups)
    return chosen * prior + (1 - chosen) * relative, dict(
        prior_weight=chosen, relative_tree=tree_info, source_validation_losses=losses,
        weight_selection='Mean source holdout log regret plus half the worst source holdout')


def fold(rows, arrays, y, split, target, args):
    folder = args.output / target
    folder.mkdir(parents=True, exist_ok=True)
    indices = {role: np.asarray([i for i, row in enumerate(rows)
                                if (row['_device'] == target) == (role == 'test')
                                and split[row['_group']] == role], dtype=np.int64)
               for role in ('train', 'validation', 'test')}
    counts = {role: dict(Counter(rows[i]['_device'] for i in ids))
              for role, ids in indices.items()}
    test = indices['test']
    test_rows = [rows[i] for i in test]
    test_groups = task_groups(test_rows, np.arange(len(test)))
    if (len(test) != 2992 or len(test_groups) != 199
            or any(set(counts[r]) != set(DEVICES) - {target} for r in ('train', 'validation'))):
        raise ValueError(f'Unexpected fold for {target}: {counts}')
    save_json(folder / 'fold.json', dict(counts=counts, test_tasks=len(test_groups),
                                       random=random_baseline(y[test], test_groups)))
    print(f'{args.mode} {target} {counts}', flush=True)
    if args.prior_blend_only:
        pred, info = fit_prior_blend(rows, arrays, y, indices, args)
        run = save_run(folder, 'validated_prior_blend', args.seeds[0], info, test_rows, pred)
        save_json(folder / 'summary.json', summarize([run], methods=('validated_prior_blend',)))
        save_json(folder / 'status.json', dict(complete=True, metric_runs=1))
        print(f'Finished {args.mode} {target} prior blend', flush=True)
        return
    calibration = np.asarray([i for i, row in enumerate(rows)
                              if row['_device'] == target and split[row['_group']] == 'validation'])
    calibration_groups = task_groups([rows[i] for i in calibration], np.arange(len(calibration)))
    from portable_rank_tree import fit_rank_tree
    print(f'Fit {args.mode} {target} rank_tree and boosted_tree', flush=True)
    rank_tree, rank_info = fit_rank_tree(rows, arrays, y, indices, args.seeds[0], args)
    _, latency_tree, tree_info = tree_fit(arrays, y, indices, args.seeds[0], args.epochs)
    for seed in args.seeds:
        save_run(folder, 'rank_tree', seed, rank_info, test_rows, rank_tree[test])
        save_run(folder, 'boosted_tree', seed, tree_info, test_rows, latency_tree[test])
        expert_predictions = {'rank_tree': rank_tree, 'boosted_tree': latency_tree}
        for method in ('three', 'flat_ranker', 'flat'):
            print(f'Fit {args.mode} {target} {method} {seed}', flush=True)
            if method == 'flat_ranker':
                _, pred, info = ranker_fit(rows, arrays, y, indices, seed, args)
            else:
                _, pred, info = fit(arrays, y, indices['train'], indices['validation'],
                                    method, seed, args)
            save_run(folder, method, seed, info, test_rows, pred[test])
            if method in EXPERTS:
                expert_predictions[method] = pred
        ranks = np.asarray([percentile_ranks(expert_predictions[key][test], test_groups)
                            for key in EXPERTS])
        save_run(folder, 'rank_consensus', seed, {'equal_weight': EXPERTS}, test_rows,
                 ranks.mean(axis=0))
        prior_ranks = np.asarray([
            percentile_ranks(expert_predictions[key][calibration], calibration_groups)
            for key in EXPERTS
        ])
        weights = calibration_weights(prior_ranks, y[calibration], calibration_groups)
        save_run(folder, 'calibrated_consensus', seed,
                 dict(weights=dict(zip(EXPERTS, weights)), target_calibration_rows=CALIBRATION_ROWS),
                 test_rows, weights @ ranks)

    runs = [json.loads(path.read_text()) for path in sorted(folder.glob('*.metrics.json'))]
    methods = sorted({run['method'] for run in runs})
    if set(methods) != set(BASELINES) | {'rank_tree', 'rank_consensus', 'calibrated_consensus'}:
        raise ValueError(f'Incomplete methods for {target}')
    save_json(folder / 'summary.json', summarize(runs, methods=methods))
    save_json(folder / 'status.json', dict(complete=True, metric_runs=len(runs)))
    print(f'Finished {args.mode} {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    parser.add_argument('--prior-blend-only', action='store_true')
    parser.add_argument('--targets', choices=DEVICES, nargs='+', default=DEVICES)
    parser.add_argument('--seeds', type=int, nargs='+', default=(2026, 2027, 2028))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows, hardware, hashes = load_matched(args.inputs, args.t4, args.mode)
    split = json.loads(args.split.read_text())['assignment']
    y = np.asarray([r['_y'] for r in rows])
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('The verified grid must have finite positive timings')
    arrays = matrices(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(timing_mode=args.mode, dtype='float16', seeds=list(args.seeds), epochs=args.epochs,
                    patience=args.patience, budgets=list(BUDGETS),
                    target_calibration_rows=(0 if args.prior_blend_only else
                                             {'calibrated_consensus': CALIBRATION_ROWS, 'other': 0}),
                    device_passes={'H100': 3, 'RTX3070-WSL': 3, 'RTXA5000': 1, 'T4': 1},
                    split_sha256=sha(args.split.read_bytes()), archive_sha256=hashes, hardware=hardware,
                    device_software_shift='T4 uses an older compiler stack',
                    deterministic_tree_seeds=[args.seeds[0]],
                    consensus='Equal-weight within-workload percentile ranks')
    path = args.output / 'protocol.json'
    if path.exists() and protocol != json.loads(path.read_text()):
        raise ValueError(f'Protocol mismatch at {args.output}')
    save_json(path, protocol)
    for target in args.targets:
        fold(rows, arrays, y, split, target, args)


if __name__ == '__main__':
    main()
