"""Strict, zero-shot leave-one-GPU-out comparison on graph-replay labels."""

import argparse
import copy
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from train_matched import SEEDS, training_rows, tree_fit
from test_unseen_a5000 import load_a5000, sha256
from compare_surrogates import Network, Processor, fit, matrices, save_json, scores

DEVICES = ('H100', 'RTX3070-WSL', 'RTXA5000')
METHODS = ('flat_ranker', 'flat', 'boosted_tree', 'three')
BUDGETS = (1, 4, 8)
RANK_METRICS = ('spearman', *(f'{metric}_b{b}' for b in BUDGETS
                             for metric in ('success5', 'regret')))


def task_groups(rows, indices):
    groups = defaultdict(list)
    for index in indices:
        row = rows[index]
        groups[row['_device'], row['kernel_family'], row['_task']].append(index)
    return {key: np.asarray(values, dtype=np.int64) for key, values in sorted(groups.items())}


def make_pairs(y, groups, tie_margin=.01):
    """Sample only within the same source GPU, workload and dtype."""
    result = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        local = y[indices]
        fastest, slower = np.nonzero(local[None, :] > (1 + tie_margin) * local[:, None])
        if not len(fastest):
            continue
        boundary = (local[fastest] <= 1.05 * local.min()) & (local[slower] > 1.05 * local.min())
        weights = 1. + boundary.astype(float)
        result.append((indices[fastest], indices[slower], weights / weights.sum()))
    return result


def validation_regret(labels, predicted, groups):
    families = defaultdict(list)
    for (device, family, _), ids in groups.items():
        actual = labels[ids]
        chosen = actual[np.argsort(predicted[ids], kind='stable')[:max(BUDGETS)]]
        regrets = [chosen[:b].min() / actual.min() - 1 for b in BUDGETS]
        families[device, family].append(float(np.mean(np.log1p(regrets))))
    return statistics.mean(statistics.mean(v) for v in families.values())


def raw_scores(model, x):
    model.eval()
    with torch.no_grad():
        parts = [model([a[i:i+2048] for a in x]) for i in range(0, len(x[0]), 2048)]
    result = torch.cat(parts).numpy()
    if not np.isfinite(result).all():
        raise ValueError('Non-finite ranking scores')
    return result


def ranker_fit(rows, arrays, y, indices, seed, args, *, kind='flat', model_factory=Network):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train, val = indices['train'], indices['validation']
    processor = Processor().fit(arrays, train, y)
    x = processor.transform(arrays)
    model = model_factory([a.shape[1] for a in arrays], kind)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_groups = task_groups(rows, train)
    pairs = make_pairs(y, train_groups)
    if not pairs:
        raise ValueError('No distinguishable source pairs')
    # Local validation arrays avoid evaluating any target row during training.
    x_val = [a[val] for a in x]
    val_rows = [rows[i] for i in val]
    val_groups = task_groups(val_rows, np.arange(len(val)))
    best, best_epoch, best_value, stale = None, 0, float('inf'), 0
    pairs_per_task = 12
    for epoch in range(1, args.epochs+1):
        model.train()
        sampled = [rng.choice(len(fast), size=pairs_per_task, p=w) for fast, _, w in pairs]
        fast = np.concatenate([f[sel] for (f, _, _), sel in zip(pairs, sampled)])
        slow = np.concatenate([s[sel] for (_, s, _), sel in zip(pairs, sampled)])
        order = rng.permutation(len(fast))
        for start in range(0, len(order), args.batch_size):
            sample = order[start:start+args.batch_size]
            ids = np.concatenate((fast[sample], slow[sample]))
            opt.zero_grad(set_to_none=True)
            good, bad = model([a[ids] for a in x]).chunk(2)
            loss = F.softplus(good - bad).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
        value = validation_regret(y[val], raw_scores(model, x_val), val_groups)
        if value < best_value:
            best, best_epoch, best_value, stale = copy.deepcopy(model.state_dict()), epoch, value, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best is None:
        raise ValueError('Ranker produced no valid checkpoint')
    model.load_state_dict(best)
    return (model, processor), raw_scores(model, x), dict(
        best_epoch=best_epoch, epochs_run=epoch, source_tasks=len(pairs),
        validation_mean_log_regret=best_value, pairs_per_task_per_epoch=pairs_per_task,
        parameters=sum(p.numel() for p in model.parameters()),
    )


def summarize(runs, methods=METHODS):
    metrics = (*RANK_METRICS, 'mape_pct', 'mae_us')
    result = {}
    for method in methods:
        selected = [run for run in runs if run['method'] == method]
        if not selected:
            continue
        result[method] = {}
        for metric in metrics:
            values = []
            for run in selected:
                groups = defaultdict(list)
                for t in run['tasks']:
                    value = t.get(metric)
                    if value is not None:
                        groups[t['family']].append(value)
                if groups:
                    values.append(statistics.mean(statistics.mean(v) for v in groups.values()))
            if values:
                result[method][metric] = dict(
                    mean=statistics.mean(values),
                    seed_sd=statistics.stdev(values) if len(values) > 1 else None,
                    per_seed=values,
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=DEVICES, required=True)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.inputs/'hardware.json').read_text())
    rows, input_hashes = training_rows(args.inputs, 'graph', metadata)
    a5000_path = args.inputs/'a5000-pass00-raw.tar.gz'
    a5000, a5000_device = load_a5000(a5000_path, args.inputs/'a5000-pass00-verified.json',
                                    args.inputs/'h100.zip')
    rows.extend(a5000)
    input_hashes[str(a5000_path)] = sha256(a5000_path.read_bytes())
    metadata['RTXA5000'] = {'architecture': a5000_device, 'passes': 1}
    split = json.loads(args.split.read_text())['assignment']
    indices = {role: np.asarray([i for i, row in enumerate(rows)
                                 if split[row['_group']] == role
                                 and (row['_device'] == args.target) == (role == 'test')])
               for role in ('train', 'validation', 'test')}
    if (len(indices['test']) != 5984 or not all(len(ix) for ix in indices.values())
            or {rows[i]['_device'] for i in indices['train']} != set(DEVICES)-{args.target}):
        raise ValueError('Incomplete strict device/shape split')
    test_groups = task_groups(rows, indices['test'])
    if len(test_groups) != 398:
        raise ValueError('Unexpected held-out workload/dtype count')
    y = np.asarray([r['_y'] for r in rows])
    arrays = matrices(rows)
    save_json(args.output/'protocol.json', dict(
        target=args.target, target_calibration_rows=0, graph_timing_passes={
            'H100': 3, 'RTX3070-WSL': 3, 'RTXA5000': 1},
        data_sha256=input_hashes, split_sha256=sha256(args.split.read_bytes()),
        code_sha256=sha256(Path(__file__).read_bytes()),
        device_features=metadata, seeds=list(args.seeds), budgets=list(BUDGETS),
        train_val_test_rows={role: dict(Counter(rows[i]['_device'] for i in ids))
                             for role, ids in indices.items()},
        ranker='flat MLP, pairwise logistic loss, source task balanced, ignoring gaps <=1% and doubling sampling across the 5%-of-best boundary',
        checkpoint_selection={'flat_ranker': 'source validation mean log regret at budgets 1, 4, 8',
                              'regressors': 'source validation MAPE'},
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
    ))
    print('Verified target', args.target, 'with', len(test_groups), 'test tasks', flush=True)
    runs = []
    for seed in args.seeds:
        for method in METHODS:
            print('Training', args.target, seed, method, flush=True)
            if method == 'flat_ranker':
                (model, processor), pred, info = ranker_fit(rows, arrays, y, indices, seed, args)
            elif method == 'boosted_tree':
                _, pred, info = tree_fit(arrays, y, indices, seed, args.epochs)
            else:
                _, pred, info = fit(arrays, y, indices['train'], indices['validation'], method, seed, args)
            if method == 'flat_ranker':
                torch.save(dict(weights=model.state_dict(), processor=vars(processor),
                                branch_sizes=[a.shape[1] for a in arrays], output='lower_is_better'),
                           args.output/f'{method}-{seed}.pt')
            tasks = scores(rows, indices['test'], pred, budgets=BUDGETS)
            if method == 'flat_ranker':
                for task in tasks:
                    task.pop('mape_pct')
                    task.pop('mae_us')
            run = dict(seed=seed, method=method, target=args.target, training=info, tasks=tasks)
            runs.append(run)
            save_json(args.output/f'{method}-{seed}.metrics.json', run)
            with (args.output/f'{method}-{seed}.predictions.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                score_name = 'ranking_score' if method == 'flat_ranker' else 'prediction_ms'
                writer.writerow(('workload', 'configuration', 'graph_ms', score_name))
                for i in indices['test']:
                    writer.writerow((rows[i]['workload_id'], rows[i]['config_id'], y[i], float(pred[i])))
            save_json(args.output/'summary.json', summarize(runs))
            print('Completed', args.target, seed, method, info, flush=True)
    save_json(args.output/'status.json', dict(complete=True, target=args.target, model_fits=len(runs)))


if __name__ == '__main__':
    main()
