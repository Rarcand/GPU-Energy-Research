"""Compare three surrogate representations on a single verified latency target."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

from analysis.surrogate.scripts.features.gather_timing import checked_rows, sha

ROOT = Path(__file__).resolve().parents[4]
from analysis.surrogate.scripts.models.compare_surrogates import Processor, fit, matrices, normalized, save_json, scores

SEEDS = (2026, 2027, 2028, 2029, 2030)
MODELS = ('three', 'flat', 'boosted_tree')
METRICS = ('mape_pct', 'mae_us', 'spearman', 'success5_b1', 'success5_b4',
           'regret_b1', 'regret_b4')


def training_rows(inputs, mode, metadata):
    files = {d: inputs / f'{name}.zip' for d, name in (('H100', 'h100'), ('RTX3070-WSL', 'rtx3070'))}
    measured = {d: checked_rows(path) for d, path in files.items()}
    h100, rtx = measured['H100'], measured['RTX3070-WSL']
    if len(h100) != 30000 or h100.keys() != rtx.keys():
        raise ValueError('Require a matched, 30,000-candidate registry')
    if any(h100[k]['input_sha256'] != rtx[k]['input_sha256'] for k in h100):
        raise ValueError('Inputs differ across devices')
    rows = []
    for device, times in measured.items():
        arch = metadata[device]['architecture']
        seen = set()
        for key, record in sorted(times.items()):
            row = normalized({**arch, **record, 'is_valid': True,
                              'latency_median_ms': record[f'{mode}_median_ms']})
            row['_device'] = device
            if row['_key'] in seen:
                raise ValueError(f'Duplicate model candidate on {device}: {key}')
            seen.add(row['_key'])
            rows.append(row)
    if {r['_key'] for r in rows if r['_device'] == 'H100'} != {
            r['_key'] for r in rows if r['_device'] == 'RTX3070-WSL'}:
        raise ValueError('Model feature identities differ across devices')
    return rows, {str(path): sha(path.read_bytes()) for path in files.values()}


def tree_fit(arrays, y, indices, seed, epochs):
    processor = Processor().fit(arrays, indices['train'], y)
    x = np.column_stack([t.numpy() for t in processor.transform(arrays)])
    train, val = (indices[k] for k in ('train', 'validation'))
    best = None
    best_score = float('inf')
    for leaves, l2 in ((15, 1.), (31, 1.), (31, 10.)):
        estimator = HistGradientBoostingRegressor(
            max_iter=epochs, max_leaf_nodes=leaves, l2_regularization=l2,
            early_stopping=False, random_state=seed,
        )
        estimator.fit(x[train], np.log(y[train]))
        loss = float(np.mean(np.abs(np.exp(estimator.predict(x[val])) / y[val] - 1)))
        if loss < best_score:
            best, best_score = estimator, loss
    cfg = dict(leaves=best.max_leaf_nodes, l2=best.l2_regularization,
               iterations=epochs, validation_mape_pct=100 * best_score)
    return (best, processor), np.exp(best.predict(x)), cfg


def summarize(runs):
    def macro(tasks, metric):
        groups = defaultdict(list)
        for task in tasks:
            value = task[metric]
            if value is not None:
                groups[task['device'], task['family']].append(value)
        return statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None

    def aggregate(values):
        values = [value for value in values if value is not None]
        return (dict(mean=statistics.mean(values), seed_sd=statistics.stdev(values) if len(values)>1 else 0.)
                if values else None)

    output = {}
    for method in MODELS:
        selected = [r for r in runs if r['method'] == method]
        if not selected:
            continue
        split_tasks = {}
        for device in ('all', 'H100', 'RTX3070-WSL', *sorted({t['family'] for r in selected for t in r['tasks']})):
            split_tasks[device] = [
                [t for t in r['tasks'] if device in ('all', t['device'], t['family'])]
                for r in selected
            ]
        output[method] = dict(seeds=[r['seed'] for r in selected], summaries={
            device: {m: aggregate([macro(ts, m) for ts in tasks]) for m in METRICS}
            for device, tasks in split_tasks.items()
        })
    paired = {}
    for baseline in MODELS[1:]:
        baseline_runs = {r['seed']: r for r in runs if r['method'] == baseline}
        common = [(r, baseline_runs[r['seed']]) for r in runs
                  if r['method'] == 'three' and r['seed'] in baseline_runs]
        paired[baseline] = {m: aggregate([macro(x['tasks'], m)-macro(y['tasks'], m)
                                          for x, y in common
                                          if macro(x['tasks'], m) is not None and macro(y['tasks'], m) is not None])
                            for m in METRICS}
    return dict(models=output, paired_three_minus_baseline=paired)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('graph', 'individual'), required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--hardware', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.hardware.read_text())
    rows, input_digests = training_rows(args.inputs, args.mode, metadata)
    assignment = json.loads(args.split.read_text())['assignment']
    observed_groups = {r['_group'] for r in rows}
    if not observed_groups <= assignment.keys():
        raise ValueError(f'{len(observed_groups-assignment.keys())} workload shapes absent from frozen split')
    roles = ('train', 'validation', 'test')
    if not {assignment[group] for group in observed_groups} <= set(roles):
        raise ValueError('Unrecognized partition in frozen shape assignment')
    indices = {role: np.array([i for i, r in enumerate(rows) if assignment[r['_group']] == role])
               for role in roles}
    if any(not len(ix) for ix in indices.values()):
        raise ValueError('All three partitions must contain data')
    sizes = {role: dict(rows=len(ix), shapes=len({rows[i]['_group'] for i in ix}),
                         device_rows=dict(Counter(rows[i]['_device'] for i in ix)))
             for role, ix in indices.items()}
    y = np.asarray([r['_y'] for r in rows])
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Invalid training label')
    arrays = matrices(rows)
    save_json(args.output/'protocol.json', dict(
        mode=args.mode, labels='median of three correct pass medians',
        evaluation='held-out shapes on both devices; 5%-of-oracle selection at B=1 and B=4',
        input_archives_sha256=input_digests, hardware=metadata,
        split_sha256=sha(args.split.read_bytes()), sizes=sizes,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        versions=dict(torch=torch.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, python=platform.python_version()),
        code_sha256={str(f.relative_to(ROOT)): sha(f.read_bytes()) for f in
                     (Path(__file__).resolve(), Path(__file__).with_name('compare_surrogates.py').resolve(),
                      Path(__file__).resolve().parents[1] / 'features' / 'gather_timing.py')},
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
    ))
    print('Setup complete', args.mode, sizes, flush=True)
    runs = []
    for seed in args.seeds:
        for method in MODELS:
            print('Training', args.mode, seed, method, flush=True)
            if method == 'boosted_tree':
                (model, processor), predicted, info = tree_fit(arrays, y, indices, seed, args.epochs)
                joblib.dump((model, processor), args.output/f'{method}-{seed}.joblib')
            else:
                (model, processor), predicted, info = fit(arrays, y, indices['train'],
                                                          indices['validation'], method, seed, args)
                torch.save(dict(weights=model.state_dict(), processor=vars(processor),
                                branch_sizes=[a.shape[1] for a in arrays], method=method),
                           args.output/f'{method}-{seed}.pt')
            tasks = scores(rows, indices['test'], predicted)
            record = dict(mode=args.mode, seed=seed, method=method, training=info, tasks=tasks)
            runs.append(record)
            save_json(args.output/f'{method}-{seed}.metrics.json', record)
            with (args.output/f'{method}-{seed}.predictions.csv').open('w', newline='') as dst:
                writer = csv.writer(dst)
                writer.writerow(('device', 'workload_id', 'config_id', 'group', 'latency_ms', 'prediction_ms'))
                for i in indices['test']:
                    row = rows[i]
                    writer.writerow((row['_device'], row['workload_id'], row['config_id'],
                                     row['_group'], y[i], predicted[i]))
            save_json(args.output/'summary.json', summarize(runs))
            print('Completed', args.mode, seed, method, info, flush=True)
    save_json(args.output/'status.json', dict(complete=True, model_fits=len(runs)))
    print('Completed all comparisons', args.mode, flush=True)


if __name__ == '__main__':
    main()
