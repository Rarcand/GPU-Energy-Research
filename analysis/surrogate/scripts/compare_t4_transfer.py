"""FP16-only transfer to a held-out Colab T4, with separate timing modes."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from compare_ranker import BUDGETS, METHODS, ranker_fit, summarize, task_groups
from compare_surrogates import ARCH, fit, matrices, normalized, save_json, scores
from gather_timing import FAMILY_COUNTS, FEATURES, KEYS, checked_rows, sha
from test_unseen_a5000 import load_a5000
from train_matched import training_rows, tree_fit

SOURCE_DEVICES = ('H100', 'RTX3070-WSL', 'RTXA5000')


def read_t4(path, peer, mode):
    expected = {k for k, r in peer.items() if r['dtype'] == 'float16'}
    rows, hardware = {}, None
    with zipfile.ZipFile(path) as export:
        files = {Path(n).name: n for n in export.namelist() if not n.endswith('/')}
        receipts = sorted(n for n in files if n.endswith('.complete.json'))
        status = json.loads(export.read(files['status.json']))
        if (len(receipts) != 70 or status != {'phase': 'complete', 'expected_rows': 15000}):
            raise ValueError('Expected a complete 70-shard T4 production export')
        for name in receipts:
            label = name.removesuffix('.complete.json')
            receipt = json.loads(export.read(files[name]))
            data = export.read(files[label + '.zip'])
            if sha(data) != receipt['archive_sha256']:
                raise ValueError(f'T4 archive hash mismatch: {label}')
            with zipfile.ZipFile(io.BytesIO(data)) as unit:
                protocol = json.loads(unit.read(label + '.protocol.json'))
                summary = json.loads(unit.read(label + '.summary.json'))
                raw = unit.read(label + '.jsonl')
            arch = protocol['architecture']
            if hardware is None:
                hardware = arch
            if ((protocol['device'], protocol['torch'], protocol['triton'], protocol['pass_index'])
                    != ('T4', '2.6.0+cu124', '3.2.0', 0)
                    or any(arch.get(k) != hardware.get(k) for k in ('gpu_name', *ARCH))
                    or summary['complete'] is not True
                    or summary['rows'] != len(raw.splitlines())
                    or receipt['rows'] != summary['rows']
                    or sha(raw) != summary['output_sha256']
                    or receipt['output_sha256'] != summary['output_sha256']):
                raise ValueError(f'T4 protocol or checkpoint mismatch: {label}')
            for row in map(json.loads, raw.splitlines()):
                key = tuple(row[name] for name in KEYS)
                if (key not in expected or key in rows
                        or any(row[field] != peer[key][field] for field in FEATURES)
                        or row['timing_status'] != 'ok' or not row['correctness_pass']):
                    raise ValueError(f'Invalid or unmatched T4 row: {key}')
                record = normalized({**arch, **row, 'latency_median_ms': row[mode]['median_ms'],
                                     'is_valid': True})
                record['_device'] = 'T4'
                rows[key] = record
    if rows.keys() != expected or Counter(k[0] for k in rows) != {
            k: count // 2 for k, count in FAMILY_COUNTS.items()}:
        raise ValueError('T4 candidate keys differ from the planned FP16 grid')
    return [rows[key] for key in sorted(rows)], hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--t4', required=True, type=Path)
    parser.add_argument('--split', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('graph', 'individual'))
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=(2026, 2027, 2028))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    metadata = json.loads((args.inputs / 'hardware.json').read_text())
    source_rows, hashes = training_rows(args.inputs, args.mode, metadata)
    h100 = args.inputs / 'h100.zip'
    a5000 = args.inputs / 'a5000-pass00-raw.tar.gz'
    a5000_rows, a5000_arch = load_a5000(
        a5000, args.inputs / 'a5000-pass00-verified.json', h100, mode=args.mode)
    t4_rows, t4_arch = read_t4(args.t4, checked_rows(h100), args.mode)
    rows = [r for r in source_rows + a5000_rows if r['dtype'] == 'float16'] + t4_rows

    assignment = json.loads(args.split.read_text())['assignment']
    roles = ('train', 'validation', 'test')
    indices = {
        role: np.asarray([i for i, r in enumerate(rows)
                          if (r['_device'] == 'T4') == (role == 'test')
                          and assignment[r['_group']] == role], dtype=np.int64)
        for role in roles
    }
    test = indices['test']
    counts = {role: dict(Counter(rows[i]['_device'] for i in ids))
              for role, ids in indices.items()}
    if (len(test) != 2992 or len(task_groups(rows, test)) != 199
            or any(set(counts[role]) != set(SOURCE_DEVICES) for role in roles[:2])):
        raise ValueError(f'Unexpected T4 transfer split: {counts}')
    y = np.asarray([r['_y'] for r in rows])
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Found an invalid timing label')
    metadata['RTXA5000'] = {'architecture': a5000_arch}
    metadata['T4'] = {'architecture': t4_arch, 'torch': '2.6.0+cu124', 'triton': '3.2.0'}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'protocol.json', dict(
        mode=args.mode, dtype='float16', target='T4', target_calibration_rows=0,
        source_devices=SOURCE_DEVICES,
        rows=counts, test_tasks=199, seeds=args.seeds, budgets=BUDGETS, epochs=args.epochs,
        passes={'H100': 3, 'RTX3070-WSL': 3, 'RTXA5000': 1, 'T4': 1},
        software_shift='T4 compiler stack differs from the source devices',
        model_versions={'torch': str(torch.__version__), 'numpy': np.__version__},
        split_sha256=sha(args.split.read_bytes()), hardware=metadata,
        archive_sha256={**hashes, str(a5000): sha(a5000.read_bytes()),
                        str(args.t4): sha(args.t4.read_bytes())},
    ))
    print(f'Verified {args.mode} FP16 split: {counts}', flush=True)
    arrays = matrices(rows)
    runs = []
    for seed in args.seeds:
        for method in METHODS:
            print(f'Training {args.mode} {method} seed {seed}', flush=True)
            if method == 'flat_ranker':
                _, predictions, info = ranker_fit(rows, arrays, y, indices, seed, args)
            elif method == 'boosted_tree':
                _, predictions, info = tree_fit(arrays, y, indices, seed, args.epochs)
            else:
                _, predictions, info = fit(arrays, y, indices['train'], indices['validation'],
                                           method, seed, args)
            tasks = scores(rows, test, predictions, budgets=BUDGETS)
            if method == 'flat_ranker':
                for task in tasks:
                    task.pop('mape_pct')
                    task.pop('mae_us')
            run = dict(seed=seed, method=method, target='T4', training=info, tasks=tasks)
            runs.append(run)
            save_json(args.output / f'{method}-{seed}.metrics.json', run)
            save_json(args.output / 'summary.json', summarize(runs))
            print(f'Completed {args.mode} {method} seed {seed}', flush=True)
    save_json(args.output / 'status.json', dict(complete=True, model_fits=len(runs)))


if __name__ == '__main__':
    main()
