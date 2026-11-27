"""Zero-shot A5000 evaluation of frozen H100/RTX 3070 graph-replay surrogates."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

import joblib
import numpy as np
import torch

from analysis.surrogate.scripts.features.gather_timing import FAMILY_COUNTS, checked_rows
from analysis.surrogate.scripts.models.train_matched import MODELS, SEEDS, summarize
from analysis.surrogate.scripts.models.compare_surrogates import ARCH, Network, Processor, matrices, normalized, predict, scores, save_json


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_a5000(archive, audit, peers, mode='graph'):
    evidence = json.loads(audit.read_text())
    if evidence['archive_sha256'] != sha256(archive.read_bytes()):
        raise ValueError('A5000 archive hash differs from the desktop export audit')
    peer = checked_rows(peers)
    rows = {}
    hardware = None
    with tarfile.open(archive) as a5000:
        names = sorted(member.name.removesuffix('.summary.json') for member in a5000.getmembers()
                       if member.name.startswith('published/') and member.name.endswith('.summary.json'))
        if len(names) != 70 or set(names) != set(evidence['shard_hashes']):
            raise ValueError('Expected the 70 verified A5000 shards')
        for name in names:
            metadata = json.load(a5000.extractfile(name+'.protocol.json'))
            if (metadata['device'], metadata['torch'], metadata['triton']) != (
                    'RTXA5000', '2.8.0+cu128', '3.4.0'):
                raise ValueError(f'Unexpected A5000 protocol: {name}')
            architecture = metadata['architecture']
            if hardware is None:
                hardware = architecture
            elif any(hardware.get(key) != architecture.get(key) for key in ('gpu_name', *ARCH)):
                raise ValueError(f'Architecture metadata changed: {name}')
            raw = a5000.extractfile(name+'.jsonl').read()
            if sha256(raw) != evidence['shard_hashes'][name]:
                raise ValueError(f'Published A5000 shard checksum mismatch: {name}')
            for entry in map(json.loads, raw.splitlines()):
                key = tuple(entry[field] for field in ('kernel_family', 'workload_id', 'config_id'))
                label = entry[mode]['median_ms']
                if (key in rows or not entry['correctness_pass'] or entry['timing_status'] != 'ok'
                        or not np.isfinite(label) or label <= 0
                        or entry['input_sha256'] != peer[key]['input_sha256']):
                    raise ValueError(f'Invalid or unmatched A5000 candidate: {key}')
                row = normalized({**architecture, **entry, 'latency_median_ms': label, 'is_valid': True})
                row['_device'] = 'RTXA5000'
                rows[key] = row
    if rows.keys() != peer.keys() or len(rows) != sum(FAMILY_COUNTS.values()):
        raise ValueError('A5000 candidate grid differs from training GPU grid')
    return [rows[k] for k in sorted(rows)], hardware


def paired_intervals(runs, metric, seed=9402, repetitions=2000):
    """Paired, family-stratified bootstrap over seeds and workload shapes."""
    index = {(r['method'], r['seed']): {(t['family'], t['group'], t['task']): t[metric]
                                           for t in r['tasks']} for r in runs}
    keys = sorted(index['three', SEEDS[0]])
    grouped = defaultdict(lambda: defaultdict(list))
    for position, (family, shape, _) in enumerate(keys):
        grouped[family][shape].append(position)
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(len(SEEDS), size=(repetitions, len(SEEDS)))
    by_family = [list(groups.values()) for groups in grouped.values()]
    shape_draws = [rng.integers(len(groups), size=(repetitions, len(groups))) for groups in by_family]
    output = {}
    for baseline in MODELS[1:]:
        delta = np.array([[index['three', s][k]-index[baseline, s][k] for k in keys] for s in SEEDS])
        sampled = np.zeros(repetitions)
        for groups, draws in zip(by_family, shape_draws):
            group_sum = np.column_stack([delta[:, positions].sum(axis=1) for positions in groups])
            group_size = np.array([len(positions) for positions in groups])
            sampled += (group_sum[seed_draws[:, :, None], draws[:, None, :]].sum(axis=2)
                        / group_size[draws].sum(axis=1)[:, None]).mean(axis=1) / len(by_family)
        output[baseline] = dict(ci95=np.quantile(sampled, [.025, .975]).tolist(),
                                bootstrap_repetitions=repetitions, workload_shapes=sum(map(len, by_family)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('models', type=Path)
    parser.add_argument('split', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((a.models/'protocol.json').read_text())
    status = json.loads((a.models/'status.json').read_text())
    if (protocol['mode'] != 'graph' or not status['complete'] or status['model_fits'] != 15
            or protocol['split_sha256'] != sha256(a.split.read_bytes())
            or set(protocol['hardware']) != {'H100', 'RTX3070-WSL'}):
        raise ValueError('Expected complete frozen graph-replay fits on H100 and 3070')
    archive = a.data/'a5000-pass00-raw.tar.gz'
    rows, hardware = load_a5000(archive, a.data/'a5000-pass00-verified.json', a.data/'h100.zip')
    split = json.loads(a.split.read_text())['assignment']
    test = np.asarray([i for i, row in enumerate(rows) if split[row['_group']] == 'test'])
    if len(test)*2 != protocol['sizes']['test']['rows']:
        raise ValueError('Strict test coverage differs from the training GPU test coverage')
    arrays = matrices(rows)
    outcomes, digests = {'strict': [], 'all_a5000_shapes': []}, {}
    for seed in SEEDS:
        for method in MODELS:
            path = a.models/f'{method}-{seed}.{"joblib" if method == "boosted_tree" else "pt"}'
            digests[path.name] = sha256(path.read_bytes())
            if method == 'boosted_tree':
                model, processor = joblib.load(path)
                merged = np.column_stack([p.numpy() for p in processor.transform(arrays)])
                pred = np.exp(model.predict(merged))
            else:
                # The prior run created and stores these local research checkpoints.
                state = torch.load(path, map_location='cpu', weights_only=False)
                if state['method'] != method or state['branch_sizes'] != [x.shape[1] for x in arrays]:
                    raise ValueError('Frozen model schema mismatch')
                model = Network(state['branch_sizes'], method)
                model.load_state_dict(state['weights'])
                processor = Processor()
                vars(processor).update(state['processor'])
                pred = predict(model, processor.transform(arrays), processor)
            record = {'seed': seed, 'method': method}
            for label, indices in (('strict', test), ('all_a5000_shapes', np.arange(len(rows)))):
                outcomes[label].append({**record, 'tasks': scores(rows, indices, pred)})
            with (a.output/f'{method}-{seed}.predictions.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['workload', 'candidate', 'shape_role', 'graph_ms', 'prediction_ms'])
                for row, prediction in zip(rows, pred):
                    w.writerow([row['workload_id'], row['config_id'], split[row['_group']],
                                row['_y'], float(prediction)])
            print('Evaluated', method, seed, flush=True)

    report = dict(
        target='RTXA5000', target_label_passes=1, target_calibration_rows=0,
        hardware=hardware, source_hardware=protocol['hardware'], checkpoint_sha256=digests,
        a5000_archive_sha256=sha256(archive.read_bytes()), split_sha256=protocol['split_sha256'],
        slurm_job_id=os.getenv('SLURM_JOB_ID'),
    )
    for label, runs in outcomes.items():
        report[label] = dict(
            **summarize(runs), tasks=len(runs[0]['tasks']),
            paired_intervals={metric: paired_intervals(runs, metric)
                              for metric in ('success5_b1', 'success5_b4', 'mape_pct')},
        )
        save_json(a.output/(label+'.tasks.json'), runs)
    save_json(a.output/'summary.json', report)


if __name__ == '__main__':
    main()
