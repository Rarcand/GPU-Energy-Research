"""Export verified per-candidate timing medians and audit frozen surrogate rankings."""

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys
import zipfile

FAMILY_COUNTS = dict(attention=4000, conv2d=4992, cross_entropy=3408, gather=4000,
                     gemm=7200, residual_layernorm=2992, softmax=3408)
MODES = ('graph', 'individual')
KEYS = ('kernel_family', 'workload_id', 'config_id')
FEATURES = ('dtype', 'workload_features_json', 'config_features_json', 'input_sha256')
# Frozen candidate identity used by the November 18 model prediction exports.
SHAPE = ('m', 'n', 'k', 'rows', 'reduction_width', 'batch', 'height', 'in_channels',
         'out_channels', 'kernel', 'stride', 'heads', 'q_len', 'kv_len', 'head_dim',
         'causal', 'vocab', 'lookups', 'width')
CONFIG = ('block_m', 'block_n', 'block_k', 'group_m', 'block_size', 'block_multiplier',
          'block_rows', 'block_cols', 'num_warps', 'num_stages')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gather(root, device, output):
    passes = defaultdict(dict)
    manifest = {}
    for summary in sorted(root.rglob('*.summary.json')):
        if summary.parent.name not in ('pass-00', 'pass-01', 'pass-02'):
            continue
        raw = summary.with_name(summary.name.replace('.summary.json', '.jsonl'))
        protocol = raw.with_suffix('.protocol.json')
        summary_text, protocol_text = summary.read_bytes(), protocol.read_bytes()
        meta, settings = json.loads(summary_text), json.loads(protocol_text)
        round_number = int(summary.parent.name[-2:])
        data = raw.read_bytes()
        if (not meta['complete'] or sha(data) != meta['output_sha256']
                or settings['pass_index'] != round_number or settings['device'] != device):
            raise ValueError(f'Invalid published timing file: {raw}')
        records = data.splitlines()
        if len(records) != meta['rows']:
            raise ValueError(f'Row count mismatch: {raw}')
        for row in map(json.loads, records):
            key = tuple(row[name] for name in KEYS)
            if (key in passes[round_number] or row['pass_index'] != round_number
                    or row['timing_status'] != 'ok' or not row['correctness_pass']):
                raise ValueError(f'Duplicate or unverified row: {key}')
            times = tuple(row[mode]['median_ms'] for mode in MODES)
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError(f'Invalid latency: {key}')
            passes[round_number][key] = tuple(row[name] for name in FEATURES) + times
        manifest[str(raw.relative_to(root))] = dict(
            data_sha256=sha(data), summary_sha256=sha(summary_text),
            protocol_sha256=sha(protocol_text), rows=len(records),
        )
    if set(passes) != {0, 1, 2} or len(manifest) != 210:
        raise ValueError('Require 70 published shards in each of three passes')
    keys = set(passes[0])
    if any(set(rows) != keys for rows in passes.values()) or Counter(k[0] for k in keys) != FAMILY_COUNTS:
        raise ValueError('Candidate keys mismatch across passes')
    data = io.StringIO(newline='')
    fields = ('device', *KEYS, *FEATURES)
    for mode in MODES:
        fields += tuple(f'{mode}_pass{i}_ms' for i in range(3))
        fields += (f'{mode}_median_ms', f'{mode}_relative_spread')
    table = csv.writer(data)
    table.writerow(fields)
    by_family = defaultdict(lambda: defaultdict(list))
    for key in sorted(keys):
        values = [passes[p][key] for p in range(3)]
        identity = values[0][:len(FEATURES)]
        if any(record[:len(FEATURES)] != identity for record in values):
            raise ValueError(f'Input or features changed across passes: {key}')
        record = [device, *key, *identity]
        for i, mode in enumerate(MODES):
            samples = [v[len(FEATURES) + i] for v in values]
            median = statistics.median(samples)
            spread = (max(samples) - min(samples)) / median
            record.extend([*samples, median, spread])
            by_family[key[0]][mode].append(spread)
        table.writerow(record)
    csv_data = data.getvalue().encode()
    metadata = dict(
        device=device, candidates=len(keys), passes=3, correct_rows=len(keys)*3,
        timing_targets=['legacy adaptive CUDA graph replay (amortized ms per launch)',
                        'legacy individual launch (ms per launch)'],
        csv_sha256=sha(csv_data), source_files=manifest,
        family_median_relative_spread={
            k: {mode: statistics.median(v) for mode, v in values.items()} for k, values in by_family.items()
        },
    )
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('measurements.csv', csv_data)
        archive.writestr('manifest.json', json.dumps(metadata, indent=2) + '\n')
    print(device, len(keys), 'candidates, three verified passes', file=sys.stderr)


def checked_rows(path):
    with zipfile.ZipFile(path) as archive:
        summary = json.loads(archive.read('manifest.json'))
        data = archive.read('measurements.csv')
    if sha(data) != summary['csv_sha256']:
        raise ValueError(f'Export digest mismatch: {path}')
    rows = {}
    for record in csv.DictReader(io.StringIO(data.decode())):
        key = tuple(record[k] for k in KEYS)
        if key in rows or record['device'] != summary['device']:
            raise ValueError(f'Duplicate candidate or wrong device in {path}: {key}')
        rows[key] = record
    if len(rows) != summary['candidates'] or Counter(k[0] for k in rows) != FAMILY_COUNTS:
        raise ValueError(f'Incomplete export: {path}')
    return rows


def frozen_candidate(raw):
    row = dict(raw)
    for field in ('workload_features_json', 'config_features_json'):
        row.update(json.loads(row.get(field) or '{}'))

    def numeric(fields):
        result = {}
        for key in fields:
            value = row.get(key)
            name = str(value).lower()
            try:
                number = float(name == 'true') if name in ('true', 'false') else float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result[key] = number
        return result

    shape = numeric(SHAPE)
    if row.get('locality'):
        shape['locality'] = row['locality']
    return ':'.join((row['kernel_family'], json.dumps(shape, sort_keys=True), row['dtype'],
                     json.dumps(numeric(CONFIG), sort_keys=True)))


def normalize_registry(registry, measurements):
    join = {}
    for file in sorted(registry.glob('*-matched30k-rtx3070-wsl-v2-*.csv')):
        with file.open(newline='') as src:
            for row in csv.DictReader(src):
                key = tuple(row[k] for k in KEYS)
                record = measurements[key]
                label = frozen_candidate(row)
                if label in join:
                    raise ValueError(f'Duplicate normalized candidate: {label}')
                join[label] = (record, float(row['latency_median_ms']))
    if len(join) != len(measurements):
        raise ValueError('Historical candidate registry and timing export have different counts')
    return join


def score_predictions(file, lookup):
    tasks = defaultdict(list)
    with file.open(newline='') as src:
        for row in csv.DictReader(src):
            if row['device'] != 'RTX3070-WSL':
                continue
            target, historical = lookup[row['candidate']]
            if historical != float(row['latency_ms']):
                raise ValueError(f'Historical label mismatch in {file}: {row["candidate"]}')
            tasks[row['task']].append((row, target))
    if not tasks:
        raise ValueError(f'No held-out tasks: {file}')
    metrics = defaultdict(lambda: defaultdict(list))
    choices = defaultdict(set)
    flags = defaultdict(list)
    for entries in tasks.values():
        ordered = sorted(entries, key=lambda e: float(e[0]['prediction_ms']))
        family = entries[0][1]['kernel_family']
        choices[family].add(entries[0][0]['task'])
        for measure in ('historical', *MODES):
            y = [float(row['latency_ms'] if measure == 'historical' else target[f'{measure}_median_ms'])
                 for row, target in ordered]
            optimal = min(y)
            for budget in (1, 4):
                metrics[measure, budget][family].append(min(y[:budget]) <= 1.05 * optimal)
        for row, target in ordered:
            flags[family].append(float(target['graph_relative_spread']))
    return {
        'tasks': sum(map(len, choices.values())),
        'families': {family: len(tasks) for family, tasks in choices.items()},
        'macro': {f'{mode}_S{b}': statistics.mean(statistics.mean(v) for v in families.values())
                  for (mode, b), families in metrics.items()},
        'family': {f'{mode}_S{b}': {f: statistics.mean(v) for f, v in families.items()}
                   for (mode, b), families in metrics.items()},
        'per_family_test_graph_median_spread': {f: statistics.median(v) for f, v in flags.items()},
    }


def audit(root, rtx_path, h100_path, scores_path):
    rtx, h100 = checked_rows(rtx_path), checked_rows(h100_path)
    if set(rtx) != set(h100):
        raise ValueError('H100 and 3070 candidate keys differ')
    equal = sum(rtx[key]['input_sha256'] == h100[key]['input_sha256'] for key in rtx)
    original = root / 'results/nvidia_geforce_rtx_3070_wsl'
    lookup = normalize_registry(original, rtx)
    base = root / 'results/comparisons/corrected-devices-20261118'
    models = {}
    for method in ('three', 'flat', 'boosted_tree', 'separate_families', 'random_search'):
        files = sorted((base / 'joint/predictions').glob(f'joint-*-{method}-.csv'))
        if len(files) != 5:
            raise ValueError(f'Expected five frozen predictions for {method}')
        results = [score_predictions(f, lookup) for f in files]
        models[method] = dict(
            test_tasks=results[0]['tasks'], families=results[0]['families'],
            metrics={metric: statistics.mean(r['macro'][metric] for r in results)
                     for metric in results[0]['macro']},
            per_seed=results,
        )
    output = dict(
        device='RTX3070-WSL', matched_h100_rtx3070_candidates=len(rtx),
        identical_input_hashes_across_devices=equal, models=models,
        note='Frozen joint predictions use historical latency labels. Rescoring changes the test target only. The oracle is restricted to the same held-out candidate sets as the original predictions; graph and individual timings are separate targets.',
    )
    with scores_path.open('x') as dest:
        json.dump(output, dest, indent=2)
        dest.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='command', required=True)
    collect = modes.add_parser('gather')
    collect.add_argument('root', type=Path)
    collect.add_argument('device')
    collect.add_argument('output', type=Path)
    compare = modes.add_parser('audit')
    for name in ('root', 'rtx_path', 'h100_path', 'scores_path'):
        compare.add_argument(name, type=Path)
    args = p.parse_args()
    if args.command == 'gather':
        gather(args.root, args.device, args.output)
    else:
        audit(args.root, args.rtx_path, args.h100_path, args.scores_path)


if __name__ == '__main__':
    main()
