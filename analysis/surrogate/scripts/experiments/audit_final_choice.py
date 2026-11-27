"""Measure whether hardware improves the final pick when the first probe is shared."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.surrogate.scripts.experiments.audit_timing_protocol_transfer import measure
from analysis.surrogate.scripts.experiments.benchmark_few_shot_surrogate import ShortlistSurrogate, metric_for_order, replay, source_scenarios
from analysis.surrogate.scripts.experiments.capacity_bfloat16_transfer import DEVICES as BF16_DEVICES, verify_matched_grid
from analysis.surrogate.scripts.experiments.capacity_transport import centered, physical_phases, shift_bank
from analysis.surrogate.scripts.experiments.compare_gemm_resource_residual import aligned_grid
from analysis.surrogate.scripts.experiments.compare_portable_surrogates import DEVICES as FP16_DEVICES, load_matched
from analysis.surrogate.scripts.models.compare_surrogates import FAMILIES, save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.experiments.test_unseen_a5000 import load_a5000
from analysis.surrogate.scripts.models.train_matched import training_rows


METRICS = ('regret_b4', 'regret_b8', 'success5_b4', 'success5_b8')
COMPARISONS = (
    ('hardware', 'blind'),
    ('hardware', 'neutral'),
    ('hardware_after_blind_first', 'blind'),
    ('hardware_after_blind_first', 'neutral_after_blind_first'),
    ('hardware', 'hardware_first_then_blind'),
    ('hardware_first_then_blind', 'blind'),
)
EXPERIMENTS = {
    ('float16', 'largest'): ('split-largest-shapes.json', 'capacity_transport_graph.json'),
    ('float16', 'next_largest'): ('capacity_split_upper_middle.json', 'capacity_transport_upper_middle_graph.json'),
    ('bfloat16', 'largest'): ('split-largest-shapes.json', 'capacity_bf16_largest.json'),
    ('bfloat16', 'next_largest'): ('capacity_split_upper_middle.json', 'capacity_bf16_upper_middle.json'),
}


def grids(rows, dtype, devices):
    align = aligned_grid if dtype == 'float16' else verify_matched_grid
    for family in FAMILIES:
        shapes, configs, ordered = align([row for row in rows if row['kernel_family'] == family])
        dims = len(devices), len(shapes), len(configs)
        times = np.asarray([measure(row, 'graph') for row in ordered]).reshape(dims)
        # Both formats use the same two-byte storage geometry.
        features = centered(physical_phases([{**row, 'dtype': 'float16'} for row in ordered], dims))
        yield family, shapes, times, features


def replay_fold(shapes, times, features, assignment, target, previous):
    sources = [i for i in range(len(times)) if i != target]
    train, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                   for role in ('train', 'test'))
    blind = source_scenarios(times, shapes, sources, train, test)
    model = np.asarray(previous['coefficient'])
    strength = previous['strength']
    hardware = shift_bank(blind, features, sources, test, target, model, strength)
    neutral = shift_bank(blind, features, sources, test, target, model, strength, neutral=True)
    weights = np.full(len(sources), 1. / len(sources))
    results = []
    for i, shape_index in enumerate(test):
        actual = times[target, shape_index]
        blind_first = ShortlistSurrogate(blind[i], weights, adaptive=True).next_candidate()
        hardware_first = ShortlistSurrogate(hardware[i], weights, adaptive=True).next_candidate()
        settings = {
            'blind': (blind[i], None),
            'neutral': (neutral[i], None),
            'hardware': (hardware[i], None),
            'hardware_after_blind_first': (hardware[i], blind_first),
            'neutral_after_blind_first': (neutral[i], blind_first),
            'hardware_first_then_blind': (blind[i], hardware_first),
        }
        outcomes = {}
        for name, (prior, first) in settings.items():
            order = replay(prior, weights, actual, adaptive=True, first_candidate=first)
            values = metric_for_order(order, actual)
            outcomes[name] = [values[k] for k in METRICS]
            if name in ('blind', 'neutral', 'hardware'):
                original = previous['scores'][name][i]
                if (original['shape'] != shapes[shape_index]
                        or not np.allclose(outcomes[name], [original[k] for k in METRICS],
                                           rtol=0, atol=1e-12)):
                    raise ValueError(f'Original replay differs for {name}, {shapes[shape_index]}')
        results.append(outcomes)
    return results


def summaries(results, *, minimum_candidates):
    selected = {name: rows for name, (count, rows) in results.items() if count >= minimum_candidates}
    methods = tuple(next(iter(selected.values()))[0][0])
    arrays = {f: {m: np.asarray([[r[m] for r in fold] for fold in folds]).mean(axis=0)
                  for m in methods} for f, folds in selected.items()}
    rng = np.random.default_rng(48127)
    draws = {family: rng.integers(len(data['blind']), size=(10000, len(data['blind'])))
             for family, data in arrays.items()}
    point = {m: np.mean([a[m].mean(axis=0) for a in arrays.values()], axis=0) for m in methods}
    tests = {}
    for method, base in COMPARISONS:
        delta = {f: a[method] - a[base] for f, a in arrays.items()}
        trial = np.mean([v[draws[f]].mean(axis=1) for f, v in delta.items()], axis=0)
        limits = np.quantile(trial, (.025, .975), axis=0).T
        tests[f'{method}_minus_{base}'] = {
            key: {'difference': float(point[method][i] - point[base][i]), 'ci95': limits[i].tolist()}
            for i, key in enumerate(METRICS)
        }
    return {'families': list(selected),
            'shapes': sum(len(a['blind']) for a in arrays.values()),
            'means': {m: dict(zip(METRICS, v.tolist())) for m, v in point.items()},
            'comparisons': tests}


def load_precision_inputs(inputs, t4):
    fp16, _, _ = load_matched(inputs, t4, 'graph')
    hardware = json.loads((inputs / 'hardware.json').read_text())
    source, _ = training_rows(inputs, 'graph', hardware)
    a5000, _ = load_a5000(inputs / 'a5000-pass00-raw.tar.gz',
                          inputs / 'a5000-pass00-verified.json', inputs / 'h100.zip')
    bf16 = [row for row in source + a5000 if row['dtype'] == 'bfloat16']
    return (('float16', fp16, FP16_DEVICES), ('bfloat16', bf16, BF16_DEVICES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 't4', 'reports', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'protocol': ('FP16 and BF16 graph replay on held-out GPUs and workload sizes. '
                     'Reuse the original source-only fit for every comparison. '
                     'Share only the first candidate in the matched-first controls. '
                     'Select the fastest of the probed candidates. '
                     'Paired 95% shape bootstrap with equal family and fixed-GPU weights.'),
        'experiments': {},
    }
    for dtype, rows, devices in load_precision_inputs(args.inputs, args.t4):
        data = list(grids(rows, dtype, devices))
        for size in ('largest', 'next_largest'):
            split_name, report_name = EXPERIMENTS[dtype, size]
            split_path, report_path = args.reports / split_name, args.reports / report_name
            assignment = json.loads(split_path.read_bytes())['assignment']
            report = json.loads(report_path.read_bytes())
            archived = report['splits'][split_path.stem] if 'splits' in report else report
            if tuple(report['devices']) != tuple(devices):
                raise ValueError('Archived device order differs from replay')
            results = {
                family: (times.shape[-1], [replay_fold(shapes, times, features, assignment, d,
                                                       archived['families'][family][device])
                                           for d, device in enumerate(devices)])
                for family, shapes, times, features in data
            }
            result['experiments'][f'{dtype}_{size}'] = {
                'devices': devices, 'split_sha256': sha(split_path.read_bytes()),
                'original_report_sha256': sha(report_path.read_bytes()),
                'candidates_by_family': {f: count for f, (count, _) in results.items()},
                'all_families': summaries(results, minimum_candidates=1),
                'non_exhaustive_at_b8': summaries(results, minimum_candidates=9),
            }
            print(f'Verified original measurements and replayed controls for {dtype}, {size}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == '__main__':
    main()
