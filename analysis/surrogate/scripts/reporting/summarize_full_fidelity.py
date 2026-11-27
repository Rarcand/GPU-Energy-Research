"""Combine held-out precision/size splits and render the fidelity curves."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import gzip
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

MODE_NAMES = {'graph': 'CUDA graph replay', 'individual': 'Individual launch'}
METHOD_LABELS = {'adaptive_ridge': 'Adaptive ridge', 'geometry_adaptive_ridge': 'Geometry-only adaptive ridge',
                 'absolute_adaptive_ridge': 'Absolute adaptive ridge', 'absolute_ridge': 'Absolute ridge',
                 'boosted_tree': 'Boosted tree', 'three': 'Three encoders',
                 'film': 'FiLM latency', 'film_ranked': 'FiLM ranking', 'flat': 'Flat MLP',
                 'hybrid_reference': 'Ridge + GP reference',
                 'world_greedy': 'World model', 'world_lookahead': 'World model + look-ahead',
                 'world_open_loop': 'World model open loop', 'random': 'Random'}
COLORS = {'adaptive_ridge': '#135b98', 'geometry_adaptive_ridge': '#22a79a', 'absolute_ridge': '#7a57b4',
          'absolute_adaptive_ridge': '#b33728',
          'boosted_tree': '#d77920', 'three': '#cc4c68', 'film': '#80910c', 'film_ranked': '#4a6d22',
          'flat': '#5f6a7d',
          'hybrid_reference': '#2c343e', 'world_greedy': '#7f40a0', 'world_lookahead': '#d17a00',
          'world_open_loop': '#a673bc', 'random': '#849093'}
PAIR_METRICS = ('selection_regret', 'selection_within_5pct', 'prediction_mape', 'prediction_within_10pct')
SUPPLEMENTAL = {
    'absolute': ('absolute_adaptive_ridge',),
    'world': ('world_greedy', 'world_lookahead', 'world_open_loop'),
    'film': ('film_ranked',),
}


def average_methods(splits):
    methods = {}
    for m in splits[0]:
        rows = [e[m] for e in splits]
        fidelity = {t: np.mean([s['within_by_budget'][t] for s in rows], axis=0).tolist()
                    for t in rows[0]['within_by_budget']}
        prediction = defaultdict(lambda: defaultdict(list))
        for row in rows:
            for budget, metrics in row['prediction'].items():
                for metric, value in metrics.items():
                    prediction[budget][metric].append(value)
        methods[m] = {
            'mean_regret_by_budget': np.mean([s['mean_regret_by_budget'] for s in rows], axis=0).tolist(),
            'within_by_budget': fidelity,
            'first_budget_at_coverage': {
                t: {str(q): next((i + 1 for i, v in enumerate(c) if v >= q / 100 - 1e-12), None)
                    for q in (50, 90, 95, 100)} for t, c in fidelity.items()
            },
            'prediction': {b: {k: float(np.mean(v)) for k, v in metrics.items()}
                           for b, metrics in prediction.items()},
        }
    return methods


def read_traces(folder, filename, supplements=None):
    if supplements is None:
        supplements = [(folder / name, methods) for name, methods in SUPPLEMENTAL.items()]
    with ExitStack() as resources:
        streams = [resources.enter_context(gzip.open(folder / filename, 'rt'))]
        groups = []
        for path, methods in supplements:
            extra = path / filename
            if extra.exists():
                streams.append(resources.enter_context(gzip.open(extra, 'rt')))
                groups.append(methods)
        for batch in zip_longest(*streams):
            if any(line is None for line in batch):
                raise ValueError(f'Trace sizes differ in {filename}')
            a, *other = map(json.loads, batch)
            for b, methods in zip(other, groups):
                if ({k: v for k, v in a.items() if k != 'models'}
                        != {k: v for k, v in b.items() if k != 'models'}
                        or ('adaptive_ridge' in b['models']
                            and a['models']['adaptive_ridge'] != b['models']['adaptive_ridge'])):
                    raise ValueError(f'Trace task differs in {filename}')
                a['models'].update({m: b['models'][m] for m in methods})
            yield a


def paired_at_four(folder, mode, expected_cases, supplements=None, reference_pairs=None):
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dtype in ('fp16', 'bf16'):
        filename = f'{mode}-{dtype}-predictions.jsonl.gz'
        for row in read_traces(folder, filename, supplements):
            if row['mode'] != mode or row['device'] in records[row['split'], row['family']][row['shape']][dtype]:
                raise ValueError(f'Duplicate task or mismatched timing mode in {filename}')
            actual = np.asarray(row['measured_us'])
            observed = set(row['common_probe_order'][:4])
            unseen = np.asarray([i not in observed for i in range(len(actual))])
            if len(row['candidate_keys']) != len(actual) or not np.all(actual > 0):
                raise ValueError(f'Invalid measured grid in {filename}')
            outcome = {}
            for name, model in row['models'].items():
                order = model['probe_order']
                if sorted(order) != list(range(len(actual))):
                    raise ValueError(f'Invalid probe order in {filename}')
                regret = actual[order[:4]].min() / actual.min() - 1.
                vector = model['forecasts']['4']
                if {i for i, x in enumerate(vector) if x is None} != observed:
                    raise ValueError(f'Forecast leaked measured configs in {filename}')
                estimate = np.asarray([x for x in vector if x is not None])
                if not np.isfinite(estimate).all():
                    raise ValueError(f'Nonfinite forecast in {filename}')
                ape = np.abs(estimate / actual[unseen] - 1.)
                outcome[name] = np.array((regret, regret <= .05 + 1e-12,
                                          ape.mean(), (ape <= .10).mean()))
            records[row['split'], row['family']][row['shape']][dtype][row['device']] = outcome
    groups = defaultdict(list)
    checked = 0
    for by_shape in records.values():
        values = defaultdict(list)
        for by_dtype in by_shape.values():
            if set(by_dtype) != {'fp16', 'bf16'} or sorted(len(d) for d in by_dtype.values()) != [3, 4]:
                raise ValueError('Missing matched dtype or device in a test shape')
            shape = defaultdict(list)
            for by_device in by_dtype.values():
                checked += len(by_device)
                for method in next(iter(by_device.values())):
                    shape[method].append(np.mean([d[method] for d in by_device.values()], axis=0))
            for method, pair in shape.items():
                values[method].append(np.mean(pair, axis=0))
        for method, outcomes in values.items():
            groups[method].append(np.asarray(outcomes))
    if checked != expected_cases:
        raise ValueError(f'Expected {expected_cases} trace cases, validated {checked}')
    baseline = groups['adaptive_ridge']
    rng = np.random.default_rng(10023)
    draws = [rng.integers(len(v), size=(10000, len(v))) for v in baseline]
    contrasts = {}
    if reference_pairs is None:
        reference_pairs = [(m, 'adaptive_ridge') for m in groups if m != 'adaptive_ridge']
        reference_pairs.extend((m, b) for m, b in (('world_greedy', 'world_open_loop'),
                                                   ('world_lookahead', 'world_greedy'),
                                                   ('world_greedy', 'film_ranked'), ('film_ranked', 'film'))
                               if m in groups and b in groups)
    for method, reference in reference_pairs:
        paired = [m - b for m, b in zip(groups[method], groups[reference])]
        point = np.mean([r.mean(axis=0) for r in paired], axis=0)
        boot = np.mean([r[d].mean(axis=1) for r, d in zip(paired, draws)], axis=0)
        interval = np.quantile(boot, (.025, .975), axis=0).T
        contrasts[f'{method}_minus_{reference}'] = {
            k: {'difference': float(x), 'ci95': i.tolist()} for k, x, i in zip(PAIR_METRICS, point, interval)}
    return {'validated_trace_cases': checked, 'contrasts': contrasts}


def combine(folder):
    result = {
        'protocol': ('Average each FP16/BF16 and largest/next-largest test split equally within timing mode. '
                     'All metrics first average within workload family, with equal weight across families. '
                     'Forecast metrics at budget B score only configurations outside the common probe prefix '
                     'and families with more than B configurations. Selection metrics use each policy’s '
                     'own probe sequence against the oracle minimum of the full measured grid.'),
        'modes': {},
    }
    for mode in MODE_NAMES:
        inputs = []
        for dtype in ('fp16', 'bf16'):
            filename = f'{mode}-{dtype}.json'
            report = json.loads((folder / filename).read_bytes())
            for name, methods in SUPPLEMENTAL.items():
                extra_path = folder / name / filename
                if not extra_path.exists():
                    continue
                extra = json.loads(extra_path.read_bytes())
                for size, part in report['experiments'].items():
                    other = extra['experiments'][size]
                    if part['split_sha256'] != other['split_sha256']:
                        raise ValueError(f'Supplemental comparison used a different split in {filename}')
                    part['macro'].update({m: other['macro'][m] for m in methods})
                    for f, section in part['families'].items():
                        if section['device_shape_cases'] != other['families'][f]['device_shape_cases']:
                            raise ValueError(f'Supplemental case count changed in {filename}, {f}')
                        section['methods'].update({m: other['families'][f]['methods'][m] for m in methods})
            inputs.append(report)
        parts = [part for file in inputs for part in file['experiments'].values()]
        families = {}
        for family in parts[0]['families']:
            rows = [p['families'][family] for p in parts]
            families[family] = {'candidates_per_shape': rows[0]['candidates_per_shape'],
                                'device_shape_cases': sum(r['device_shape_cases'] for r in rows),
                                'methods': average_methods([r['methods'] for r in rows])}
        result['modes'][mode] = {
            'macro': average_methods([p['macro'] for p in parts]), 'families': families,
            'paired_at_four': paired_at_four(folder, mode, sum(f['device_shape_cases'] for f in families.values())),
        }
    return result


def plot(overview, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    fig.suptitle('Autotuning fidelity versus measurements', fontsize=17, fontweight='bold')
    for idx, (mode, section) in enumerate(overview['modes'].items()):
        left, right = axes[idx]
        curves = section['macro']
        plot_methods = ('adaptive_ridge', 'geometry_adaptive_ridge', 'film', 'film_ranked', 'world_greedy',
                        'world_lookahead', 'world_open_loop', 'hybrid_reference', 'random')
        for name, label in METHOD_LABELS.items():
            if name not in curves or name not in plot_methods:
                continue
            m = curves[name]
            y = np.asarray(m['within_by_budget']['5'][:16]) * 100
            left.plot(np.arange(1, len(y) + 1), y, label=label, color=COLORS[name],
                      linewidth=2 if name.startswith('adaptive') else 1.5,
                      linestyle='--' if name in ('random', 'world_open_loop') else '-')
            # With identical observed history the greedy and look-ahead model make the same forecast.
            if name in ('random', 'world_lookahead'):
                continue
            budgets = [b for b in (0, 1, 2, 4, 8) if str(b) in m['prediction']]
            coverage = [100 * m['prediction'][str(b)]['within_10pct'] for b in budgets]
            if budgets:
                right.plot(budgets, coverage, label=label, color=COLORS[name], marker='o', markersize=4)
        left.axhline(90, color='#a0a5ad', linewidth=.8, linestyle=':')
        left.set(xlabel='Configs measured', ylabel='Cases within 5% of fastest (%)',
                 title=f'{MODE_NAMES[mode]} · configuration found', xticks=(1, 2, 4, 8, 12, 16), ylim=(0, 102))
        right.set(xlabel='Shared timings revealed', ylabel='Unmeasured latencies within 10% (%)',
                  title=f'{MODE_NAMES[mode]} · prediction accuracy', xticks=(0, 1, 2, 4, 8), ylim=(0, 102))
        for panel in (left, right):
            panel.grid(axis='y', alpha=.22)
            panel.spines[['top', 'right']].set_visible(False)
        if idx == 0:
            left.legend(loc='lower right', fontsize=7.8, ncol=2, frameon=False)
    fig.savefig(output, dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = combine(args.folder)
    (args.folder / 'overview.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot(result, args.folder / 'fidelity-curves.png')


if __name__ == '__main__':
    main()
