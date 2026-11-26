"""Build the workshop tables and figures directly from matched replay traces."""

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter, PercentFormatter
import numpy as np

from summarize_full_fidelity import read_traces
from workshop_feedback import feedback_figure, summarize_feedback, independent_summary


MODES = {'graph': 'CUDA graph replay', 'individual': 'Individual launch'}
METHODS = {
    'simple_world_gaussian': ('Gaussian surrogate', '#1f77b4', '-', 'o'),
    'geometry_ridge': ('Geometry ridge', '#c45d00', (0, (4, 2.5)), 's'),
    'film_ranked': ('Ranked FiLM', '#238254', '-.', '^'),
    'boosted_tree': ('Boosted tree', '#7952a1', (0, (2, 2)), 'D'),
    'target_only_gp_ei': ('Target-only GP--EI', '#aa3d49', (0, (5, 2, 1, 2, 1, 2)), 'P'),
    'random': ('Random search', '#606060', ':', 'x'),
    'simple_world_no_feedback': ('Gaussian, no feedback', '#506a68', '--', 'v'),
    'simple_world_offset_only': ('Gaussian, offset-only forecast', '#506a68', '--', 'v'),
}
PAPER_METHODS = tuple(METHODS)[:6]
FORECAST_METHODS = PAPER_METHODS[:-1]
OFFSET_CONTROL = 'simple_world_offset_only'
FAMILY_LABELS = {'gemm': 'GEMM', 'conv2d': 'Conv2D', 'attention': 'Attention', 'softmax': 'Softmax',
                 'cross_entropy': 'Cross-entropy', 'gather': 'Gather', 'residual_layernorm': 'Residual LayerNorm'}
SUPPLEMENTS = {'workshop-belief': ('simple_world_gaussian', 'simple_world_no_feedback', OFFSET_CONTROL),
               'policy_world': ('geometry_ridge',), 'film': ('film_ranked',),
               'target-only-gp': ('target_only_gp_ei',)}
HORIZON = 48
METRICS = {'regret': HORIZON, 'mape': HORIZON + 1, 'within10': HORIZON + 2}


def case_metrics(row):
    actual = np.asarray(row['measured_us'])
    candidates = len(actual)
    if not 4 < candidates <= HORIZON or len(row['candidate_keys']) != candidates or np.any(actual <= 0):
        raise ValueError('Invalid held-out candidate grid')
    best = actual.min()
    acceptable = actual <= best * (1.05 + 1e-12)
    four = set(row['common_probe_order'][:4])
    unseen = np.array([i not in four for i in range(candidates)])
    budgets = np.minimum(np.arange(1, HORIZON + 1), candidates)
    scores = np.zeros((len(METHODS), len(budgets) + len(METRICS)))
    for i, name in enumerate(METHODS):
        if name == 'random':
            other = candidates - sum(map(int, acceptable))
            scores[i, :HORIZON] = [1. - comb(other, int(b)) / comb(candidates, int(b)) for b in budgets]
            continue
        entry = row['models'][name]
        order = entry['probe_order']
        if sorted(order) != list(range(candidates)):
            raise ValueError(f'Invalid probe order for {name}')
        scores[i, :HORIZON] = np.maximum.accumulate(acceptable[order])[budgets - 1]
        values = entry['forecasts']['4']
        if len(values) != candidates or {i for i, v in enumerate(values) if v is None} != four:
            raise ValueError(f'Wrong held-out forecast mask for {name}')
        predicted = np.asarray([v for v in values if v is not None])
        if not np.isfinite(predicted).all() or np.any(predicted <= 0):
            raise ValueError(f'Invalid forecast for {name}')
        ape = np.abs(predicted / actual[unseen] - 1.)
        scores[i, HORIZON:] = (actual[order[:4]].min() / best - 1., ape.mean(), (ape <= .10).mean())
    return scores


def summarize(folder, mode):
    supplements = [(folder / path, names) for path, names in SUPPLEMENTS.items()]
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    cases = timings = 0
    for dtype in ('fp16', 'bf16'):
        filename = f'{mode}-{dtype}-predictions.jsonl.gz'
        for path, _ in supplements:
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
        for row in read_traces(folder, filename, supplements):
            cell = strata[row['split'], row['family']][row['shape']][dtype]
            if row['mode'] != mode or row['device'] in cell:
                raise ValueError(f'Unexpected or duplicate case in {filename}')
            cell[row['device']] = case_metrics(row)
            cases += 1
            timings += len(row['measured_us'])
    if (cases, timings) != (2786, 41888):
        raise ValueError(f'Unexpected workshop test grid: {cases} cases, {timings} timings')
    rng = np.random.default_rng(51163)
    family_values = defaultdict(list)
    family_bootstrap = {}
    resampled = np.zeros((10000, len(METHODS), HORIZON + len(METRICS)))
    for (_, family), cases_by_shape in sorted(strata.items()):
        vectors = []
        for per_dtype in cases_by_shape.values():
            if {dtype: len(devices) for dtype, devices in per_dtype.items()} != {'fp16': 4, 'bf16': 3}:
                raise ValueError('An expected precision or held-out GPU is absent')
            vectors.append(np.mean([np.mean(list(devices.values()), axis=0) for devices in per_dtype.values()], axis=0))
        block = np.stack(vectors)
        n = len(block)
        family_values[family].append(block.mean(axis=0))
        draws = rng.multinomial(n, np.full(n, 1. / n), size=10000)
        sampled = (draws @ block.reshape(n, -1) / n).reshape(resampled.shape)
        resampled += sampled
        if family not in family_bootstrap:
            family_bootstrap[family] = sampled[:, :2, :HORIZON].copy()
        else:
            family_bootstrap[family] += sampled[:, :2, :HORIZON]
    if set(family_values) != FAMILY_LABELS.keys() or any(len(v) != 2 for v in family_values.values()):
        raise ValueError('Expected two workload-size strata per family')
    family_values = {name: np.mean(values, axis=0) for name, values in family_values.items()}
    macro = np.mean(list(family_values.values()), axis=0)
    resampled /= len(strata)
    methods = {}
    for m, name in enumerate(METHODS):
        scores = macro[m]
        family_hit = {f: int(np.flatnonzero(v[m, :HORIZON] >= .90 - 1e-12)[0]) + 1
                      for f, v in family_values.items()}
        methods[name] = {'success_by_budget': scores[:HORIZON].tolist(), 'p90_by_family': family_hit,
                         'worst_family_p90': max(family_hit.values())}
        if name in (*FORECAST_METHODS, OFFSET_CONTROL):
            methods[name].update({key: float(scores[i]) for key, i in METRICS.items()})
            methods[name]['family_mape'] = {f: float(v[m, METRICS['mape']]) for f, v in family_values.items()}
    intervals = np.quantile(resampled[:, 0, :HORIZON], (.025, .975), axis=0)
    paired = {}
    for name in ('geometry_ridge', 'target_only_gp_ei', 'simple_world_no_feedback'):
        m = list(METHODS).index(name)
        delta = (resampled[:, 0, 3] - resampled[:, m, 3]) * 100.
        paired[name] = [float((macro[0, 3] - macro[m, 3]) * 100.), *np.quantile(delta, (.025, .975)).tolist()]
    b90 = np.maximum.reduce([np.argmax(values >= 1.80 - 1e-12, axis=-1) + 1
                            for values in family_bootstrap.values()])
    b90_delta = b90[:, 0] - b90[:, 1]
    mape_index = METRICS['mape']
    offset_index = list(METHODS).index(OFFSET_CONTROL)
    mape_delta = 100. * (resampled[:, 0, mape_index] - resampled[:, offset_index, mape_index])
    return {'cases': cases, 'timings': timings, 'methods': methods, 'band': intervals.tolist(),
            'paired_s4_pp': paired, 'paired_b90_vs_ridge': np.quantile(b90_delta, (.025, .975)).tolist(),
            'paired_mape_vs_offset_only_pp': np.quantile(mape_delta, (.025, .975)).tolist()}


def figures(results, paper):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    feedback_figure(results['feedback'], FAMILY_LABELS, paper / 'search_benchmark.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.82), sharex=True, sharey=True)
    y = np.arange(len(FAMILY_LABELS))
    display_order = ('simple_world_gaussian', 'film_ranked', 'boosted_tree',
                     'target_only_gp_ei', 'geometry_ridge')
    row_offset = dict(zip(display_order, np.linspace(-.4, .4, len(display_order))))
    for ax, (mode, title) in zip(axes, MODES.items()):
        for name in FORECAST_METHODS:
            label, color, _, mark = METHODS[name]
            error = results[mode]['methods'][name]['family_mape']
            ax.scatter([100. * error[f] for f in FAMILY_LABELS], y + row_offset[name],
                       facecolor=color, edgecolor='white', linewidth=.25,
                       marker=mark, s=16, label=label.replace('--', '–'), zorder=3)
        ax.set(title=title, xlabel='Latency prediction MAPE', yticks=y, yticklabels=list(FAMILY_LABELS.values()))
        ax.set_xscale('log')
        ax.set_xticks([10, 15, 20, 30, 50], [f'{v}%' for v in (10, 15, 20, 30, 50)])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(8, 64)
        ax.grid(axis='both', color='#e0e4e8', lw=.55)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, columnspacing=1.55, handletextpad=.4)
    fig.subplots_adjust(left=.19, right=.985, bottom=.16, top=.75, wspace=.12)
    fig.savefig(paper / 'forecast_benchmark.pdf')
    plt.close(fig)


def tables(results, paper):
    def write(name, spec, rows):
        content = '\\begin{tabular}{' + spec + '}\n\\toprule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
        (paper / name).write_text(content)

    rows = [r' & \multicolumn{4}{c}{Graph replay} & \multicolumn{4}{c}{Individual launch} \\',
            r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
            r'Method & $S_4$ & $B_{90}$ & MAPE & Within 10\% & $S_4$ & $B_{90}$ & MAPE & Within 10\% \\', r'\midrule']
    for name in (*PAPER_METHODS[:-1], 'simple_world_no_feedback', OFFSET_CONTROL, 'random'):
        cells = [METHODS[name][0]]
        for mode in MODES:
            a = results[mode]['methods'][name]
            cells += ['--', '--'] if name == OFFSET_CONTROL else [f"{a['success_by_budget'][3]*100:.1f}", str(a['worst_family_p90'])]
            cells += [f"{a['mape']*100:.1f}", f"{a['within10']*100:.1f}"] if name in (*FORECAST_METHODS, OFFSET_CONTROL) else ['--', '--']
        rows.append(' & '.join(cells) + r' \\')
    write('benchmark_main.tex', 'lrrrrrrrr', rows)
    rows = [r' & \multicolumn{2}{c}{Graph replay} & \multicolumn{2}{c}{Individual launch} \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
            r'Family & Gaussian & Ridge & Gaussian & Ridge \\', r'\midrule']
    for f, label in FAMILY_LABELS.items():
        values = [results[mode]['methods'][name]['p90_by_family'][f] for mode in MODES for name in PAPER_METHODS[:2]]
        rows.append(' & '.join((label, *map(str, values))) + r' \\')
    write('benchmark_families.tex', 'lrrrr', rows)

    rows = [r' & \multicolumn{3}{c}{Graph replay} & \multicolumn{3}{c}{Individual launch} \\',
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
            r'GPU & On & Off & Ridge & On & Off & Ridge \\', r'\midrule']
    for device, label in [('H100','H100'), ('A100','A100'), ('RTXA5000','A5000'),
                          ('RTX3070-WSL','3070 (WSL)'), ('T4','T4')]:
        scores = results['independent_devices'][device]['values']
        cells = [label] + [f'{value:.1f}' for mode in scores for value in mode[:3]]
        rows.append(' & '.join(cells) + r' \\')
    write('benchmark_passes.tex','lrrrrrr',rows)
    commands = []
    for mode, prefix in (('graph', 'Graph'), ('individual', 'Launch')):
        for method, name in (('geometry_ridge', 'Ridge'), ('target_only_gp_ei', 'Cold'), ('simple_world_no_feedback', 'NoFeedback')):
            point, lo, hi = results[mode]['paired_s4_pp'][method]
            commands.append('\\newcommand{\\' + prefix + name + 'Delta}{' + f'{point:+.2f}' + '}')
            commands.append('\\newcommand{\\' + prefix + name + 'CI}{[' + f'{lo:+.2f},{hi:+.2f}' + ']}')
        lo, hi = results[mode]['paired_b90_vs_ridge']
        commands.append('\\newcommand{\\' + prefix + 'BudgetCI}{[' + f'{lo:.0f},{hi:+.0f}' + ']}')
        offset_mape = results[mode]['methods'][OFFSET_CONTROL]['mape']
        commands.append('\\newcommand{\\' + prefix + 'OffsetMAPE}{' + f'{100. * offset_mape:.1f}' + '}')
        lo, hi = results[mode]['paired_mape_vs_offset_only_pp']
        commands.append('\\newcommand{\\' + prefix + 'OffsetCI}{[' + f'{lo:+.2f},{hi:+.2f}' + ']}')
    (paper / 'benchmark_effects.tex').write_text('\n'.join(commands) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('paper', type=Path)
    args = parser.parse_args()
    report = {mode: summarize(args.benchmark, mode) for mode in MODES}
    report['feedback'] = summarize_feedback(args.benchmark)
    report['independent'] = independent_summary(args.benchmark)
    report['independent_devices'] = dict(report['independent']['devices'])
    for device, n in [('T4', 398), ('RTXA5000', 796), ('A100', 796)]:
        filename = f'independent-winners-{device}.jsonl.gz'
        passes = 3
        if device == 'A100' and not (args.benchmark / filename).exists():
            passes = 2
            filename = 'independent-winners-A100-2pass.jsonl.gz'
        repeat = independent_summary(args.benchmark, filename, (device,), n, pass_count=passes)
        report['independent_devices'][device] = repeat['devices'][device]
    (args.benchmark / 'workshop-benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    args.paper.mkdir(parents=True, exist_ok=True)
    tables(report, args.paper)
    figures(report, args.paper)


if __name__ == '__main__':
    main()
