"""Summarize independent-pass tuning success against a fixed source-trained portfolio."""
import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.surrogate.scripts.reporting.workshop_feedback import MODES, paired_summary

CASE = ('family', 'split', 'shape', 'dtype', 'device', 'mode')
BUDGETS = (1, 2, 4, 8)
PAIRS = tuple(itertools.permutations(range(3), 2))
FROZEN = (
    'independent-winners.jsonl.gz', 'independent-winners-A100.jsonl.gz',
    'independent-winners-RTXA5000.jsonl.gz', 'independent-winners-T4.jsonl.gz',
)


def stream(path):
    with gzip.open(path, 'rt') as f:
        yield from map(json.loads, f)


def read_panel(root):
    frozen = {}
    for filename in FROZEN:
        for row in stream(root / filename):
            key = tuple(row[k] for k in CASE)
            pair = (row['select'], row['score'])
            if (key, pair) in frozen:
                raise ValueError(f'Duplicate reference {key} {pair}')
            frozen[key, pair] = np.asarray(row['regret'][:2])
    if len(frozen) != 3582 * len(MODES) * len(PAIRS):
        raise ValueError('Incomplete reference panel')
    sums = defaultdict(lambda: np.zeros(4))
    masks = defaultdict(int)
    for row in stream(root / 'independent-portfolio.jsonl.gz'):
        key = tuple(row[k] for k in CASE)
        b = row['budget']
        pair = (row['select'], row['score'])
        if b not in BUDGETS or (key, pair) not in frozen:
            raise ValueError(f'Unknown case, pair or budget: {key} {pair} {b}')
        cell = (b, key)
        bit = 1 << PAIRS.index(pair)
        if bit & masks[cell]:
            raise ValueError(f'Duplicate {cell} {pair}')
        masks[cell] |= bit
        regret = np.asarray(row['regret'])
        if regret.shape != (4,) or not np.isfinite(regret).all() or np.any(regret < -1e-12):
            raise ValueError(f'Invalid regret: {cell} {pair}')
        if b == 4 and not np.allclose(regret[:2], frozen[key, pair], rtol=1e-8, atol=1e-10):
            raise ValueError(f'Four-probe replay differs from frozen policy: {cell} {pair}')
        sums[cell] += regret <= 0.05 + 1e-12
    if len(sums) != len(frozen) * len(BUDGETS) // len(PAIRS):
        raise ValueError('Incomplete evaluation panel')
    panel = defaultdict(lambda: defaultdict(dict))
    for (b, key), total in sums.items():
        if masks[b, key] != (1 << len(PAIRS)) - 1:
            raise ValueError(f'Missing ordered pass pair for {b} {key}')
        # The shared bootstrap computes contrasts against slots 1 and 2.
        # Here those slots hold global and local portfolios respectively.
        panel[b][key[:-1]][key[-1]] = total[[0, 2, 3, 1]] / len(PAIRS)
    return panel


def rows(scope, name, b, report, feedback_report, n):
    for i, mode in enumerate(MODES):
        on, portfolio, local, off = report['values'][i]
        lo, hi = report['ci'][i]
        # The shared routine names slot 2 after the original ridge comparator.
        local_lo, local_hi = report['gaussian_minus_geometry_ridge_ci95_pp'][i]
        yield dict(scope=scope, group=name, budget=b, mode=mode, cases=n,
                   gaussian_pct=on, portfolio_pct=portfolio, feedback_disabled_pct=off,
                   gaussian_minus_portfolio_pp=report['gain'][i], ci_low_pp=lo, ci_high_pp=hi, local_portfolio_pct=local,
                   gaussian_minus_local_portfolio_pp=report['gaussian_minus_geometry_ridge_pp'][i],
                   local_ci_low_pp=local_lo, local_ci_high_pp=local_hi,
                   gaussian_minus_feedback_disabled_pp=feedback_report['gain'][i],
                   feedback_ci_low_pp=feedback_report['ci'][i][0],
                   feedback_ci_high_pp=feedback_report['ci'][i][1])


def paired_contrasts(cases):
    portfolios = paired_summary(cases, compare_ridge=True)
    feedback_cases = {key: {mode: values[[0, 3]] for mode, values in cell.items()}
                      for key, cell in cases.items()}
    return portfolios, paired_summary(feedback_cases)


def estimates(panel):
    out, device_families = [], []
    for b, cases in sorted(panel.items()):
        (families, full), (feedback_families, feedback) = paired_contrasts(cases)
        out.extend(rows('overall', 'all', b, full, feedback, len(cases)))
        for family, report in sorted(families.items()):
            out.extend(rows('family', family, b, report, feedback_families[family],
                            sum(k[0] == family for k in cases)))
        for scope, index in [('device', 4), ('precision', 3)]:
            for name in sorted({k[index] for k in cases}):
                subset = {k: v for k, v in cases.items() if k[index] == name}
                (subfamilies, group), (subfeedback, group_feedback) = paired_contrasts(subset)
                out.extend(rows(scope, name, b, group, group_feedback, len(subset)))
                if scope == 'device':
                    for family in sorted(subfamilies):
                        n = sum(k[0] == family for k in subset)
                        for entry in rows('device_family', family, b, subfamilies[family], subfeedback[family], n):
                            device_families.append(dict(device=name, family=family, **entry))
    return out, device_families


def tex_table(out):
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Mode & $B$ & Gaussian & Global & Local & $\Delta_{\rm local}$ [95\% CI] \\', r'\midrule']
    for mode, label in zip(MODES, ('Graph', 'Individual')):
        for r in sorted((r for r in out if r['scope'] == 'overall' and r['mode'] == mode),
                        key=lambda r: r['budget']):
            lines.append(f"{label} & {r['budget']} & {r['gaussian_pct']:.1f} & "
                         f"{r['portfolio_pct']:.1f} & {r['local_portfolio_pct']:.1f} & {r['gaussian_minus_local_portfolio_pp']:+.1f} "
                         f"[{r['local_ci_low_pp']:+.1f}, {r['local_ci_high_pp']:+.1f}] " + r'\\')
            label = ''
        if mode == 'graph':
            lines.append(r'\midrule')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('table', type=Path)
    args = parser.parse_args()
    out, device_families = estimates(read_panel(args.benchmark))
    for name, estimates_table in [('summary', out), ('device-family', device_families)]:
        path = args.benchmark / f'independent-portfolio-{name}.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=estimates_table[0])
            writer.writeheader()
            writer.writerows(estimates_table)
    report = dict(threshold=0.05, budgets=list(BUDGETS), heldout_shape_split_cases=398,
                  six_ordered_rotations=True, frozen_four_probe_replay_checked=True,
                  bootstrap='10,000 paired shape resamples within family and size range. '
                            'Equal weights for families, splits, precisions, and devices within precision. '
                            'Intervals are pointwise and conditional on the measured environments.',
                  estimates=out)
    (args.benchmark / 'independent-portfolio-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(tex_table(out))
