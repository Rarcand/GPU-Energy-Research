"""Compare held-out policy replays with matched workload shapes, precision, and pass count."""
import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from workshop_feedback import MODES, paired_summary


def score(root, passes):
    files = ['independent-winners.jsonl.gz', 'independent-winners-T4.jsonl.gz',
             'independent-winners-RTXA5000.jsonl.gz',
             'independent-winners-A100' + ('-2pass' if passes == 2 else '') + '.jsonl.gz']
    expected_pairs = set(itertools.permutations(range(passes), 2))
    rotations = defaultdict(dict)
    for filename in files:
        with gzip.open(root/filename, 'rt') as stream:
            for r in map(json.loads, stream):
                pair = (r['select'], r['score'])
                if pair not in expected_pairs:
                    continue
                key = tuple(r[k] for k in ('family', 'split', 'shape', 'dtype', 'device', 'mode'))
                cell = rotations[key]
                if pair in cell:
                    raise ValueError(f'Duplicate scoring pass pair for {key}')
                regrets = np.asarray(r['regret'])
                if regrets.shape != (3,) or not np.isfinite(regrets).all() or np.any(regrets < -1e-12):
                    raise ValueError(f'Invalid regret for {key}')
                cell[pair] = regrets <= 0.05 + 1e-12
    by_dtype = defaultdict(lambda: defaultdict(dict))
    for (*key, mode), values in rotations.items():
        if values.keys() != expected_pairs:
            raise ValueError(f'Incomplete rotation for {key}')
        by_dtype[key[3]][tuple(key)][mode] = np.mean(list(values.values()), axis=0)
    report = {}
    expected_shapes = None
    for dtype, records in by_dtype.items():
        devices = {'A100', 'H100', 'RTXA5000', 'RTX3070-WSL'}
        if dtype == 'float16':
            devices.add('T4')
        grid = defaultdict(set)
        for (family, split, shape, _, device), modes in records.items():
            if modes.keys() != set(MODES):
                raise ValueError('Missing timing mode')
            grid[family, split, shape].add(device)
        if len(grid) != 398 or any(measured != devices for measured in grid.values()):
            raise ValueError(f'Unmatched held-out shapes for {dtype}')
        if expected_shapes is not None and set(grid) != expected_shapes:
            raise ValueError('The FP16 and BF16 shape sets differ')
        expected_shapes = set(grid)
        _, overall = paired_summary(records, compare_ridge=True)
        report[dtype] = dict(devices=sorted(devices), heldout_shape_split_cases=len(grid),
                            device_shape_split_cases=len(records), overall=overall)
    if set(report) != {'float16', 'bfloat16'}:
        raise ValueError('Missing precision')
    matched = {}
    for (family, split, shape, _, device), bf16 in by_dtype['bfloat16'].items():
        fp16 = by_dtype['float16'][family, split, shape, 'float16', device]
        matched[family, split, shape, 'paired', device] = {
            mode: np.array([bf16[mode][0] - bf16[mode][1], fp16[mode][0] - fp16[mode][1]])
            for mode in MODES}
    _, contrast = paired_summary(matched)
    return dict(passes=passes, ordered_pairs=len(expected_pairs), methods=['Gaussian', 'feedback disabled', 'geometry ridge'],
                modes=list(MODES), budget=4, precisions=report,
                matched_four_gpu_precision_contrast=dict(
                    cases=len(matched), bf16_minus_fp16_feedback_gain_pp=contrast['gain'],
                    ci95_pp=contrast['ci'], feedback_gains_bf16_then_fp16_pp=contrast['values']),
                interpretation='Conditional paired shape intervals within each precision. Source and target GPU pools '
                               'vary between FP16 and BF16. The two precision estimates do not isolate a datatype effect.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--passes', type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    out = score(args.benchmark, args.passes)
    dest = args.benchmark / f'independent-policy-by-precision-{args.passes}pass.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    for dtype, result in out['precisions'].items():
        print(dtype, result['device_shape_split_cases'], result['overall'])
