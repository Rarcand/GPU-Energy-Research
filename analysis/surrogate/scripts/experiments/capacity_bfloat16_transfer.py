"""Stress-test the FP16 capacity correction on matched BF16 timing labels."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from analysis.surrogate.scripts.experiments.audit_timing_protocol_transfer import measure
from analysis.surrogate.scripts.experiments.benchmark_few_shot_surrogate import source_scenarios
from analysis.surrogate.scripts.experiments.capacity_transport import (centered, evaluate_bank, learn_transport, physical_phases,
                                shift_bank, summarize)
from analysis.surrogate.scripts.models.compare_surrogates import FAMILIES, save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.models.portable_rank_prior import config_key
from analysis.surrogate.scripts.experiments.test_unseen_a5000 import load_a5000
from analysis.surrogate.scripts.models.train_matched import training_rows

DEVICES = ('H100', 'RTX3070-WSL', 'RTXA5000')


def verify_matched_grid(rows):
    groups = sorted({r['_group'] for r in rows})
    configs = sorted({config_key(r) for r in rows})
    lookup = {(r['_device'], r['_group'], config_key(r)): r for r in rows}
    aligned = [lookup[key] for key in product(DEVICES, groups, configs)]
    if not len(rows) == len(aligned) == len(lookup):
        raise ValueError('Incomplete or duplicate BF16 candidates')
    return groups, configs, aligned


def evaluate_fold(times, phases, shapes, assignment, target, fp16_fold):
    sources = [i for i in range(len(DEVICES)) if i != target]
    train, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                   for role in ('train', 'test'))
    if not train or not test:
        raise ValueError('Both train and test shapes are required')
    model = learn_transport(phases, centered(np.log(times)), sources, train)
    bank = source_scenarios(times, shapes, sources, train, test)
    strength = fp16_fold['strength']
    source_validation = fp16_fold['source_validation']
    forced = min((.25, .50, 1.), key=lambda s: np.mean(source_validation[str(s)]))
    actual = times[target, test]
    banks = {
        'blind': bank,
        'hardware': shift_bank(bank, phases, sources, test, target, model, strength),
        'neutral': shift_bank(bank, phases, sources, test, target, model, strength, neutral=True),
        'hardware_unscreened': shift_bank(bank, phases, sources, test, target, model, forced),
    }
    scores = {name: [dict(shape=shapes[i], **value) for i, value in
                     zip(test, evaluate_bank(cases, actual))] for name, cases in banks.items()}
    return dict(strength=strength, coefficient=model.tolist(), scores=scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--fp16-validation', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assignment = json.loads(args.split.read_text())['assignment']
    validation = json.loads(args.fp16_validation.read_text())['splits'][args.split.stem]['families']
    mode = 'graph'
    metadata = json.loads((args.inputs / 'hardware.json').read_text())
    source, hashes = training_rows(args.inputs, mode, metadata)
    paths = (args.inputs / 'a5000-pass00-raw.tar.gz', args.inputs / 'a5000-pass00-verified.json')
    a5000, _ = load_a5000(*paths, args.inputs / 'h100.zip', mode=mode)
    hashes.update({str(p): sha(p.read_bytes()) for p in paths})
    rows = [r for r in source + a5000 if r['dtype'] == 'bfloat16']
    if len(rows) != 45000:
        raise ValueError('Expected 15,000 BF16 candidates per supported GPU')
    folds = {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            shapes, configs, ordered = verify_matched_grid([r for r in rows if r['kernel_family'] == family])
            dims = (len(DEVICES), len(shapes), len(configs))
            # Reuse the same two-byte workload geometry for both native formats.
            features = centered(physical_phases([{**r, 'dtype': 'float16'} for r in ordered], dims))
            times = np.asarray([measure(r, mode) for r in ordered]).reshape(dims)
            folds[family] = {device: evaluate_fold(times, features, shapes, assignment, d, validation[family][device])
                             for d, device in enumerate(DEVICES)}
    report = {'timing': mode, 'dtype': 'bfloat16', 'devices': DEVICES, 'candidates': len(rows),
              'protocol': ('Stress test with graph-replay BF16 outcomes. Fix transport strength from FP16 '
                           'source-GPU validation of the corresponding outer fold. Refit phase coefficients '
                           'on the remaining two BF16 GPUs only. Matched test shapes across devices.'),
              'split_sha256': sha(args.split.read_bytes()), 'archive_sha256': hashes,
              'fp16_validation_sha256': sha(args.fp16_validation.read_bytes()),
              'summary': summarize(folds), 'families': folds}
    save_json(args.output, report)


if __name__ == '__main__':
    main()
