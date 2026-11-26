"""Replay a physical-capacity correction of source timing scenarios on unseen GPUs."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from audit_timing_protocol_transfer import measure
from benchmark_few_shot_surrogate import metric_for_order, replay, source_scenarios
from compare_gemm_resource_residual import aligned_grid
from compare_portable_surrogates import DEVICES, load_matched
from compare_surrogates import FAMILIES, save_json
from gather_timing import sha
from gemm_resource_features import _shared_capacity
from kernel_resource_features import features_for_family


# Dense FP16 tensor arithmetic with FP32 accumulation, memory bandwidth,
# and peak FP32 SIMD rate. The A5000 tensor estimate applies the dense
# FP32-accumulate to SIMD ratio of the RTX 3070's GA10x SM (2.0) to 27.8.
SPECS = {'H100': (756.5, 2000., 51.), 'RTX3070-WSL': (40.6, 448., 20.3),
         'RTXA5000': (55.6, 768., 27.8), 'T4': (65., 320., 8.1)}
SPEC_SOURCES = {
    'H100': ('https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306',
             'https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf'),
    'RTX3070-WSL': ('https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf',),
    'RTXA5000': ('https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf',
                'https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf'),
    'T4': ('https://www.nvidia.com/en-us/data-center/tesla-t4/',),
}
STRENGTHS = (0., .25, .50, 1.)
METHODS = ('blind', 'hardware', 'neutral', 'hardware_unscreened')
METRICS = ('success5_b1', 'success5_b4', 'success5_b8', 'regret_b1', 'regret_b4', 'regret_b8')


def physical_phases(rows, dims):
    geometry, _, names = features_for_family(rows)
    g = dict(zip(names['geometry'], geometry.T))
    family = rows[0]['kernel_family']
    ctas, warps = g['cta_count'], g['num_warps']
    zeros = np.zeros_like(ctas)
    tensor_ops, scalar_ops, traffic, stages, iterations = zeros, zeros, zeros, zeros, np.ones_like(ctas)
    if family == 'gemm':
        tensor_ops = g['valid_fma_flops'] / g['useful_fma_fraction']
        working = g['fp16_input_bytes'] + 2 * g['m'] * g['n']
        traffic = g['cta_logical_read_bytes'] + 2 * g['m'] * g['n']
        iterations = g['k_steps']
        stages = g['staged_panel_bytes_proxy']
        values = g['block_m'] * g['block_n']
    else:
        working = g['working_set_bytes_proxy']
        traffic = working
        values = 32 * warps * g['values_per_thread_proxy']
        if family == 'conv2d':
            tensor_ops = g['useful_flops'] / g['useful_tile_fraction']
            iterations = g['reduction_steps']
            traffic = (ctas * iterations * g['panel_bytes_proxy'] + 2 * ctas * values)
            # The kernel computes input addresses and masks for the implicit im2col tile.
            scalar_ops = 8 * ctas * iterations * g['block_m'] * g['block_k']
            stages = g['staged_bytes_proxy']
        elif family == 'attention':
            tensor_ops = g['padded_dot_work']
            iterations = g['kv_steps']
            traffic = 4 * (ctas * g['block_m'] + ctas * iterations * g['block_n']) * g['head_dim']
            scalar_ops = 10 * ctas * iterations * g['block_m'] * g['block_n']
            stages = g['staged_bytes_proxy']
        elif family == 'gather':
            traffic = 4 * g['lookups'] * g['width'] + 4 * g['lookups'] * np.ceil(g['width'] / g['block_cols'])
            scalar_ops = 3 * ctas * values
        elif family == 'residual_layernorm':
            scalar_ops = 10 * ctas * values
            traffic = 6 * g['rows'] * g['width'] + 4 * ctas * g['width']
        else:
            scalar_ops = (8 if family == 'cross_entropy' else 7) * ctas * values

    tensor_rate, memory_rate, scalar_rate = np.asarray([SPECS[row['_device']] for row in rows]).T
    sm, l2, threads, registers = np.asarray(
        [[float(row[k]) for k in ('sm_count', 'l2_cache_bytes', 'max_threads_per_sm', 'registers_per_sm')]
         for row in rows]).T
    shared = np.asarray([_shared_capacity(row) for row in rows])
    regs = 16 * (32 * warps) + values
    occupancy = np.maximum(1, np.minimum.reduce((threads // (32 * warps), registers // regs,
                                                 shared // np.maximum(stages, 1))))
    waves = np.ceil(ctas / (sm * occupancy))
    resident_share = np.minimum(ctas / (sm * waves * np.minimum(occupancy, 2)), 1.)
    active_sm_share = np.minimum(ctas / (sm * np.ceil(ctas / sm)), 1.)
    # Smooth interpolation from compulsory traffic to repeated CTA loads as the
    # working-set estimate grows past L2. These are proxies, not traffic counters.
    spill = (working / (working + l2)) ** 2
    dram_bytes = working + spill * np.maximum(traffic - working, 0.)
    tensor_us = tensor_ops / (tensor_rate * 1e6 * resident_share)
    scalar_us = scalar_ops / (scalar_rate * 1e6 * resident_share)
    dram_us = dram_bytes / (memory_rate * 1e3 * active_sm_share)
    onchip_us = traffic / (memory_rate * 1e3 * 4. * resident_share)
    issue_proxy = ctas * warps / (sm * 64.)
    iteration_proxy = waves * iterations / (occupancy * 16.)
    xs = np.log1p(np.column_stack((tensor_us, scalar_us, dram_us, onchip_us, issue_proxy,
                                   iteration_proxy, np.maximum.reduce((tensor_us, scalar_us, dram_us)))))
    return xs.reshape((*dims, -1))


def centered(terms):
    return terms - terms.mean(axis=2, keepdims=True)


def learn_transport(xs, ys, sources, train):
    pairs = tuple(combinations(sources, 2))
    x = np.concatenate([(xs[a, train] - xs[b, train]).reshape(-1, xs.shape[-1]) for a, b in pairs])
    y = np.concatenate([(ys[a, train] - ys[b, train]).ravel() for a, b in pairs])
    regularizer = np.sqrt(.02 * len(x)) * np.eye(x.shape[-1])
    return nnls(np.vstack((x, regularizer)), np.r_[y, np.zeros(x.shape[-1])])[0]


def shift_bank(bank, xs, sources, queries, target, coefficients, strength, neutral=False):
    source = xs[sources][:, queries]
    destination = source.mean(axis=0) if neutral else xs[target, queries]
    delta = np.clip((destination - source) @ coefficients, -.75, .75)
    transported = bank + strength * delta.transpose(1, 0, 2)[:, :, None, :]
    return transported - transported.min(axis=-1, keepdims=True)


def evaluate_bank(bank, labels):
    weights = np.full(bank.shape[1], 1. / bank.shape[1])
    return [metric_for_order(replay(cases, weights, y, adaptive=True), y)
            for cases, y in zip(bank, labels)]


def evaluate_fold(times, xs, shapes, assignment, target):
    sources = [i for i in range(len(DEVICES)) if i != target]
    roles = ('train', 'validation', 'test')
    train, valid, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                          for role in roles)
    if min(map(len, (train, valid, test))) == 0 or any(
            assignment[shape] not in (*roles, 'excluded') for shape in shapes):
        raise ValueError('Expected three nonempty shape partitions, with optional excluded shapes')
    ys = centered(np.log(times))
    validation = {strength: [] for strength in STRENGTHS}
    for hidden in sources:
        inner = [i for i in sources if i != hidden]
        model = learn_transport(xs, ys, inner, train)
        bank = source_scenarios(times, shapes, inner, train, valid)
        for strength in STRENGTHS:
            adjusted = shift_bank(bank, xs, inner, valid, hidden, model, strength)
            scores = evaluate_bank(adjusted, times[hidden, valid])
            validation[strength].append(float(np.mean([np.log1p(s['regret_b4']) + .25 *
                                                       np.log1p(s['regret_b1']) for s in scores])))
    selected = min(STRENGTHS, key=lambda strength: np.mean(validation[strength]))
    forced = min(STRENGTHS[1:], key=lambda strength: np.mean(validation[strength]))
    model = learn_transport(xs, ys, sources, train)
    bank = source_scenarios(times, shapes, sources, train, test)
    labels = times[target, test]
    banks = {'blind': bank, 'hardware': shift_bank(bank, xs, sources, test, target, model, selected),
             'neutral': shift_bank(bank, xs, sources, test, target, model, selected, neutral=True),
             'hardware_unscreened': shift_bank(bank, xs, sources, test, target, model, forced)}
    scores = {method: [dict(shape=shapes[i], **score) for i, score in
                       zip(test, evaluate_bank(variants, labels))] for method, variants in banks.items()}
    return dict(coefficient=model.tolist(), strength=selected, source_validation=validation, scores=scores)


def summarize(families):
    records = {m: {k: [np.asarray([[r[k] for r in fold['scores'][m]] for fold in f.values()])
                       for f in families.values()] for k in METRICS} for m in METHODS}
    means = {m: {k: float(np.mean([v.mean() for v in all_values]))
                 for k, all_values in metrics.items()} for m, metrics in records.items()}
    comparisons = {}
    rng = np.random.default_rng(273581)
    for method, control in (('hardware', 'blind'), ('hardware', 'neutral'), ('hardware_unscreened', 'blind')):
        contrast = {}
        for metric in METRICS:
            deltas = [(a - b).mean(axis=0) for a, b in zip(records[method][metric], records[control][metric])]
            bootstrap = [v[rng.integers(len(v), size=(10000, len(v)))].mean(axis=1) for v in deltas]
            contrast[metric] = dict(delta=float(np.mean([v.mean() for v in deltas])),
                                    ci95=np.quantile(np.mean(bootstrap, axis=0), (.025, .975)).tolist(),
                                    by_family={f: float(v.mean()) for f, v in zip(families, deltas)})
        comparisons[f'{method}_minus_{control}'] = contrast
    return dict(macro=means, contrasts=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--splits', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing', choices=('graph', 'individual'), default='graph')
    args = parser.parse_args()
    rows, _, hashes = load_matched(args.inputs, args.t4, args.timing)
    report = dict(timing=args.timing, candidates=len(rows), archive_hashes=hashes, devices=DEVICES,
                  capacities=SPECS, sources=SPEC_SOURCES,
                  protocol='FP16 leave-one-GPU-and-shape-out replay. Source-only nested selection. '
                           'Primary endpoint B4 success within 5% of the measured oracle. '
                           'Paired bootstrap over shapes within families averaged over four fixed devices.',
                  splits={p.stem: {} for p in args.splits},
                  split_hashes={p.stem: sha(p.read_bytes()) for p in args.splits})
    assignments = {p.stem: json.loads(p.read_text())['assignment'] for p in args.splits}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            shapes, configs, ordered = aligned_grid([r for r in rows if r['kernel_family'] == family])
            dims = (len(DEVICES), len(shapes), len(configs))
            xs = centered(physical_phases(ordered, dims))
            times = np.asarray([measure(r, args.timing) for r in ordered]).reshape(dims)
            for name, assignment in assignments.items():
                report['splits'][name][family] = {
                    device: evaluate_fold(times, xs, shapes, assignment, index)
                    for index, device in enumerate(DEVICES)}
            print(f'{args.timing}: {family}', flush=True)
    for name, families in report['splits'].items():
        report['splits'][name] = {'summary': summarize(families), 'families': families}
    save_json(args.output, report)


if __name__ == '__main__':
    main()
