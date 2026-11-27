"""Qualify native A100 timing on a fixed two-round panel before matched collection."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import clock_timing as timer


def configure():
    shared = Path(__file__).resolve().parents[1] / 'shared'
    frozen = shared / 'kernels'
    sys.path[:0] = [str(shared), str(frozen)]
    import pilot_control as control
    import timing_validation as source
    import torch
    import triton
    import benchmark_matmul as gemm
    import benchmark_reductions as reductions
    import extra_triton_kernels as extra
    from benchmark_extra_kernels import cases

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected an NVIDIA A100')
    return control, source, torch, triton, (gemm, reductions, extra, cases), frozen


def load_prior(path):
    if not path:
        return {}
    with path.open() as stream:
        return {
            (row['kernel_family'], row['workload_id'], row['config_id'], row['round']): row
            for row in map(json.loads, stream)
        }


def run_rows(rows, rounds, file, env, clock, pilot=False):
    control, source, torch, _, dependencies, _ = env
    groups = defaultdict(list)
    for row in rows:
        groups[row['workload_id']].append(row)

    hashes = {}
    records = []
    with file.open('x', buffering=1) as dest:
        for round_index in rounds:
            rng = random.Random(889321 + round_index)
            workloads = sorted(groups)
            rng.shuffle(workloads)
            print(file.stem, 'round', round_index, 'started', flush=True)
            for key in workloads:
                members = sorted(groups[key], key=lambda row: row['config_id'])
                rng.shuffle(members)
                execute, output, reference, atol, rtol, digest = source.workload(
                    members[0], torch, *dependencies,
                )
                if hashes.setdefault(key, digest) != digest:
                    raise RuntimeError(f'Input changed across rounds: {key}')
                for row in members:
                    config = json.loads(row['config_features_json'])

                    def launch():
                        execute(config)

                    launch()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output.float(), reference.float(), atol=atol, rtol=rtol)
                    item = {k: row[k] for k in (
                        'kernel_family', 'workload_id', 'config_id', 'dtype',
                        'workload_features_json', 'config_features_json',
                    )}
                    item.update(round=round_index, input_sha256=digest)
                    if pilot:
                        # Replicate the short-warmup timing on this GPU before the
                        # clock-verified measurement. Cooling the pilot control is
                        # deliberate and kept out of the production code path.
                        time.sleep(.25)
                        item['short_warm'] = control.controlled(launch, torch)
                        item['short_warm_end_clocks_mhz'] = clock()
                    item['controlled'] = timer.measure(launch, torch, clock, audit_trace=pilot)
                    torch.testing.assert_close(output.float(), reference.float(), atol=atol, rtol=rtol)
                    item['correctness_pass'] = True
                    records.append(item)
                    dest.write(json.dumps(item, allow_nan=False) + '\n')
                    if not item['controlled']['valid']:
                        print('Unstable clocks', item['kernel_family'], key, item['config_id'],
                              round_index, flush=True)
    return records


def paired_report(records, families, control, expected_count):
    failed = [r for r in records if not r['controlled']['valid']]
    failed_keys = {(r['kernel_family'], r['workload_id'], r['config_id']) for r in failed}
    valid = [r for r in records if (r['kernel_family'], r['workload_id'], r['config_id']) not in failed_keys]
    scores = control.summarize(valid)
    passed = (len(records) == 2 * expected_count and not failed
              and control.passes_gate(scores, families))
    return dict(passed=passed, candidates=len(records)//2, clock_failures=[
        {k: r[k] for k in ('kernel_family', 'workload_id', 'config_id', 'round')} for r in failed
    ], controlled=scores, median_limit=control.MEDIAN_LIMIT, p90_limit=control.P90_LIMIT)


def size_digest(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--prior', type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.root = args.root.resolve()
    env = configure()
    control, source, torch, triton, _, frozen = env
    units = source.registry(args.root, 'A100')
    gpu = control.assigned_gpu(torch)
    clock = timer.ClockReader(gpu)
    fingerprint = subprocess.check_output(
        ['nvidia-smi', '-i', gpu, '--query-gpu=name,uuid,pci.bus_id,driver_version',
         '--format=csv,noheader'], text=True,
    ).strip()
    paths = [Path(__file__), Path(timer.__file__), Path(control.__file__),
             Path(source.__file__), *sorted(frozen.glob('*.py'))]
    source.atomic(args.output / 'protocol.json', dict(
        mode='pilot', uuid=gpu, gpu=fingerprint, hostname=platform.node(),
        job=os.getenv('SLURM_JOB_ID'), partition=os.getenv('SLURM_JOB_PARTITION'),
        torch=torch.__version__, triton=triton.__version__,
        interval_depth=timer.DEPTH, intervals_per_graph=timer.INTERVALS,
        measured_replays=timer.REPLAYS, minimum_warm_seconds=timer.MIN_WARM_S,
        plateau_window_seconds=timer.CLOCK_WINDOW_S, maximum_warm_seconds=timer.MAX_WARM_S,
        clock_tolerance=timer.SM_TOLERANCE, warm_latency_tolerance=timer.LATENCY_TOLERANCE,
        code={str(p.resolve().relative_to(args.root)): size_digest(p) for p in paths},
        inputs={str(p.relative_to(args.root)): size_digest(p) for p, _ in units.values()},
    ))
    print('GPU', fingerprint, 'host', platform.node(), flush=True)
    with control.telemetry(args.output / 'gpu-telemetry.csv', gpu):
        selection = control.select_pilot(units)
        expected = len(selection)
        pilot = run_rows(selection, (0, 1), args.output / 'pilot.jsonl', env, clock, pilot=True)
        report = paired_report(pilot, source.FAMILIES, control, expected)
        report['short_warm_same_node'] = control.summarize(pilot, field='short_warm')
        prior = load_prior(args.prior)
        if prior:
            matched = []
            for r in pilot:
                key = tuple(r[k] for k in ('kernel_family', 'workload_id', 'config_id', 'round'))
                old = prior[key]
                if old['input_sha256'] != r['input_sha256']:
                    raise RuntimeError(f'Prior pilot input mismatch: {key}')
                matched.append(old)
            report['prior_pilot_other_allocation'] = control.summarize(matched)
        warm = [r['controlled']['warmup_seconds'] for r in pilot if r['controlled']['valid']]
        if warm:
            report['median_warmup_seconds'] = statistics.median(warm)
        source.atomic(args.output / 'pilot.summary.json', report)
        print('Pilot', json.dumps(report), flush=True)
        if not report['passed']:
            raise RuntimeError('Pilot exceeded the unchanged repeatability gate')


if __name__ == '__main__':
    main()
