#!/usr/bin/env python3
"""Benchmark Triton row-softmax and fused cross-entropy/reduction families."""

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from measurement_utils import (
    CandidateEvaluationError, architecture_metadata, atomic_write_csv,
    candidate_failure_record, candidate_success_record, dynamic_telemetry,
    power_window, run_metadata, safe_ratio, timing_summary,
)


@triton.jit
def row_softmax_kernel(x_ptr, y_ptr, rows: tl.constexpr, cols: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(y_ptr + row * cols + offsets, numerator / denominator, mask=mask)


@triton.jit
def cross_entropy_kernel(logits_ptr, target_ptr, loss_ptr,
                         rows: tl.constexpr, cols: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    logits = tl.load(logits_ptr + row * cols + offsets, mask=mask, other=-float("inf"))
    maximum = tl.max(logits, axis=0)
    log_normalizer = tl.log(tl.sum(tl.exp(logits - maximum), axis=0)) + maximum
    target = tl.load(target_ptr + row)
    target_logit = tl.load(logits_ptr + row * cols + target)
    tl.store(loss_ptr + row, log_normalizer - target_logit)


def configs(seed, max_configs, profile):
    domains = ((1, 2, 4), (2, 4, 8), (1, 2, 3, 4, 5)) if profile == "boundary" \
        else ((1, 2), (4, 8), (1, 2, 3))
    values = list(itertools.product(*domains))
    random.Random(seed).shuffle(values)
    return values[:max_configs] if max_configs else values


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        try:
            rows, cols = (int(part) for part in item.lower().split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid reduction shape: {}".format(item)) from exc
        shapes.append((rows, cols))
    return shapes


def launch(family, x, output, target, config):
    block_multiplier, num_warps, num_stages = config
    block_size = triton.next_power_of_2(x.shape[1]) * block_multiplier
    if block_size > 65536:
        raise ValueError("BLOCK_SIZE {} exceeds supported study limit".format(block_size))
    if family == "softmax":
        row_softmax_kernel[(x.shape[0],)](
            x, output, x.shape[0], x.shape[1], BLOCK_SIZE=block_size,
            num_warps=num_warps, num_stages=num_stages)
    else:
        cross_entropy_kernel[(x.shape[0],)](
            x, target, output, x.shape[0], x.shape[1], BLOCK_SIZE=block_size,
            num_warps=num_warps, num_stages=num_stages)
    return block_size


def measure(family, x, output, target, config, warmup, repetitions, check):
    try:
        compile_start = time.perf_counter()
        block_size = launch(family, x, output, target, config)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - compile_start) * 1e3
    except Exception as exc:
        stage = "precheck" if isinstance(exc, ValueError) else "compile_or_first_launch"
        raise CandidateEvaluationError(stage, exc) from exc
    correctness_pass, max_abs_error, max_rel_error = "", "", ""
    if check:
        try:
            expected = torch.softmax(x, dim=1) if family == "softmax" else F.cross_entropy(
                x.float(), target, reduction="none")
            difference = (output.float() - expected.float()).abs()
            max_abs_error = difference.max().item()
            max_rel_error = (difference / expected.float().abs().clamp_min(1e-6)).max().item()
            torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)
            correctness_pass = True
        except Exception as exc:
            raise CandidateEvaluationError("correctness", exc) from exc
    try:
        for _ in range(warmup):
            launch(family, x, output, target, config)
        torch.cuda.synchronize()
        timings = []
        for _ in range(repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); launch(family, x, output, target, config); end.record(); end.synchronize()
            timings.append(start.elapsed_time(end))
    except Exception as exc:
        raise CandidateEvaluationError("runtime", exc) from exc
    return block_size, compile_ms, correctness_pass, max_abs_error, max_rel_error, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families", default="softmax,cross_entropy")
    parser.add_argument("--shapes", default="256x512,1024x1024,4096x2048,2048x4096,512x8192")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--max-configs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config-seed", type=int, default=2026,
                        help="Fixed seed for the sampled configuration subset")
    parser.add_argument("--config-profile", choices=("safe", "boundary"), default="safe")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--power-window-ms", type=int, default=250)
    parser.add_argument("--skip-correctness-check", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is visible; submit this to a GPU queue")
    families = [name.strip() for name in args.families.split(",")]
    unknown = set(families) - {"softmax", "cross_entropy"}
    if unknown:
        raise SystemExit("Unsupported families: {}".format(sorted(unknown)))
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        dtypes = [(name, dtype_map[name]) for name in args.dtypes.split(",")]
    except KeyError as exc:
        raise SystemExit("Unsupported dtype: {}".format(exc.args[0])) from exc

    architecture = architecture_metadata()
    run = run_metadata(args.seed, args.session_id); run["triton_version"] = triton.__version__
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    randomizer = random.Random(args.seed)
    candidates = list(itertools.product(
        families, parse_shapes(args.shapes), dtypes,
        configs(args.config_seed, args.max_configs, args.config_profile)))
    randomizer.shuffle(candidates)
    rows = []
    for order_index, (family, shape, dtype_item, config) in enumerate(candidates):
        row_count, col_count = shape
        dtype_name, dtype = dtype_item
        block_multiplier, num_warps, num_stages = config
        x = torch.randn((row_count, col_count), device="cuda", dtype=dtype)
        target = torch.randint(0, col_count, (row_count,), device="cuda")
        output = torch.empty_like(x) if family == "softmax" else torch.empty(
            (row_count,), device="cuda", dtype=torch.float32)
        element_size = x.element_size()
        estimated_operations = row_count * col_count * (5 if family == "softmax" else 4)
        estimated_bytes = x.numel() * element_size
        estimated_bytes += output.numel() * output.element_size()
        if family == "cross_entropy":
            estimated_bytes += target.numel() * target.element_size()
        block_size = triton.next_power_of_2(col_count) * block_multiplier
        threads_per_program = num_warps * 32
        workload_features = {
            "rows": row_count, "reduction_width": col_count, "dtype": dtype_name,
            "masked_fraction": (block_size - col_count) / block_size,
        }
        config_features = {
            "block_size": block_size, "block_multiplier": block_multiplier,
            "num_warps": num_warps, "num_stages": num_stages,
        }
        base = {
            **run, **architecture, "candidate_order": order_index,
            "kernel_family": family,
            "workload_id": "{}-{}x{}-{}".format(family, row_count, col_count, dtype_name),
            "config_id": "bs{}-w{}-s{}".format(block_size, num_warps, num_stages),
            "sampling_policy": args.config_profile,
            "workload_features_json": json.dumps(workload_features, sort_keys=True),
            "config_features_json": json.dumps(config_features, sort_keys=True),
            "rows": row_count, "reduction_width": col_count, "dtype": dtype_name,
            "block_size": block_size, "block_multiplier": block_multiplier,
            "num_warps": num_warps, "num_stages": num_stages,
            "estimated_operations": estimated_operations, "estimated_bytes": estimated_bytes,
            "arithmetic_intensity": safe_ratio(estimated_operations, estimated_bytes),
            "working_set_fraction_of_memory": safe_ratio(
                estimated_bytes, architecture["total_memory_bytes"]),
            "grid_blocks": row_count,
            "programs_per_sm": safe_ratio(row_count, architecture["sm_count"]),
            "execution_waves_estimate": safe_ratio(row_count, architecture["sm_count"]),
            "threads_per_program": threads_per_program,
            "threads_fraction_of_block_limit": safe_ratio(
                threads_per_program, architecture["max_threads_per_block"]),
            "masked_lane_fraction": (block_size - col_count) / block_size,
        }
        try:
            _, compile_ms, correct, max_abs, max_rel, timings = measure(
                family, x, output, target, config, args.warmup, args.repetitions,
                not args.skip_correctness_check)
            summary = timing_summary(timings)
            energy = power_window(
                lambda: launch(family, x, output, target, config), args.power_window_ms)
            rows.append({
                **base, **summary, **energy, **dynamic_telemetry(), "compile_ms": compile_ms,
                "correctness_pass": correct, "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "effective_bandwidth_gbps": estimated_bytes / (summary["latency_median_ms"] * 1e6),
                **candidate_success_record(),
            })
        except Exception as exc:
            stage = exc.stage if isinstance(exc, CandidateEvaluationError) else "runtime"
            rows.append({**base, **dynamic_telemetry(),
                         **candidate_failure_record(
                             stage, exc, not args.skip_correctness_check)})
        finally:
            del x, target, output
            torch.cuda.empty_cache()
    atomic_write_csv(args.output, rows)
    print("Wrote {} reduction-family rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
