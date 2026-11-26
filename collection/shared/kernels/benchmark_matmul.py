#!/usr/bin/env python3
"""Collect randomized, architecture-aware Triton GEMM measurements."""

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from measurement_utils import (
    CandidateEvaluationError, architecture_metadata, atomic_write_csv,
    candidate_failure_record, candidate_success_record, dynamic_telemetry,
    power_window, run_metadata, safe_ratio, timing_summary,
)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m_size, BLOCK_M)
    num_pid_n = tl.cdiv(n_size, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m_size) &
                    (offs_k[None, :] + k_block * BLOCK_K < k_size), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k_block * BLOCK_K < k_size) &
                    (offs_n[None, :] < n_size), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


def make_configs(max_configs, seed, profile):
    """Sample independent tile, warp, and stage knobs to reduce confounding."""
    if profile == "boundary":
        # These are legitimate Triton tuning choices, but combinations near the
        # resource boundary can fail compilation on a particular architecture.
        domains = ((32, 64, 128, 256), (32, 64, 128, 256),
                   (16, 32, 64, 128), (4, 8), (2, 3, 4, 5))
    else:
        domains = ((32, 64, 128), (32, 64, 128),
                   (16, 32, 64), (4, 8), (2, 3, 4))
    configs = list(itertools.product(
        *domains,
    ))
    random.Random(seed).shuffle(configs)
    return configs[:max_configs] if max_configs else configs


def launch(a, b, output, config):
    block_m, block_n, block_k, num_warps, num_stages = config
    grid = (triton.cdiv(a.shape[0], block_m) * triton.cdiv(b.shape[1], block_n),)
    matmul_kernel[grid](
        a, b, output, a.shape[0], b.shape[1], a.shape[1],
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=8, num_warps=num_warps, num_stages=num_stages,
    )


def measure(a, b, output, config, warmup, repetitions, check):
    try:
        compile_start = time.perf_counter()
        launch(a, b, output, config)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - compile_start) * 1e3
    except Exception as exc:
        raise CandidateEvaluationError("compile_or_first_launch", exc) from exc
    correctness_pass, max_abs_error, max_rel_error = "", "", ""
    if check:
        try:
            expected = torch.matmul(a, b)
            difference = (output.float() - expected.float()).abs()
            max_abs_error = difference.max().item()
            max_rel_error = (difference / expected.float().abs().clamp_min(1e-6)).max().item()
            tolerance = 5e-2 if a.dtype == torch.bfloat16 else 2e-2
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
            correctness_pass = True
        except Exception as exc:
            raise CandidateEvaluationError("correctness", exc) from exc
    try:
        for _ in range(warmup):
            launch(a, b, output, config)
        torch.cuda.synchronize()
        timings = []
        for _ in range(repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); launch(a, b, output, config); end.record(); end.synchronize()
            timings.append(start.elapsed_time(end))
    except Exception as exc:
        raise CandidateEvaluationError("runtime", exc) from exc
    return compile_ms, correctness_pass, max_abs_error, max_rel_error, timings


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        try:
            shape = tuple(int(part) for part in item.lower().split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid shape: {}".format(item)) from exc
        if len(shape) != 3:
            raise argparse.ArgumentTypeError("GEMM shapes must be MxNxK")
        shapes.append(shape)
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shapes", default=(
        "256x4096x1024,4096x256x1024,768x1024x3072,"
        "1024x1024x1024,2048x2048x2048,4096x4096x1024"
    ))
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--max-configs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config-seed", type=int, default=2026,
                        help="Fixed seed for the sampled configuration subset")
    parser.add_argument("--config-profile", choices=("safe", "boundary"), default="safe")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--power-window-ms", type=int, default=250)
    parser.add_argument("--skip-correctness-check", action="store_true")
    args = parser.parse_args()
    # Correctness reference: retain FP32 intermediate accumulation. Reduced-
    # precision cuBLAS reductions can otherwise reject accurate Triton results.
    # These flags affect the PyTorch reference, not the timed Triton kernel.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is visible; submit this to a GPU queue")
    if args.warmup < 1 or args.repetitions < 2:
        raise SystemExit("Use at least 1 warmup and 2 measured repetitions")
    shapes = parse_shapes(args.shapes)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        dtypes = [(name, dtype_map[name]) for name in args.dtypes.split(",")]
    except KeyError as exc:
        raise SystemExit("Unsupported dtype: {}".format(exc.args[0])) from exc

    configs = make_configs(args.max_configs, args.config_seed, args.config_profile)
    randomizer = random.Random(args.seed)
    architecture = architecture_metadata()
    run = run_metadata(args.seed, args.session_id)
    run["triton_version"] = triton.__version__
    run["reference_policy"] = "torch_gemm_fp32_accumulation_v2"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rows = []
    work_items = list(itertools.product(shapes, dtypes)); randomizer.shuffle(work_items)
    for (m_size, n_size, k_size), (dtype_name, dtype) in work_items:
        a = torch.randn((m_size, k_size), device="cuda", dtype=dtype)
        b = torch.randn((k_size, n_size), device="cuda", dtype=dtype)
        output = torch.empty((m_size, n_size), device="cuda", dtype=dtype)
        element_size = a.element_size()
        flop_count = 2 * m_size * n_size * k_size
        working_set_bytes = (
            m_size * k_size + k_size * n_size + m_size * n_size
        ) * element_size
        config_order = list(configs); randomizer.shuffle(config_order)
        for order_index, config in enumerate(config_order):
            block_m, block_n, block_k, num_warps, num_stages = config
            grid_blocks = math.ceil(m_size / block_m) * math.ceil(n_size / block_n)
            threads_per_program = num_warps * 32
            estimated_shared_bytes = (
                (block_m * block_k + block_k * block_n) * element_size * num_stages
            )
            workload_features = {
                "m": m_size, "n": n_size, "k": k_size, "dtype": dtype_name,
                "aspect_m_n": m_size / n_size, "aspect_k_n": k_size / n_size,
            }
            config_features = {
                "block_m": block_m, "block_n": block_n, "block_k": block_k,
                "num_warps": num_warps, "num_stages": num_stages, "group_m": 8,
            }
            base = {
                **run, **architecture, "candidate_order": order_index,
                "kernel_family": "gemm",
                "workload_id": "gemm-{}x{}x{}-{}".format(m_size, n_size, k_size, dtype_name),
                "config_id": "bm{}-bn{}-bk{}-w{}-s{}".format(*config),
                "sampling_policy": args.config_profile,
                "workload_features_json": json.dumps(workload_features, sort_keys=True),
                "config_features_json": json.dumps(config_features, sort_keys=True),
                "m": m_size, "n": n_size, "k": k_size, "dtype": dtype_name,
                "block_m": block_m, "block_n": block_n, "block_k": block_k,
                "num_warps": num_warps, "num_stages": num_stages,
                "flop_count": flop_count, "estimated_bytes": working_set_bytes,
                "arithmetic_intensity": safe_ratio(flop_count, working_set_bytes),
                "working_set_fraction_of_memory": safe_ratio(
                    working_set_bytes, architecture["total_memory_bytes"]),
                "grid_blocks": grid_blocks,
                "programs_per_sm": safe_ratio(grid_blocks, architecture["sm_count"]),
                "execution_waves_estimate": safe_ratio(grid_blocks, architecture["sm_count"]),
                "threads_per_program": threads_per_program,
                "threads_fraction_of_block_limit": safe_ratio(
                    threads_per_program, architecture["max_threads_per_block"]),
                "estimated_shared_bytes": estimated_shared_bytes,
                "shared_fraction_of_sm": safe_ratio(
                    estimated_shared_bytes, architecture["shared_memory_per_sm_bytes"]),
            }
            try:
                compile_ms, correct, max_abs, max_rel, timings = measure(
                    a, b, output, config, args.warmup, args.repetitions,
                    not args.skip_correctness_check)
                summary = timing_summary(timings)
                energy = power_window(lambda: launch(a, b, output, config), args.power_window_ms)
                rows.append({
                    **base, **summary, **energy, **dynamic_telemetry(), "compile_ms": compile_ms,
                    "correctness_pass": correct, "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "achieved_tflops": flop_count / (summary["latency_median_ms"] * 1e9),
                    **candidate_success_record(),
                })
            except Exception as exc:
                stage = exc.stage if isinstance(exc, CandidateEvaluationError) else "runtime"
                rows.append({**base, **dynamic_telemetry(),
                             **candidate_failure_record(
                                 stage, exc, not args.skip_correctness_check)})
            finally:
                torch.cuda.empty_cache()
    atomic_write_csv(args.output, rows)
    print("Wrote {} GEMM rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
