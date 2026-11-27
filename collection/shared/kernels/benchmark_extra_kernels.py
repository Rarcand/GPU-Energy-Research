#!/usr/bin/env python3
"""Collect matched Triton timings for four additional kernel families."""

import argparse
import itertools
import json
import random
import time
from pathlib import Path


FAMILIES = ("conv2d", "attention", "gather", "residual_layernorm")
SHARDS = 10


def cases(family, expanded=False):
    if family == "conv2d":
        # NHWC input; ResNet-style spatial sizes, channel counts and strides.
        shapes = [(b, h, ci, co, k, s) for b in (1, 4, 16)
                  for h, ci, co, k, s in (
                      (56, 64, 256, 1, 1), (56, 64, 64, 3, 1),
                      (56, 128, 128, 3, 2), (28, 128, 512, 1, 1),
                      (28, 128, 128, 3, 1), (14, 256, 256, 3, 1),
                      (14, 256, 1024, 1, 1), (7, 512, 2048, 1, 1))]
        names = ("batch", "height", "in_channels", "out_channels", "kernel", "stride")
        knobs = ("block_m", "block_n", "block_k", "num_warps")
        configs = itertools.product((16, 32, 64), (32, 64), (16, 32), (4, 8))
    elif family == "attention":
        # Batch 1, multiple heads; q_len=1 represents a single decode query.
        shapes = [(heads, q, kv, d, causal)
                  for heads in (4, 8) for d in (32, 64, 128)
                  for q, kv, causal in ((1, 256, False), (1, 2048, False),
                                        (128, 128, False), (256, 256, True),
                                        (512, 512, False), (1024, 1024, True))]
        names = ("heads", "q_len", "kv_len", "head_dim", "causal")
        knobs = ("block_m", "block_n", "num_warps")
        configs = itertools.product((16, 32), (32, 64), (4, 8))
    elif family == "gather":
        shapes = list(itertools.product(
            (4096, 65536), (128, 1024, 4096), (64, 128, 256, 513),
            ("uniform", "clustered", "repeated"),
        ))
        names = ("vocab", "lookups", "width", "locality")
        knobs = ("block_rows", "block_cols", "num_warps")
        configs = itertools.product((1, 4, 8, 16), (32, 64, 128), (4, 8))
    else:
        shapes = list(itertools.product(
            (1, 32, 128, 512, 2048, 4096),
            (128, 256, 511, 768, 1024, 1536, 2048, 3072),
        ))
        names = ("rows", "width")
        knobs = ("block_rows", "num_warps")
        configs = itertools.product((1, 2, 4, 8), (4, 8))
    configs = list(configs)
    if expanded:
        additional, count = {
            "conv2d": (itertools.product(
                (1, 2, 4, 8), (7, 14, 28, 56, 112), (32, 64, 96, 128, 256),
                (32, 64, 128, 256), (1, 3), (1, 2)), 104),
            "attention": ((
                (h, q, kv, d, causal) for h in (1, 4, 8, 12)
                for d in (32, 64, 128)
                for q, kv, causal in [
                    *((1, k, False) for k in (63, 128, 257, 768, 1024, 1536, 3072)),
                    *((n, n, mask) for n in (63, 128, 192, 257, 384, 768, 1024)
                      for mask in (False, True)),
                ]), 250),
            "gather": (itertools.product(
                (8192, 32768, 65536, 131072), (32, 256, 1536, 4096),
                (63, 128, 257, 512), ("uniform", "clustered", "repeated")), 100),
            "residual_layernorm": (itertools.product(
                (1, 8, 16, 64, 256, 768, 1536, 3072, 4096),
                (63, 96, 128, 192, 255, 256, 384, 511, 512, 640, 768,
                 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096)),
                187),
        }[family]
        existing = set(shapes)
        pool = [shape for shape in additional if shape not in existing]
        random.Random(20261113).shuffle(pool)
        shapes = (shapes + pool)[:count]
        if family == "gather":
            random.Random(20261113).shuffle(configs)
            configs = configs[:20]
    return names, shapes, knobs, configs


def measure(launch, output, reference, warmup, repetitions, torch):
    from measurement_utils import CandidateEvaluationError, timing_summary

    try:
        started = time.perf_counter()
        launch()
        torch.cuda.synchronize()
        compile_ms = 1000 * (time.perf_counter() - started)
    except Exception as exc:
        raise CandidateEvaluationError("compile_or_first_launch", exc) from exc
    try:
        torch.testing.assert_close(output, reference, atol=0.02, rtol=0.03)
    except AssertionError as exc:
        raise CandidateEvaluationError("correctness", exc) from exc
    try:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(repetitions):
            start.record()
            launch()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    except Exception as exc:
        raise CandidateEvaluationError("runtime", exc) from exc
    return {"compile_ms": compile_ms, "correctness_pass": True, **timing_summary(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=61205)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    fields, shapes, knobs, configs = cases(args.family, expanded=args.expanded)
    if args.shard is not None:
        shapes = shapes[args.shard::SHARDS]
    if args.smoke:
        if args.shard is not None:
            parser.error("--smoke uses a fixed cross-family sample, without --shard")
        picks = {
            "conv2d": (0, 1, 2),  # 1x1, padded 3x3, stride 2
            "attention": (0, 3, 5),  # decode, causal prefill
            "gather": (0, 1, 2, 9),  # three locality modes and a masked width
            "residual_layernorm": (0, 10),  # single row and masked width
        }
        shapes = [shapes[index] for index in picks[args.family]]
        configs = [configs[0], configs[-1]]
    if args.plan:
        print(f"{args.family}: {len(shapes)} shapes, {len(configs)} configs, "
              f"{len(shapes) * len(configs) * 2} attempts across FP16 and BF16")
        return
    if args.output is None or args.output.exists():
        parser.error("--output must name a new CSV file")
    if args.warmup < 1 or args.repetitions < 2:
        parser.error("use at least one warmup and two repetitions")

    import torch
    import triton
    from extra_triton_kernels import make_workload
    from measurement_utils import (
        CandidateEvaluationError, architecture_metadata, atomic_write_csv,
        candidate_failure_record, candidate_success_record, run_metadata,
    )

    if not torch.cuda.is_available():
        raise SystemExit("Run collection on a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {**run_metadata(args.seed, args.session_id), **architecture_metadata(),
                "triton_version": triton.__version__, "power_measurement_status": "disabled"}
    records = []
    for shape in shapes:
        workload = dict(zip(fields, shape))
        for dtype_name in ("float16", "bfloat16"):
            workload_id = "-".join(map(str, (args.family, *shape, dtype_name)))
            generator = torch.Generator().manual_seed(args.seed)
            launch, output, reference = make_workload(
                args.family, shape, getattr(torch, dtype_name), generator
            )
            ordered = configs.copy()
            random.Random(f"{args.seed}:{workload_id}").shuffle(ordered)
            for order, config in enumerate(ordered):
                parameters = dict(zip(knobs, config))
                record = {
                    **metadata, **workload, **parameters,
                    "kernel_family": args.family, "dtype": dtype_name,
                    "workload_id": workload_id, "candidate_order": order,
                    "config_id": "-".join(f"{key}{value}" for key, value in parameters.items()),
                    "workload_features_json": json.dumps(workload, sort_keys=True),
                    "config_features_json": json.dumps(parameters, sort_keys=True),
                }
                try:
                    record.update(measure(
                        lambda: launch(*config), output, reference,
                        args.warmup, args.repetitions, torch,
                    ))
                except CandidateEvaluationError as exc:
                    record.update(candidate_failure_record(exc.stage, exc))
                else:
                    record.update(candidate_success_record())
                records.append(record)
    atomic_write_csv(args.output, records)
    print(f"Wrote {len(records)} attempted candidates to {args.output}")


if __name__ == "__main__":
    main()
