#!/usr/bin/env python3
"""Run a fixed seven-family candidate grid, identical on each GPU."""

import argparse
import csv
import itertools
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark_extra_kernels import FAMILIES, SHARDS, cases


SEED = 20261113
DTYPES = ("float16", "bfloat16")
EXPECTED = {
    "gemm": 7200, "softmax": 3408, "cross_entropy": 3408, "conv2d": 4992,
    "attention": 4000, "gather": 4000, "residual_layernorm": 2992,
}


def base_shapes():
    gemm = list(itertools.product(
        (128, 256, 384, 768, 1024, 1536, 3072),
        (192, 384, 768, 1024, 1536, 3072),
        (320, 640, 960, 1600, 2560),
    ))
    reductions = list(itertools.product(
        (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        (127, 192, 255, 257, 383, 511, 513, 767, 1023, 1025, 1535, 2047,
         2049, 3071, 4095, 6143, 8191, 12287),
    ))
    random.Random(SEED).shuffle(gemm)
    random.Random(SEED).shuffle(reductions)
    return gemm[:75], reductions[:142]


def shape_arg(shapes):
    return ",".join("x".join(map(str, shape)) for shape in shapes)


def steps(shard, smoke):
    gemm, reductions = base_shapes()
    if smoke:
        gemm, reductions = gemm[:1], reductions[:1]
    else:
        gemm, reductions = gemm[shard::SHARDS], reductions[shard::SHARDS]
    gemm_configs, reduction_configs = (2, 2) if smoke else (48, 12)
    options = ["--dtypes", ",".join(DTYPES), "--config-seed", str(SEED),
               "--power-window-ms", "0"]
    yield ("gemm", "benchmark_matmul.py",
           options + ["--shapes", shape_arg(gemm), "--max-configs", str(gemm_configs)],
           len(gemm) * gemm_configs * len(DTYPES))
    for family in ("softmax", "cross_entropy"):
        yield (family, "benchmark_reductions.py",
               options + ["--families", family, "--shapes", shape_arg(reductions),
                          "--max-configs", str(reduction_configs)],
               len(reductions) * reduction_configs * len(DTYPES))
    for family in FAMILIES:
        _, shapes, _, configs = cases(family, expanded=True)
        if smoke:
            yield family, "benchmark_extra_kernels.py", [family, "--smoke"], None
        else:
            yield (family, "benchmark_extra_kernels.py",
                   [family, "--expanded", "--shard", str(shard)],
                   len(shapes[shard::SHARDS]) * len(configs) * len(DTYPES))


def verify(path, family, expected, smoke):
    keys = set()
    valid_dtypes = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["workload_id"], row["config_id"])
            if row["kernel_family"] != family or key in keys:
                raise ValueError(f"Unexpected family or duplicate candidate in {path}: {key}")
            keys.add(key)
            if row["is_valid"].lower() == "true":
                valid_dtypes.add(row["dtype"])
    if not keys or (expected is not None and len(keys) != expected):
        raise ValueError(f"{path}: expected {expected}, got {len(keys)} candidates")
    if smoke and valid_dtypes != set(DTYPES):
        raise RuntimeError(f"{family} smoke run missing successful timings: {valid_dtypes}")
    return len(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--shards", nargs="+", type=int, choices=range(SHARDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--session-id", default="matched30k")
    args = parser.parse_args()
    if args.plan:
        counts = Counter()
        for shard in range(SHARDS):
            for family, _, _, count in steps(shard, False):
                counts[family] += count
        if dict(counts) != EXPECTED or sum(counts.values()) != 30000:
            raise SystemExit(f"Unexpected grid: {counts}")
        for family, count in counts.items():
            print(f"{family}: {count:,}")
        print(f"Total: {sum(counts.values()):,} per GPU")
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    shards = [0] if args.smoke else args.shards
    if len(set(shards)) != len(shards):
        parser.error("shard IDs must be distinct")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.publish_dir is not None:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
    mode_label = "smoke" if args.smoke else "matched30k"
    script_dir = Path(__file__).resolve().parent
    for shard in shards:
        for family, script, options, count in steps(shard, args.smoke):
            path = args.output_dir / f"{family}-{mode_label}-{args.session_id}-{shard:02d}.csv"
            if path.exists():
                raise FileExistsError(path)
            subprocess.run([
                sys.executable, str(script_dir / script), *options,
                "--output", str(path), "--session-id", args.session_id,
                "--seed", str(SEED + shard),
            ], check=True)
            verified = verify(path, family, count, args.smoke)
            if args.publish_dir is not None:
                destination = args.publish_dir / path.name
                with destination.with_suffix(".part").open("xb") as output, path.open("rb") as source:
                    shutil.copyfileobj(source, output)
                destination.with_suffix(".part").rename(destination)
            print(f"{family} shard {shard}: verified {verified:,} attempts", flush=True)


if __name__ == "__main__":
    main()
