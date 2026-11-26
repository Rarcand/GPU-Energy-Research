#!/usr/bin/env python3
"""Check that completed GPU collections cover the same candidate keys."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def inventory(directory, pattern="*-scale-*.csv"):
    attempts = set()
    valid = set()
    models = set()
    duplicates = 0
    files = sorted(directory.glob(pattern))
    if not files:
        raise SystemExit(f"No scale collection files in {directory}")
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                models.add(row["gpu_name"])
                key = tuple(row[field] for field in ("kernel_family", "workload_id", "config_id"))
                if not all(key):
                    raise SystemExit(f"Missing candidate identifier in {path}")
                duplicates += key in attempts
                attempts.add(key)
                try:
                    latency = float(row.get("latency_median_ms", ""))
                except (TypeError, ValueError):
                    continue
                flag = row.get("is_valid", "").strip().lower()
                if flag in {"1", "true", "yes"} and math.isfinite(latency) and latency > 0:
                    valid.add(key)
    if len(models) != 1 or "" in models:
        raise SystemExit(f"Expected one GPU model in {directory}, found {sorted(models)}")
    return models.pop(), attempts, valid, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_dirs", type=Path, nargs="+")
    parser.add_argument("--matched30k", action="store_true",
                        help="audit the seven-family 30,000-candidate sweep")
    args = parser.parse_args()
    if len(args.gpu_dirs) < 2:
        parser.error("provide at least two GPU result directories")
    pattern = "*-matched30k-*.csv" if args.matched30k else "*-scale-*.csv"
    expected_count = 30000 if args.matched30k else 36000
    inventories = [inventory(directory, pattern) for directory in args.gpu_dirs]
    names = [name for name, _, _, _ in inventories]
    if len(set(names)) != len(names):
        parser.error("each directory must represent a distinct GPU model")
    union = set.union(*(attempts for _, attempts, _, _ in inventories))
    common_attempts = set.intersection(*(attempts for _, attempts, _, _ in inventories))
    common_valid = set.intersection(*(valid for _, _, valid, _ in inventories))
    report = {
        "devices": {
            name: {
                "unique_attempts": len(attempts), "valid_latency_rows": len(valid),
                "missing_vs_union": len(union - attempts), "duplicates": duplicates,
            }
            for name, attempts, valid, duplicates in inventories
        },
        "matched_attempts_per_gpu": len(common_attempts),
        "common_valid_per_gpu": len(common_valid),
        "common_valid_by_family": dict(sorted(Counter(key[0] for key in common_valid).items())),
    }
    print(json.dumps(report, indent=2))
    if any(len(attempts) != expected_count or attempts != union or duplicates
           for _, attempts, _, duplicates in inventories):
        raise SystemExit("Collection incomplete, unmatched, or duplicated.")


if __name__ == "__main__":
    main()
