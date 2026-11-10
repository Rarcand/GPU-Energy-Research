#!/usr/bin/env python3
"""Evaluate architecture transfer, few-shot adaptation, and tuning efficiency.

The target GPU is never used for normalization or model selection.  Target
workload IDs are split into a calibration pool and a fixed held-out test pool.
Few-shot rows come only from the calibration pool, while every reported target
metric is computed on the same held-out target workloads.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_latency_baseline import (
    ARCH_FEATURES, ARCH_LOG_FEATURES, CONFIG_FEATURES, CONFIG_LOG_FEATURES,
    WORKLOAD_FEATURES, WORKLOAD_LOG_FEATURES, LatencySurrogate, Standardizer,
    as_float, encoded, load_rows, make_vocab, numeric_matrix,
    regression_metrics,
)


def architecture_matches(row, token):
    text = "{} {}".format(row.get("gpu_name", ""), row.get("architecture_id", ""))
    return token.lower() in text.lower()


def workload_group(row):
    return "{}::{}".format(row.get("kernel_family", "unknown"), row.get("workload_id", ""))


def stratified_groups(rows, seed, test_fraction):
    by_family = defaultdict(set)
    for row in rows:
        by_family[row.get("kernel_family", "unknown")].add(workload_group(row))
    rng = random.Random(seed)
    test_groups = set()
    calibration_groups = set()
    for family, groups in sorted(by_family.items()):
        values = sorted(groups)
        rng.shuffle(values)
        if len(values) < 2:
            raise SystemExit("Need at least two target workloads for {}".format(family))
        count = max(1, round(len(values) * test_fraction))
        count = min(count, len(values) - 1)
        test_groups.update(values[:count])
        calibration_groups.update(values[count:])
    return test_groups, calibration_groups


def source_split(rows, source_indices, seed, validation_fraction):
    by_family = defaultdict(set)
    for index in source_indices:
        row = rows[index]
        by_family[row.get("kernel_family", "unknown")].add(workload_group(row))
    rng = random.Random(seed)
    validation_groups = set()
    for family, groups in sorted(by_family.items()):
        values = sorted(groups)
        rng.shuffle(values)
        count = max(1, round(len(values) * validation_fraction))
        count = min(count, len(values) - 1)
        validation_groups.update(values[:count])
    train = np.asarray([i for i in source_indices if workload_group(rows[i]) not in validation_groups])
    validation = np.asarray([i for i in source_indices if workload_group(rows[i]) in validation_groups])
    return train, validation


def sample_calibration(rows, pool_indices, fraction, seed):
    if fraction <= 0:
        return np.asarray([], dtype=np.int64)
    by_family = defaultdict(list)
    for index in pool_indices:
        by_family[rows[index].get("kernel_family", "unknown")].append(index)
    rng = random.Random(seed)
    selected = []
    for family, values in sorted(by_family.items()):
        rng.shuffle(values)
        count = max(1, round(len(values) * fraction))
        selected.extend(values[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def prepare(rows, scaler_fit_indices, no_architecture=False):
    architecture = numeric_matrix(rows, ARCH_FEATURES, ARCH_LOG_FEATURES)
    workload = numeric_matrix(rows, WORKLOAD_FEATURES, WORKLOAD_LOG_FEATURES)
    configuration = numeric_matrix(rows, CONFIG_FEATURES, CONFIG_LOG_FEATURES)
    scalers = {}
    for name, values in (("architecture", architecture), ("workload", workload),
                         ("configuration", configuration)):
        scaler = Standardizer().fit(values[scaler_fit_indices])
        values[:] = scaler.transform(values)
        scalers[name] = scaler
    if no_architecture:
        architecture[:] = 0.0
    family_vocab = make_vocab(row.get("kernel_family", "unknown") for row in rows)
    dtype_vocab = make_vocab(row.get("dtype", "unknown") for row in rows)
    family = encoded([row.get("kernel_family", "unknown") for row in rows], family_vocab)
    dtype = encoded([row.get("dtype", "unknown") for row in rows], dtype_vocab)
    latency = np.asarray([row["_target"] for row in rows], dtype=np.float32)
    return architecture, workload, configuration, family, dtype, latency, family_vocab, dtype_vocab


def tensor_batches(tensors, indices, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(*(tensor[indices] for tensor in tensors))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def model_predictions(model, features, indices, target_mean, target_std, device):
    model.eval()
    output = []
    tensors = tuple(torch.from_numpy(value) for value in features)
    with torch.no_grad():
        for batch in tensor_batches(tensors, indices, 512, False, 0):
            batch = [value.to(device) for value in batch]
            output.append(model(*batch).cpu().numpy())
    return np.exp(np.concatenate(output) * target_std + target_mean)


def train_model(rows, prepared, train_indices, validation_indices, args, seed):
    architecture, workload, configuration, family, dtype, latency, family_vocab, dtype_vocab = prepared
    log_latency = np.log(latency)
    target_mean = float(log_latency[train_indices].mean())
    target_std = max(float(log_latency[train_indices].std()), 1e-8)
    target = ((log_latency - target_mean) / target_std).astype(np.float32)
    features = (architecture, workload, configuration, family, dtype)
    tensors = tuple(torch.from_numpy(value) for value in (*features, target))
    torch.manual_seed(seed)
    model = LatencySurrogate(
        architecture.shape[1], workload.shape[1], configuration.shape[1],
        len(family_vocab), len(dtype_vocab), args.hidden_size, args.embedding_size,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    best_state, best_mape, stale = None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tensor_batches(tensors, train_indices, args.batch_size, True, seed + epoch):
            arch, work, config, family_batch, dtype_batch, target_batch = [
                value.to(args.device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(arch, work, config, family_batch, dtype_batch), target_batch)
            loss.backward()
            optimizer.step()
        prediction = model_predictions(
            model, features, validation_indices, target_mean, target_std, args.device)
        score = regression_metrics(latency[validation_indices], prediction)["mape_pct"]
        if score < best_mape - 1e-5:
            best_mape = score
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, features, latency, target_mean, target_std, epoch, best_mape


def per_family_metrics(rows, indices, actual, predicted):
    result = regression_metrics(actual, predicted)
    result["families"] = {}
    for family in sorted(set(rows[index].get("kernel_family", "unknown") for index in indices)):
        mask = np.asarray([rows[index].get("kernel_family", "unknown") == family
                           for index in indices])
        result["families"][family] = regression_metrics(actual[mask], predicted[mask])
    return result


def family_median_prediction(rows, latency, train_indices, test_indices):
    medians = {}
    for family in sorted(set(rows[i].get("kernel_family", "unknown") for i in train_indices)):
        medians[family] = float(np.median([
            latency[i] for i in train_indices if rows[i].get("kernel_family") == family]))
    return np.asarray([medians[rows[i].get("kernel_family", "unknown")] for i in test_indices])


def config_vector(row):
    values = []
    for name in CONFIG_FEATURES:
        value = as_float(row.get(name))
        values.append(0.0 if value is None else value)
    return np.asarray(values, dtype=np.float64)


def gp_order(rows, candidate_indices, budget, seed):
    """Small dependency-free Gaussian-process lower-confidence-bound search."""
    rng = random.Random(seed)
    remaining = list(candidate_indices)
    rng.shuffle(remaining)
    chosen = remaining[:min(3, budget)]
    remaining = remaining[len(chosen):]
    all_x = np.stack([config_vector(rows[i]) for i in candidate_indices])
    scale = all_x.std(axis=0); scale[scale < 1e-12] = 1.0
    center = all_x.mean(axis=0)
    vectors = {i: (config_vector(rows[i]) - center) / scale for i in candidate_indices}
    while len(chosen) < budget and remaining:
        x_train = np.stack([vectors[i] for i in chosen])
        y_train = np.log(np.asarray([rows[i]["_target"] for i in chosen]))
        distances = np.sqrt(((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
        positive = distances[distances > 0]
        length = float(np.median(positive)) if len(positive) else 1.0
        kernel = np.exp(-0.5 * (distances / max(length, 1e-6)) ** 2)
        kernel.flat[::len(kernel) + 1] += 1e-5
        inverse_y = np.linalg.solve(kernel, y_train)
        x_test = np.stack([vectors[i] for i in remaining])
        cross_dist = np.sqrt(((x_test[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
        cross = np.exp(-0.5 * (cross_dist / max(length, 1e-6)) ** 2)
        mean = cross @ inverse_y
        solved = np.linalg.solve(kernel, cross.T)
        variance = np.maximum(1.0 - np.sum(cross * solved.T, axis=1), 1e-9)
        score = mean - 1.5 * np.sqrt(variance)
        position = int(np.argmin(score))
        chosen.append(remaining.pop(position))
    return chosen


def budget_curves(rows, test_indices, surrogate_prediction, budgets, repeats, seed):
    groups = defaultdict(list)
    for local_position, row_index in enumerate(test_indices):
        groups[workload_group(rows[row_index])].append((row_index, local_position))
    records = []
    for group, pairs in sorted(groups.items()):
        indices = [pair[0] for pair in pairs]
        positions = [pair[1] for pair in pairs]
        actual = np.asarray([rows[i]["_target"] for i in indices])
        oracle = float(actual.min())
        family = rows[indices[0]].get("kernel_family", "unknown")
        predicted = surrogate_prediction[positions]
        surrogate_order = np.argsort(predicted)
        for requested in budgets:
            budget = min(requested, len(indices))
            if budget < 1:
                continue
            surrogate_best = float(actual[surrogate_order[:budget]].min())
            records.append({
                "workload_group": group, "family": family, "method": "surrogate",
                "budget": requested, "effective_budget": budget, "repeat": 0,
                "oracle_latency_ms": oracle, "best_latency_ms": surrogate_best,
                "oracle_performance_recovered_pct": 100.0 * oracle / surrogate_best,
                "regret_pct": 100.0 * (surrogate_best - oracle) / oracle,
            })
            for repeat in range(repeats):
                repeat_seed = seed + repeat * 1009 + sum(ord(c) for c in group)
                random_indices = list(range(len(indices)))
                random.Random(repeat_seed).shuffle(random_indices)
                random_best = float(actual[random_indices[:budget]].min())
                gp_indices = gp_order(rows, indices, budget, repeat_seed)
                gp_best = min(float(rows[index]["_target"]) for index in gp_indices)
                for method, best in (("random", random_best), ("bayesian_gp_lcb", gp_best)):
                    records.append({
                        "workload_group": group, "family": family, "method": method,
                        "budget": requested, "effective_budget": budget, "repeat": repeat,
                        "oracle_latency_ms": oracle, "best_latency_ms": best,
                        "oracle_performance_recovered_pct": 100.0 * oracle / best,
                        "regret_pct": 100.0 * (best - oracle) / oracle,
                    })
    return records


def summarize_budget(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["budget"])].append(row)
    return [{
        "method": method, "budget": budget, "observations": len(values),
        "mean_oracle_performance_recovered_pct": float(np.mean([
            value["oracle_performance_recovered_pct"] for value in values])),
        "mean_regret_pct": float(np.mean([value["regret_pct"] for value in values])),
        "median_regret_pct": float(np.median([value["regret_pct"] for value in values])),
    } for (method, budget), values in sorted(grouped.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-token", default="A100")
    parser.add_argument("--target-token", default="H100")
    parser.add_argument("--target-test-fraction", type=float, default=0.40)
    parser.add_argument("--source-validation-fraction", type=float, default=0.20)
    parser.add_argument("--few-shot-fractions", default="0,0.01,0.05,0.10,0.25")
    parser.add_argument("--budgets", default="1,3,5,10,25,50")
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows([path for path in args.inputs if path.is_file()], "latency_median_ms")
    source = np.asarray([i for i, row in enumerate(rows)
                         if architecture_matches(row, args.source_token)], dtype=np.int64)
    target = np.asarray([i for i, row in enumerate(rows)
                         if architecture_matches(row, args.target_token)], dtype=np.int64)
    if not len(source) or not len(target):
        raise SystemExit("Could not identify both source and target rows")
    source_train, source_validation = source_split(
        rows, source, args.seed, args.source_validation_fraction)
    target_test_groups, target_calibration_groups = stratified_groups(
        [rows[i] for i in target], args.seed, args.target_test_fraction)
    target_test = np.asarray([i for i in target if workload_group(rows[i]) in target_test_groups])
    target_calibration = np.asarray([
        i for i in target if workload_group(rows[i]) in target_calibration_groups])
    fractions = [float(value) for value in args.few_shot_fractions.split(",")]
    budgets = [int(value) for value in args.budgets.split(",")]
    prepared = prepare(rows, source_train, no_architecture=False)
    experiment_rows = []
    predictions_by_fraction = {}
    latency = prepared[5]
    for fraction in fractions:
        adaptation = sample_calibration(
            rows, target_calibration, fraction, args.seed + round(fraction * 10000))
        train_indices = np.concatenate((source_train, adaptation))
        trained = train_model(
            rows, prepared, train_indices, source_validation, args,
            args.seed + round(fraction * 10000))
        model, features, latency, target_mean, target_std, epochs, validation_mape = trained
        prediction = model_predictions(
            model, features, target_test, target_mean, target_std, args.device)
        predictions_by_fraction[fraction] = prediction
        metrics = per_family_metrics(rows, target_test, latency[target_test], prediction)
        experiment_rows.append({
            "experiment": "zero_shot" if fraction == 0 else "few_shot",
            "target_fraction": fraction, "target_adaptation_rows": int(len(adaptation)),
            "source_train_rows": int(len(source_train)), "target_test_rows": int(len(target_test)),
            "epochs": epochs, "source_validation_mape_pct": validation_mape,
            "target_metrics": metrics,
        })

    largest_fraction = max(fractions)
    largest_adaptation = sample_calibration(
        rows, target_calibration, largest_fraction,
        args.seed + round(largest_fraction * 10000))
    ablation_train = np.concatenate((source_train, largest_adaptation))
    prepared_no_arch = prepare(rows, source_train, no_architecture=True)
    trained = train_model(
        rows, prepared_no_arch, ablation_train, source_validation, args, args.seed + 90000)
    model, features, no_arch_latency, target_mean, target_std, epochs, validation_mape = trained
    no_arch_prediction = model_predictions(
        model, features, target_test, target_mean, target_std, args.device)
    family_prediction = family_median_prediction(rows, latency, ablation_train, target_test)
    ablations = {
        "full_architecture_encoder": next(
            row["target_metrics"] for row in experiment_rows
            if row["target_fraction"] == largest_fraction),
        "no_architecture_encoder": per_family_metrics(
            rows, target_test, no_arch_latency[target_test], no_arch_prediction),
        "family_median": per_family_metrics(
            rows, target_test, latency[target_test], family_prediction),
    }

    budget_prediction = predictions_by_fraction[largest_fraction]
    budget_records = budget_curves(
        rows, target_test, budget_prediction, budgets, args.random_repeats, args.seed)
    budget_summary = summarize_budget(budget_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "transfer_metrics.json").open("w") as handle:
        json.dump({
            "protocol": {
                "source_token": args.source_token, "target_token": args.target_token,
                "source_rows": int(len(source)), "target_rows": int(len(target)),
                "source_train_rows": int(len(source_train)),
                "source_validation_rows": int(len(source_validation)),
                "target_calibration_pool_rows": int(len(target_calibration)),
                "target_test_rows": int(len(target_test)),
                "target_test_workloads": len(target_test_groups),
                "normalization_fit": "source training rows only",
                "model_selection": "source validation workloads only",
            },
            "transfer": experiment_rows, "ablations": ablations,
            "budget_summary": budget_summary,
        }, handle, indent=2, allow_nan=False)
    with (args.output_dir / "budget_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_records[0]))
        writer.writeheader(); writer.writerows(budget_records)
    with (args.output_dir / "budget_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_summary[0]))
        writer.writeheader(); writer.writerows(budget_summary)
    print(json.dumps({
        "output_dir": str(args.output_dir), "transfer": experiment_rows,
        "ablations": ablations, "budget_summary": budget_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
