"""Portable configuration ranking with task-relative latency targets."""

from collections import Counter
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from compare_ranker import BUDGETS, task_groups, validation_regret


def fit_rank_tree(rows, arrays, y, indices, seed, args, *, include_validation=False):
    """Fit a tree on source log-regret and predict a score (lower is better)."""
    train = indices['train']
    validation = indices['validation']
    devices = np.asarray([row['_device'] for row in rows])
    source_devices = sorted(set(devices[train]))
    if len(source_devices) < 2 or set(source_devices) != set(devices[validation]):
        raise ValueError('Rank-tree validation needs at least two source GPUs in both shape splits')
    groups = task_groups(rows, train)
    tasks_per_family = Counter((device, family) for device, family, _ in groups)
    families_per_device = Counter(device for device, _ in tasks_per_family)
    targets = np.empty(len(y), dtype=float)
    weights = np.empty(len(y), dtype=float)
    for (device, family, _), ids in groups.items():
        target = np.log(y[ids])
        targets[ids] = target - target.min()
        weights[ids] = 1 / (families_per_device[device] * tasks_per_family[device, family] * len(ids))

    folds = []
    for device in source_devices:
        seen = train[devices[train] != device]
        held_out = validation[devices[validation] == device]
        local_groups = task_groups([rows[i] for i in held_out], np.arange(len(held_out)))
        folds.append((device, seen, held_out, local_groups))

    configurations = {
        'workload_and_config': np.column_stack(arrays[1:]),
        'architecture_workload_and_config': np.column_stack(arrays),
    }

    def fit(x, ids):
        estimator = HistGradientBoostingRegressor(
            max_iter=args.epochs, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        )
        sample_weights = weights[ids] / weights[ids].mean()
        return estimator.fit(x[ids], targets[ids], sample_weight=sample_weights)

    losses, validation_predictions = {}, {}
    for variant, x in configurations.items():
        validation_predictions[variant] = {
            device: (held_out, fit(x, seen).predict(x[held_out]))
            for device, seen, held_out, _ in folds
        }
        losses[variant] = {
            device: validation_regret(y[held_out], validation_predictions[variant][device][1],
                                      local_groups)
            for device, _, held_out, local_groups in folds
        }
    selected = min(losses, key=lambda variant: mean(losses[variant].values()))
    x = configurations[selected]
    predictions = fit(x, train).predict(x)
    info = {
        'target': 'log latency relative to fastest configuration within source GPU/workload/dtype',
        'task_weighting': 'equal source GPUs, families within GPU, and tasks within family',
        'selected_features': selected,
        'validation': 'held source GPU and held shape, averaged across pseudo-target GPUs',
        'validation_budgets': list(BUDGETS),
        'validation_mean_log_regret_by_device': losses,
        'iterations': args.epochs,
        'max_leaf_nodes': 31,
        'l2_regularization': 1.0,
    }
    if include_validation:
        return predictions, info, validation_predictions[selected]
    return predictions, info
