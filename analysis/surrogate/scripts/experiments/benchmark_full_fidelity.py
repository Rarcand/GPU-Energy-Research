"""Measure prediction accuracy and search-to-fidelity on fully measured grids."""

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import torch

from analysis.surrogate.scripts.experiments.audit_final_choice import EXPERIMENTS, load_precision_inputs
from analysis.surrogate.scripts.experiments.audit_timing_protocol_transfer import measure
from analysis.surrogate.scripts.experiments.benchmark_candidate_response import conditional_order, contrast_covariance, contrast_posterior
from analysis.surrogate.scripts.experiments.benchmark_few_shot_surrogate import source_scenarios
from analysis.surrogate.scripts.experiments.capacity_bfloat16_transfer import verify_matched_grid
from analysis.surrogate.scripts.experiments.capacity_transport import shift_bank
from analysis.surrogate.scripts.experiments.compare_gemm_resource_residual import aligned_grid, make_model
from analysis.surrogate.scripts.models.compare_neural_encoders import make_network
from analysis.surrogate.scripts.models.compare_single_surrogates import choose_scale, fit as fit_relative
from analysis.surrogate.scripts.models.compare_surrogate_architectures import empirical_gp_prior, prepare_families
from analysis.surrogate.scripts.models.compare_surrogates import fit as fit_neural, matrices, save_json
from analysis.surrogate.scripts.features.gather_timing import sha


PROBE_BUDGETS = (1, 2, 4, 8, 16)
SELECTION_THRESHOLDS = (.01, .05, .10)
PREDICTION_THRESHOLDS = (.05, .10, .20)
NEURAL_MODELS = ('flat', 'three', 'film')
SEEDS = (2026, 2027)


def absolute_predictions(x, times, sources, train, target, tests):
    features = x[sources][:, train].reshape(-1, x.shape[-1])
    labels = np.log(times[sources][:, train].ravel())
    query = x[target, tests]
    fitted = {}
    for name, kind in (('absolute_ridge', 'ridge'), ('boosted_tree', 'tree')):
        predicted = make_model(kind).fit(features, labels).predict(query.reshape(-1, x.shape[-1]))
        fitted[name] = predicted.reshape(query.shape[:2])
    return fitted


def neural_predictions(rows, arrays, y, assignment, target, args):
    training, validation = (np.asarray([i for i, row in enumerate(rows)
                                        if row['_device'] != target and assignment[row['_group']] == role])
                            for role in ('train', 'validation'))
    predictions, fits = {}, {}
    for kind in NEURAL_MODELS:
        values, infos = [], []
        for seed in SEEDS:
            _, predicted, info = fit_neural(arrays, y, training, validation, kind, seed, args,
                                            model_factory=make_network)
            values.append(np.log(predicted))
            infos.append({'seed': seed, **info})
        predictions[kind], fits[kind] = np.mean(values, axis=0), infos
    return predictions, fits


def error_metrics(predicted, actual, *, relative=False):
    error = predicted - actual
    ape = np.abs(np.expm1(np.clip(error, -20., 20.)))
    stats = {'mape': float(ape.mean()), 'log_rmse': float(np.sqrt(np.mean(error**2)))}
    for tolerance in PREDICTION_THRESHOLDS:
        stats[f'within_{int(tolerance * 100)}pct'] = float(np.mean(ape <= tolerance))
    if not relative:
        absolute = np.abs(np.exp(np.clip(predicted, -20., 20.)) - np.exp(actual))
        stats['mae_us'] = float(1000. * absolute.mean())
    ranks = rankdata(np.stack((predicted, actual)), axis=1)
    if len(actual) > 1 and np.all(np.std(ranks, axis=1) > 0.):
        stats['spearman'] = float(np.corrcoef(ranks)[0, 1])
    return stats


def forecast(prior, covariance, labels, measured):
    if covariance is None:
        return prior + np.median(labels[measured] - prior[measured])
    anchor, *others = measured
    anchored = contrast_covariance(covariance, anchor)
    contrast = labels[others] - labels[anchor]
    posterior, _ = contrast_posterior(prior - prior[anchor], anchored, others, contrast)
    return posterior + labels[anchor]


def evaluate_case(actual, models):
    labels, oracle = np.log(actual), actual.min()
    orders, stats, trace = {}, {}, {}
    for name, (prior, covariance, _, anchor) in models.items():
        order = (np.argsort(prior, kind='stable') if covariance is None else
                 conditional_order(prior, covariance, actual, anchor, budget=len(actual)))
        if sorted(order) != list(range(len(actual))):
            raise ValueError('A full-budget policy must measure each configuration exactly once')
        orders[name] = np.asarray(order)
    common = orders['adaptive_ridge']
    for name, (prior, covariance, absolute, _) in models.items():
        relative = prior - prior.mean()
        errors = {'zero_relative': error_metrics(relative, labels - labels.mean(), relative=True)}
        predictions = {'zero_relative_ratio': np.exp(np.clip(relative, -20., 20.)).tolist()}
        if absolute:
            errors['0'] = error_metrics(prior, labels)
            errors['0']['best_estimate_ape'] = abs(np.exp(np.clip(prior.min(), -20., 20.)) / oracle - 1.)
            predictions['0'] = (1000. * np.exp(np.clip(prior, -20., 20.))).tolist()
        for budget in PROBE_BUDGETS:
            if budget >= len(actual):
                continue
            measured = common[:budget].tolist()
            unseen = np.ones(len(actual), dtype=bool)
            unseen[measured] = False
            fitted = forecast(prior, covariance, labels, measured)
            errors[str(budget)] = error_metrics(fitted[unseen], labels[unseen])
            values = 1000. * np.exp(np.clip(fitted, -20., 20.))
            best = min(values[unseen].min() / 1000., actual[measured].min())
            errors[str(budget)]['best_estimate_ape'] = float(abs(best / oracle - 1.))
            predictions[str(budget)] = [float(v) if u else None for v, u in zip(values, unseen)]
        regret = np.minimum.accumulate(actual[orders[name]]) / oracle - 1.
        stats[name] = {'accuracy': errors, 'regret': regret}
        trace[name] = {'probe_order': orders[name].tolist(), 'forecasts': predictions}
    stats['random'] = {'accuracy': {}, 'regret': actual / oracle - 1.}
    return stats, {'measured_us': (1000. * actual).tolist(), 'common_probe_order': common.tolist(),
                   'models': trace}


@lru_cache(maxsize=None)
def random_minimum_weights(n, budget):
    total = math.comb(n, budget)
    return np.asarray([math.comb(n - i - 1, budget - 1) / total for i in range(n)])


def family_summary(records):
    counts = {len(record['random']['regret']) for record in records}
    if len(counts) != 1:
        raise ValueError('Candidate count varies within a family')
    n = counts.pop()
    output = {}
    for method in records[0]:
        values = np.asarray([r[method]['regret'] for r in records])
        if method == 'random':
            values = np.sort(values, axis=1)
            weights = np.asarray([random_minimum_weights(n, b) for b in range(1, n + 1)])
            mean_regret = weights @ values.mean(axis=0)
            coverage = {int(100 * t): (weights @ (values <= t + 1e-12).mean(axis=0)).tolist()
                        for t in SELECTION_THRESHOLDS}
        else:
            mean_regret = values.mean(axis=0)
            coverage = {int(100 * t): (values <= t + 1e-12).mean(axis=0).tolist()
                        for t in SELECTION_THRESHOLDS}
        accuracy = defaultdict(lambda: defaultdict(list))
        for record in records:
            for budget, metrics in record[method]['accuracy'].items():
                for key, value in metrics.items():
                    accuracy[budget][key].append(value)
        output[method] = {'mean_regret_by_budget': mean_regret.tolist(), 'within_by_budget': coverage,
                          'prediction': {b: {k: float(np.mean(a)) for k, a in v.items()}
                                         for b, v in accuracy.items()}}
    return {'device_shape_cases': len(records), 'candidates_per_shape': n, 'methods': output}


def hit_budgets(coverage):
    return {t: {str(q): next((i + 1 for i, x in enumerate(curve) if x >= q / 100 - 1e-12), None)
                for q in (50, 90, 95, 100)} for t, curve in coverage.items()}


def macro_summary(families):
    output = {}
    max_size = max(v['candidates_per_shape'] for v in families.values())
    for method in next(iter(families.values()))['methods']:
        all_cases = [f['methods'][method] for f in families.values()]

        def padded_mean(curves):
            return np.asarray([np.pad(c, (0, max_size - len(c)), mode='edge') for c in curves]).mean(0).tolist()

        coverage = {t: padded_mean([r['within_by_budget'][t] for r in all_cases])
                    for t in all_cases[0]['within_by_budget']}
        errors = defaultdict(lambda: defaultdict(list))
        for r in all_cases:
            for budget, metrics in r['prediction'].items():
                for k, v in metrics.items():
                    errors[budget][k].append(v)
        output[method] = {'mean_regret_by_budget': padded_mean([r['mean_regret_by_budget'] for r in all_cases]),
                          'within_by_budget': coverage, 'first_budget_at_coverage': hit_budgets(coverage),
                          'prediction': {b: {k: float(np.mean(a)) for k, a in v.items()}
                                         for b, v in errors.items()}}
    return output


def run_fold(groups, config_grid, row_indices, neural, assignment, target, archived, writer, identity,
             *, additional_absolute=False):
    scores = {}
    for family, shapes, times, physical, geometry, predictors in groups:
        sources = [d for d in range(len(times)) if d != target]
        train, validation, test = ([i for i, s in enumerate(shapes) if assignment[s] == role]
                                  for role in ('train', 'validation', 'test'))
        ridge = fit_relative(predictors, times, sources, train)
        scale, _ = choose_scale(predictors, times, sources, train, validation)
        means = ridge.mean(predictors[target, test])
        if additional_absolute:
            absolute_model = fit_relative(predictors, times, sources, train, absolute=True)
            absolute_scale, _ = choose_scale(predictors, times, sources, train, validation, absolute=True)
            absolute_means = absolute_model.mean(predictors[target, test])
        else:
            geometric = fit_relative(geometry, times, sources, train)
            geo_scale, _ = choose_scale(geometry, times, sources, train, validation)
            geo_means = geometric.mean(geometry[target, test])
            absolute = absolute_predictions(predictors, times, sources, train, target, test)
        if archived is not None:
            prior = archived['families'][family][identity['device']]
            bank = source_scenarios(times, shapes, sources, train, test)
            bank = shift_bank(bank, physical, sources, test, target, prior['coefficient'], prior['strength'])
        scores[family] = []
        for j, i in enumerate(test):
            actual, mean = times[target, i], means[j]
            anchor = int(mean.argmin())
            models = {'adaptive_ridge': (mean, scale * ridge.residual_covariance, False, anchor)}
            if additional_absolute:
                prediction = absolute_means[j]
                models['absolute_adaptive_ridge'] = (prediction, absolute_scale * absolute_model.residual_covariance,
                                                     True, int(prediction.argmin()))
            else:
                geo_mean = geo_means[j]
                models.update({
                    'geometry_adaptive_ridge': (geo_mean, geo_scale * geometric.residual_covariance,
                                                False, int(geo_mean.argmin())),
                    'static_ridge': (mean, None, False, anchor),
                })
                for method, values in absolute.items():
                    models[method] = (values[j], None, True, None)
            ids = row_indices[family][target, i]
            for method, values in neural.items():
                models[method] = (values[ids], None, True, None)
            if archived is not None:
                gp_mean, covariance, _, _ = empirical_gp_prior(bank[j])
                models['hybrid_reference'] = (gp_mean, covariance, False, anchor)
            stats, trace = evaluate_case(actual, models)
            scores[family].append(stats)
            writer.write(json.dumps({**identity, 'family': family, 'shape': shapes[i],
                                     'candidate_keys': config_grid[family], **trace}, separators=(',', ':')) + '\n')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 't4', 'reports', 'output', 'traces'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--additional-absolute', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, rows, devices = next(t for t in load_precision_inputs(args.inputs, args.t4) if t[0] == args.dtype)
    groups = list(prepare_families(rows, args.dtype, devices, mode=args.mode))
    ids = {id(row): i for i, row in enumerate(rows)}
    group_index, configs = {}, {}
    align = aligned_grid if args.dtype == 'float16' else verify_matched_grid
    for f, shapes, times, _, _, _ in groups:
        order, configs[f], ordered = align([row for row in rows if row['kernel_family'] == f])
        if order != shapes:
            raise ValueError(f'Feature alignment changed for {f}')
        group_index[f] = np.asarray([ids[id(row)] for row in ordered]).reshape(times.shape)
    if not args.additional_absolute:
        arrays = matrices(rows)
        labels = np.asarray([measure(row, args.mode) for row in rows])
    report = {
        'protocol': ('Hold out the target GPU and workload shapes. Score predictions on the complete held-out grid '
                     'at zero probes. At nonzero budgets, all models see the same prefix chosen by hardware-aware '
                     'adaptive ridge and are scored on the same remaining configs. Absolute regressors report '
                     'both zero-probe and calibrated errors. Static predictions are calibrated with a shared log '
                     'offset. Relative models report zero-probe geometry-normalized errors and use a measured '
                     'anchor for absolute latency. Independently replay each model’s own full candidate order '
                     'and record when its fastest measured config reaches each tolerance of the measured oracle. '
                     'Average shapes and fixed GPUs within family, then weight families equally. Neural architectures '
                     'are refit as latency regressors with identical raw feature packs and averaged over two seeds; '
                     'tree and ridge models use engineered features. The archived hybrid is included for graph replay.'),
        'dtype': args.dtype, 'mode': args.mode, 'devices': devices, 'seeds': SEEDS,
        'prediction_budgets': PROBE_BUDGETS, 'experiments': {},
        'additional_absolute': args.additional_absolute,
        'trace_units': 'Microseconds except zero_relative_ratio, which is relative to geometric-mean latency.',
    }
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), gzip.open(args.traces, 'wt', compresslevel=3) as writer:
        for size in ('largest', 'next_largest'):
            split_name, ref_name = EXPERIMENTS[args.dtype, size]
            split_path = args.reports / split_name
            assignment = json.loads(split_path.read_bytes())['assignment']
            archived = None
            if args.mode == 'graph' and not args.additional_absolute:
                archived = json.loads((args.reports / ref_name).read_bytes())
                if tuple(archived['devices']) != devices:
                    raise ValueError('Archived device order differs from replay')
                if 'splits' in archived:
                    archived = archived['splits'][split_path.stem]
            per_family, fits_by_device = defaultdict(list), {}
            for target, device in enumerate(devices):
                neural = {}
                if not args.additional_absolute:
                    neural, fits_by_device[device] = neural_predictions(rows, arrays, labels, assignment, device, args)
                identity = {'split': size, 'dtype': args.dtype, 'mode': args.mode, 'device': device}
                scores = run_fold(groups, configs, group_index, neural, assignment, target, archived, writer, identity,
                                  additional_absolute=args.additional_absolute)
                for family, records in scores.items():
                    per_family[family].extend(records)
                print(args.mode, args.dtype, size, device, 'complete', flush=True)
            summaries = {f: family_summary(r) for f, r in per_family.items()}
            for f in summaries.values():
                for method in f['methods'].values():
                    method['first_budget_at_coverage'] = hit_budgets(method['within_by_budget'])
            report['experiments'][size] = {'split_sha256': sha(split_path.read_bytes()),
                                           'neural_fits': fits_by_device, 'families': summaries,
                                           'macro': macro_summary(summaries)}
            save_json(args.output, report)


if __name__ == '__main__':
    main()
