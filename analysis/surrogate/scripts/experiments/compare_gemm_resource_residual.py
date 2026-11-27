"""Does a static hardware correction improve cross-GPU FP16 GEMM ranking?"""

import argparse
from itertools import product
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analysis.surrogate.scripts.experiments.compare_portable_surrogates import DEVICES, load_matched
from analysis.surrogate.scripts.models.compare_surrogates import save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.models.portable_rank_prior import BUDGETS, NEIGHBORS, config_key, score_task, shape_distances

FEATURE_SETS = ('geometry_residual', 'resource_residual')
MODELS = ('ridge', 'tree')
BLEND_STRENGTHS = (.25, .5, 1.)
METHODS = ('rank_prior', 'nearest_slowdown', *FEATURE_SETS, 'geometry_matched',
           'neutral_resources', 'source_selected')


def aligned_grid(rows):
    lookup = {(r['_device'], r['_group'], config_key(r)): r for r in rows}
    shapes = sorted({r['_group'] for r in rows})
    configs = sorted({config_key(r) for r in rows})
    ordered = [lookup[k] for k in product(DEVICES, shapes, configs)]
    if len(ordered) != len(lookup) or len(ordered) != len(rows):
        raise ValueError('The source and target registries must form one matched grid')
    return shapes, configs, ordered


def neighbor_ranks(ranks, shapes, sources, reference, query):
    if not sources:
        raise ValueError('At least one source device is required')
    reference_names = [shapes[i] for i in reference]
    query_names = [shapes[i] for i in query]
    distances = shape_distances(reference_names, query_names)
    for row, shape in enumerate(query_names):
        if shape in reference_names:
            distances[row, reference_names.index(shape)] = np.inf
    closest = np.argsort(distances, axis=1, kind='stable')[:, :NEIGHBORS]
    if not np.isfinite(np.take_along_axis(distances, closest, axis=1)).all():
        raise ValueError('Insufficient distinct source shapes')
    source_mean = ranks[list(sources)][:, reference].mean(axis=0)
    return (source_mean[closest].mean(axis=1) - 1.) / (ranks.shape[2] - 1.)


def make_model(name):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    return HistGradientBoostingRegressor(
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=32,
        l2_regularization=2.0, early_stopping=False, random_state=2026)


def features_with_prior(features, device, shapes, prior):
    return np.column_stack((features[device, shapes].reshape(prior.size, -1), prior.ravel()))


def fit_residuals(ranks, timings, features, shapes, sources, train):
    xs, ys, weights = [], [], []
    for gpu in sources:
        # The rank prior for each training row comes from other GPUs and other shapes.
        peers = [d for d in sources if d != gpu]
        prior = neighbor_ranks(ranks, shapes, peers, train, train)
        xs.append(features_with_prior(features, gpu, train, prior))
        observed_rank = (ranks[gpu, train] - 1.) / (ranks.shape[2] - 1.)
        ys.append((observed_rank - prior).ravel())
        labels = timings[gpu, train]
        weights.append((1. + 3. * (labels <= 1.05 * labels.min(axis=1, keepdims=True))).ravel())
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    results = {}
    for name in MODELS:
        estimator = make_model(name)
        if name == 'ridge':
            estimator.fit(x, y, ridge__sample_weight=weight)
        else:
            estimator.fit(x, y, sample_weight=weight)
        results[name] = estimator
    return results


def correction(estimator, features, device, shape_ids, prior):
    estimate = estimator.predict(features_with_prior(features, device, shape_ids, prior))
    return np.clip(estimate.reshape(prior.shape), -.5, .5)


def neutral_correction(estimator, features, sources, shape_ids, prior):
    averaged = features[sources][:, shape_ids].mean(axis=0)
    matrix = np.column_stack((averaged.reshape(prior.size, -1), prior.ravel()))
    return np.clip(estimator.predict(matrix).reshape(prior.shape), -.5, .5)


def validation_loss(labels, predictions):
    best = labels.min(axis=1)
    top = np.take_along_axis(labels, np.argsort(predictions, axis=1, kind='stable')[:, :4], axis=1)
    return float(np.mean(np.log(top[:, 0] / best) + .25 * np.log(top.min(axis=1) / best)))


def choose_corrections(ranks, timings, feature_sets, shapes, sources, train, validation,
                       validate_counterfactual=False):
    choices = {kind: {p: [] for p in product(MODELS, BLEND_STRENGTHS)} for kind in FEATURE_SETS}
    neutral_losses = {p: [] for p in product(MODELS, BLEND_STRENGTHS)}
    prior_losses = []
    for held in sources:
        inner_sources = [d for d in sources if d != held]
        prior = neighbor_ranks(ranks, shapes, inner_sources, train, validation)
        labels = timings[held, validation]
        prior_losses.append(validation_loss(labels, prior))
        for kind in FEATURE_SETS:
            x = feature_sets[kind]
            estimates = fit_residuals(ranks, timings, x, shapes, inner_sources, train)
            for name, estimator in estimates.items():
                residual = correction(estimator, x, held, validation, prior)
                if validate_counterfactual and kind == 'resource_residual':
                    neutral = neutral_correction(estimator, x, inner_sources, validation, prior)
                    for amount in BLEND_STRENGTHS:
                        neutral_losses[name, amount].append(
                            validation_loss(labels, prior + amount * neutral))
                for amount in BLEND_STRENGTHS:
                    loss = validation_loss(labels, prior + amount * residual)
                    choices[kind][name, amount].append(loss)
    selected = {kind: min(grid, key=lambda trial: np.mean(grid[trial]))
                for kind, grid in choices.items()}
    validation_means = {'rank_prior': float(np.mean(prior_losses)), **{
        kind: float(np.mean(choices[kind][trial])) for kind, trial in selected.items()}}
    audit = {}
    if validate_counterfactual:
        trial = selected['resource_residual']
        audit = {
            'resource': choices['resource_residual'][trial],
            'geometry_matched': choices['geometry_residual'][trial],
            'neutral_resources': neutral_losses[trial],
            'rank_prior': prior_losses,
            'training_gpu_order': [DEVICES[i] for i in sources],
        }
    return selected, validation_means, min(validation_means, key=validation_means.get), audit


def evaluate_fold(ranks, timings, feature_sets, shapes, assignment, target, *,
                  retain_shortlists=False, validate_counterfactual=False):
    sources = [d for d in range(len(DEVICES)) if d != target]
    train, validation, test = (np.array([i for i, shape in enumerate(shapes) if assignment[shape] == role])
                               for role in ('train', 'validation', 'test'))
    selected, source_validation, source_choice, audit = choose_corrections(
        ranks, timings, feature_sets, shapes, sources, train, validation, validate_counterfactual)
    prior = neighbor_ranks(ranks, shapes, sources, train, test)
    source_labels = timings[sources][:, train]
    relative_slowdown = (source_labels / source_labels.min(axis=2, keepdims=True)).mean(axis=0)
    nearest = shape_distances([shapes[i] for i in train], [shapes[i] for i in test]).argmin(axis=1)
    predictions = {'rank_prior': prior, 'nearest_slowdown': relative_slowdown[nearest]}
    final_models = {}
    for kind, (estimator_name, amount) in selected.items():
        x = feature_sets[kind]
        estimates = final_models[kind] = fit_residuals(ranks, timings, x, shapes, sources, train)
        estimator = estimates[estimator_name]
        residual = correction(estimator, x, target, test, prior)
        predictions[kind] = prior + amount * residual
        if kind == 'resource_residual':
            neutral_delta = neutral_correction(estimator, x, sources, test, prior)
            predictions['neutral_resources'] = prior + amount * neutral_delta
    estimator_name, amount = selected['resource_residual']
    predictions['geometry_matched'] = prior + amount * correction(
        final_models['geometry_residual'][estimator_name], feature_sets['geometry_residual'],
        target, test, prior)
    predictions['source_selected'] = predictions[source_choice]
    labels = timings[target, test]
    task_scores = {
        method: [dict(shape=shapes[i], **score_task(label, prediction))
                 for i, label, prediction in zip(test, labels, predicted)]
        for method, predicted in predictions.items()
    }
    if retain_shortlists:
        for method, predicted in predictions.items():
            for task, order in zip(task_scores[method],
                                   np.argsort(predicted, axis=1, kind='stable')[:, :max(BUDGETS)]):
                task['shortlist'] = order.tolist()
    random = {}
    size = labels.shape[1]
    near_best = np.sum(labels <= 1.05 * labels.min(axis=1, keepdims=True), axis=1)
    for b in BUDGETS:
        random[f'success5_b{b}'] = float(np.mean([
            1 - comb(size - int(near), min(size, b)) / comb(size, min(size, b))
            for near in near_best
        ]))
    return {
        'training_shapes_per_source': len(train), 'validation_shapes_per_source': len(validation),
        'test_shapes': len(test), 'candidates_per_shape': size,
        'source_choice': source_choice, 'source_validation_loss': source_validation,
        'counterfactual_source_validation': audit,
        'best_corrections': {kind: dict(model=model, amount=amount)
                             for kind, (model, amount) in selected.items()},
        'task_scores': task_scores, 'random': random,
        'metrics': {method: {key: float(np.mean([t[key] for t in tasks]))
                             for key in tasks[0] if key not in ('shape', 'shortlist')}
                    for method, tasks in task_scores.items()},
        'regret_quantiles': {method: {f'p{p}_b{b}': float(np.percentile(
            [t[f'regret_b{b}'] for t in tasks], p)) for p in (50, 90) for b in BUDGETS}
                             for method, tasks in task_scores.items()},
    }


def paired_differences(folds):
    rng = np.random.default_rng(49278)
    shape_count = next(iter(folds.values()))['test_shapes']
    draws = rng.integers(shape_count, size=(5000, shape_count))
    out = {}
    for method, baseline in (('geometry_residual', 'rank_prior'),
                             ('resource_residual', 'rank_prior'),
                             ('resource_residual', 'geometry_residual'),
                             ('resource_residual', 'geometry_matched'),
                             ('resource_residual', 'neutral_resources'),
                             ('source_selected', 'rank_prior')):
        out[f'{method}_minus_{baseline}'] = {}
        for key in ('success5_b1', 'regret_b1', 'success5_b4', 'regret_b4'):
            per_shape = np.mean([
                [a[key] - b[key] for a, b in zip(item['task_scores'][method],
                                                   item['task_scores'][baseline])]
                for item in folds.values()], axis=0)
            out[f'{method}_minus_{baseline}'][key] = {
                'difference': float(per_shape.mean()),
                'ci95': np.quantile(per_shape[draws].mean(axis=1), (.025, .975)).tolist(),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--t4', type=Path, required=True)
    parser.add_argument('--splits', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()

    from analysis.surrogate.scripts.features.gemm_resource_features import gemm_features

    rows, _, hashes = load_matched(args.inputs, args.t4, args.mode)
    gemm = [row for row in rows if row['kernel_family'] == 'gemm']
    shapes, configs, ordered = aligned_grid(gemm)
    dims = (len(DEVICES), len(shapes), len(configs))
    timings = np.asarray([r['_y'] for r in ordered]).reshape(dims)
    if not np.isfinite(timings).all() or np.any(timings <= 0):
        raise ValueError('GEMM timings must be finite and positive')
    ranks = rankdata(timings, axis=2)
    geometry, resources, feature_names = gemm_features(ordered)
    features = {
        'geometry_residual': geometry.reshape((*dims, -1)),
        'resource_residual': np.column_stack((geometry, resources)).reshape((*dims, -1)),
    }
    results = {}
    with threadpool_limits(limits=args.threads):
        for path in args.splits:
            split = json.loads(path.read_text())['assignment']
            folds = {}
            for target, device in enumerate(DEVICES):
                folds[device] = evaluate_fold(ranks, timings, features, shapes, split, target)
                print(args.mode, path.stem, device, folds[device]['source_choice'], {
                    name: round(v['success5_b1'] * 100, 2)
                    for name, v in folds[device]['metrics'].items()}, flush=True)
            results[path.stem] = dict(
                split_sha256=sha(path.read_bytes()), folds=folds, paired=paired_differences(folds),
                macro={method: {key: float(np.mean([f['metrics'][method][key]
                                                    for f in folds.values()]))
                                for key in next(iter(folds.values()))['metrics'][method]}
                       for method in METHODS},
                random={key: float(np.mean([f['random'][key] for f in folds.values()]))
                        for key in next(iter(folds.values()))['random']},
            )
    save_json(args.output, dict(mode=args.mode, dtype='float16', target_calibration_rows=0,
                               verified_candidates=len(gemm), devices=DEVICES, feature_names=feature_names,
                               archive_sha256=hashes,
                               code_sha256=sha(Path(__file__).read_bytes()),
                               bootstrap='Paired by shape across four fixed held-out GPUs',
                               neutral_resources='Replaces target static features with same-shape, same-config source feature means',
                               residual_label='Measured fractional rank minus cross-fitted source prior',
                               validation='Inner held source GPU and disjoint validation shapes; mean log regret at top one plus 0.25 times top-four log regret',
                               splits=results))


if __name__ == '__main__':
    main()
