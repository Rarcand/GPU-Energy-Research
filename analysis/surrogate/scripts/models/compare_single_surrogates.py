"""Compare one-model Bayesian ridge tuning with the ridge/GP hybrid."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from analysis.surrogate.scripts.experiments.audit_final_choice import EXPERIMENTS, load_precision_inputs
from analysis.surrogate.scripts.experiments.benchmark_candidate_response import conditional_order
from analysis.surrogate.scripts.experiments.benchmark_few_shot_surrogate import metric_for_order, source_scenarios
from analysis.surrogate.scripts.experiments.capacity_transport import shift_bank
from analysis.surrogate.scripts.experiments.compare_gemm_resource_residual import make_model
from analysis.surrogate.scripts.models.compare_surrogate_architectures import METRICS, empirical_gp_prior, prepare_families
from analysis.surrogate.scripts.models.compare_surrogates import save_json
from analysis.surrogate.scripts.features.gather_timing import sha
from analysis.surrogate.scripts.models.portable_rank_prior import score_task


METHODS = ('hybrid', 'static_ridge', 'bayesian_ridge', 'ridge_with_residuals',
           'ridge_with_residuals_open_loop', 'ridge_with_residuals_geometry', 'listwise_ridge',
           'listwise_adaptive_ridge')
PAIRS = tuple((m, 'hybrid') for m in ('static_ridge', 'bayesian_ridge', 'ridge_with_residuals')) + (
    ('ridge_with_residuals', 'static_ridge'),
    ('ridge_with_residuals', 'ridge_with_residuals_open_loop'),
    ('ridge_with_residuals', 'ridge_with_residuals_geometry'),
    ('listwise_ridge', 'static_ridge'),
    ('listwise_adaptive_ridge', 'ridge_with_residuals'),
    ('listwise_adaptive_ridge', 'hybrid'),
)
SCALES = (.25, 1., 4.)
TEMPERATURE = .10


def fit_listwise_weights(features, targets, ridge):
    """Add a whole-task ranking loss to the same linear latency model."""
    shape = (-1, targets.shape[-1])
    logits = -targets.reshape(shape) / TEMPERATURE
    expected = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    labels = targets.ravel() - ridge.intercept_
    penalty = ridge.alpha / len(labels)

    def loss(weights):
        predictions = features @ weights
        error = predictions - labels
        scores = -predictions.reshape(shape) / TEMPERATURE
        normalizer = logsumexp(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores - normalizer)
        ranking = -(expected * (scores - normalizer)).sum(axis=1).mean()
        objective = (.5 * (error @ error / len(labels) + penalty * (weights @ weights))
                     + TEMPERATURE ** 2 * ranking)
        gradient = error / len(labels) + TEMPERATURE * (expected - probabilities).ravel() / len(expected)
        return objective, features.T @ gradient + penalty * weights

    fitted = minimize(loss, ridge.coef_, jac=True, method='L-BFGS-B', options={'maxiter': 300})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f'Listwise optimizer failed: {fitted.message}')
    return fitted.x


class BayesianRidgeSurrogate:
    """A single log-latency regression with Gaussian uncertainty for each task."""

    def __init__(self, predictors, times, *, listwise=False, absolute=False):
        self.model = make_model('ridge')
        target = np.log(times) if absolute else np.log(times / times.min(axis=-1, keepdims=True))
        fit_rows = predictors.reshape(-1, predictors.shape[-1])
        self.model.fit(fit_rows, target.ravel())
        scaled = self.model.named_steps['standardscaler'].transform(fit_rows)
        ridge = self.model.named_steps['ridge']
        if listwise:
            ridge.coef_ = fit_listwise_weights(scaled, target, ridge)
        alpha = ridge.alpha
        precision = scaled.T @ scaled + alpha * np.eye(scaled.shape[-1])
        residual = target - self.model.predict(fit_rows).reshape(target.shape)
        centered = residual - residual.mean(axis=-1, keepdims=True)
        fit_residual = centered.reshape(-1, centered.shape[-1])
        covariance = fit_residual.T @ fit_residual / len(fit_residual)
        variance = np.maximum(covariance.diagonal(), .025 ** 2)
        self.beta_covariance = variance.mean() * np.linalg.inv(precision)
        self.residual_covariance = .75 * covariance + .25 * np.diag(variance)

    def mean(self, features):
        flat = features.reshape(-1, features.shape[-1])
        return self.model.predict(flat).reshape(features.shape[:-1])

    def covariance(self, features):
        x = self.model.named_steps['standardscaler'].transform(features)
        return x @ self.beta_covariance @ x.T


def fit(predictors, times, devices, shapes, *, listwise=False, absolute=False):
    return BayesianRidgeSurrogate(predictors[devices][:, shapes], times[devices][:, shapes],
                                  listwise=listwise, absolute=absolute)


def choose_scale(predictors, times, sources, train, validation, *, listwise=False, absolute=False):
    """Validate the covariance scale on source-only unseen GPU/shape pairs."""
    results = {s: [] for s in SCALES}
    for hidden in sources:
        peers = [d for d in sources if d != hidden]
        model = fit(predictors, times, peers, train, listwise=listwise, absolute=absolute)
        means = model.mean(predictors[hidden, validation])
        labels = times[hidden, validation]
        for scale in SCALES:
            scores = []
            covariance = scale * model.residual_covariance
            for prior, actual in zip(means, labels):
                anchor = int(prior.argmin())
                order = conditional_order(prior, covariance, actual, anchor, budget=4)
                value = actual[order].min() / actual.min()
                scores.append(float(np.log(value)))
            results[scale].append(float(np.mean(scores)))
    return min(SCALES, key=lambda s: np.mean(results[s])), results


def evaluate_fold(data, assignment, target, previous):
    _, shapes, times, physical, geometry, predictors = data
    sources = [d for d in range(len(times)) if d != target]
    train, validation, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                              for role in ('train', 'validation', 'test'))
    model = fit(predictors, times, sources, train)
    means = model.mean(predictors[target, test])
    scale, source_loss = choose_scale(predictors, times, sources, train, validation)
    residual = scale * model.residual_covariance
    geometry_model = fit(geometry, times, sources, train)
    geometry_means = geometry_model.mean(geometry[target, test])
    geometry_scale, geometry_loss = choose_scale(geometry, times, sources, train, validation)
    geometry_covariance = geometry_scale * geometry_model.residual_covariance
    ranking_model = fit(predictors, times, sources, train, listwise=True)
    ranking_means = ranking_model.mean(predictors[target, test])
    ranking_scale, ranking_loss = choose_scale(predictors, times, sources, train, validation, listwise=True)
    ranking_covariance = ranking_scale * ranking_model.residual_covariance
    bank = source_scenarios(times, shapes, sources, train, test)
    bank = shift_bank(bank, physical, sources, test, target, previous['coefficient'], previous['strength'])
    result = []
    for i, prior, geometric_prior, rank_prior, cases in zip(test, means, geometry_means, ranking_means, bank):
        actual = times[target, i]
        anchor = int(prior.argmin())
        gp_mean, gp_covariance, _, _ = empirical_gp_prior(cases)
        vanilla_covariance = model.covariance(predictors[target, i])
        policies = {
            'hybrid': conditional_order(gp_mean, gp_covariance, actual, anchor),
            'bayesian_ridge': conditional_order(prior, vanilla_covariance, actual, anchor),
            'ridge_with_residuals': conditional_order(prior, residual, actual, anchor),
            'ridge_with_residuals_open_loop': conditional_order(prior, residual, actual, anchor, feedback=False),
            'ridge_with_residuals_geometry': conditional_order(geometric_prior, geometry_covariance, actual,
                                                              int(geometric_prior.argmin())),
            'listwise_adaptive_ridge': conditional_order(rank_prior, ranking_covariance, actual,
                                                        int(rank_prior.argmin())),
        }
        scores = {m: metric_for_order(order, actual) for m, order in policies.items()}
        scores['static_ridge'] = score_task(actual, prior)
        scores['listwise_ridge'] = score_task(actual, rank_prior)
        result.append([[scores[m][k] for k in METRICS] for m in METHODS])
    return np.asarray(result), {'scale': scale, 'source_validation_log_regret': source_loss,
                                'geometry_scale': geometry_scale,
                                'geometry_source_validation_log_regret': geometry_loss,
                                'ranking_scale': ranking_scale,
                                'ranking_source_validation_log_regret': ranking_loss}


def summarize(families, sizes):
    rng = np.random.default_rng(7133)
    result = {'means': {m: {} for m in METHODS}, 'paired': {f'{a}_minus_{b}': {} for a, b in PAIRS}}
    for index, metric in enumerate(METRICS):
        selected = [y[..., index] for family, y in families.items()
                    if not (metric.endswith('b8') and sizes[family] <= 8)]
        means = np.mean([v.mean(axis=0) for v in selected], axis=0)
        draws = [rng.integers(len(v), size=(10000, len(v))) for v in selected]
        for m, mean in zip(METHODS, means):
            result['means'][m][metric] = float(mean)
        for m, b in PAIRS:
            i, j = METHODS.index(m), METHODS.index(b)
            sample = np.mean([(v[:, i] - v[:, j])[d].mean(axis=1) for v, d in zip(selected, draws)], axis=0)
            result['paired'][f'{m}_minus_{b}'][metric] = {
                'difference': float(means[i] - means[j]),
                'ci95': np.quantile(sample, (.025, .975)).tolist(),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 't4', 'reports', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    args = parser.parse_args()
    report = {
        'protocol': ('One linear regression preserves the ridge prior and makes every prediction. '
                     'Its Gaussian posterior conditions on observed log-latency contrasts. '
                     'The plain Bayesian uncertainty comes from the ridge coefficient precision. '
                     'The residual model estimates cross-candidate covariance from centered errors '
                     'on source training tasks and selects an uncertainty scale on held-out source '
                     'GPUs and validation shapes. It uses no timing scenarios at test time. '
                     'The geometry-only refit uses the same learning and validation with hardware features removed. '
                     'The listwise variant adds a softmax cross-entropy loss to the squared log-latency loss. '
                     'The soft target favors fast candidates within each training task with temperature 0.10. '
                     'The hybrid reference uses the same ridge anchor and its original GP. '
                     'All policies use the same held-out test grids and budgets. Eight-probe scores '
                     'exclude families with no more than eight candidates. Paired 95% intervals '
                     'resample workload shapes across fixed GPU folds.'),
        'experiments': {},
    }
    with threadpool_limits(limits=1):
        for dtype, rows, devices in load_precision_inputs(args.inputs, args.t4):
            groups = list(prepare_families(rows, dtype, devices))
            sizes = {data[0]: data[2].shape[-1] for data in groups}
            for size in ('largest', 'next_largest'):
                split_name, ref_name = EXPERIMENTS[dtype, size]
                split_path, ref_path = args.reports / split_name, args.reports / ref_name
                assignment = json.loads(split_path.read_bytes())['assignment']
                ref = json.loads(ref_path.read_bytes())
                if tuple(ref['devices']) != tuple(devices):
                    raise ValueError('Archived device order differs from replay')
                if 'splits' in ref:
                    ref = ref['splits'][split_path.stem]
                results, validation = {}, {}
                for data in groups:
                    family = data[0]
                    folds = [evaluate_fold(data, assignment, i, ref['families'][family][d])
                             for i, d in enumerate(devices)]
                    results[family] = np.asarray([scores for scores, _ in folds]).mean(axis=0)
                    validation[family] = {d: audit for d, (_, audit) in zip(devices, folds)}
                report['experiments'][f'{dtype}_{size}'] = {
                    'devices': devices, 'split_sha256': sha(split_path.read_bytes()),
                    'original_report_sha256': sha(ref_path.read_bytes()),
                    'device_shape_cases': len(devices) * sum(len(v) for v in results.values()),
                    'source_validation': validation, **summarize(results, sizes),
                }
                print(f'Completed {dtype}, {size}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)


if __name__ == '__main__':
    main()
