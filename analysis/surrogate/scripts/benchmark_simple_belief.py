"""Replay a compact belief state for static GPU configuration latencies."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from audit_final_choice import EXPERIMENTS
from benchmark_candidate_response import NOISE, DENSITY_FACTOR
from benchmark_full_fidelity import PROBE_BUDGETS, error_metrics, family_summary, hit_budgets, macro_summary
from benchmark_world_fidelity import read_primary
from compare_single_surrogates import fit
from compare_surrogates import save_json
from gather_timing import sha


SETTINGS = {
    'simple_world_gaussian': (False, False, 1.),
    'simple_world_robust': (True, False, 1.),
    'simple_world_robust_tilt': (True, True, 1.),
    'simple_world_damped': (False, False, .5),
    'simple_world_offset_only': (False, False, 0.),
}
SCALE_VARIANCE = 25.


def condition(prior, covariance, observations, robust=False, profile_weight=1.):
    """Condition a joint log-latency prior using the same unordered set of probes."""
    if not observations:
        return prior, covariance.diagonal()
    known = sorted(observations)
    labels = np.asarray([observations[i] for i in known])
    residual = labels - prior[known]
    cross = covariance[:, known]
    observed = cross[known].copy()
    diagonal = np.diag_indices(len(known))
    observed[diagonal] += NOISE ** 2
    factor = cho_factor(observed, lower=True, check_finite=False)
    if robust and len(known) >= 3:
        # A leave-one-out innovation measures surprise without conditioning on the timing itself.
        precision = cho_solve(factor, np.eye(len(known)), check_finite=False)
        z2 = (precision @ residual) ** 2 / precision.diagonal()
        inflation = np.clip(z2 / 2.25, 1., 64.)
        if np.any(inflation > 1.):
            observed[diagonal] += NOISE ** 2 * (inflation - 1.)
            factor = cho_factor(observed, lower=True, check_finite=False)
    regressor = cho_solve(factor, cross.T, check_finite=False)
    mean = prior + regressor.T @ residual
    if profile_weight < 1.:
        calibrated_prior = prior + np.median(residual)
        mean = profile_weight * mean + (1. - profile_weight) * calibrated_prior
    variance = np.maximum(covariance.diagonal() - np.sum(cross * regressor.T, axis=1), 1e-12)
    return mean, variance


def probe_order(prior, covariance, labels, robust, profile_weight=1., *, feedback=True, budget=None):
    observations = {}
    for _ in range(len(labels) if budget is None else min(budget, len(labels))):
        if observations:
            mean, variance = condition(prior, covariance, observations, robust, profile_weight)
            delta = min(observations.values()) - mean
            deviation = np.sqrt(variance)
            z = delta / deviation
            score = delta * ndtr(z) + deviation * np.exp(-.5 * z ** 2) / DENSITY_FACTOR
            score[list(observations)] = -np.inf
        else:
            score = -prior
        choice = int(np.argmax(score))
        observations[choice] = labels[choice] if feedback else prior[choice]
    return list(observations)


def score_case(prior, covariance, actual, common):
    labels = np.log(actual)
    oracle = actual.min()
    contrast = prior - prior.mean()
    stats, traces = {}, {}
    budgets = [b for b in PROBE_BUDGETS if b < len(actual)]
    default = 'simple_world_gaussian'
    for method, (robust, tilt, profile_weight) in SETTINGS.items():
        # A shared intercept calibrates target scale. Optional tilt lets probes adjust the prior's spread.
        joint = covariance + SCALE_VARIANCE
        if tilt:
            joint += .25 ** 2 * np.outer(contrast, contrast)
        order = (traces[default]['probe_order'] if profile_weight == 0.
                 else probe_order(prior, joint, labels, robust, profile_weight))
        accuracy = {'zero_relative': error_metrics(contrast, labels - labels.mean(), relative=True)}
        forecasts = {'zero_relative_ratio': np.exp(np.clip(contrast, -20., 20.)).tolist()}
        for b in budgets:
            prefix = common[:b]
            observations = {i: labels[i] for i in prefix}
            posterior, _ = condition(prior, joint, observations, robust, profile_weight)
            unseen = np.ones(len(labels), dtype=bool)
            unseen[prefix] = False
            observed_error = error_metrics(posterior[unseen], labels[unseen])
            predicted_ms = np.exp(np.clip(posterior, -20., 20.))
            best = min(predicted_ms[unseen].min(), actual[prefix].min())
            observed_error['best_estimate_ape'] = float(abs(best / oracle - 1.))
            accuracy[str(b)] = observed_error
            forecasts[str(b)] = [float(v * 1000.) if keep else None
                                  for v, keep in zip(predicted_ms, unseen)]
        stats[method] = {'accuracy': accuracy, 'regret': np.minimum.accumulate(actual[order]) / oracle - 1.}
        traces[method] = {'probe_order': order, 'forecasts': forecasts}
    order = probe_order(prior, covariance + SCALE_VARIANCE, labels, False, feedback=False)
    stats['simple_world_no_feedback'] = {
        'accuracy': stats[default]['accuracy'],
        'regret': np.minimum.accumulate(actual[order]) / oracle - 1.,
    }
    traces['simple_world_no_feedback'] = {'probe_order': order, 'forecasts': traces[default]['forecasts']}
    stats['random'] = {'accuracy': {}, 'regret': actual / oracle - 1.}
    return stats, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('prepared', 'reports', 'primary-traces', 'output', 'traces'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    args = parser.parse_args()
    with args.prepared.open('rb') as stream:
        groups, devices = pickle.load(stream)[args.mode, args.dtype]
    primary = read_primary(args.primary_traces)
    result = {
        'protocol': ('A geometry ridge prior and its residual covariance supply a posterior over static log '
                     'latencies. All probes jointly update a global scale and correlated candidate latencies. '
                     'The robust update adds observation noise for surprising leave-one-out innovations. '
                     'The tilt variant also lets observations calibrate the relative spread of the prior. '
                     'The damped variant blends the Gaussian mean with a median scale-calibrated prior. '
                     'All variants use expected improvement and are scored on the same forecast prefixes. '
                     'The no-feedback policy substitutes prior means when selecting the next probe, but reuses '
                     'the measured-prefix forecasts for the separate accuracy experiment. The offset-only '
                     'variant is a forecast control that adds the median observed log residual to the source '
                     'profile and reuses the default selection order.'),
        'dtype': args.dtype, 'mode': args.mode, 'devices': devices,
        'primary_trace_sha256': sha(args.primary_traces.read_bytes()), 'experiments': {},
    }
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), gzip.open(args.traces, 'wt', compresslevel=3) as writer:
        for size in ('largest', 'next_largest'):
            filename, _ = EXPERIMENTS[args.dtype, size]
            split_data = (args.reports / filename).read_bytes()
            assignment = json.loads(split_data)['assignment']
            cases = defaultdict(list)
            for target, device in enumerate(devices):
                sources = [d for d in range(len(devices)) if d != target]
                for family, shapes, times, _, geometry, _ in groups:
                    train, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                                   for role in ('train', 'test'))
                    fitted = fit(geometry, times, sources, train)
                    predictions = fitted.mean(geometry[target, test])
                    for i, prior in zip(test, predictions):
                        entry = primary[size, device, family, shapes[i]]
                        actual = times[target, i]
                        np.testing.assert_array_equal(entry['measured_us'], actual * 1000.)
                        stats, traces = score_case(prior, fitted.residual_covariance, actual,
                                                   entry['common_probe_order'])
                        cases[family].append(stats)
                        writer.write(json.dumps({**entry, 'models': traces}, separators=(',', ':')) + '\n')
                print(args.mode, args.dtype, size, device, 'complete', flush=True)
            families = {name: family_summary(records) for name, records in cases.items()}
            for family in families.values():
                for metrics in family['methods'].values():
                    metrics['first_budget_at_coverage'] = hit_budgets(metrics['within_by_budget'])
            result['experiments'][size] = {'split_sha256': sha(split_data), 'families': families,
                                            'macro': macro_summary(families)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == '__main__':
    main()
