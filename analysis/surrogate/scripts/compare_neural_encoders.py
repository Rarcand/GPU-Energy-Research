"""Compare matched neural encoders as static rankers and first-probe selectors."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from audit_final_choice import EXPERIMENTS, grids, load_precision_inputs
from benchmark_candidate_response import conditional_order
from benchmark_few_shot_surrogate import metric_for_order, source_scenarios
from capacity_bfloat16_transfer import verify_matched_grid
from capacity_transport import shift_bank
from compare_gemm_resource_residual import aligned_grid
from compare_ranker import ranker_fit
from compare_surrogate_architectures import METRICS, empirical_gp_prior
from compare_surrogates import Network, matrices, save_json
from gather_timing import sha
from portable_rank_prior import score_task


ENCODERS = ('three', 'flat', 'film')
METHODS = (*ENCODERS, *(f'{name}_then_gp' for name in ENCODERS))
PAIRS = (('three', 'flat'), ('film', 'flat'), ('film', 'three'),
         ('three_then_gp', 'flat_then_gp'), ('film_then_gp', 'flat_then_gp'),
         ('film_then_gp', 'three_then_gp'), *((f'{m}_then_gp', m) for m in ENCODERS))
SEEDS = (2026, 2027, 2028)


class HardwareFiLM(nn.Module):
    """Use hardware features to scale and shift the workload/config features."""

    def __init__(self, sizes):
        super().__init__()
        hardware, workload, config = sizes
        hardware_width = 48
        with torch.random.fork_rng(devices=[]):
            target = sum(p.numel() for p in Network(sizes, 'three').parameters())

        def count(width):
            return ((hardware + 1) * hardware_width + (hardware_width + 1) * 2 * width
                    + (workload + config + 1) * width + 2 * width + (width + 1) * width
                    + (width + 1) * 64 + 65)

        width = min(range(16, 257), key=lambda w: abs(count(w) - target))
        self.hardware = nn.Sequential(nn.Linear(hardware, hardware_width), nn.ReLU(),
                                      nn.Linear(hardware_width, 2 * width))
        self.task = nn.Sequential(nn.Linear(workload + config, width), nn.ReLU(), nn.LayerNorm(width),
                                  nn.Dropout(.05), nn.Linear(width, width))
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(.1), nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, arrays):
        hardware, workload, config = arrays
        scale, offset = self.hardware(hardware).chunk(2, dim=1)
        features = self.task(torch.cat((workload, config), dim=1))
        return self.head((1. + .5 * torch.tanh(scale)) * features + offset).squeeze(1)


def make_network(sizes, kind):
    return HardwareFiLM(sizes) if kind == 'film' else Network(sizes, kind)


def train_and_predict(rows, arrays, labels, assignment, device, args):
    indices = {role: np.asarray([i for i, row in enumerate(rows)
                                if row['_device'] != device and assignment[row['_group']] == role])
               for role in ('train', 'validation')}
    predictions, runs = {}, {}
    for kind in ENCODERS:
        predictions[kind], runs[kind] = [], []
        for seed in SEEDS:
            _, predicted, info = ranker_fit(rows, arrays, labels, indices, seed, args,
                                            kind=kind, model_factory=make_network)
            predictions[kind].append(predicted)
            runs[kind].append({'seed': seed, **info})
        predictions[kind] = np.asarray(predictions[kind])
    return predictions, runs


def evaluate_family(predictions, data, row_indices, assignment, target, previous):
    _, shapes, times, features = data
    sources = [i for i in range(len(times)) if i != target]
    train, test = ([i for i, shape in enumerate(shapes) if assignment[shape] == role]
                   for role in ('train', 'test'))
    bank = source_scenarios(times, shapes, sources, train, test)
    bank = shift_bank(bank, features, sources, test, target, previous['coefficient'], previous['strength'])
    result = []
    for case, i in zip(bank, test):
        labels, ids = times[target, i], row_indices[target, i]
        prior, covariance, _, _ = empirical_gp_prior(case)
        scores = {}
        for method, seed_predictions in predictions.items():
            ranking = rankdata(seed_predictions[:, ids], axis=-1).mean(axis=0)
            scores[method] = score_task(labels, ranking)
            order = conditional_order(prior, covariance, labels, int(ranking.argmin()))
            scores[f'{method}_then_gp'] = metric_for_order(order, labels)
        result.append([[scores[m][metric] for metric in METRICS] for m in METHODS])
    return np.asarray(result)


def summarize(families, grids_by_family):
    rng = np.random.default_rng(9133)
    report = {'means': {m: {} for m in METHODS}, 'paired': {f'{a}_minus_{b}': {} for a, b in PAIRS}}
    for index, metric in enumerate(METRICS):
        values = [a[..., index] for family, a in families.items()
                  if not (metric.endswith('b8') and grids_by_family[family] <= 8)]
        means = np.mean([rows.mean(axis=0) for rows in values], axis=0)
        draws = [rng.integers(len(a), size=(10000, len(a))) for a in values]
        for m, score in zip(METHODS, means):
            report['means'][m][metric] = float(score)
        for a, b in PAIRS:
            i, j = METHODS.index(a), METHODS.index(b)
            bootstrap = np.mean([(v[:, i] - v[:, j])[d].mean(axis=1) for v, d in zip(values, draws)], axis=0)
            report['paired'][f'{a}_minus_{b}'][metric] = {
                'difference': float(means[i] - means[j]),
                'ci95': np.quantile(bootstrap, (.025, .975)).tolist(),
            }
    report['by_family'] = {f: {m: dict(zip(METRICS, row.tolist()))
                               for m, row in zip(METHODS, values.mean(axis=0))}
                           for f, values in families.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 't4', 'reports', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.002)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, rows, devices = next(t for t in load_precision_inputs(args.inputs, args.t4) if t[0] == args.dtype)
    arrays = matrices(rows)
    labels = np.asarray([row['_y'] for row in rows])
    ids = {id(row): i for i, row in enumerate(rows)}
    align = aligned_grid if args.dtype == 'float16' else verify_matched_grid
    families = list(grids(rows, args.dtype, devices))
    sizes = {family: times.shape[-1] for family, _, times, _ in families}
    row_indices = {}
    for family, shapes, times, _ in families:
        order_shapes, _, ordered = align([r for r in rows if r['kernel_family'] == family])
        if shapes != order_shapes:
            raise ValueError(f'Alignment changed for {family}')
        row_indices[family] = np.asarray([ids[id(r)] for r in ordered]).reshape(times.shape)
    report = {
        'protocol': ('Same raw hardware, workload and config feature groups for all neural architectures. '
                     'Pairwise ranking loss and validation on source GPU shapes only. Ensemble three fixed '
                     'seeds by mean within-shape rank. Static policies evaluate rank prefixes. GP policies '
                     'share the same empirical prior and acquisition, varying only the anchor config. '
                     'Equal family and fixed-GPU weights; paired 95% shape intervals. Eight-probe results '
                     'exclude fully exhausted grids. Ridge references in the earlier architecture study '
                     'use engineered resource proxies and a per-family fit.'),
        'devices': devices, 'dtype': args.dtype, 'seeds': SEEDS,
        'training': {k: getattr(args, k) for k in ('epochs', 'patience', 'batch_size', 'lr')},
        'experiments': {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for size in ('largest', 'next_largest'):
            split_name, report_name = EXPERIMENTS[args.dtype, size]
            split_path, ref_path = args.reports / split_name, args.reports / report_name
            assignment = json.loads(split_path.read_bytes())['assignment']
            ref = json.loads(ref_path.read_bytes())
            if tuple(ref['devices']) != devices:
                raise ValueError('Archived device order differs from replay')
            previous = ref['splits'][split_path.stem] if 'splits' in ref else ref
            scores = {f: [] for f in sizes}
            runs = {}
            for target, device in enumerate(devices):
                predicted, runs[device] = train_and_predict(rows, arrays, labels, assignment, device, args)
                for data in families:
                    f = data[0]
                    scores[f].append(evaluate_family(predicted, data, row_indices[f], assignment,
                                                     target, previous['families'][f][device]))
                print(args.dtype, size, device, 'complete', flush=True)
            averaged = {f: np.asarray(a).mean(axis=0) for f, a in scores.items()}
            report['experiments'][size] = {
                'split_sha256': sha(split_path.read_bytes()), 'original_report_sha256': sha(ref_path.read_bytes()),
                'device_shape_cases': len(devices) * sum(len(a) for a in averaged.values()),
                'training': runs, **summarize(averaged, sizes),
            }
            save_json(args.output, report)


if __name__ == '__main__':
    main()
