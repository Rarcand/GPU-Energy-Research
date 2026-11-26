"""Train a recurrent latent model of what an autotuning probe can reveal."""

import argparse
import copy
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from torch import nn

from audit_final_choice import EXPERIMENTS, load_precision_inputs
from benchmark_candidate_response import conditional_order
from benchmark_few_shot_surrogate import metric_for_order, source_scenarios
from capacity_transport import shift_bank
from compare_gemm_resource_residual import make_model
from compare_surrogate_architectures import empirical_gp_prior, prepare_families
from compare_surrogates import FAMILIES, save_json
from gather_timing import sha


METHODS = ('ridge_gp', 'world_greedy', 'world_open_loop', 'world_rollout')
METRICS = ('regret_b1', 'regret_b4', 'success5_b4')
PAIRS = (('world_greedy', 'ridge_gp'), ('world_rollout', 'ridge_gp'),
         ('world_greedy', 'world_open_loop'), ('world_rollout', 'world_greedy'))
SEEDS = (2026, 2027)


class Episodes(NamedTuple):
    features: torch.Tensor
    mask: torch.Tensor
    ridge: torch.Tensor
    log_times: torch.Tensor

    def take(self, ids):
        return Episodes(*(x[ids] for x in self))


class ProbeWorldModel(nn.Module):
    """Predict observations and update a compact belief about the latency surface."""

    def __init__(self, inputs, width=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs + 1, width), nn.SiLU(), nn.Linear(width, width))
        self.initial = nn.Linear(2 * width, width)
        self.transition = nn.GRUCell(width + 1, width)
        self.decoder = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 2))

    def start(self, features, mask, ridge, anchor):
        rows = torch.arange(len(anchor))
        base = ridge - ridge[rows, anchor, None]
        encoded = self.encoder(torch.cat((features, base[..., None]), dim=-1))
        context = (encoded * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        belief = torch.tanh(self.initial(torch.cat((context, encoded[rows, anchor]), dim=-1)))
        return encoded, base, belief

    def observe(self, encoded, belief, candidate, contrast):
        action = encoded[torch.arange(len(candidate)), candidate]
        return self.transition(torch.cat((action, contrast[:, None]), dim=-1), belief)

    def predict(self, encoded, base, belief, anchor):
        context = belief[:, None, :].expand(encoded.shape)
        residual, scale = self.decoder(torch.cat((encoded, context), dim=-1)).unbind(dim=-1)
        mean = base + residual - residual[torch.arange(len(anchor)), anchor, None]
        return mean, .025 + 1.25 * torch.sigmoid(scale)


def ridge_scores(times, predictors, sources, train, queries, device):
    features = predictors[sources][:, train]
    labeled = times[sources][:, train]
    response = np.log(labeled / labeled.min(axis=-1, keepdims=True)).ravel()
    model = make_model('ridge').fit(features.reshape(-1, features.shape[-1]), response)
    instances = predictors[device, queries]
    return model.predict(instances.reshape(-1, instances.shape[-1])).reshape(instances.shape[:2])


def pack(features, actual, ridge, family, padding):
    task_count, candidates, features_count = features.shape
    max_candidates, max_features = padding
    padded = np.zeros((task_count, max_candidates, max_features + len(FAMILIES)), dtype=np.float32)
    padded[:, :candidates, :features_count] = features
    padded[:, :candidates, max_features + FAMILIES.index(family)] = 1.
    mask = np.broadcast_to(np.arange(max_candidates) < candidates, padded.shape[:2]).copy()
    pad = ((0, 0), (0, max_candidates - candidates))
    return Episodes(torch.from_numpy(padded), torch.from_numpy(mask),
                    torch.from_numpy(np.pad(ridge, pad).astype(np.float32)),
                    torch.from_numpy(np.pad(np.log(actual), pad).astype(np.float32)))


def prepare_split(families, assignment, target):
    padding = max(d[2].shape[-1] for d in families), max(d[-1].shape[-1] for d in families)
    training, validation, tests = [], [], {}
    for family, shapes, times, physical, _, predictors in families:
        train, valid, test = (np.array([i for i, s in enumerate(shapes) if assignment[s] == role])
                              for role in ('train', 'validation', 'test'))
        sources = [i for i in range(len(times)) if i != target]
        features = predictors[sources][:, train].reshape(-1, predictors.shape[-1])
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale = np.where(scale < 1e-6, np.maximum(np.abs(mean), 1.), scale)

        def episode(device, ids, ridge):
            x = np.clip((predictors[device, ids] - mean) / scale, -10., 10.)
            return pack(x, times[device, ids], ridge, family, padding)

        for device in sources:
            peers = [i for i in sources if i != device]
            for queries in np.array_split(train, 3):
                references = np.setdiff1d(train, queries)
                ridge = ridge_scores(times, predictors, peers, references, queries, device)
                training.append(episode(device, queries, ridge))
            ridge = ridge_scores(times, predictors, peers, train, valid, device)
            validation.append(episode(device, valid, ridge))
        ridge = ridge_scores(times, predictors, sources, train, test, target)
        tests[family] = (episode(target, test, ridge), ridge.argmin(axis=1),
                         times, physical, sources, train, test, shapes)
    return (Episodes(*(torch.cat(v) for v in zip(*training))),
            Episodes(*(torch.cat(v) for v in zip(*validation))), tests)


def sample_sequences(batch, rng):
    mask, ridge = batch.mask.numpy(), batch.ridge.numpy()
    candidates = rng.uniform(-.05, .05, ridge.shape) + ridge
    count = len(ridge)
    rank = np.where(mask, candidates, np.inf).argsort(axis=1)[:, :3]
    anchor = rank[np.arange(count), rng.integers(0, 3, size=count)]
    uniform = rng.random(ridge.shape)
    uniform[~mask] = 2.
    random_anchor = rng.random(count) < .15
    anchor[random_anchor] = uniform.argmin(axis=1)[random_anchor]
    uniform[np.arange(count), anchor] = 3.
    return torch.from_numpy(np.column_stack((anchor, uniform.argsort(axis=1)[:, :2])))


def episode_loss(model, batch, sequence):
    anchor, rows = sequence[:, 0], torch.arange(len(sequence))
    encoded, base, state = model.start(batch.features, batch.mask, batch.ridge, anchor)
    relative = batch.log_times - batch.log_times[rows, anchor, None]
    best = relative.masked_fill(~batch.mask, torch.inf).min(dim=1, keepdim=True).values
    weights = 1. + (relative <= best + np.log1p(.05)).float()
    available = batch.mask.clone()
    losses = []
    for step, candidate in enumerate(sequence.T):
        if step:
            state = model.observe(encoded, state, candidate, relative[rows, candidate])
        available[rows, candidate] = False
        mean, scale = model.predict(encoded, base, state, anchor)
        error = relative - mean
        point_loss = .5 * (error / scale).square() + scale.log() + .1 * error.square()
        # Clone this per-step mask because later probes update it in place.
        selected = available.clone().float() * weights
        losses.append((point_loss * selected).sum() / selected.sum())
    return torch.stack(losses).mean()


def fit_world(train, validation, epochs, batch_size):
    validation_sequence = sample_sequences(validation, np.random.default_rng(1107))
    selected, best_value, audit = None, float('inf'), []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ProbeWorldModel(train.features.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        best_epoch, stale, best_seed = 0, 0, float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            sequence = sample_sequences(train, rng)
            for offset in np.array_split(rng.permutation(len(train.mask)),
                                         int(np.ceil(len(train.mask) / batch_size))):
                optimizer.zero_grad(set_to_none=True)
                loss = episode_loss(model, train.take(offset), sequence[offset])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                score = float(episode_loss(model, validation, validation_sequence))
            if score < best_seed:
                best_seed, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if score < best_value:
                best_value, selected = score, copy.deepcopy(model)
            if stale >= 6:
                break
        audit.append({'seed': seed, 'best_epoch': best_epoch, 'epochs': epoch, 'validation_loss': best_seed})
    if selected is None:
        raise ValueError('World model did not produce a finite validation loss')
    return selected.eval(), audit


def acquisition(mean, scale, incumbent, available):
    gain = incumbent[:, None] - mean
    z = gain / scale
    return np.where(available, gain * ndtr(z) + scale * np.exp(-.5 * z**2) / np.sqrt(2 * np.pi), -np.inf)


def plan_third(model, encoded, base, state, anchor, mean, scale, incumbent, available):
    """Use the learned observation model to branch on three possible outcomes."""
    batches, candidates = available.shape
    shortlisted = np.argsort(-acquisition(mean, scale, incumbent, available), axis=1)[:, :min(8, candidates - 2)]
    chosen, values = shortlisted[:, 0], np.full(batches, -np.inf)
    rows = np.arange(batches)
    for third in shortlisted.T:
        terminal_improvement = np.zeros(batches)
        valid = available.copy()
        valid[rows, third] = False
        for location, weight in ((-3**.5, 1/6), (0., 2/3), (3**.5, 1/6)):
            fantasy = mean[rows, third] + location * scale[rows, third]
            future = model.observe(encoded, state, torch.from_numpy(third), torch.from_numpy(fantasy).float())
            future_mean, future_std = (x.numpy() for x in model.predict(encoded, base, future, anchor))
            updated = np.minimum(incumbent, fantasy)
            final_gain = acquisition(future_mean, future_std, updated, valid).max(axis=1)
            terminal_improvement += weight * (incumbent - updated + final_gain)
        better = terminal_improvement > values
        chosen, values = np.where(better, third, chosen), np.maximum(terminal_improvement, values)
    return chosen


@torch.no_grad()
def rollout(model, features, pad_mask, ridge, labels, prefix, *, feedback=True, lookahead=False):
    """Read only the selected timings while making the two remaining choices."""
    task_count, candidate_count = labels.shape
    rows = np.arange(task_count)
    anchor, second = prefix.T
    anchor_tensor = torch.from_numpy(anchor)
    encoded, base, state = model.start(features, pad_mask, ridge, anchor_tensor)
    available = np.ones((task_count, candidate_count), dtype=bool)
    available[rows, anchor] = available[rows, second] = False
    mean, scale = (x.numpy()[:, :candidate_count] for x in model.predict(encoded, base, state, anchor_tensor))
    anchor_value = np.log(labels[rows, anchor]) if feedback else 0.
    observed = np.log(labels[rows, second]) - anchor_value if feedback else mean[rows, second]
    incumbent = np.minimum(0., observed)
    state = model.observe(encoded, state, torch.from_numpy(second), torch.from_numpy(observed).float())
    mean, scale = (x.numpy()[:, :candidate_count] for x in model.predict(encoded, base, state, anchor_tensor))
    if lookahead:
        third = plan_third(model, encoded[:, :candidate_count], base[:, :candidate_count], state,
                           anchor_tensor, mean, scale, incumbent, available)
    else:
        third = acquisition(mean, scale, incumbent, available).argmax(axis=1)
    observed = np.log(labels[rows, third]) - anchor_value if feedback else mean[rows, third]
    incumbent = np.minimum(incumbent, observed)
    available[rows, third] = False
    state = model.observe(encoded, state, torch.from_numpy(third), torch.from_numpy(observed).float())
    mean, scale = (x.numpy()[:, :candidate_count] for x in model.predict(encoded, base, state, anchor_tensor))
    fourth = acquisition(mean, scale, incumbent, available).argmax(axis=1)
    return np.column_stack((prefix, third, fourth))


def compare_family(model, data, target, previous):
    episodes, anchors, times, physical, sources, train, test, shapes = data
    bank = shift_bank(source_scenarios(times, shapes, sources, train, test), physical, sources,
                      test, target, previous['coefficient'], previous['strength'])
    observed = times[target, test]
    gp = []
    for case, anchor, labels in zip(bank, anchors, observed):
        prior, covariance, _, _ = empirical_gp_prior(case)
        gp.append(conditional_order(prior, covariance, labels, int(anchor), budget=4))
    gp = np.asarray(gp)
    inputs = (model, *episodes[:3], observed, gp[:, :2].copy())
    orders = (gp, rollout(*inputs), rollout(*inputs, feedback=False), rollout(*inputs, lookahead=True))
    task_scores = []
    for order in orders:
        assert np.array_equal(order[:, :2], gp[:, :2])
        if any(len(set(r)) != 4 for r in order):
            raise ValueError('A policy selected a duplicate candidate')
        task = [metric_for_order(o, y) for o, y in zip(order, observed)]
        task_scores.append([[score[k] for k in METRICS] for score in task])
    return np.asarray(task_scores).transpose(1, 0, 2)


def summarize(scores):
    means = np.mean([row.mean(axis=0) for row in scores.values()], axis=0)
    rng = np.random.default_rng(5037)
    draws = {f: rng.integers(len(y), size=(10000, len(y))) for f, y in scores.items()}
    paired = {}
    for m, b in PAIRS:
        i, j = METHODS.index(m), METHODS.index(b)
        samples = [(y[:, i] - y[:, j])[draws[f]].mean(axis=1) for f, y in scores.items()]
        ci = np.quantile(np.mean(samples, axis=0), (.025, .975), axis=0).T
        paired[f'{m}_minus_{b}'] = {k: {'difference': float(d), 'ci95': v.tolist()}
                                   for k, d, v in zip(METRICS, means[i] - means[j], ci)}
    return {'means': {m: dict(zip(METRICS, y.tolist())) for m, y in zip(METHODS, means)}, 'paired': paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 't4', 'reports', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, rows, devices = next(row for row in load_precision_inputs(args.inputs, args.t4) if row[0] == args.dtype)
    families = list(prepare_families(rows, args.dtype, devices))
    result = {
        'protocol': ('Learn a latent belief over a fixed workload’s latency surface from random probe sequences '
                     'on source GPUs and training shapes. Use cross-fitted ridge predictions to initialize '
                     'source training episodes. Predict per-candidate log latency relative to the first probe '
                     'and its predictive scale, and update belief with a GRU conditioned on probe and outcome. '
                     'Test all policies with the same ridge/GP opening two probes, four total measurements, '
                     'and the same hardware and shape holdouts. The open-loop control uses model forecasts '
                     'in place of actual updates. The rollout policy simulates probe three and chooses it '
                     'for its expected downstream utility at probe four. Bootstrap over workload shapes '
                     'with equal family and fixed-GPU weights. No physical GPU state is modeled.'),
        'dtype': args.dtype, 'devices': devices, 'seeds': SEEDS,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'experiments': {},
    }
    with threadpool_limits(limits=1):
        for size in ('largest', 'next_largest'):
            split_name, report_name = EXPERIMENTS[args.dtype, size]
            split_path, report_path = args.reports / split_name, args.reports / report_name
            assignment = json.loads(split_path.read_bytes())['assignment']
            previous = json.loads(report_path.read_bytes())
            if tuple(previous['devices']) != devices:
                raise ValueError('Device order changed')
            if 'splits' in previous:
                previous = previous['splits'][split_path.stem]
            scores, audits = {f: [] for f in FAMILIES}, {}
            for target, device in enumerate(devices):
                train, valid, tests = prepare_split(families, assignment, target)
                model, training = fit_world(train, valid, args.epochs, args.batch_size)
                audits[device] = {'training_episodes': len(train.mask), 'validation_episodes': len(valid.mask),
                                  'parameters': sum(p.numel() for p in model.parameters()), 'fits': training}
                for family, episode in tests.items():
                    scores[family].append(compare_family(model, episode, target, previous['families'][family][device]))
                print(args.dtype, size, device, 'complete', flush=True)
            averaged = {f: np.asarray(ds).mean(axis=0) for f, ds in scores.items()}
            result['experiments'][size] = {
                'split_sha256': sha(split_path.read_bytes()), 'original_report_sha256': sha(report_path.read_bytes()),
                'device_shape_cases': len(devices) * sum(len(s) for s in averaged.values()),
                'training': audits, **summarize(averaged),
            }
            save_json(args.output, result)


if __name__ == '__main__':
    main()
