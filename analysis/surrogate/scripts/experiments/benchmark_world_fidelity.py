"""Train probe world models through full grids and score held-out forecast fidelity."""

import argparse
import copy
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from analysis.surrogate.scripts.experiments.audit_final_choice import EXPERIMENTS, load_precision_inputs
from analysis.surrogate.scripts.experiments.benchmark_full_fidelity import PROBE_BUDGETS, error_metrics, family_summary, hit_budgets, macro_summary
from analysis.surrogate.scripts.experiments.benchmark_probe_world_model import ProbeWorldModel, acquisition, plan_third, prepare_split
from analysis.surrogate.scripts.models.compare_surrogate_architectures import prepare_families
from analysis.surrogate.scripts.models.compare_surrogates import save_json
from analysis.surrogate.scripts.features.gather_timing import sha


METHODS = ('world_greedy', 'world_lookahead', 'world_open_loop')
SEEDS = (2026, 2027)


class FullGridWorldModel(ProbeWorldModel):
    """Share recurrent weights between training on entire histories and single probes."""

    def __init__(self, inputs, width=32):
        super().__init__(inputs, width)
        self.transition = nn.GRU(width + 1, width, batch_first=True)

    def observe(self, encoded, belief, candidate, contrast):
        action = encoded[torch.arange(len(candidate)), candidate]
        transition = torch.cat((action, contrast[:, None]), dim=-1)
        _, state = self.transition(transition[:, None], belief[None])
        return state[0]


def sample_history(batch, rng):
    mask, ridge = batch.mask.numpy(), batch.ridge.numpy()
    rows = np.arange(len(mask))
    rank = np.where(mask, ridge + rng.uniform(-.05, .05, ridge.shape), np.inf).argsort(axis=1)[:, :3]
    anchor = rank[rows, rng.integers(0, 3, len(rows))]
    uniform = rng.random(ridge.shape)
    uniform[~mask] = 2.
    replace = rng.random(len(rows)) < .15
    anchor[replace] = uniform.argmin(axis=1)[replace]
    uniform[rows, anchor] = -1.
    order = uniform.argsort(axis=1)
    last = mask.sum(axis=1) - 1
    later = 4 + (rng.random((len(rows), 2)) * (last[:, None] - 3)).astype(int)
    budgets = np.column_stack((np.ones(len(rows), dtype=int), np.full(len(rows), 2),
                               np.full(len(rows), 3), later))
    return torch.from_numpy(order), torch.from_numpy(budgets)


def history_loss(model, batch, order, budgets):
    rows = torch.arange(len(order))
    anchor = order[:, 0]
    encoded, base, initial = model.start(batch.features, batch.mask, batch.ridge, anchor)
    relative = batch.log_times - batch.log_times[rows, anchor, None]
    actions = order[:, 1:]
    sequence = torch.cat((encoded[rows[:, None], actions], relative[rows[:, None], actions, None]), dim=-1)
    after, _ = model.transition(sequence, initial[None])
    states = torch.cat((initial[:, None], after), dim=1)
    rank = order.argsort(dim=1)
    best = relative.masked_fill(~batch.mask, torch.inf).min(dim=1, keepdim=True).values
    weights = 1. + (relative <= best + np.log1p(.05)).float()
    losses = []
    for budget in budgets.T:
        mean, scale = model.predict(encoded, base, states[rows, budget - 1], anchor)
        error = relative - mean
        weight = weights * (batch.mask & (rank >= budget[:, None]))
        likelihood = .5 * (error / scale).square() + scale.log() + .1 * error.square()
        losses.append((likelihood * weight).sum() / weight.sum())
    return torch.stack(losses).mean()


def fit_world(train, valid, epochs, batch_size):
    valid_history = sample_history(valid, np.random.default_rng(1107))
    best_model, best_value, audits = None, float('inf'), []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = FullGridWorldModel(train.features.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        best_seed, best_epoch, stale = float('inf'), 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            order, budgets = sample_history(train, rng)
            for indices in np.array_split(rng.permutation(len(order)), int(np.ceil(len(order) / batch_size))):
                optimizer.zero_grad(set_to_none=True)
                loss = history_loss(model, train.take(indices), order[indices], budgets[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(history_loss(model, valid, *valid_history))
            if valid_loss < best_seed:
                best_seed, best_epoch, stale = valid_loss, epoch, 0
            else:
                stale += 1
            if valid_loss < best_value:
                best_model, best_value = copy.deepcopy(model), valid_loss
            if stale >= 5:
                break
        audits.append({'seed': seed, 'best_epoch': best_epoch, 'epochs': epoch, 'validation_loss': best_seed})
    if best_model is None:
        raise ValueError('World model failed source validation')
    return best_model.eval(), audits


@torch.no_grad()
def rollout(model, batch, actual, shared, *, feedback, plan):
    """Return complete measured order and predictions at the same shared prefixes."""
    rows = np.arange(len(actual))
    n = actual.shape[-1]
    label = np.log(actual)
    first = torch.from_numpy(shared[:, 0].copy())
    encoded, base, initial = model.start(batch.features, batch.mask, batch.ridge, first)
    encoded, base = encoded[:, :n], base[:, :n]

    def predict(state):
        return (x.numpy() for x in model.predict(encoded, base, state, first))

    def observe(state, candidate, contrast):
        return model.observe(encoded, state, torch.from_numpy(candidate.copy()), torch.from_numpy(contrast).float())

    initial_mean, _ = predict(initial)
    anchor = label[rows, shared[:, 0]]
    state = initial
    remaining = np.ones_like(actual, dtype=bool)
    remaining[rows, shared[:, 0]] = False
    forecast = {'zero_relative': initial_mean}
    evaluated = [b for b in PROBE_BUDGETS if b < n]
    for count in range(1, max(evaluated) + 1):
        mean, _ = predict(state)
        if count in evaluated:
            forecast[str(count)] = mean + anchor[:, None]
        if count < max(evaluated):
            chosen = shared[:, count]
            contrast = label[rows, chosen] - anchor if feedback else mean[rows, chosen]
            state = observe(state, chosen, contrast)
    state = initial
    incumbent = np.zeros(len(actual))
    orders = shared[:, :2].copy()
    for step in range(1, n - 1):
        mean, _ = predict(state)
        chosen = orders[:, step]
        contrast = label[rows, chosen] - anchor if feedback else mean[rows, chosen]
        incumbent = np.minimum(incumbent, contrast)
        remaining[rows, chosen] = False
        state = observe(state, chosen, contrast)
        mean, scale = predict(state)
        if plan and step == 1:
            next_probe = plan_third(model, encoded, base, state, first, mean, scale, incumbent, remaining)
        else:
            next_probe = acquisition(mean, scale, incumbent, remaining).argmax(axis=1)
        orders = np.column_stack((orders, next_probe))
    return orders, forecast


def score_family(model, data, meta, writer):
    batch, _, times, _, _, _, test, shapes = data
    actual = times[meta['target'], test]
    rows = [meta['rows'][meta['size'], meta['device'], meta['family'], shapes[i]] for i in test]
    np.testing.assert_array_equal(np.asarray([r['measured_us'] for r in rows]), 1000. * actual)
    shared = np.asarray([r['common_probe_order'] for r in rows])
    choices = {
        'world_greedy': rollout(model, batch, actual, shared, feedback=True, plan=False),
        'world_lookahead': rollout(model, batch, actual, shared, feedback=True, plan=True),
        'world_open_loop': rollout(model, batch, actual, shared, feedback=False, plan=False),
    }
    scores = []
    for i, data in enumerate(rows):
        values = actual[i]
        observed_log = np.log(values)
        oracle = values.min()
        trace, metrics = {}, {}
        for name, (orders, forecasts) in choices.items():
            if sorted(orders[i]) != list(range(len(values))):
                raise ValueError('World policy did not produce a full permutation')
            relative = forecasts['zero_relative'][i].astype(float)
            relative -= relative.mean()
            by_budget = {'zero_relative_ratio': np.exp(np.clip(relative, -20., 20.)).tolist()}
            accuracy = {'zero_relative': error_metrics(relative, observed_log - observed_log.mean(), relative=True)}
            for b, predictions in forecasts.items():
                if b == 'zero_relative':
                    continue
                unseen = np.ones(len(values), dtype=bool)
                unseen[shared[i, :int(b)]] = False
                predicted = predictions[i]
                accuracy[b] = error_metrics(predicted[unseen], observed_log[unseen])
                us = 1000. * np.exp(np.clip(predicted, -20., 20.))
                best = min(us[unseen].min() / 1000., values[~unseen].min())
                accuracy[b]['best_estimate_ape'] = float(abs(best / oracle - 1.))
                by_budget[b] = [float(v) if keep else None for v, keep in zip(us, unseen)]
            trace[name] = {'probe_order': orders[i].tolist(), 'forecasts': by_budget}
            metrics[name] = {'accuracy': accuracy, 'regret': np.minimum.accumulate(values[orders[i]]) / oracle - 1.}
        # Reuse the benchmark's exact family aggregation; random is omitted on merge.
        metrics['random'] = {'accuracy': {}, 'regret': values / oracle - 1.}
        scores.append(metrics)
        writer.write(json.dumps({**data, 'models': trace}, separators=(',', ':')) + '\n')
    return scores


def read_primary(path):
    rows = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            r.pop('models')
            key = tuple(r[k] for k in ('split', 'device', 'family', 'shape'))
            if key in rows:
                raise ValueError('Duplicate primary replay key')
            rows[key] = r
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 't4', 'reports', 'primary-traces', 'output', 'traces'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--mode', choices=('graph', 'individual'), required=True)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, rows, devices = next(d for d in load_precision_inputs(args.inputs, args.t4) if d[0] == args.dtype)
    groups = list(prepare_families(rows, args.dtype, devices, mode=args.mode))
    primary = read_primary(args.primary_traces)
    result = {
        'protocol': ('Source-only recurrent belief model trained on permuted complete grids, with losses at '
                     'probes 1, 2, 3 and two variable later probes. All world variants share the primary '
                     'adaptive ridge opening two probes. Greedy and lookahead use actual feedback after '
                     'the opening; open loop uses forecasts. Lookahead simulates only the third/fourth '
                     'decision and then resumes greedy. Forecasts use the primary common measurement '
                     'prefix for all models. The first measurement anchors absolute latency; zero-probe '
                     'forecasts are relative. This models learning about a static latency grid.'),
        'dtype': args.dtype, 'mode': args.mode, 'devices': devices, 'seeds': SEEDS, 'epochs': args.epochs,
        'primary_trace_sha256': sha(args.primary_traces.read_bytes()), 'experiments': {},
    }
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), gzip.open(args.traces, 'wt', compresslevel=3) as writer:
        for size in ('largest', 'next_largest'):
            split_name, _ = EXPERIMENTS[args.dtype, size]
            path = args.reports / split_name
            assignment = json.loads(path.read_bytes())['assignment']
            cases, fits = defaultdict(list), {}
            for target, device in enumerate(devices):
                train, valid, tests = prepare_split(groups, assignment, target)
                model, fits[device] = fit_world(train, valid, args.epochs, args.batch_size)
                for family, data in tests.items():
                    meta = dict(size=size, device=device, family=family, target=target, rows=primary)
                    cases[family].extend(score_family(model, data, meta, writer))
                print(args.mode, args.dtype, size, device, 'complete', flush=True)
            totals = {f: family_summary(v) for f, v in cases.items()}
            for f in totals.values():
                for method in f['methods'].values():
                    method['first_budget_at_coverage'] = hit_budgets(method['within_by_budget'])
            result['experiments'][size] = {'split_sha256': sha(path.read_bytes()), 'families': totals,
                                            'training': fits, 'macro': macro_summary(totals)}
            save_json(args.output, result)


if __name__ == '__main__':
    main()
