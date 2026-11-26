"""Static workload and hardware features for the frozen FP16 Triton kernels."""

import numpy as np

from gemm_resource_features import _positive, _ratio, _shared_capacity, gemm_features


def columns(rows, keys):
    values = np.asarray([[float(r[k]) for k in keys] for r in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f'Expected finite positive fields {keys}')
    return values.T


def features_for_family(rows):
    family = rows[0]['kernel_family']
    if any(r['kernel_family'] != family or r['dtype'] != 'float16' for r in rows):
        raise ValueError('Expected a single FP16 kernel family')
    if family == 'gemm':
        return gemm_features(rows)

    warps, = columns(rows, ('num_warps',))
    threads = 32 * warps
    # Shared-memory ratios below are only added where operand staging is plausible.
    staged = None
    if family == 'conv2d':
        keys = ('batch', 'height', 'in_channels', 'out_channels', 'kernel', 'stride',
                'block_m', 'block_n', 'block_k')
        batch, h, ci, co, k, stride, bm, bn, bk = columns(rows, keys)
        out_h = np.floor((h + 2 * np.floor(k / 2) - k) / stride) + 1
        m, depth = batch * out_h**2, k**2 * ci
        gm, gn, steps = np.ceil(m / bm), np.ceil(co / bn), np.ceil(depth / bk)
        ctas = gm * gn
        fill = m * co * depth / (ctas * bm * bn * steps * bk)
        working = 2 * (batch * h**2 * ci + co * depth + m * co)
        panel = 2 * bk * (bm + bn)
        registers = bm * bn
        staged = 2 * panel
        extra = {'output_height': out_h, 'reduction_steps': steps,
                 'useful_flops': 2 * m * co * depth}
    elif family == 'attention':
        keys = ('heads', 'q_len', 'kv_len', 'head_dim', 'block_m', 'block_n')
        heads, q, kv, d, bm, bn = columns(rows, keys)
        qm, kn = np.ceil(q / bm), np.ceil(kv / bn)
        ctas = heads * qm
        fill = q * kv / (qm * bm * kn * bn)
        working = 4 * heads * (q + kv) * d
        panel = 2 * d * (bm + 2 * bn)
        registers = bm * (d + bn)
        staged = 4 * d * bn
        causal = np.asarray([float(str(r['causal']).lower() in ('1', 'true')) for r in rows])
        # This kernel still iterates over every KV tile even in causal mode.
        extra = {'causal': causal, 'kv_steps': kn, 'padded_dot_work': 4 * ctas * kn * bm * bn * d}
    elif family == 'gather':
        keys = ('vocab', 'lookups', 'width', 'block_rows', 'block_cols')
        vocab, n, width, br, bc = columns(rows, keys)
        gr, gc = np.ceil(n / br), np.ceil(width / bc)
        ctas = gr * gc
        fill = n * width / (ctas * br * bc)
        locality = np.asarray([r['locality'] for r in rows])
        active_rows = np.minimum(n, vocab)
        active_rows = np.where(locality == 'repeated', np.minimum(active_rows, 16), active_rows)
        working = 2 * (n + active_rows) * width + 4 * n
        panel = 4 * br * bc + 4 * br
        registers = br * bc
        extra = {'uniform': locality == 'uniform', 'clustered': locality == 'clustered',
                 'repeated': locality == 'repeated', 'active_rows_upper_bound': active_rows}
    elif family == 'residual_layernorm':
        keys = ('rows', 'width', 'block_rows')
        n, width, br = columns(rows, keys)
        bc = np.exp2(np.ceil(np.log2(width)))
        ctas = np.ceil(n / br)
        fill = n * width / (ctas * br * bc)
        working = 2 * width * (3 * n + 2)
        panel = 2 * width * (3 * br + 2)
        registers = 2 * br * bc
        extra = {'padded_width': bc}
    elif family in ('cross_entropy', 'softmax'):
        keys = ('rows', 'reduction_width', 'block_multiplier', 'num_stages')
        n, width, multiplier, _ = columns(rows, keys)
        bc = np.exp2(np.ceil(np.log2(width))) * multiplier
        ctas = n
        fill = width / bc
        panel = 4 * width if family == 'softmax' else 2 * width + 12
        working = n * panel
        registers = bc
        extra = {'padded_width': bc}
    else:
        raise ValueError(f'Unsupported kernel family {family}')

    geometry = dict(zip(keys, columns(rows, keys)))
    geometry.update(extra, cta_count=ctas, useful_tile_fraction=fill,
                    working_set_bytes_proxy=working, panel_bytes_proxy=panel,
                    values_per_thread_proxy=registers / threads, num_warps=warps)
    hardware = ('sm_count', 'l2_cache_bytes', 'max_threads_per_sm', 'registers_per_sm')
    sm, l2, thread_limit, sm_registers = np.asarray(
        [[_positive(r.get(k)) for k in hardware] for r in rows], dtype=float).T
    resident_upper_bound = np.floor(thread_limit / threads)
    wave_slots = sm * resident_upper_bound
    waves = np.ceil(_ratio(ctas, wave_slots))
    resources = {
        'ctas_per_sm': _ratio(ctas, sm),
        'one_cta_per_sm_wave_fill_proxy': _ratio(ctas, sm * np.ceil(_ratio(ctas, sm))),
        'thread_limited_wave_count_proxy': waves,
        'thread_limited_wave_fill_proxy': _ratio(ctas, wave_slots * waves),
        'cta_threads_fraction_of_sm_limit': _ratio(threads, thread_limit),
        'working_set_over_l2_proxy': _ratio(working, l2),
        'panel_bytes_over_l2_proxy': _ratio(panel, l2),
        'cta_values_over_sm_registers_proxy': _ratio(registers, sm_registers),
        **{f'has_{k}': v > 0 for k, v in zip(hardware, (sm, l2, thread_limit, sm_registers))},
    }
    if staged is not None:
        geometry['staged_bytes_proxy'] = staged
        shared = np.asarray([_shared_capacity(r) for r in rows])
        resources['staged_bytes_over_shared_per_sm_proxy'] = _ratio(staged, shared)
        resources['has_shared_memory_limit'] = shared > 0
    return (np.column_stack(tuple(geometry.values())), np.column_stack(tuple(resources.values())),
            {'geometry': tuple(geometry), 'resources': tuple(resources)})
