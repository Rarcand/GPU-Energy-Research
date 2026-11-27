"""Static launch geometry and resource-pressure proxies for the frozen FP16 GEMM."""

import math

import numpy as np


GEMM_FIELDS = ('m', 'n', 'k', 'block_m', 'block_n', 'block_k',
               'num_warps', 'num_stages')


def _positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.
    return number if math.isfinite(number) and number > 0 else 0.


def _shared_capacity(row):
    reported = _positive(row.get('shared_memory_per_sm_bytes'))
    # T4 reports no shared-memory limit in this export. NVIDIA documents
    # a 64 KiB architectural maximum per SM on compute capability 7.5.
    # https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    turing_maximum = 64 * 1024 if _positive(row.get('compute_capability')) == 7.5 else 0.
    return reported or turing_maximum


def _ratio(numerator, denominator):
    """Use zero for missing denominators and supply corresponding availability flags."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 0)


def gemm_features(rows):
    """Return geometry-only and hardware-relative matrices, with ordered names.

    Logical loads assume every requesting tile loads each valid input element once.
    They omit actual cache reuse and memory transaction granularity. Panel storage,
    register pressure, and wave packing are static proxies for the Triton launch.
    """
    if any(row.get('kernel_family') != 'gemm' or row.get('dtype') != 'float16'
           for row in rows):
        raise ValueError('Static GEMM features require FP16 GEMM rows')
    workload = np.asarray(
        [[_positive(row.get(field)) for field in GEMM_FIELDS] for row in rows],
        dtype=float,
    ).reshape(-1, len(GEMM_FIELDS))
    if (workload <= 0).any() or (np.floor(workload) != workload).any():
        raise ValueError('Expected positive integer GEMM dimensions and launch parameters')
    m, n, k, bm, bn, bk, warps, stages = workload.T
    grid_m, grid_n, k_steps = np.ceil(m / bm), np.ceil(n / bn), np.ceil(k / bk)
    ctas = grid_m * grid_n
    fill_m, fill_n, fill_k = m / (grid_m * bm), n / (grid_n * bn), k / (k_steps * bk)
    threads = 32 * warps
    # The frozen kernel groups output programs across up to eight row tiles.
    group_m = np.minimum(grid_m, 8)
    fp16_input_bytes = 2 * k * (m + n)
    fp16_output_bytes = 2 * m * n
    fp16_cta_read_bytes = 2 * k * (m * grid_n + n * grid_m)
    flops = 2 * m * n * k
    panel_bytes = 2 * bk * (bm + bn)
    grouped_panel_bytes = 2 * bk * (bm * group_m + bn)
    staged_panel_bytes = 256 * np.ceil(panel_bytes * stages / 256)
    accumulator_values = bm * bn

    geometry = {
        **dict(zip(GEMM_FIELDS, workload.T)),
        'cta_count': ctas,
        'cta_grid_m': grid_m,
        'cta_grid_n': grid_n,
        'k_steps': k_steps,
        'm_fill': fill_m,
        'n_fill': fill_n,
        'k_fill': fill_k,
        'useful_fma_fraction': fill_m * fill_n * fill_k,
        'cta_group_m': group_m,
        'fp16_input_bytes': fp16_input_bytes,
        'valid_fma_flops': flops,
        'cta_logical_read_bytes': fp16_cta_read_bytes,
        'compulsory_arithmetic_intensity': flops / (fp16_input_bytes + fp16_output_bytes),
        'cta_logical_arithmetic_intensity': flops / (fp16_cta_read_bytes + fp16_output_bytes),
        'staged_panel_bytes_proxy': staged_panel_bytes,
        'fp32_accumulator_values_per_thread_proxy': accumulator_values / threads,
    }

    hardware_fields = ('sm_count', 'l2_cache_bytes', 'max_threads_per_sm', 'registers_per_sm')
    sm, l2, thread_limit, registers = np.asarray(
        [[_positive(row.get(field)) for field in hardware_fields] for row in rows],
        dtype=float,
    ).reshape(-1, len(hardware_fields)).T
    shared = np.asarray([_shared_capacity(row) for row in rows], dtype=float)
    thread_slots_per_sm = np.floor(thread_limit / threads)
    thread_wave_capacity = sm * thread_slots_per_sm
    single_wave_count = np.ceil(_ratio(ctas, sm))
    thread_wave_count = np.ceil(_ratio(ctas, thread_wave_capacity))

    resources = {
        'ctas_per_sm': _ratio(ctas, sm),
        'single_cta_wave_fill_proxy': _ratio(ctas, sm * single_wave_count),
        'thread_limited_ctas_per_sm_upper_bound': thread_slots_per_sm,
        'thread_limited_wave_count_proxy': thread_wave_count,
        'thread_limited_wave_fill_proxy': _ratio(ctas, thread_wave_capacity * thread_wave_count),
        'cta_threads_fraction_of_sm_limit': _ratio(threads, thread_limit),
        'input_bytes_over_l2': _ratio(fp16_input_bytes, l2),
        'grouped_panel_bytes_over_l2': _ratio(grouped_panel_bytes, l2),
        'staged_panel_bytes_over_sm_shared_limit_proxy': _ratio(staged_panel_bytes, shared),
        'cta_accumulator_over_sm_registers_proxy': _ratio(accumulator_values, registers),
        **{f'has_{field}': feature > 0 for field, feature in (
            *zip(hardware_fields, (sm, l2, thread_limit, registers)),
            ('shared_memory_per_sm_bytes', shared),
        )},
    }
    return (np.column_stack(tuple(geometry.values())),
            np.column_stack(tuple(resources.values())),
            {'geometry': tuple(geometry), 'resources': tuple(resources)})
