"""Short-warmup control and unchanged pass thresholds for native A100 qualification."""
import ctypes
import json
import math
import statistics
import subprocess
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager

DEPTH = 128
SAMPLES = 20
WARMUP_SECONDS = .030
MEDIAN_LIMIT = .05
P90_LIMIT = .10


def assigned_gpu(torch):
    """Resolve CUDA ordinal 0 through the driver, even under Slurm remapping."""
    torch.cuda.set_device(0)
    driver = ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0):
        raise RuntimeError('Cannot resolve CUDA ordinal 0')
    if driver.cuDeviceGetUuid(ctypes.byref(raw), device):
        raise RuntimeError('Cannot resolve CUDA UUID')
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


@contextmanager
def telemetry(path, gpu):
    columns = ('timestamp,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,'
               'power.draw,utilization.gpu')
    with path.open('x') as out:
        process = subprocess.Popen(
            ['nvidia-smi', '-i', gpu, '--query-gpu=' + columns, '--format=csv',
             '-lms', '100'], stdout=out, stderr=out,
        )
        try:
            yield
            if process.poll() is not None:
                raise RuntimeError('GPU telemetry exited early')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def capture(launch, torch, stream, depth):
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        begin.record()
        for _ in range(depth):
            launch()
        end.record()
    return graph, begin, end


def controlled(launch, torch):
    stream = torch.cuda.Stream()
    graph, begin, end = capture(launch, torch, stream, DEPTH)
    values = []
    with torch.cuda.stream(stream):
        warm_end = time.monotonic() + WARMUP_SECONDS
        while True:
            for _ in range(16):
                graph.replay()
            stream.synchronize()
            if time.monotonic() >= warm_end:
                break
        start_ns = time.time_ns()
        for _ in range(SAMPLES):
            graph.replay()
            stream.synchronize()
            values.append(begin.elapsed_time(end) / DEPTH)
        end_ns = time.time_ns()
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise RuntimeError('Invalid CUDA event interval')
    return dict(samples_ms=values, median_ms=statistics.median(values),
                start_ns=start_ns, end_ns=end_ns)


def select_pilot(units):
    grouped = defaultdict(lambda: defaultdict(list))
    for _, rows in units.values():
        for row in rows:
            grouped[row['kernel_family'], row['dtype']][row['workload_id']].append(row)
    chosen = []
    for (family, dtype), workloads in sorted(grouped.items()):
        def size(rows):
            shape = json.loads(rows[0]['workload_features_json'])
            return math.prod(v for v in shape.values() if type(v) is int and v > 0)
        ordered = sorted(workloads.values(), key=lambda rows: (size(rows), rows[0]['workload_id']))
        for pos in sorted({0, len(ordered) // 2, len(ordered) - 1}):
            configs = sorted(ordered[pos], key=lambda r: (r['config_features_json'], r['config_id']))
            chosen.extend(configs[i] for i in sorted({0, len(configs)//3, 2*len(configs)//3, len(configs)-1}))
    return chosen


def relative_spread(a, b):
    return abs(a - b) / ((a + b) / 2)


def summarize(records, field='controlled'):
    spreads = defaultdict(dict)
    for record in records:
        pair = spreads[record['kernel_family'], record['workload_id'], record['config_id']]
        if record['round'] in pair:
            raise ValueError('Duplicate timing round for candidate')
        pair[record['round']] = record[field]['median_ms']
    families = defaultdict(list)
    for (family, _, _), pair in spreads.items():
        if len(pair) != 2:
            raise ValueError('Each candidate needs exactly two timing rounds')
        families[family].append(relative_spread(*pair.values()))
    result = {}
    for family, values in sorted(families.items()):
        p90 = sorted(values)[math.ceil(.9 * len(values)) - 1]
        result[family] = dict(candidates=len(values), median=statistics.median(values), p90=p90)
    return result


def passes_gate(result, families):
    return set(result) == set(families) and all(
        item['median'] <= MEDIAN_LIMIT and item['p90'] <= P90_LIMIT for item in result.values()
    )
