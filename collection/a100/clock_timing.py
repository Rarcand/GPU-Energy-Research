"""CUDA graph timing at a verified, naturally reached operating point."""

import ctypes
import math
import statistics
import time

DEPTH = 128
INTERVALS = 4
REPLAYS = 5
MIN_WARM_S = .60
CLOCK_INTERVAL_S = .06
CLOCK_WINDOW_S = .25
# Give heavy kernels time to reach the existing plateau criteria.
MAX_WARM_S = 20.0
SM_TOLERANCE = .04
LATENCY_TOLERANCE = .04


class ClockReader:
    """Read clock domains through NVML without changing device settings."""

    def __init__(self, gpu):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.device = ctypes.c_void_p()
        self.check(self.lib.nvmlInit_v2(), 'initialization')
        self.check(
            self.lib.nvmlDeviceGetHandleByUUID(gpu.encode('ascii'), ctypes.byref(self.device)),
            'device lookup',
        )

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f'NVML {operation} failed with code {code}')

    def __call__(self):
        value = ctypes.c_uint()
        result = []
        for domain in (1, 2):  # NVML_CLOCK_SM and NVML_CLOCK_MEM
            self.check(self.lib.nvmlDeviceGetClockInfo(self.device, domain, ctypes.byref(value)),
                       'clock read')
            result.append(value.value)
        if min(result) == 0:
            raise RuntimeError(f'NVML reported a zero active clock: {result}')
        return tuple(result)


def consistent(values, tolerance):
    return max(values) - min(values) <= tolerance * statistics.median(values)


def clocks_stable(clocks):
    sm, mem = zip(*clocks)
    return consistent(sm, SM_TOLERANCE) and min(mem) == max(mem)


def capture(launch, torch, stream, intervals):
    events = [
        (torch.cuda.Event(enable_timing=True, external=True),
         torch.cuda.Event(enable_timing=True, external=True))
        for _ in range(intervals)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for begin, end in events:
            begin.record()
            for _ in range(DEPTH):
                launch()
            end.record()
    return graph, events


def measure(launch, torch, clock, audit_trace=False):
    stream = torch.cuda.Stream()
    probe, probe_events = capture(launch, torch, stream, 1)
    graph, events = capture(launch, torch, stream, INTERVALS)
    started_ns = time.time_ns()
    trace = []
    checks = []
    observed = []
    attempts = 0

    def read_intervals(event_pairs):
        values = [begin.elapsed_time(end) / DEPTH for begin, end in event_pairs]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise RuntimeError('Invalid CUDA graph event interval')
        return values

    with torch.cuda.stream(stream):
        warm_started = time.monotonic()
        next_read = warm_started
        batch = 1
        while time.monotonic() - warm_started < MAX_WARM_S:
            tick = time.monotonic()
            for _ in range(batch):
                probe.replay()
            stream.synchronize()
            now = time.monotonic()
            # Adapt only how much warmup we enqueue at once. All measured intervals
            # always contain DEPTH launches and exclude CPU-side dispatch.
            batch = max(1, min(8, int(.014 * batch / max(.000001, now - tick))))
            if now < next_read:
                continue

            clocks = clock()
            latency = read_intervals(probe_events)[0]
            reading = (now, clocks, latency)
            trace.append((round(now - warm_started, 6), clocks, latency))
            observed = [entry for entry in observed if now - entry[0] <= 1.25]
            observed.append(reading)
            next_read = now + CLOCK_INTERVAL_S
            window = observed[-6:]
            if (now - warm_started < MIN_WARM_S or len(window) < 4
                    or now - window[0][0] < CLOCK_WINDOW_S):
                continue
            clock_window = [entry[1] for entry in window]
            if (not clocks_stable(clock_window)
                    or not consistent([entry[2] for entry in window], LATENCY_TOLERANCE)):
                continue

            attempts += 1
            values, measured_clocks = [], [clock()]
            for _ in range(REPLAYS):
                graph.replay()
                stream.synchronize()
                values.extend(read_intervals(events))
                measured_clocks.append(clock())
            # Retain evidence for rejected windows, including their timing samples.
            accepted = clocks_stable(clock_window + measured_clocks) and consistent(
                [entry[2] for entry in window] + [statistics.median(values)], LATENCY_TOLERANCE
            )
            if audit_trace or not accepted:
                checks.append(dict(samples_ms=values, clocks_mhz=measured_clocks, accepted=accepted))
            if accepted:
                return dict(
                    valid=True, median_ms=statistics.median(values), samples_ms=values,
                    clocks_mhz=measured_clocks, warm_clocks_mhz=clock_window,
                    warmup_seconds=round(now - warm_started, 6),
                    rejected_windows=checks[:-1] if audit_trace else checks,
                    warm_trace=trace if audit_trace else [], attempts=attempts,
                    start_ns=started_ns, end_ns=time.time_ns(),
                )
            observed.clear()
        elapsed = time.monotonic() - warm_started

    return dict(valid=False, reason='No stable active timing window', warm_trace=trace,
                rejected_windows=checks, attempts=attempts, warmup_seconds=elapsed,
                start_ns=started_ns, end_ns=time.time_ns())
