"""Profiler and benchmarking utilities for PyTorch functions."""

from __future__ import annotations

import os
import sys
from typing import Callable, Literal

import torch
import tilelang
import tilelang.language as T

from tilelang.utils.device import Event, device_synchronize, get_current_device


@tilelang.jit(out_idx=[], target="ascend")
def get_msprof_cache_flush_kernel(numel: int, dtype: T.dtype):
    elems_per_block = (numel + 63) // 64
    vec = 2048
    repeat = (elems_per_block + vec - 1) // vec

    @T.prim_func
    def _tilelang_profiler_cache_flush(Cache: T.Tensor((numel,), dtype)):
        with T.Kernel(64) as bx:
            for r in range(repeat):
                with T.SimtVF(threads=vec):
                    for i in T.Parallel(vec):
                        offset = bx * elems_per_block + r * vec + i
                        if offset < numel:
                            Cache[offset] = T.cast(0, dtype)

    return _tilelang_profiler_cache_flush


class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr output.

    Source: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
    """

    def __enter__(self):
        # Open null device files
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Save original stdout/stderr objects
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Redirect file descriptors and streams to null device
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

        return self

    def __exit__(self, *_):
        # Restore original stdout/stderr objects
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        # Close duplicated file descriptors
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        # Close null device files
        self.outnull_file.close()
        self.errnull_file.close()


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti", "cudagraph", "msprof", "msprof_detail"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
) -> float | list[float]:
    """Benchmark the runtime of a PyTorch function with L2 cache management.

    This function provides accurate GPU kernel timing by:
    - Clearing L2 cache between runs for consistent measurements
    - Auto-calculating warmup and repeat counts based on kernel runtime
    - Supporting multiple profiling backends (CUDA events, CUPTI, or CUDA graph replay)
    - Offering flexible result aggregation (mean/median/min/max/quantiles)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (default: 25)
        rep: Target total benchmark time in milliseconds (default: 100)
        _n_warmup: Manual override for warmup iterations (default: 0 = auto)
        _n_repeat: Manual override for benchmark iterations (default: 0 = auto)
        quantiles: Performance percentiles to compute (e.g., [0.5, 0.95])
        fast_flush: Use faster L2 cache flush with int32 vs int8 (default: True)
        backend: Profiler backend - "event", "cupti", "cudagraph", or "msprof" (default: "event")
        return_mode: Result aggregation method - "mean", "median", "min", or "max"

    Returns:
        Runtime in milliseconds (float) or list of quantile values if quantiles specified
    """
    assert return_mode in ["min", "max", "mean", "median"], f"Invalid return_mode: {return_mode}"

    device = get_current_device()

    # Initial function call and synchronization
    fn()
    device_synchronize()

    # Create L2 cache flush buffer (256 MB)
    # Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
    cache_size = int(256e6 // 4) if fast_flush else int(256e6)
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_size, dtype=cache_dtype, device=device)

    # Estimate kernel runtime with 5 iterations
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    start_event.synchronize()
    end_event.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Calculate warmup and repeat counts (minimum 1 iteration each)
    n_warmup = _n_warmup if _n_warmup > 0 else max(1, int(warmup / estimate_ms))
    n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

    # Warmup phase
    for _ in range(n_warmup):
        fn()

    # Benchmarking phase
    if backend == "event":
        return _bench_with_cuda_events(fn, cache, n_repeat, quantiles, return_mode)
    elif backend == "cupti":
        return _bench_with_cupti(fn, cache, n_repeat)
    elif backend == "cudagraph":
        return _bench_with_cudagraph(fn, cache, n_repeat, quantiles, return_mode)
    elif backend.startswith("msprof"):
        return _bench_with_msprof(fn, cache, n_repeat, detailed=backend == "msprof_detail")
    else:
        raise ValueError(f"Unknown profiler backend: {backend}")


def _bench_with_cuda_events(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    """Benchmark using device events for timing."""
    # Create timing events
    start_events = [Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [Event(enable_timing=True) for _ in range(n_repeat)]

    # Run benchmark iterations
    for i in range(n_repeat):
        cache.zero_()  # Clear L2 cache
        start_events[i].record()
        fn()
        end_events[i].record()

    # Synchronize and collect timings
    device_synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float,
    )

    # Return quantiles if requested
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantile_values[0] if len(quantile_values) == 1 else quantile_values

    # Return aggregated result
    return getattr(torch, return_mode)(times).item()


def _bench_with_cupti(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
) -> float:
    """Benchmark using CUPTI profiler for detailed kernel timing."""
    with suppress_stdout_stderr():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
        )

        with profiler:
            for _ in range(2):
                for _ in range(n_repeat):
                    cache.zero_()
                    fn()
                profiler.step()

    # Calculate average kernel time, excluding cache-clearing overhead
    total_cuda_time = 0.0
    excluded_time = 0.0
    excluded_kernels = "at::native::vectorized_elementwise"

    for event in profiler.key_averages():
        total_cuda_time += event.self_device_time_total
        if excluded_kernels in event.key:
            excluded_time += event.self_device_time_total

    kernel_time_us = (total_cuda_time - excluded_time) / n_repeat
    return kernel_time_us * 1e-3  # Convert microseconds to milliseconds


def _bench_with_msprof(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    detailed: bool = False,
) -> float:
    """Benchmark using torch_npu msprof timing."""
    import shutil
    import signal
    import struct
    import tempfile
    from collections import defaultdict
    from contextlib import contextmanager, redirect_stderr, redirect_stdout
    from dataclasses import dataclass
    from pathlib import Path

    import torch_npu.profiler

    @dataclass
    class KernelProfile:
        """Per-kernel profiling result with AIC and AIV pipe utilization.

        All pipe fields are ratios in [0, 1] relative to their own total_cycles.
        """

        dur_ns: float
        # AIC pipes
        aic_total_cycles: int = 0
        aic_mad: float = 0.0
        aic_scalar: float = 0.0
        aic_mte1: float = 0.0
        aic_mte2: float = 0.0
        aic_mte3: float = 0.0
        aic_fixpipe: float = 0.0
        # AIV pipes
        aiv_total_cycles: int = 0
        aiv_vec: float = 0.0
        aiv_scalar: float = 0.0
        aiv_mte2: float = 0.0
        aiv_mte3: float = 0.0

        def __add__(self, other: KernelProfile) -> KernelProfile:
            """Sum two profiles: dur_ns adds, pipe ratios are cycle-weighted averages."""
            aic_c = self.aic_total_cycles + other.aic_total_cycles
            aiv_c = self.aiv_total_cycles + other.aiv_total_cycles
            wa = self.aic_total_cycles / aic_c if aic_c else 0
            wb = other.aic_total_cycles / aic_c if aic_c else 0
            va = self.aiv_total_cycles / aiv_c if aiv_c else 0
            vb = other.aiv_total_cycles / aiv_c if aiv_c else 0
            return KernelProfile(
                dur_ns=self.dur_ns + other.dur_ns,
                aic_total_cycles=aic_c,
                aic_mad=self.aic_mad * wa + other.aic_mad * wb,
                aic_scalar=self.aic_scalar * wa + other.aic_scalar * wb,
                aic_mte1=self.aic_mte1 * wa + other.aic_mte1 * wb,
                aic_mte2=self.aic_mte2 * wa + other.aic_mte2 * wb,
                aic_mte3=self.aic_mte3 * wa + other.aic_mte3 * wb,
                aic_fixpipe=self.aic_fixpipe * wa + other.aic_fixpipe * wb,
                aiv_total_cycles=aiv_c,
                aiv_vec=self.aiv_vec * va + other.aiv_vec * vb,
                aiv_scalar=self.aiv_scalar * va + other.aiv_scalar * vb,
                aiv_mte2=self.aiv_mte2 * va + other.aiv_mte2 * vb,
                aiv_mte3=self.aiv_mte3 * va + other.aiv_mte3 * vb,
            )

        @property
        def dur_us(self) -> float:
            return self.dur_ns / 1000

        def tflops(self, flops: float) -> float:
            return flops / self.dur_ns / 1000

        def gbps(self, bytes: float) -> float:
            return bytes / self.dur_ns / 1e3

        def __repr__(self):
            def _fmt(label, val):
                return f"{label}={val * 100:.1f}%" if val > 0.001 else None

            parts = [f"{self.dur_ns / 1000:.1f}us"]
            for s in [
                _fmt("mad", self.aic_mad),
                _fmt("aic_mte2", self.aic_mte2),
                _fmt("aic_mte1", self.aic_mte1),
                _fmt("aic_fix", self.aic_fixpipe),
                _fmt("aiv_vec", self.aiv_vec),
                _fmt("aiv_mte2", self.aiv_mte2),
                _fmt("aiv_mte3", self.aiv_mte3),
            ]:
                if s:
                    parts.append(s)
            if self.aic_total_cycles:
                parts.append(f"aic_cycles={self.aic_total_cycles}")
            if self.aiv_total_cycles:
                parts.append(f"aiv_cycles={self.aiv_total_cycles}")
            return f"KernelProfile({', '.join(parts)})"

    @contextmanager
    def _suppress_output(suppress: bool):
        if not suppress:
            yield
            return

        with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
            yield

    @contextmanager
    def _defer_signals():
        pending_signal = None

        def _handler(signum, frame):
            nonlocal pending_signal
            pending_signal = signum

        old_sigint = signal.signal(signal.SIGINT, _handler)
        old_sigterm = signal.signal(signal.SIGTERM, _handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if pending_signal == signal.SIGINT:
                raise KeyboardInterrupt()
            if pending_signal == signal.SIGTERM:
                raise SystemExit(128 + signal.SIGTERM)

    @contextmanager
    def _prof_work_path():
        key = "ASCEND_WORK_PATH"
        old = os.environ.get(key)
        tmp_dir = tempfile.mkdtemp(prefix="tilelang_prof_")
        os.environ[key] = tmp_dir
        try:
            yield tmp_dir
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _nop_trace_ready(_prof):
        pass

    def _parse_ffts_profile(prof_path: Path):
        hash_to_name: dict[int, str] = {}
        for f in prof_path.rglob("*hash_dic.slice_*"):
            if f.name.endswith(".done"):
                continue
            text = f.read_bytes().decode("utf-8", errors="replace")
            for line in text.strip().split("\n"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        h = int(parts[0])
                        if h >= 2**63:
                            h -= 2**64
                        hash_to_name[h] = parts[1]
                    except ValueError:
                        pass

        seq_to_name: dict[int, str] = {}
        for f in prof_path.rglob("*task_track.slice_*"):
            if f.name.endswith(".done"):
                continue
            data = f.read_bytes()
            for i in range(len(data) // 64):
                vals = struct.unpack_from("<8q", data, i * 64)
                seq = (vals[3] >> 32) & 0xFFFF
                name_hash = vals[5]
                if name_hash != 0:
                    seq_to_name[seq] = hash_to_name.get(name_hash, f"<unknown {name_hash}>")

        aic_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        aiv_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        for f in prof_path.rglob("ffts_profile*"):
            if f.name.endswith(".done") or f.stat().st_size == 0:
                continue
            data = f.read_bytes()
            for i in range(len(data) // 128):
                vals = struct.unpack_from("<16q", data, i * 128)
                total_cycles = vals[1]
                if total_cycles == 0:
                    continue
                seq = (vals[0] >> 32) & 0xFFFF
                is_aiv = vals[2] >= (1 << 31)
                dur_ns = float(vals[15] - vals[14])
                name = seq_to_name.get(seq, "<unknown>")
                metrics = {
                    "dur_ns": dur_ns,
                    "total_cycles": float(total_cycles),
                    "vec": vals[4] / total_cycles,
                    "mad": vals[5] / total_cycles,
                    "scalar": vals[6] / total_cycles,
                    "mte1": vals[7] / total_cycles,
                    "mte2": vals[8] / total_cycles,
                    "mte3": vals[9] / total_cycles,
                    "fixpipe": vals[12] / total_cycles,
                }
                if is_aiv:
                    aiv_records[name].append(metrics)
                else:
                    aic_records[name].append(metrics)

        profiles: dict[str, KernelProfile] = {}
        for name in set(aic_records.keys()) | set(aiv_records.keys()):
            aic_list = aic_records.get(name, [])
            aiv_list = aiv_records.get(name, [])
            dur_list = aic_list or aiv_list
            n_aic = len(aic_list) or 1
            n_aiv = len(aiv_list) or 1
            profiles[name] = KernelProfile(
                dur_ns=sum(m["dur_ns"] for m in dur_list) / len(dur_list),
                aic_total_cycles=int(sum(m["total_cycles"] for m in aic_list) / n_aic),
                aic_mad=sum(m["mad"] for m in aic_list) / n_aic,
                aic_scalar=sum(m["scalar"] for m in aic_list) / n_aic,
                aic_mte1=sum(m["mte1"] for m in aic_list) / n_aic,
                aic_mte2=sum(m["mte2"] for m in aic_list) / n_aic,
                aic_mte3=sum(m["mte3"] for m in aic_list) / n_aic,
                aic_fixpipe=sum(m["fixpipe"] for m in aic_list) / n_aic,
                aiv_total_cycles=int(sum(m["total_cycles"] for m in aiv_list) / n_aiv),
                aiv_vec=sum(m["vec"] for m in aiv_list) / n_aiv,
                aiv_scalar=sum(m["scalar"] for m in aiv_list) / n_aiv,
                aiv_mte2=sum(m["mte2"] for m in aiv_list) / n_aiv,
                aiv_mte3=sum(m["mte3"] for m in aiv_list) / n_aiv,
            )
        return profiles

    def _sum_profiles(profiles):
        total = profiles[0]
        for profile in profiles[1:]:
            total += profile
        return total

    cache_dtype = "int32" if cache.dtype == torch.int32 else "int8"
    cache_flush_kernel = get_msprof_cache_flush_kernel(int(cache.numel()), cache_dtype)
    cache_flush_kernel(cache)
    torch.npu.synchronize()

    with _defer_signals(), _prof_work_path(), _suppress_output(True):
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=_nop_trace_ready,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            ),
        ) as prof:
            for _ in range(n_repeat):
                cache_flush_kernel(cache)
                fn()
            torch.npu.synchronize()
            prof.step()

        kernel_profiles = _parse_ffts_profile(Path(prof.prof_if.prof_path))

    kernel_profiles = {name: profile for name, profile in kernel_profiles.items() if name != "_tilelang_profiler_cache_flush_kernel"}
    if not kernel_profiles:
        raise RuntimeError("No profiled kernels remain after excluding the msprof flush kernel.")

    profile = _sum_profiles(list(kernel_profiles.values()))
    if detailed:
        return profile
    return profile.dur_ns / 1e6


def _bench_with_cudagraph(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    """Benchmark using CUDA graph for minimal launch overhead.

    This implementation follows triton.testing.do_bench_cudagraph.
    It captures the kernel execution in a CUDA graph and replays it multiple
    times to minimize host overhead and provide accurate timing measurements.

    Note: Cache flushing is done before graph replay, not within the graph,
    since CUDA graphs require fixed execution patterns.
    """
    n_retries = 10
    with torch.cuda.stream(torch.cuda.Stream()):
        # Construct a CUDA graph with `n_repeat` unrolled function calls to minimize host overhead.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()

        torch.cuda.synchronize()

        # Measure time by replaying the graph multiple times.
        # Clear cache before each replay for consistent measurements.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        for i in range(n_retries):
            cache.zero_()  # Clear L2 cache before replay
            start_events[i].record()
            g.replay()
            end_events[i].record()

        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) / n_repeat for s, e in zip(start_events, end_events)],
            dtype=torch.float,
        )

        # Return quantiles if requested
        if quantiles is not None:
            quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
            return quantile_values[0] if len(quantile_values) == 1 else quantile_values

        # Return aggregated result
        return getattr(torch, return_mode)(times).item()
