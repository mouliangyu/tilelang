"""
asckit.bench — NPU kernel benchmarking
from kx.zhou
"""

import struct
import shutil
import signal
import tempfile
import os
import csv
from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from collections import defaultdict

__all__ = ["KernelProfile", "bench_msprof_slow", "bench_msprof_quick", "bench_msprof", "bench"]


@dataclass
class KernelProfile:
    """Per-kernel profiling result with AIC and AIV pipe utilization.
    All pipe fields are ratios in [0, 1] relative to their own total_cycles."""

    dur_ns: float  # duration in nanoseconds
    # AIC pipes
    aic_total_cycles: int = 0
    aic_mad: float = 0.0  # Cube MAD
    aic_scalar: float = 0.0
    aic_mte1: float = 0.0  # L1 -> L0
    aic_mte2: float = 0.0  # GM -> L1
    aic_mte3: float = 0.0  # UB -> GM (on AIC, rarely used)
    aic_fixpipe: float = 0.0  # L0C -> out
    # AIV pipes
    aiv_total_cycles: int = 0
    aiv_vec: float = 0.0  # Vector compute
    aiv_scalar: float = 0.0
    aiv_mte2: float = 0.0  # GM -> UB
    aiv_mte3: float = 0.0  # UB -> GM

    def __add__(self, other: "KernelProfile") -> "KernelProfile":
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
def suppress_stdout_stderr(suppress: bool):
    if not suppress:
        yield
        return

    with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
        yield


@contextmanager
def _defer_signals():
    """Defer SIGINT/SIGTERM during a critical section.

    The profiler hangs if interrupted mid-operation (e.g. during
    finalize_trace or stop_trace). This context manager blocks delivery of
    SIGINT and SIGTERM until the critical section exits, then re-raises
    KeyboardInterrupt if one was received.
    """
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
        elif pending_signal == signal.SIGTERM:
            raise SystemExit(128 + signal.SIGTERM)


@contextmanager
def _prof_work_path():
    """Create a temporary directory for ASCEND_WORK_PATH during profiling.

    Prevents the profiler from dumping temporary files into cwd.
    Uses a fresh tempdir (guaranteed writable) and cleans it up on exit.
    Restores the original ASCEND_WORK_PATH value (or unsets) on exit.
    """
    key = "ASCEND_WORK_PATH"
    old = os.environ.get(key)
    tmp_dir = tempfile.mkdtemp(prefix="asckit_prof_")
    os.environ[key] = tmp_dir
    try:
        yield tmp_dir
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
        shutil.rmtree(tmp_dir, ignore_errors=True)


# A no-op callback passed as ``on_trace_ready`` to torch_npu.profiler.profile.
# When on_trace_ready is None the profiler enters "export_only_mode"
# and creates an ``export_only_prof_dir`` directory in the cwd as a side-effect.
# Passing any callable — even a no-op — prevents that.
def _nop_trace_ready(_prof):
    pass


def bench_msprof_quick(
    fn,
    kernel_names: str | list[str] = None,
    num_warmups: int = 30,
    num_tests: int = 100,
    suppress_verbose_output: bool = True,
    return_all_kernels: bool = False,
    flush_l2: bool = True,
) -> KernelProfile | dict[str, KernelProfile]:
    """Profile kernel(s) with pipe utilization via torch_npu.profiler.

    Uses Level1 + PipeUtilization, parses ffts_profile.data directly
    (skips CSV analyse). Overhead: ~0.6-2s.

    Args:
        kernel_names: str  -> match one kernel by substring, return KernelProfile
                      None -> require exactly one kernel, return KernelProfile
        return_all_kernels: If True, return dict[str, KernelProfile].
        flush_l2: If True, flush L2 cache between iterations.

    Returns:
        KernelProfile or dict[str, KernelProfile]
    """
    import torch
    import torch_npu.profiler

    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    flush_l2_size = int(8e9 // 4)

    with _defer_signals(), _prof_work_path(), suppress_stdout_stderr(suppress_verbose_output):
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
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device="npu").zero_()
                fn()
            torch.npu.synchronize()
            prof.step()

        prof_path = Path(prof.prof_if.prof_path)
        profiles = _parse_ffts_profile(prof_path)

    if return_all_kernels:
        return profiles

    # Filter
    if kernel_names is None:
        if len(profiles) != 1:
            summary = "\n".join(f"  - {n}" for n in profiles) or "  (none)"
            raise RuntimeError(f"Expected unique kernel, but found {len(profiles)}:\n{summary}")
        return next(iter(profiles.values()))
    if isinstance(kernel_names, str):
        matched = {n: p for n, p in profiles.items() if kernel_names in n}
        if len(matched) != 1:
            summary = "\n".join(f"  - {n}" for n in profiles) or "  (none)"
            raise RuntimeError(f"No unique kernel matching '{kernel_names}':\n{summary}")
        return next(iter(matched.values()))
    return [next(p for n, p in profiles.items() if name in n) for name in kernel_names]


def _parse_ffts_profile(prof_path: Path) -> dict[str, KernelProfile]:
    """Parse ffts_profile.data and correlate with kernel names.

    Returns {kernel_name: KernelProfile} with averaged metrics across iterations.
    """
    # Build name lookup from hash_dic + task_track
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

    # Parse ffts_profile records (128 bytes each)
    # [0]=header (high16=seq), [1]=total_cycles, [2]=block_config (high bit=AIV),
    # [5]=mac, [6]=scalar, [7]=mte1, [8]=mte2, [9]=mte3,
    # [12]=fixpipe, [14]=start_ts, [15]=end_ts
    # Mix kernels produce 2 records per launch: AIC (block_config < 2^31) + AIV (block_config >= 2^31)
    aic_records: dict[str, list[dict]] = defaultdict(list)
    aiv_records: dict[str, list[dict]] = defaultdict(list)
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
                "total_cycles": total_cycles,
                "vec": vals[4] / total_cycles,  # AIV vec pipe (0 on AIC)
                "mad": vals[5] / total_cycles,  # AIC mad pipe (0 on AIV)
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

    # Merge AIC + AIV into a single KernelProfile per kernel (average across iterations)
    all_names = set(aic_records.keys()) | set(aiv_records.keys())
    result: dict[str, KernelProfile] = {}
    for name in all_names:
        aic_list = aic_records.get(name, [])
        aiv_list = aiv_records.get(name, [])
        # Duration from AIC (or AIV if no AIC)
        dur_list = aic_list or aiv_list
        n_aic = len(aic_list) or 1
        n_aiv = len(aiv_list) or 1
        result[name] = KernelProfile(
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
    return result


def _parse_kernel_details_csv(prof_path: Path) -> dict[str, KernelProfile]:
    """Parse kernel_details.csv produced by msprof analyse.

    Returns {kernel_name: KernelProfile} with averaged metrics across iterations.
    The CSV has per-task rows with columns like:
      Name, Duration(us), aic_total_cycles, aic_mac_ratio, aic_scalar_ratio,
      aic_mte1_ratio, aic_mte2_ratio, aic_mte3_ratio, aic_fixpipe_ratio,
      aiv_total_cycles, aiv_vec_ratio, aiv_scalar_ratio, aiv_mte2_ratio, aiv_mte3_ratio
    """
    # Find kernel_details.csv (in ASCEND_PROFILER_OUTPUT/)
    csv_files = list(prof_path.rglob("kernel_details.csv"))
    if not csv_files:
        raise FileNotFoundError(f"kernel_details.csv not found under {prof_path}")
    csv_file = csv_files[0]

    # Parse CSV rows grouped by kernel Name
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if name:
                records[name].append(row)

    def _float(row, key, default=0.0):
        v = row.get(key, "").strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _int(row, key, default=0):
        v = row.get(key, "").strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    result: dict[str, KernelProfile] = {}
    for name, rows in records.items():
        n = len(rows)
        result[name] = KernelProfile(
            dur_ns=sum(_float(r, "Duration(us)") for r in rows) / n * 1000,  # us -> ns
            aic_total_cycles=int(sum(_int(r, "aic_total_cycles") for r in rows) / n),
            aic_mad=sum(_float(r, "aic_mac_ratio") for r in rows) / n,
            aic_scalar=sum(_float(r, "aic_scalar_ratio") for r in rows) / n,
            aic_mte1=sum(_float(r, "aic_mte1_ratio") for r in rows) / n,
            aic_mte2=sum(_float(r, "aic_mte2_ratio") for r in rows) / n,
            aic_mte3=sum(_float(r, "aic_mte3_ratio") for r in rows) / n,
            aic_fixpipe=sum(_float(r, "aic_fixpipe_ratio") for r in rows) / n,
            aiv_total_cycles=int(sum(_int(r, "aiv_total_cycles") for r in rows) / n),
            aiv_vec=sum(_float(r, "aiv_vec_ratio") for r in rows) / n,
            aiv_scalar=sum(_float(r, "aiv_scalar_ratio") for r in rows) / n,
            aiv_mte2=sum(_float(r, "aiv_mte2_ratio") for r in rows) / n,
            aiv_mte3=sum(_float(r, "aiv_mte3_ratio") for r in rows) / n,
        )
    return result


def bench_msprof_slow(
    fn,
    kernel_names: str | list[str] = None,
    num_warmups: int = 30,
    num_tests: int = 100,
    suppress_verbose_output: bool = True,
    return_all_kernels: bool = False,
    flush_l2: bool = True,
) -> KernelProfile | dict[str, KernelProfile]:
    """Full torch_npu.profiler with on_trace_ready (generates trace_view.json etc).

    Same return type as bench_msprof_quick but slower (~6s).
    Parses the kernel_details.csv produced by msprof analyse (the official
    CSV output) rather than re-parsing raw binary ffts_profile data.
    """
    import torch
    import torch_npu.profiler

    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()

    flush_l2_size = int(8e9 // 4)

    trace_dir = Path(tempfile.mkdtemp(prefix="asckit_bench_"))

    with _defer_signals(), _prof_work_path(), suppress_stdout_stderr(suppress_verbose_output):
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=True,
            with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            ),
        ) as prof:
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device="npu").zero_()
                fn()
            torch.npu.synchronize()
            prof.step()

        prof_path = Path(prof.prof_if.prof_path)
        profiles = _parse_kernel_details_csv(prof_path)
        shutil.rmtree(trace_dir, ignore_errors=True)

    if return_all_kernels:
        return profiles

    if kernel_names is None:
        if len(profiles) != 1:
            summary = "\n".join(f"  - {n}" for n in profiles) or "  (none)"
            raise RuntimeError(f"Expected unique kernel, but found {len(profiles)}:\n{summary}")
        return next(iter(profiles.values()))
    if isinstance(kernel_names, str):
        matched = {n: p for n, p in profiles.items() if kernel_names in n}
        if len(matched) != 1:
            summary = "\n".join(f"  - {n}" for n in profiles) or "  (none)"
            raise RuntimeError(f"No unique kernel matching '{kernel_names}':\n{summary}")
        return next(iter(matched.values()))
    return [next(p for n, p in profiles.items() if name in n) for name in kernel_names]


bench_msprof = bench_msprof_quick


def bench(fn, warmup: int = 30, iters: int = 100) -> KernelProfile:
    """Benchmark a kernel on NPU. Returns KernelProfile with duration + pipe utilization.

    Example:
        >>> p = bench(lambda: ker(x, y, z, N))
        >>> print(f"{p.dur_us:.1f} us, {p.tflops(2*M*N*K):.1f} TFLOPS")
    """
    return bench_msprof_quick(fn, kernel_names=None, num_warmups=warmup, num_tests=iters, flush_l2=False)
