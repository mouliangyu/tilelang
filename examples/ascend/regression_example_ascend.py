"""Ascend perf-regression driver with per-entry process isolation.

Run via `maint/scripts/run_perf_regression_ascend.sh` (which runs this file once per
ref through `regression_all.py`), or directly: `python regression_example_ascend.py`.

Why this file is self-isolating
--------------------------------
`regression_all.py` runs this whole file as a SINGLE subprocess, and many Ascend
examples can fail *hard* on the baseline ref — a C++ `LOG(FATAL)`/`abort`, an NPU/bisheng
crash, an import-time failure, or a hang — none of which a Python `try/except` can catch.
A single such failure would otherwise kill the process and discard EVERY result for that
ref. So instead of running all entries in one process, this driver re-invokes itself once
per entry (`TL_PERF_REGRESSION_ENTRY=<name>`) in a child subprocess with a timeout. A
crash/timeout/import-failure in one entry only costs that entry's result; the parent logs
it and moves on.

This logic is intentionally kept inside this Ascend-only file. The shared CUDA-path
modules (`tilelang/testing/perf_regression.py`, `maint/scripts/regression_all.py`) are not
modified; we only reuse their public JSON-marker contract.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

import tilelang.testing

_HERE = os.path.dirname(os.path.abspath(__file__))

# Must match the marker the shared collectors (`regression_all.py`,
# `run_perf_regression_ascend.sh`) grep for.
_MARKER = "__TILELANG_PERF_RESULTS_JSON__="
_ENV_ENTRY = "TL_PERF_REGRESSION_ENTRY"  # child: run only this single entry
_ENV_TIMEOUT = "TL_PERF_REGRESSION_TIMEOUT"  # parent: per-entry timeout (seconds)
_DEFAULT_ENTRY_TIMEOUT = 900.0

# (entry name, example path relative to this file, run_regression_perf kwargs).
# One row == one isolated child. The path may live in a subfolder (e.g. flash_attention/);
# the child puts its directory on sys.path so intra-folder imports (``from core import …``)
# resolve without any dotted-module gymnastics.
_ENTRIES: list[tuple[str, str, dict]] = [
    ("ascend_gemm_bf16", "example_gemm.py", {"dtype": "bfloat16"}),
    ("ascend_gemm_fp32", "example_gemm.py", {"dtype": "float32"}),
    ("ascend_gemm_fp32_hf32", "example_gemm.py", {"dtype": "float32", "hf32": "nearest_even"}),
    ("ascend_gemm_l0", "example_gemm_l0.py", {}),
    ("ascend_gemm_mix_manual", "example_gemm_mix_manual.py", {}),
    ("ascend_gemm_mixedkernel", "example_gemm_mixedkernel.py", {}),
    ("ascend_gemm_bypass_l2", "example_gemm_bypass_l2.py", {}),
    ("ascend_gemm_various_shapes", "example_gemm_various_shapes.py", {}),
    ("ascend_blockscaled_gemm", "example_blockscaled_gemm.py", {}),
    ("ascend_mha", "flash_attention/example_mha.py", {}),
    ("ascend_gqa", "flash_attention/example_gqa.py", {}),
    ("ascend_rmsnorm", "example_rmsnorm.py", {}),
    ("ascend_buffer_version_annotation", "example_buffer_version_annotation.py", {}),
    ("ascend_atomic_add", "example_atomic.py", {}),
    ("ascend_simdvf_topk_gate", "example_simdvf_topk_gate.py", {}),
    ("ascend_simdvf_vecadd", "example_simdvf_vecadd.py", {}),
    ("ascend_simdvf_vecadd_lower", "example_simdvf_vecadd_lower.py", {}),
    ("ascend_simtvf_vecadd", "example_simtvf_vecadd.py", {}),
    ("ascend_simtvf_vecadd_mutex", "example_simtvf_vecadd_mutex.py", {}),
    ("ascend_vmi_vecadd", "example_vmi_vecadd.py", {}),
    ("ascend_simdvf_per_token_cast_to_fp8", "example_simdvf_per_token_cast_to_fp8.py", {}),
    ("ascend_simtvf_per_token_cast_to_fp8", "example_simtvf_per_token_cast_to_fp8.py", {}),
    ("ascend_crosslevel_multibuffer", "example_crosslevel_multibuffer.py", {}),
    ("ascend_compress", "example_compress.py", {}),
]


def _load_example(rel_path: str):
    """Import an example by file path, with its own directory on sys.path[0].

    Loading by path (rather than dotted module name) lets a subfolder example
    (e.g. ``flash_attention/example_mha.py``) use plain ``from core import …``
    against a sibling ``core.py`` — the same way it resolves when run directly.
    """
    file_path = os.path.join(_HERE, rel_path)
    example_dir = os.path.dirname(file_path)
    sys.path.insert(0, example_dir)
    mod_name = "_regression_example_" + os.path.splitext(os.path.basename(rel_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_child(entry_name: str) -> int:
    """Child mode: import + run exactly one entry, emit its JSON marker, return exit code.

    Importing the example module is deferred to here so that an import-time hard failure is
    contained to this one child process instead of aborting the whole suite.
    """
    spec = next((e for e in _ENTRIES if e[0] == entry_name), None)
    if spec is None:
        print(f"  ⚠️  unknown perf regression entry: {entry_name}", flush=True)
        print(_MARKER + json.dumps([]), flush=True)
        return 1

    _, rel_path, kwargs = spec
    results: list[dict] = []
    try:
        module = _load_example(rel_path)
        # process_func records into the shared _RESULTS and retries non-positive latency.
        from tilelang.testing import perf_regression as pr

        pr._reset_results()
        tilelang.testing.process_func(module.run_regression_perf, entry_name, **kwargs)
        results = [{"name": r.name, "latency": r.latency} for r in pr._RESULTS]
    except Exception as e:  # noqa: BLE001 - catchable failures still skip cleanly
        print(f"  ⚠️  {entry_name} failed, skipping: {type(e).__name__}: {e}", flush=True)

    print(_MARKER + json.dumps(results, separators=(",", ":")), flush=True)
    return 0


def _run_parent() -> int:
    """Parent mode: run each entry in its own subprocess; merge surviving results."""
    try:
        timeout = float(os.environ.get(_ENV_TIMEOUT, _DEFAULT_ENTRY_TIMEOUT))
    except ValueError:
        timeout = _DEFAULT_ENTRY_TIMEOUT

    total = len(_ENTRIES)
    merged: list[dict] = []
    print(f"\n{'=' * 60}")
    print("  Ascend Performance Regression Suite (isolated)")
    print(f"  {total} entries, {timeout:.0f}s per-entry timeout")
    print(f"{'=' * 60}")

    for idx, (entry_name, _path, _kwargs) in enumerate(_ENTRIES, 1):
        print(f"\n  ├─ [{idx}/{total}] {entry_name}", end="", flush=True)
        child_env = {**os.environ, _ENV_ENTRY: entry_name, "TL_PERF_REGRESSION_FORMAT": "json"}
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__)],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"\n  ⚠️  {entry_name} timed out after {timeout:.0f}s, skipping", flush=True)
            continue
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            print(
                f"\n  ⚠️  {entry_name} crashed (exit {proc.returncode}), skipping. Last output:\n{tail}",
                flush=True,
            )
            continue

        items = _parse_marker(proc.stdout or "")
        if not items:
            print(f"\n  ⚠️  {entry_name} produced no results, skipping", flush=True)
            continue
        merged.extend(items)
        print(f" ({elapsed:.2f}s)", flush=True)

    print(f"\n{'=' * 60}")
    print(f"  Collected {len(merged)}/{total} entries")
    print(f"{'=' * 60}\n")
    # The single marker line the parent collector parses.
    print(_MARKER + json.dumps(merged, separators=(",", ":")), flush=True)
    return 0


def _parse_marker(output: str) -> list[dict]:
    for line in reversed(output.splitlines()):
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER) :].strip())
            except json.JSONDecodeError:
                return []
    return []


if __name__ == "__main__":
    entry = os.environ.get(_ENV_ENTRY)
    sys.exit(_run_child(entry) if entry else _run_parent())
