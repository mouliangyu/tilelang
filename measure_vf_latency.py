#!/usr/bin/env python3
"""Measure SimtVF/SimdVF latency via cannsim and inject results back into source.

This script compiles a TileLang Ascend kernel, extracts the generated VF (Vector
Function) code from the .asc output, runs each VF block independently under the
Ascend simulator (cannsim) to get cycle-accurate latency, and writes the measured
``latency=<cycles>`` annotation back into the original Python source file.

The measured latency is consumed by the auto-schedule pass (Z3 pipeline scheduler)
to produce better pipeline overlapping decisions.

Prerequisites:
  - Ascend toolkit installed (bisheng compiler, cannsim on PATH)
  - torch_npu importable (for cannsim device simulation)
  - NPU device accessible (cannsim needs /dev/davinci*)
  - Typically run with sudo for device access:
      sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \\
        ASCEND_NPU_ARCH=dav-3510 TILELANG_CACHE_DIR=/tmp/tilelang_cache \\
        python3 measure_vf_latency.py <kernel.py>

Usage::

    # Auto-detect function from __main__ block, measure and write back:
    python measure_vf_latency.py debug/flash_attn.py

    # Preview results without modifying the file:
    python measure_vf_latency.py debug/flash_attn.py --dry-run

    # Explicitly specify which function to compile:
    python measure_vf_latency.py debug/flash_attn.py --call "flash_attention(D=128, SEQ_LEN=4096)"

    # Works with any kernel that has SimtVF/SimdVF blocks:
    python measure_vf_latency.py examples/ascend/example_simdvf_vecadd.py
    python measure_vf_latency.py examples/ascend/example_simtvf_vector_add.py

How it works:
  1. Import the kernel module and find the function that returns a @T.prim_func.
     Auto-detection order:
       a) Parse ``if __name__ == "__main__"`` to find the call expression
          (e.g. ``program = vector_add(N)`` where N is resolved from locals)
       b) Try calling each exported function with common argument patterns
       c) Fall back to --call if provided
  2. Compile with tilelang.compile() (with auto_schedule; falls back to without
     if Z3 scheduling fails, e.g. for kernels with manual flag sync).
  3. Extract all ``__simt_vf__`` / ``__simd_vf__`` function definitions from the
     generated .asc source using regex + brace matching.
  4. For each VF function, generate a minimal standalone .asc that only calls
     that one VF (no DMA, no pipe_barrier, no cross-core sync), compile it with
     bisheng, run under ``cannsim record``, and parse the instruction log for
     the cycle count.
  5. Map extracted VF functions back to Python source SimtVF/SimdVF blocks by
     index (simt_vf_0 = 1st SimtVF in source, simd_vf_0 = 1st SimdVF, etc.).
  6. Inject ``latency=<cycles>`` into the original Python source file.

Options:
  --call EXPR    Expression that returns the @T.prim_func to compile.
                 If omitted, auto-detected from the module.
  --out-idx N    Output tensor index for tilelang.compile (default: -1).
  --soc SOC      cannsim SoC model name (default: Ascend950).
  --work-dir D   Directory for intermediate files (default: .cannsim_work/).
  --dry-run      Print measured latencies without modifying the source file.
"""

from __future__ import annotations

import ast
import contextlib
import glob as _glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Environment bootstrap
# ---------------------------------------------------------------------------

_CANN_SET_ENV = (sorted(_glob.glob("/usr/local/Ascend/cann-*/set_env.sh")) or ["/usr/local/Ascend/cann-9.0.T550/set_env.sh"])[-1]

_CANN_ROOT = str(Path(_CANN_SET_ENV).parent)
_CANN_PYTHON_SITE = os.path.join(_CANN_ROOT, "python", "site-packages")

_REPO_ROOT = Path(__file__).resolve().parent
_TILELANG_TEMPLATE_PATH = str(_REPO_ROOT / "src")


def _bootstrap_env() -> None:
    if "ASCEND_NPU_ARCH" in os.environ:
        return
    if not os.path.isfile(_CANN_SET_ENV):
        sys.exit(f"Ascend set_env.sh not found at {_CANN_SET_ENV}. Set env manually.")
    shell = os.environ.get("SHELL", "/bin/bash")
    output = subprocess.check_output(f'. "{_CANN_SET_ENV}" && env', shell=True, executable=shell, text=True)
    for line in output.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k and k not in ("_", "SHLVL"):
            os.environ[k] = v
    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-3510")


_bootstrap_env()
_CANNSIM = os.environ.get("CANNSIM", "cannsim")

# cannsim record sometimes truncates the instruction stream (the VF body never
# executes); re-recording usually succeeds. Bound the retries.
_RECORD_MAX_ATTEMPTS = int(os.environ.get("MEASURE_RECORD_ATTEMPTS", "5"))
# Seconds the launcher waits after synchronize() before os._exit(), so the
# camodel can finish flushing instr.bin (avoids truncated recordings).
_RECORD_FLUSH_WAIT = float(os.environ.get("MEASURE_FLUSH_WAIT", "5.0"))
# Seconds to wait between retries so the camodel + filesystem settle.
_RECORD_RETRY_DELAY = float(os.environ.get("MEASURE_RETRY_DELAY", "1.0"))

# ---------------------------------------------------------------------------
# 1. Compile kernel and get .asc source
# ---------------------------------------------------------------------------


def _compile_kernel(kernel_path: str, call_expr: str | None, out_idx: int) -> tuple[str, str | None]:
    """Import kernel module, get PrimFunc, compile.

    Returns (asc_source, scope_func) where scope_func is the name of the
    top-level def that produced the compiled kernel (used to scope the Python
    VF-block scan so blocks from other, uncompiled kernels in the same file are
    ignored). scope_func is None when it cannot be determined.
    """
    module_path = Path(kernel_path).resolve()
    spec = importlib.util.spec_from_file_location("_kernel_mod", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_kernel_mod"] = mod

    # Execute module but collect __main__ locals for eval context
    main_locals = _exec_module_with_main_locals(spec, mod, module_path)

    if call_expr:
        ns = {**vars(mod), **main_locals}
        func = eval(call_expr, ns)
        scope_func = _outermost_call_name(call_expr)
    else:
        func, scope_func = _auto_detect_func(mod, kernel_path, main_locals)

    if func is None:
        sys.exit("Could not find a PrimFunc. Use --call to specify.")

    import tilelang

    try:
        kernel = tilelang.compile(func, out_idx=out_idx)
    except Exception:
        kernel = tilelang.compile(func, out_idx=out_idx, pass_configs={"tl.enable_auto_schedule": False})
    return kernel.get_kernel_source(), scope_func


def _outermost_call_name(call_expr: str) -> str | None:
    """Get the function name of the outermost call in an expression string."""
    try:
        node = ast.parse(call_expr, mode="eval").body
        if isinstance(node, ast.Call):
            return _ast_attr_name(node.func)
    except (ValueError, SyntaxError):
        pass
    return None


def _exec_module_with_main_locals(spec, mod, module_path: Path) -> dict:
    """Execute the module and extract locals from its __main__ block.

    The module may fail during execution (e.g. ``tilelang.compile`` inside
    ``if __name__ == "__main__"`` raises an error on the current machine).
    We suppress *all* exceptions so we can still extract the variable
    assignments (``D = 128``, ``N = 2**30``, etc.) from the AST.
    """
    import unittest.mock

    with unittest.mock.patch.object(mod, "__name__", "_kernel_mod"), contextlib.suppress(Exception, SystemExit):
        spec.loader.exec_module(mod)

    # Now parse the __main__ block and exec variable assignments
    source = module_path.read_text()
    tree = ast.parse(source)
    main_locals: dict = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            continue
        # Execute simple assignments (N = 2**30, D = 128, etc.)
        # Only evaluate expressions that are pure literals — skip anything
        # that involves function calls (like program = flash_attention(...)).
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and not _contains_call(stmt.value):
                    try:
                        val = eval(compile(ast.Expression(body=stmt.value), "<main>", "eval"), {**vars(mod), **main_locals})
                        main_locals[target.id] = val
                    except Exception:
                        pass
        break

    return main_locals


def _auto_detect_func(mod, kernel_path: str, main_locals: dict = None):
    """Auto-detect PrimFunc by inspecting the module.

    Returns (PrimFunc, scope_func_name) — scope_func_name is the top-level def
    that produced the kernel, used to scope the Python VF-block scan. Returns
    (None, None) if no PrimFunc is found.

    Strategy:
    1. Look for functions decorated with @T.prim_func (already a PrimFunc)
    2. Look for functions that return PrimFunc when called with typical args
    3. Parse if __name__ == "__main__" block to find the call expression
    """
    import tvm.tir

    # Strategy 1: module-level PrimFunc objects
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, tvm.tir.PrimFunc):
            print(f"  auto-detected: {name} (PrimFunc object)")
            return obj, name

    # Strategy 2: parse __main__ block to find the compile call
    source = Path(kernel_path).read_text()
    call_expr = _parse_main_block(source)
    if call_expr:
        try:
            ns = {**vars(mod), **(main_locals or {})}
            func = eval(call_expr, ns)
            if isinstance(func, tvm.tir.PrimFunc):
                print(f"  auto-detected from __main__: {call_expr}")
                return func, _outermost_call_name(call_expr)
        except Exception:
            pass

    # Strategy 3: try calling each function with common arg patterns
    candidates = [
        name for name in dir(mod) if not name.startswith("_") and callable(getattr(mod, name)) and name not in ("ref_program", "main")
    ]
    common_args = [(), (8192,), (1, 8192), (128,), (1024,), (128, 4096), (2**20,), (2**30,)]
    for name in candidates:
        fn = getattr(mod, name)
        for args in common_args:
            try:
                result = fn(*args)
                if isinstance(result, tvm.tir.PrimFunc):
                    args_str = ", ".join(str(a) for a in args)
                    print(f"  auto-detected: {name}({args_str})")
                    return result, name
            except (TypeError, Exception):
                continue

    return None, None


def _parse_main_block(source: str) -> str | None:
    """Parse the if __name__ == '__main__' block to find the PrimFunc call.

    Looks for patterns like:
      program = func_name(args)
      kernel = tilelang.compile(func_name(args), ...)
    """
    tree = ast.parse(source)
    # Collect top-level function names defined in this file
    local_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            continue
        # Look for: var = local_func(...) or tilelang.compile(local_func(...), ...)
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            func_name = _ast_attr_name(value.func)
            # Direct call to a local function
            if func_name in local_funcs:
                call_src = _call_to_source(value, source)
                if call_src:
                    return call_src
            # tilelang.compile(local_func(...), ...)
            if func_name and "compile" in func_name and value.args:
                first_arg = value.args[0]
                if isinstance(first_arg, ast.Call):
                    arg_name = _ast_attr_name(first_arg.func)
                    if arg_name in local_funcs:
                        call_src = _call_to_source(first_arg, source)
                        if call_src:
                            return call_src
    return None


def _call_to_source(node, source: str) -> str | None:
    """Convert an AST Call node back to source text."""
    if not isinstance(node, ast.Call):
        return None
    try:
        lines = source.splitlines()
        # Get the source text for this call
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1
        if start_line == end_line:
            return lines[start_line][node.col_offset : node.end_col_offset]
        # Multi-line: just get the function name and args from AST
        func_name = _ast_attr_name(node.func)
        if func_name and not func_name.startswith("tilelang"):
            args = []
            for arg in node.args:
                try:
                    args.append(str(ast.literal_eval(arg)))
                except (ValueError, SyntaxError):
                    args.append(ast.unparse(arg))
            for kw in node.keywords:
                if kw.arg:
                    try:
                        args.append(f"{kw.arg}={ast.literal_eval(kw.value)}")
                    except (ValueError, SyntaxError):
                        args.append(f"{kw.arg}={ast.unparse(kw.value)}")
            return f"{func_name}({', '.join(args)})"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 2. Extract VF functions from .asc source
# ---------------------------------------------------------------------------

_SIMT_VF_PATTERN = re.compile(r"(__simt_vf__\s+__launch_bounds__\((\d+)\)\s+inline\s+void\s+(simt_vf_\d+)\s*\(([^)]*)\)\s*\{)")
_SIMD_VF_PATTERN = re.compile(r"(__simd_vf__\s+inline\s+void\s+(simd_vf_\d+)\s*\(([^)]*)\)\s*\{)")


def _find_matching_brace(source: str, open_pos: int) -> int:
    """Find the position of the closing brace matching the one at open_pos."""
    depth = 0
    i = open_pos
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(source) - 1


def _extract_vf_functions(asc_source: str) -> list[dict]:
    """Extract all VF function definitions from .asc source."""
    results = []

    for m in _SIMT_VF_PATTERN.finditer(asc_source):
        header = m.group(1)
        threads = int(m.group(2))
        name = m.group(3)
        params = m.group(4)

        brace_start = m.start() + len(header) - 1
        brace_end = _find_matching_brace(asc_source, brace_start)
        full_source = asc_source[m.start() : brace_end + 1]

        scalar_params = _extract_scalar_params(params)
        gm_params = _extract_gm_params(params)
        results.append(
            {
                "kind": "simt",
                "name": name,
                "threads": threads,
                "params": params,
                "scalar_params": scalar_params,
                "gm_params": gm_params,
                "source": full_source,
                "pos": m.start(),
            }
        )

    for m in _SIMD_VF_PATTERN.finditer(asc_source):
        header = m.group(1)
        name = m.group(2)
        params = m.group(3)

        brace_start = m.start() + len(header) - 1
        brace_end = _find_matching_brace(asc_source, brace_start)
        full_source = asc_source[m.start() : brace_end + 1]

        scalar_params = _extract_scalar_params(params)
        gm_params = _extract_gm_params(params)
        results.append(
            {
                "kind": "simd",
                "name": name,
                "threads": 0,
                "params": params,
                "scalar_params": scalar_params,
                "gm_params": gm_params,
                "source": full_source,
                "pos": m.start(),
            }
        )

    results.sort(key=lambda x: x["pos"])

    # Extract source_index from function name (e.g. "simt_vf_2" -> 2)
    _idx_pattern = re.compile(r"_(\d+)$")
    for r in results:
        m = _idx_pattern.search(r["name"])
        if m:
            r["source_index"] = int(m.group(1))

    return results


def _extract_scalar_params(params_str: str) -> list[str]:
    """Extract non-buffer parameter names (those without __ubuf__ or __gm__)."""
    scalars = []
    for param in params_str.split(","):
        param = param.strip()
        if not param or "__ubuf__" in param or "__gm__" in param:
            continue
        parts = param.split()
        if parts:
            scalars.append(parts[-1])
    return scalars


def _extract_gm_params(params_str: str) -> list[str]:
    """Extract __gm__ parameter names."""
    gm_params = []
    for param in params_str.split(","):
        param = param.strip()
        if "__gm__" in param:
            parts = param.split()
            if parts:
                gm_params.append(parts[-1].lstrip("*"))
    return gm_params


def _extract_includes(asc_source: str) -> str:
    """Extract all #include lines from the .asc source, ensuring common.h is present."""
    lines = []
    has_common = False
    for line in asc_source.splitlines():
        if line.startswith("#include"):
            lines.append(line)
            if "common.h" in line:
                has_common = True
    if not has_common:
        lines.insert(0, "#include <tl_templates/ascend/common.h>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Generate measurement .asc
# ---------------------------------------------------------------------------


def _gen_measure_asc(includes: str, vf: dict) -> str:
    """Generate a standalone .asc that only calls one VF function."""
    lines: list[str] = []
    A = lines.append

    A(includes)
    A("")
    A(vf["source"])
    A("")

    has_ubuf = "__ubuf__" in vf["params"]

    A("__global__ __vector__ void measure_kernel(__gm__ uint8_t* __dummy) {")
    if has_ubuf:
        A("    __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;")

    # Build argument list — cast each param to its declared type
    call_args = []
    for param in vf["params"].split(","):
        param = param.strip()
        if not param:
            continue
        # Extract the type (everything before the last identifier)
        parts = param.rsplit(" ", 1)
        if len(parts) < 2:
            call_args.append("0")
            continue
        ptype = parts[0].strip()  # e.g. "__ubuf__ float*" or "__gm__ float*" or "int32_t"

        if "__ubuf__" in ptype:
            call_args.append(f"({ptype})buf_dyn_shmem")
        elif "__gm__" in ptype:
            call_args.append(f"({ptype})__dummy")
        else:
            call_args.append("0")

    args_str = ", ".join(call_args)
    if vf["kind"] == "simt":
        A(f"    asc_vf_call<{vf['name']}>(cce::dim3({vf['threads']}), {args_str});")
    else:
        A(f"    {vf['name']}({args_str});")

    A("}")
    A("")
    A("#define ERROR_BUF_SIZE 1024")
    A("static char error_buf[ERROR_BUF_SIZE];")
    A('extern "C" const char* get_last_error() { return error_buf; }')
    A("extern \"C\" int init() { error_buf[0] = '\\0'; return 0; }")
    A('extern "C" int call(uint8_t* __dummy, void* stream) {')
    A("    measure_kernel<<<1, 0, stream>>>(__dummy);")
    A('    TILELANG_CHECK_LAST_ERROR("measure_kernel");')
    A("    return 0;")
    A("}")

    return "\n".join(lines)


def _gen_measure_asc_multi(includes: str, vf: dict, count: int = 16) -> str:
    """Like :func:`_gen_measure_asc` but calls the VF ``count`` times in a
    loop so that the vector region is long enough for the camodel to capture
    reliably.  The caller divides the measured cycle count by ``count``.
    """
    lines: list[str] = []
    A = lines.append

    A(includes)
    A("")
    A(vf["source"])
    A("")

    has_ubuf = "__ubuf__" in vf["params"]

    # Build argument list
    call_args = []
    for param in vf["params"].split(","):
        param = param.strip()
        if not param:
            continue
        parts = param.rsplit(" ", 1)
        if len(parts) < 2:
            call_args.append("0")
            continue
        ptype = parts[0].strip()
        if "__ubuf__" in ptype:
            call_args.append(f"({ptype})buf_dyn_shmem")
        elif "__gm__" in ptype:
            call_args.append(f"({ptype})__dummy")
        else:
            call_args.append("0")
    args_str = ", ".join(call_args)

    A("__global__ __vector__ void measure_kernel(__gm__ uint8_t* __dummy) {")
    if has_ubuf:
        A("    __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;")

    if vf["kind"] == "simt":
        A(f"    for (int32_t _mi = 0; _mi < {count}; ++_mi) {{")
        A(f"        asc_vf_call<{vf['name']}>(cce::dim3({vf['threads']}), {args_str});")
        A("    }")
    else:
        A(f"    for (int32_t _mi = 0; _mi < {count}; ++_mi) {{")
        A(f"        {vf['name']}({args_str});")
        A("    }")
    A("}")
    A("")
    A("#define ERROR_BUF_SIZE 1024")
    A("static char error_buf[ERROR_BUF_SIZE];")
    A('extern "C" const char* get_last_error() { return error_buf; }')
    A("extern \"C\" int init() { error_buf[0] = '\\0'; return 0; }")
    A('extern "C" int call(uint8_t* __dummy, void* stream) {')
    A("    measure_kernel<<<1, 0, stream>>>(__dummy);")
    A('    TILELANG_CHECK_LAST_ERROR("measure_kernel");')
    A("    return 0;")
    A("}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Compile, launch, cannsim
# ---------------------------------------------------------------------------


def _compile_asc(asc_path: Path, so_path: Path) -> None:
    """Compile .asc to .so with bisheng."""
    npu_arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
    cmd = [
        "bisheng",
        "-fPIC",
        "--shared",
        str(asc_path),
        f"--npu-arch={npu_arch}",
        "-I",
        _TILELANG_TEMPLATE_PATH,
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _gen_launcher(so_path: Path, flush_wait: float = _RECORD_FLUSH_WAIT) -> str:
    """Generate Python launcher that calls the .so via ctypes + torch_npu."""
    # Use a reasonably-sized dummy buffer so that GM accesses at offsets
    # don't hit out-of-bounds in the camodel simulator.  16 KB covers
    # typical VF GM access patterns without wasting device memory.
    _DUMMY_BYTES = 16384
    return f"""\
#!/usr/bin/env python3
import os, ctypes, sys, time
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
import torch
import torch_npu

lib = ctypes.CDLL("{so_path}")
lib.init()

dummy = torch.zeros({_DUMMY_BYTES}, dtype=torch.uint8, device="npu:0")
lib.call(ctypes.c_void_p(dummy.data_ptr()), ctypes.c_void_p(0))
torch.npu.synchronize()
# The camodel simulator flushes the recorded instruction stream (instr.bin)
# asynchronously after the kernel completes; os._exit() (needed to avoid a
# torch_npu shutdown hang) otherwise kills the process mid-flush, truncating
# instr.bin so the VF body is only partially recorded. Give the flush time.
time.sleep({flush_wait})
os._exit(0)
"""


def _run_cannsim(launcher_path: Path, output_dir: Path, soc: str) -> Path | None:
    """Run cannsim record then cannsim report to get tracing JSON."""
    wrapper = output_dir / "_run.sh"
    wrapper.write_text(f"#!/bin/bash\nexec {sys.executable} {launcher_path}\n")
    wrapper.chmod(0o755)

    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    python_path = env.get("PYTHONPATH", "")
    if _CANN_PYTHON_SITE not in python_path:
        env["PYTHONPATH"] = f"{_CANN_PYTHON_SITE}:{python_path}"

    # Ensure simulator/camodel libs are on LD_LIBRARY_PATH
    ascend_home = Path(_CANN_ROOT)
    arch = "aarch64-linux"
    sim_dirs = [
        str(ascend_home / arch / "lib64"),
        str(ascend_home / arch / "simulator" / "dav_3510" / "lib"),
        str(ascend_home / arch / "simulator" / "dav_3510" / "camodel"),
    ]
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for d in sim_dirs:
        if d not in ld_path:
            ld_path = f"{d}:{ld_path}"
    env["LD_LIBRARY_PATH"] = ld_path

    # Step 1: cannsim record
    cmd = [_CANNSIM, "record", "-o", str(output_dir), "-s", soc, "-n", "0", str(wrapper)]
    print(f"  [CMD] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, capture_output=True, text=True)
    dirs = sorted(output_dir.glob("cannsim_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        return None
    cannsim_dir = dirs[-1]

    # Step 2: generate tracing JSON via trace_tools (bypasses cannprof which
    # hangs on ARM). trace_tools needs the full per-instruction log restored
    # from instr.bin (the raw log_ca/ only has a summary line).
    if _CANN_PYTHON_SITE not in sys.path:
        sys.path.insert(0, _CANN_PYTHON_SITE)
    from cannsim.core.report.report_path import _prepare_log_ca
    from cannsim.trace_tools.rts_sequence import _gen_model_trace

    _prepare_log_ca(cannsim_dir)  # restores instr.bin -> tmp_log_ca/
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)
    _gen_model_trace(
        case_name="trace",
        run_mode=str(report_dir) + "/",
        log_ca_path=str(cannsim_dir),
        target_block_dim=(0,),
    )

    return cannsim_dir


# ---------------------------------------------------------------------------
# 5. Tracing report parser — extract VF cycles from trace JSON
# ---------------------------------------------------------------------------


def _parse_tracing_report(cannsim_dir: Path) -> int | None:
    """Extract VF body cycle count from the cannsim tracing report JSON.

    trace_tools outputs timestamps in cycles directly (no scaling needed).

    Strategies (tried in order):

    1. RVEC* instruction span — pure vector-pipe execution latency.  This is
       the primary strategy because the measurement kernel only contains the
       VF call (plus a pointer declaration that generates no instructions),
       so the vector-pipe events *are* the VF body.  The kernel prologue /
       UB preload (``DC_PRELOAD_XN_IMM``, etc.) runs on the SCALAR pipe in
       parallel — including its start timestamp would overcount.

    2. PUSHQ VF event — when the trace generator wraps the VF call in a
       named PUSHQ event (``VF_SIMT`` / ``VF_SIMD``), its duration captures
       the full dispatch+execute+retire window.  This is most useful for
       SimtVF where ``asc_vf_call`` dispatch lives on the PUSHQ/SCALAR pipe
       ahead of the first RVEC instruction.

    3. All-events span — last-resort fallback that covers every pipe.
    """
    report_dir = cannsim_dir.parent / "report"
    trace_files = list(report_dir.rglob("trace_core*.json"))
    if not trace_files:
        trace_files = list(report_dir.rglob("*tracing_report*.json"))
    if not trace_files:
        return None

    try:
        data = json.loads(trace_files[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # trace_tools outputs a list directly; cannprof wraps in {"traceEvents": [...]}
    if isinstance(data, list):
        events = data
    else:
        events = data.get("traceEvents", [])
    if not events:
        return None

    # Build process name map
    processes: dict[int, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            processes[e["pid"]] = e["args"]["name"]

    # All execution events
    exec_events = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    if not exec_events:
        return None

    # ------------------------------------------------------------------
    # Strategy 1: RVEC / VECTOR pipe span (pure vector execution)
    #
    # The measurement kernel body is *just* the VF call — there is no DMA,
    # no pipe barriers, no other compute.  Long-running SCALAR instructions
    # (e.g. ``DC_PRELOAD_XN_IMM`` for UB init) belong to the kernel
    # prologue and run *in parallel* with the vector pipe, so their start
    # timestamp should not be included in the VF latency.
    #
    # A SimtVF / SimdVF kernel always contains vector instructions.  If the
    # trace has zero RVEC / VECTOR events the recording was truncated
    # (cannsim sometimes misses instructions for very short kernels).
    # Return ``None`` so the caller can retry.
    # ------------------------------------------------------------------
    rvec_pids = [pid for pid, name in processes.items() if "RVEC" in name or "VECTOR" in name.upper() or "VEC" in name]
    rvec_events = [e for e in exec_events if e.get("pid") in rvec_pids]
    if not rvec_events:
        return None  # trace truncated — signal retry

    rvec_first = min(e["ts"] for e in rvec_events)
    rvec_last = max(e["ts"] + e["dur"] for e in rvec_events)
    rvec_span = max(1, rvec_last - rvec_first)

    # ------------------------------------------------------------------
    # Strategy 2: PUSHQ VF dispatch event (cross-check / fallback)
    #
    # The trace generator may wrap the VF call in a PUSHQ event named
    # "VF", "VF_SIMT", or "VF_SIMD".  When available, its ``dur`` is a
    # reasonable fallback.  But RVEC is preferred because PUSHQ includes
    # dispatch padding that varies between SimtVF / SimdVF.
    # ------------------------------------------------------------------
    pushq_pids = [pid for pid, name in processes.items() if "PUSHQ" in name]
    vf_dispatch_events = [e for e in exec_events if e.get("pid") in pushq_pids and ("VF" in e.get("name", ""))]
    pushq_dur = None
    if vf_dispatch_events:
        pushq_dur = max(e.get("dur", 0) for e in vf_dispatch_events)
        if pushq_dur <= 0:
            pushq_dur = None

    # When both RVEC and PUSHQ are available, RVEC is the pure compute
    # latency.  Only fall back to PUSHQ when RVEC is suspiciously small
    # (trace may be partial even though some events exist).
    if pushq_dur is not None and rvec_span < pushq_dur * 0.3:
        return pushq_dur

    return rvec_span


def _parse_instr_log(cannsim_dir: Path) -> int | None:
    """Parse VF cycles from tracing report."""
    return _parse_tracing_report(cannsim_dir)


# ---------------------------------------------------------------------------
# 6. Python source VF block finder (lightweight — just line numbers)
# ---------------------------------------------------------------------------


def _find_python_vf_blocks(source: str, scope_func: str | None = None) -> list[dict]:
    """Find SimtVF/SimdVF blocks in Python source, in source order.

    If scope_func is given, only blocks lexically inside the top-level def of
    that name are returned — so VF blocks belonging to other kernel factories
    in the same file don't get mixed into the mapping.
    """
    tree = ast.parse(source)
    if scope_func is not None:
        scoped = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == scope_func]
        if scoped:
            tree = scoped[0]
    blocks: list[dict] = []

    class Visitor(ast.NodeVisitor):
        def visit_With(self, node: ast.With):
            for item in node.items:
                ctx = item.context_expr
                if not isinstance(ctx, ast.Call):
                    continue
                name = _ast_attr_name(ctx.func)
                if name is None:
                    continue
                if "SimtVF" in name:
                    kind = "simt"
                elif "SimdVF" in name:
                    kind = "simd"
                else:
                    continue
                existing_latency = None
                for kw in ctx.keywords:
                    if kw.arg == "latency":
                        with contextlib.suppress(ValueError, SyntaxError):
                            existing_latency = ast.literal_eval(kw.value)
                blocks.append(
                    {
                        "kind": kind,
                        "lineno": node.lineno,
                        "existing_latency": existing_latency,
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return blocks


def _contains_call(node) -> bool:
    """Return True if the AST node contains a Call anywhere in its subtree."""
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _ast_attr_name(node) -> str | None:
    if isinstance(node, ast.Attribute):
        obj = _ast_attr_name(node.value)
        return f"{obj}.{node.attr}" if obj else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


# ---------------------------------------------------------------------------
# 7. Source rewriting
# ---------------------------------------------------------------------------


def _inject_latency(source_text: str, vf_blocks: list[dict]) -> str:
    """Insert/update latency= for each VF block."""
    lines = source_text.splitlines(keepends=True)

    for vf in sorted(vf_blocks, key=lambda b: b["lineno"], reverse=True):
        latency = vf.get("measured_latency")
        if latency is None:
            continue

        line_idx = vf["lineno"] - 1
        line = lines[line_idx]

        if "latency=" in line:
            line = re.sub(r"latency=\d+", f"latency={latency}", line)
        else:
            try:
                end = line.rindex("):")
            except ValueError:
                end = line.rindex(")")
            paren_open = line.rindex("(", 0, end)
            inner = line[paren_open + 1 : end].strip()
            if inner:
                line = line[:end] + f", latency={latency}" + line[end:]
            else:
                line = line[:end] + f"latency={latency}" + line[end:]

        lines[line_idx] = line

    return "".join(lines)


# ---------------------------------------------------------------------------
# 8. Main driver
# ---------------------------------------------------------------------------


def measure_kernel(
    kernel_path: str,
    *,
    call_expr: str | None = None,
    out_idx: int = -1,
    soc: str = "Ascend950",
    dry_run: bool = False,
    work_dir: str | None = None,
) -> None:
    print(f"Compiling kernel from {kernel_path} ...")
    asc_source, scope_func = _compile_kernel(kernel_path, call_expr, out_idx)
    print(f"  got .asc ({len(asc_source)} chars)")

    includes = _extract_includes(asc_source)
    vf_funcs = _extract_vf_functions(asc_source)

    if not vf_funcs:
        print("No VF functions found in generated .asc")
        return

    print(f"  found {len(vf_funcs)} VF function(s):")
    for vf in vf_funcs:
        label = "SimtVF" if vf["kind"] == "simt" else "SimdVF"
        extra = f", threads={vf['threads']}" if vf["kind"] == "simt" else ""
        print(f"    {vf['name']} ({label}{extra}, scalars={vf['scalar_params']})")

    # Find Python VF blocks for mapping, scoped to the compiled kernel so
    # blocks from other kernel factories in the same file are ignored.
    py_source = Path(kernel_path).read_text()
    py_blocks = _find_python_vf_blocks(py_source, scope_func)

    # Map vf_func → python block. tl.vf_source_index is module-monotonic, not a
    # per-file block position, so we can't index py_blocks with it directly.
    # Instead, within each kind, order the generated funcs by source_index (the
    # emission order in the .asc can differ from source order under pipelining)
    # and zip against the source-order python blocks of that kind.
    simt_blocks = [b for b in py_blocks if b["kind"] == "simt"]
    simd_blocks = [b for b in py_blocks if b["kind"] == "simd"]

    func_to_block: dict[str, dict] = {}
    for kind, blocks in (("simt", simt_blocks), ("simd", simd_blocks)):
        funcs = [v for v in vf_funcs if v["kind"] == kind]
        # Sort by source_index when present (falls back to .asc order, i.e. pos).
        funcs.sort(key=lambda v: v.get("source_index", v["pos"]))
        if len(funcs) != len(blocks):
            print(
                f"  WARNING: {kind} VF func count ({len(funcs)}) != Python block count"
                f" ({len(blocks)}). Some latencies may not be mapped correctly."
                f" Try re-running with --call to scope the correct function."
            )
        for func, block in zip(funcs, blocks):
            func_to_block[func["name"]] = block

    # Measure each VF function
    work_base = Path(work_dir) if work_dir else (_REPO_ROOT / ".cannsim_work")
    if work_base.exists():
        subprocess.run(["pkill", "-9", "-f", "cannprof"], capture_output=True)
        subprocess.run(["rm", "-rf", str(work_base)], capture_output=True)
    work_base.mkdir(exist_ok=True)

    print()
    for i, vf in enumerate(vf_funcs):
        label = "SimtVF" if vf["kind"] == "simt" else "SimdVF"
        print(f"Measuring [{i + 1}/{len(vf_funcs)}] {vf['name']} ({label}) ...", flush=True)

        wdir = Path(tempfile.mkdtemp(prefix=f"{vf['name']}_", dir=str(work_base)))
        wdir.chmod(0o755)

        try:
            measure_asc = _gen_measure_asc(includes, vf)
            asc_path = wdir / "measure.asc"
            asc_path.write_text(measure_asc)
            print(f"  [ASC] {asc_path}")

            so_path = wdir / "measure.so"
            _compile_asc(asc_path, so_path)
            print(f"  compiled: {so_path.name}")

            launcher_src = _gen_launcher(so_path)
            launcher_path = wdir / "_launcher.py"
            launcher_path.write_text(launcher_src)
            print(f"  [LAUNCHER] {launcher_path}")

            # cannsim record is non-deterministic: it occasionally truncates the
            # instruction stream mid-scalar-prologue, so the VF body never
            # executes and the trace has only SCALAR events (parse -> None).
            # Retry a few times; a correct run produces vector-pipe events.
            cycles = None
            for attempt in range(_RECORD_MAX_ATTEMPTS):
                if attempt > 0:
                    import time as _time

                    _time.sleep(_RECORD_RETRY_DELAY)
                cannsim_dir = _run_cannsim(launcher_path, wdir, soc)
                if cannsim_dir is None:
                    print(f"  [retry {attempt + 1}/{_RECORD_MAX_ATTEMPTS}] no cannsim output")
                    continue
                cycles = _parse_instr_log(cannsim_dir)
                if cycles is not None:
                    break
                print(f"  [retry {attempt + 1}/{_RECORD_MAX_ATTEMPTS}] record truncated (no VF body), re-recording")

            # If all standard retries fail, try a multi-call measurement:
            # call the VF N times in a loop so the vector region is long
            # enough for the camodel to capture reliably, then divide.
            if cycles is None:
                _MULTI_CALL_N = 16
                print(f"  [fallback] standard retries exhausted, trying {_MULTI_CALL_N}x loop ...")
                multi_asc = _gen_measure_asc_multi(includes, vf, _MULTI_CALL_N)
                multi_asc_path = wdir / "measure_multi.asc"
                multi_asc_path.write_text(multi_asc)
                multi_so_path = wdir / "measure_multi.so"
                _compile_asc(multi_asc_path, multi_so_path)
                multi_launcher_src = _gen_launcher(multi_so_path)
                multi_launcher_path = wdir / "_launcher_multi.py"
                multi_launcher_path.write_text(multi_launcher_src)
                for attempt in range(_RECORD_MAX_ATTEMPTS):
                    if attempt > 0:
                        import time as _time

                        _time.sleep(_RECORD_RETRY_DELAY)
                    cannsim_dir = _run_cannsim(multi_launcher_path, wdir, soc)
                    if cannsim_dir is None:
                        continue
                    total = _parse_instr_log(cannsim_dir)
                    if total is not None:
                        cycles = max(1, total // _MULTI_CALL_N)
                        break

            if cycles is None:
                print(f"  SKIP (could not parse cycles after {_RECORD_MAX_ATTEMPTS} standard + {_RECORD_MAX_ATTEMPTS} multi-call attempts)")
                continue

            print(f"  -> {cycles} cycles")

            if vf["name"] in func_to_block:
                func_to_block[vf["name"]]["measured_latency"] = cycles

        except subprocess.CalledProcessError as exc:
            print(f"  FAIL: {exc}")
            if exc.stderr:
                for line in exc.stderr.strip().splitlines()[:5]:
                    print(f"    {line}")

    # Inject latencies
    measured = [b for b in py_blocks if b.get("measured_latency")]
    if not measured:
        print("\nNo latencies measured.")
        return

    print()
    new_source = _inject_latency(py_source, py_blocks)

    if dry_run:
        print("=== Results (--dry-run, file not modified) ===")
        old_lines = py_source.splitlines()
        new_lines = new_source.splitlines()
        for b in measured:
            li = b["lineno"] - 1
            if li < len(old_lines) and li < len(new_lines) and old_lines[li] != new_lines[li]:
                print(f"  line {b['lineno']}:")
                print(f"    - {old_lines[li].strip()}")
                print(f"    + {new_lines[li].strip()}")
    else:
        old_lines = py_source.splitlines()
        new_lines = new_source.splitlines()
        changed = 0
        for b in measured:
            li = b["lineno"] - 1
            if li < len(old_lines) and li < len(new_lines) and old_lines[li] != new_lines[li]:
                print(f"  line {b['lineno']}:")
                print(f"    - {old_lines[li].strip()}")
                print(f"    + {new_lines[li].strip()}")
                changed += 1
        if changed:
            Path(kernel_path).write_text(new_source)
            print(f"\nWrote {changed} line(s) back to {kernel_path}")
        else:
            print("No changes to write.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Measure VF latency via compile + cannsim")
    parser.add_argument("kernel", help="Path to kernel .py file")
    parser.add_argument("--call", default=None, help="Expression returning @T.prim_func (e.g. 'flash_attention(D=128)')")
    parser.add_argument("--out-idx", type=int, default=-1)
    parser.add_argument("--soc", default="Ascend950")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    measure_kernel(
        args.kernel,
        call_expr=args.call,
        out_idx=args.out_idx,
        soc=args.soc,
        dry_run=args.dry_run,
        work_dir=args.work_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
