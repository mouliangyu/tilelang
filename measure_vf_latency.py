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

# ---------------------------------------------------------------------------
# 1. Compile kernel and get .asc source
# ---------------------------------------------------------------------------


def _compile_kernel(kernel_path: str, call_expr: str | None, out_idx: int) -> str:
    """Import kernel module, get PrimFunc, compile, return .asc source."""
    module_path = Path(kernel_path).resolve()
    spec = importlib.util.spec_from_file_location("_kernel_mod", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_kernel_mod"] = mod

    # Execute module but collect __main__ locals for eval context
    main_locals = _exec_module_with_main_locals(spec, mod, module_path)

    if call_expr:
        ns = {**vars(mod), **main_locals}
        func = eval(call_expr, ns)
    else:
        func = _auto_detect_func(mod, kernel_path, main_locals)

    if func is None:
        sys.exit("Could not find a PrimFunc. Use --call to specify.")

    import tilelang

    try:
        kernel = tilelang.compile(func, out_idx=out_idx)
    except Exception:
        kernel = tilelang.compile(func, out_idx=out_idx, pass_configs={"tl.enable_auto_schedule": False})
    return kernel.get_kernel_source()


def _exec_module_with_main_locals(spec, mod, module_path: Path) -> dict:
    """Execute the module and extract locals from its __main__ block."""
    # First, exec the module normally (with __main__ guard)
    import unittest.mock

    with unittest.mock.patch.object(mod, "__name__", "_kernel_mod"), contextlib.suppress(SystemExit):
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
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        val = eval(compile(ast.Expression(body=stmt.value), "<main>", "eval"), {**vars(mod), **main_locals})
                        main_locals[target.id] = val
                    except Exception:
                        pass
        break

    return main_locals


def _auto_detect_func(mod, kernel_path: str, main_locals: dict = None):
    """Auto-detect PrimFunc by inspecting the module.

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
            return obj

    # Strategy 2: parse __main__ block to find the compile call
    source = Path(kernel_path).read_text()
    call_expr = _parse_main_block(source)
    if call_expr:
        try:
            ns = {**vars(mod), **(main_locals or {})}
            func = eval(call_expr, ns)
            if isinstance(func, tvm.tir.PrimFunc):
                print(f"  auto-detected from __main__: {call_expr}")
                return func
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
                    return result
            except (TypeError, Exception):
                continue

    return None


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


def _gen_launcher(so_path: Path) -> str:
    """Generate Python launcher that calls the .so via ctypes + torch_npu."""
    return f"""\
#!/usr/bin/env python3
import os, ctypes, sys
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
import torch
import torch_npu

lib = ctypes.CDLL("{so_path}")
lib.init()

dummy = torch.zeros(1, dtype=torch.uint8, device="npu:0")
lib.call(ctypes.c_void_p(dummy.data_ptr()), ctypes.c_void_p(0))
torch.npu.synchronize()
sys.exit(0)
"""


def _run_cannsim(launcher_path: Path, output_dir: Path, soc: str) -> Path | None:
    """Run cannsim record with the launcher script."""
    wrapper = output_dir / "_run.sh"
    wrapper.write_text(f"#!/bin/bash\nexec {sys.executable} {launcher_path}\n")
    wrapper.chmod(0o755)

    cmd = [_CANNSIM, "record", "-o", str(output_dir), "-s", soc, "-g", "-n", "0", str(wrapper)]
    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    python_path = env.get("PYTHONPATH", "")
    if _CANN_PYTHON_SITE not in python_path:
        env["PYTHONPATH"] = f"{_CANN_PYTHON_SITE}:{python_path}"

    subprocess.run(cmd, env=env, capture_output=True, text=True)
    dirs = sorted(output_dir.glob("cannsim_*"), key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


# ---------------------------------------------------------------------------
# 5. Instruction log parser
# ---------------------------------------------------------------------------


def _parse_instr_log(cannsim_dir: Path) -> int | None:
    """Parse veccore0 instr_log for cycle count."""
    log_path = cannsim_dir / "log_ca" / "core0.veccore0.instr_log.dump"
    if not log_path.exists():
        return None

    text = log_path.read_text()
    pattern = re.compile(r"start:\s*(\d+),\s*tick:\s*(\d+),.*?blkDim:\s*(\d+)")

    last_match = None
    for m in pattern.finditer(text):
        if int(m.group(3)) == 1:
            last_match = m

    if last_match is None:
        matches = list(pattern.finditer(text))
        if matches:
            last_match = matches[-1]

    if last_match is None:
        return None

    return int(last_match.group(2)) - int(last_match.group(1))


# ---------------------------------------------------------------------------
# 6. Python source VF block finder (lightweight — just line numbers)
# ---------------------------------------------------------------------------


def _find_python_vf_blocks(source: str) -> list[dict]:
    """Find SimtVF/SimdVF blocks in Python source. Returns ordered list."""
    tree = ast.parse(source)
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
    asc_source = _compile_kernel(kernel_path, call_expr, out_idx)
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

    # Find Python VF blocks for mapping
    py_source = Path(kernel_path).read_text()
    py_blocks = _find_python_vf_blocks(py_source)

    # Map: separate simt and simd, match by index
    simt_funcs = [v for v in vf_funcs if v["kind"] == "simt"]
    simd_funcs = [v for v in vf_funcs if v["kind"] == "simd"]
    simt_blocks = [b for b in py_blocks if b["kind"] == "simt"]
    simd_blocks = [b for b in py_blocks if b["kind"] == "simd"]

    # Build mapping: vf_func → python block
    func_to_block: dict[str, dict] = {}
    for i, vf in enumerate(simt_funcs):
        if i < len(simt_blocks):
            func_to_block[vf["name"]] = simt_blocks[i]
    for i, vf in enumerate(simd_funcs):
        if i < len(simd_blocks):
            func_to_block[vf["name"]] = simd_blocks[i]

    # Measure each VF function
    work_base = Path(work_dir) if work_dir else (_REPO_ROOT / ".cannsim_work")
    work_base.mkdir(exist_ok=True)

    print()
    for i, vf in enumerate(vf_funcs):
        label = "SimtVF" if vf["kind"] == "simt" else "SimdVF"
        print(f"Measuring [{i + 1}/{len(vf_funcs)}] {vf['name']} ({label}) ...", flush=True)

        wdir = Path(tempfile.mkdtemp(prefix=f"{vf['name']}_", dir=str(work_base)))

        try:
            measure_asc = _gen_measure_asc(includes, vf)
            asc_path = wdir / "measure.asc"
            asc_path.write_text(measure_asc)

            so_path = wdir / "measure.so"
            _compile_asc(asc_path, so_path)
            print(f"  compiled: {so_path.name}")

            launcher_src = _gen_launcher(so_path)
            launcher_path = wdir / "_launcher.py"
            launcher_path.write_text(launcher_src)

            cannsim_dir = _run_cannsim(launcher_path, wdir, soc)
            if cannsim_dir is None:
                print("  SKIP (no cannsim output)")
                continue

            cycles = _parse_instr_log(cannsim_dir)
            if cycles is None:
                print("  SKIP (no instr_log)")
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
