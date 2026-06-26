from __future__ import annotations
import ctypes
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
from typing import Any
from pathlib import Path

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.transform import PassConfigKey
from tilelang.contrib.nvcc import (
    format_target_code_for_gencode,
    get_cuda_library_dirs,
    get_nvcc_compiler,
    get_target_arch_and_code,
)
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.contrib.hip_resource_info import filter_and_record

from .utils import is_ascend_target, is_cpu_target, is_cuda_target, is_hip_target, is_pto_target

logger = logging.getLogger(__name__)


class LibraryGenerator:
    srcpath: str | None = None
    libpath: str | None = None
    lib_code: str | None = None
    pto_kernel_source: str | None = None
    pto_kernel_name: str | None = None
    pass_configs: dict[str, Any] | None = None
    compile_flags: list[str] | None = None

    def __init__(self, target: Target, verbose: bool = False):
        self.target = target
        self.verbose = verbose

    def assign_pass_configs(self, pass_configs: dict[str, Any] | None = None):
        self.pass_configs = pass_configs

    def assign_compile_flags(self, compile_flags: list[str] | None = None):
        if compile_flags is None:
            compile_flags = []
        self.compile_flags = compile_flags

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    def update_pto_kernel(self, pto_kernel_source: str, pto_kernel_name: str):
        self.pto_kernel_source = pto_kernel_source
        self.pto_kernel_name = pto_kernel_name

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        target = self.target
        verbose = self.verbose
        if is_cuda_target(target):
            from tilelang.env import CUTLASS_INCLUDE_DIR

            _lib_ext = ".dll" if sys.platform == "win32" else ".so"
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cu", _lib_ext)

            enable_fast_math = self.pass_configs.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)

            ptxas_usage_level = self.pass_configs.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
            if ptxas_usage_level is not None:
                ptxas_usage_level = int(ptxas_usage_level)
            verbose_ptxas_output = self.pass_configs.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False)
            cuda_library_flags = [f"-L{lib_dir}" for lib_dir in get_cuda_library_dirs()]
            target_arch, target_code = get_target_arch_and_code(target)
            arch_flags = [f"-arch=sm_{target_arch}"]
            gencode_code = format_target_code_for_gencode(target_code)
            if gencode_code is not None:
                arch_flags = ["-gencode", f"arch=compute_{target_arch},code={gencode_code}"]

            command = [
                get_nvcc_compiler(),
                # tl_templates/cuda/reduce.h uses explicit lambda template
                # parameters (`[&]<typename T>(T) { ... }`) which are a C++20
                # feature.
                "-std=c++20",
                "-w",
                "-Xcudafe",
                "--diag_suppress=177",
                "-lineinfo",
                "--shared",
                src.name,
                *cuda_library_flags,
                "-lcuda",
                *arch_flags,
            ]
            if sys.platform == "win32":
                # /Zc:__cplusplus forces MSVC to report the actual C++ standard
                # via __cplusplus. Without it cuda.h's `alignas(128)` on
                # CUtensorMap is dropped (it is gated on
                # ``__cplusplus >= 201103L``), so NVCC emits a kernel param
                # with .align 8 and cuLaunchKernel later fails with
                # CUDA_ERROR_MISALIGNED_ADDRESS.
                command += ["-Xcompiler", "/Zc:preprocessor /Zc:__cplusplus"]
            else:
                command += ["--compiler-options", "-fPIC"]
            if enable_fast_math:
                command += ["--use_fast_math"]
            if ptxas_usage_level is not None:
                command += [f"--ptxas-options=--register-usage-level={ptxas_usage_level}"]
            if verbose_ptxas_output:
                command += ["--ptxas-options=--verbose"]
            command += [
                "-I" + CUTLASS_INCLUDE_DIR,
            ]

        elif is_hip_target(target):
            from tilelang.rocm.target import target_get_mcpu

            from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_HIP_SAVE_TEMP_FILES

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")
            rocm_path = find_rocm_path()
            arch = target_get_mcpu(target) or get_rocm_arch(rocm_path)
            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                f"--offload-arch={arch}",
                "--shared",
                src.name,
                "-Rpass-analysis=kernel-resource-usage",
            ]
            command += [
                "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
            ]
            if TILELANG_HIP_SAVE_TEMP_FILES != "0":
                command += ["--save-temps", "-g"]
        elif is_pto_target(target):
            self.compile_pto_lib()
            return
        elif is_ascend_target(target):
            from tilelang.contrib.bisheng import find_bisheng_path

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".asc", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".asc", ".so")

            npu_arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
            command = [
                find_bisheng_path(),
                "-fPIC",
                "--shared",
                src.name,
            ]
            if npu_arch:
                command += [f"--npu-arch={npu_arch}"]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        else:
            raise ValueError(f"Unsupported target: {target}")

        command += [
            "-I" + TILELANG_TEMPLATE_PATH,
        ]

        if self.compile_flags:
            command += [item for flag in self.compile_flags for item in flag.split() if item not in command]

        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        if sys.platform == "win32":
            src.close()

        # On Windows, two concerns matter for parallel autotune:
        # 1. nvcc needs MSVC's host compiler env (cl.exe, INCLUDE, LIB).
        # 2. Concurrent subprocesses sharing the parent's console handle can
        #    deadlock when their output interleaves with tqdm progress bars.
        # Pipe stdio + isolate stdin to make the launch self-contained.
        run_kwargs: dict[str, Any] = {"timeout": timeout}
        if sys.platform == "win32":
            from tilelang.contrib.nvcc import get_nvcc_subprocess_env

            run_kwargs.update(
                env=get_nvcc_subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        if is_hip_target(target):
            run_kwargs.setdefault("stdout", subprocess.PIPE)
            run_kwargs.setdefault("stderr", subprocess.STDOUT)

        try:
            if verbose:
                print(f"compile_lib compilation command: {' '.join(command)}")
            ret = subprocess.run(command, **run_kwargs)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            captured = ret.stdout.decode("utf-8", errors="replace") if ret.stdout else ""
            raise RuntimeError(f"Compilation Failed! {command}\n{captured}\n{self.lib_code}")

        if is_hip_target(target) and ret.stdout is not None:
            captured = filter_and_record(ret.stdout.decode("utf-8", errors="replace"))
            if verbose and captured.strip():
                print(captured)

        self.srcpath = src.name
        self.libpath = libpath

    @staticmethod
    def _compile_ptodsl_source_to_pto(
        ptodsl_source: str,
        kernel_name: str,
        src_path: str | Path,
        out_path: str | Path,
    ) -> None:
        src_path = Path(src_path)
        out_path = Path(out_path)
        src_path.write_text(ptodsl_source, encoding="utf-8")

        script = textwrap.dedent(
            """
            import importlib.util
            import pathlib
            import sys
            import traceback

            src_path = pathlib.Path(sys.argv[1])
            kernel_name = sys.argv[2]
            out_path = pathlib.Path(sys.argv[3])
            module_name = "_tilelang_ptodsl_compile"

            try:
                spec = importlib.util.spec_from_file_location(module_name, src_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
            except Exception:
                traceback.print_exc()
                sys.exit(2)

            try:
                kernel = getattr(module, kernel_name)
                compiled = kernel.compile()
                out_path.write_text(compiled.mlir_text(), encoding="utf-8")
            except Exception:
                traceback.print_exc()
                sys.exit(3)
            """
        )

        python_bin = sys.executable
        result = subprocess.run(
            [python_bin, "-c", script, str(src_path), kernel_name, str(out_path)],
            text=True,
            capture_output=True,
        )
        if result.returncode == 2:
            raise RuntimeError(
                f"PTODSL compiler frontend is unavailable in the current environment.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                "PTODSL compile-only lowering failed.\n"
                f"Command: {python_bin} -c <ptodsl-compile-script> {src_path} {kernel_name} {out_path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def compile_pto_lib(self):
        if self.pto_kernel_source is None:
            raise RuntimeError("PTO compilation requires a PTODSL kernel source.")
        if self.pto_kernel_name is None:
            raise RuntimeError("PTO compilation requires a kernel name.")
        if self.lib_code is None:
            raise RuntimeError("PTO compilation requires a host launch source.")

        out_dir = tempfile.mkdtemp(prefix="tilelang_pto_")
        ptodsl_path = os.path.join(out_dir, "kernel.ptodsl.py")
        pto_path = os.path.join(out_dir, "kernel.pto")
        fatobj_path = os.path.join(out_dir, "kernel.fatobj.o")
        launch_cpp = os.path.join(out_dir, "launch.cpp")
        launch_obj = os.path.join(out_dir, "launch.o")
        libpath = os.path.join(out_dir, "lib_kernel.so")

        self._compile_ptodsl_source_to_pto(self.pto_kernel_source, self.pto_kernel_name, ptodsl_path, pto_path)
        with open(launch_cpp, "w", encoding="utf-8") as file:
            file.write(self.lib_code)

        pto_arch = os.environ.get("PTO_ARCH", "a5").strip().lower()
        if pto_arch != "a5":
            raise RuntimeError(f"Unsupported PTO_ARCH for PTO JIT: {pto_arch}")
        pto_flags = shlex.split(os.environ.get("PTO_FLAGS", ""))
        if not any(flag == "--pto-backend" or flag.startswith("--pto-backend=") for flag in pto_flags):
            pto_flags.append("--pto-backend=vpto")

        pto_cmd = ["ptoas", f"--pto-arch={pto_arch}", *pto_flags, pto_path, "-o", fatobj_path]
        if self.verbose:
            print(f"PTO compile command: {' '.join(pto_cmd)}")
        result = subprocess.run(pto_cmd, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"PTO lowering failed.\nCommand: {' '.join(pto_cmd)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")

        from tilelang.contrib.bisheng import find_bisheng_path

        bisheng_bin = find_bisheng_path()
        aicore_arch = os.environ.get("PTO_AICORE_ARCH", "dav-c310")

        compile_cmd = [
            bisheng_bin,
            "-c",
            "-fPIC",
            "-O2",
            "-xcce",
            "-Xhost-start",
            "-Xhost-end",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=true",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            f"--cce-aicore-arch={aicore_arch}",
            "-std=c++17",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            launch_cpp,
            "-o",
            launch_obj,
        ]
        link_cmd = [
            bisheng_bin,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            "-o",
            libpath,
            fatobj_path,
            launch_obj,
            "-Wl,--no-as-needed",
        ]

        for cmd in (compile_cmd, link_cmd):
            if self.verbose:
                print(f"bisheng command: {' '.join(cmd)}")
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"PTO host link failed.\nCommand: {' '.join(cmd)}\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}")

        self.srcpath = launch_cpp
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
