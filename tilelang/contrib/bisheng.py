# pylint: disable=invalid-name
"""Utility to invoke bisheng compiler for Ascend NPU targets."""

from __future__ import absolute_import as _abs

import os
import shutil
import subprocess

from tvm.base import py_str
from tvm.contrib import utils


def find_bisheng_path() -> str:
    """Find the bisheng compiler binary.

    Searches ``BISHENG_HOME/bin/bisheng``, then ``PATH``.

    Returns
    -------
    path : str
        Full path to the ``bisheng`` executable.

    Raises
    ------
    RuntimeError
        If the compiler cannot be found.
    """
    bisheng_home = os.environ.get("BISHENG_HOME", "")
    if bisheng_home:
        candidate = os.path.join(bisheng_home, "bin", "bisheng")
        if os.path.isfile(candidate):
            return candidate

    path = shutil.which("bisheng")
    if path is not None:
        return path

    raise RuntimeError(
        "Cannot find the bisheng compiler.Please install it and make sure it is in PATH, or set the BISHENG_HOME environment variable."
    )


def compile_ascend(
    code,
    target_format="o",
    npu_arch=None,
    options=None,
    path_target=None,
    verbose=False,
):
    """Compile Ascend source code with the bisheng compiler.

    Parameters
    ----------
    code : str
        The Ascend kernel source code.

    target_format : str
        Output format - ``"o"`` for object file (default), ``"so"`` for shared
        library.

    npu_arch : str, optional
        NPU architecture string passed via ``--npu-arch=<arch>`` (e.g.
        ``"dav-3510"``).  If *None*, ``ASCEND_NPU_ARCH`` env-var is used.

    options : str or list of str, optional
        Extra flags forwarded to bisheng.

    path_target : str, optional
        Explicit output path.  A temporary file is used when not given.

    verbose : bool
        Print compiler output.

    Returns
    -------
    data : bytearray
        Contents of the compiled output file.
    """
    if npu_arch is None:
        npu_arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")

    temp = utils.tempdir()
    # Use .asc suffix so bisheng auto-detects the language without ``-x asc``
    temp_code = temp.relpath("tl_kernel.asc")
    suffix = "so" if target_format == "so" else "o"
    temp_target = temp.relpath(f"tl_kernel.{suffix}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target

    cmd = [find_bisheng_path()]
    cmd += ["-fPIC", "-std=c++20"]
    # Avoid using mmap to write linker output, thus more friendly for distributed FS
    cmd += ["-Wl,--no-mmap-output-file"]
    if target_format == "so":
        cmd += ["-shared"]
    else:
        cmd += ["-c"]
    if npu_arch:
        cmd += [f"--npu-arch={npu_arch}"]
    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        msg += f"\nCommand: {' '.join(cmd)}\n"
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data
