"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal
from tvm.target import Target as TVMTarget
from tvm.tirx import PrimFunc
from tilelang.jit import JITKernel
from tilelang import env
from tilelang.jit.adapter.cutedsl.kernel_cache import CuTeDSLKernelCache
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache
from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache
from tilelang.ascend.kernel_cache import AscendCythonKernelCache, AscendTVMFFIKernelCache

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

TargetLike = str | dict[str, object] | TVMTarget

# Create a map of singleton instance of KernelCaches
_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": TVMFFIKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "torch": TorchKernelCache(),
}

_ascend_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": AscendTVMFFIKernelCache(),
    "cython": AscendCythonKernelCache(),
}


def _resolve_cache_dispatch(
    target: TargetLike | None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None,
    verbose: bool | None,
):
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    from tilelang.backend.target import determine_target as _determine_target
    from tilelang.backend.execution_backend import resolve_execution_backend, allowed_backends_for_target

    norm_target = _determine_target(target, return_object=True)
    requested_backend = execution_backend
    resolved_backend = resolve_execution_backend(requested_backend, norm_target)
    if verbose:
        allowed_now = allowed_backends_for_target(norm_target, include_unavailable=False)
        if requested_backend in (None, "auto") or requested_backend != resolved_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                resolved_backend,
                requested_backend,
                norm_target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    dispatch_map = (
        _ascend_dispatch_map
        if ("ascend" in norm_target.keys or norm_target.kind.name == "ascend") and "pto" not in norm_target.keys
        else _dispatch_map
    )
    if resolved_backend not in dispatch_map:
        raise ValueError(f'Cannot find support for execution backend "{resolved_backend}"')
    return dispatch_map[resolved_backend], norm_target, resolved_backend, verbose


def cached(
    func: PrimFunc = None,
    out_idx: list[int] = None,
    *args,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    cache, norm_target, execution_backend, verbose = _resolve_cache_dispatch(target, execution_backend, verbose)
    return cache.cached(
        func,
        out_idx,
        *args,
        target=norm_target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
