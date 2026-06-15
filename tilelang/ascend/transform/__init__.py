"""Ascend-only TIR transform passes."""
# pylint: disable=invalid-name

from tilelang.transform import _ffi_api

from . import z3_scheduler  # noqa: F401
from .hoist_simd_pairs import HoistSimdPairs  # noqa: F401
from .normalize_mixed_kernel_sid import NormalizeMixedKernelSid  # noqa: F401


def IfConditionExtract():
    """Extract if condition to LetStmt variables."""
    return _ffi_api.IfConditionExtract()  # type: ignore


def AnnotateMultiBufferEligible():
    """Annotate each For loop with the set of buffers that can be multi-buffered."""
    return _ffi_api.AnnotateMultiBufferEligible()  # type: ignore


def AutoSchedule():
    """Auto schedule for high-level operations"""
    return _ffi_api.AutoSchedule()  # type: ignore


def AscendSimdVFLowerParallel():
    """Lower T.Parallel inside SIMD_VF blocks to MicroAPI register-level vector ops."""
    return _ffi_api.AscendSimdVFLowerParallel()  # type: ignore


def MergeUBAllocations(enable_aggressive_merge: bool = False, align_bytes: int = 16, disable_reuse: bool = False):
    """MergeUBAllocations

    Merge multiple Ascend Unified Buffer (and L0/L1) allocations into one,
    using synchronization-graph-based liveness analysis for UB reuse.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeUBAllocations(enable_aggressive_merge, align_bytes, disable_reuse)  # type: ignore


__all__ = [
    "AnnotateMultiBufferEligible",
    "AscendSimdVFLowerParallel",
    "AutoSchedule",
    "HoistSimdPairs",
    "IfConditionExtract",
    "MergeUBAllocations",
    "NormalizeMixedKernelSid",
]
