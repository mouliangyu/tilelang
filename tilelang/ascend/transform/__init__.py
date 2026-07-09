"""Ascend-only TIR transform passes."""
# pylint: disable=invalid-name

from tilelang.transform import _ffi_api

from . import z3_scheduler  # noqa: F401
from .hoist_simd_pairs import HoistSimdPairs  # noqa: F401
from .normalize_mixed_kernel_sid import NormalizeMixedKernelSid  # noqa: F401


def NormalizeControlFlowForSchedule():
    """Normalize control flow ahead of AutoSchedule.

    Rewrites each ``while cond: BODY`` into a bounded serial ``for`` loop
    (tagged ``synthetic_while``) so AutoSchedule can pipeline the body, then
    extracts complex ``if`` conditions into temporary ``Bind`` variables.
    :func:`RestoreWhileLoops` undoes the while-rewrite after scheduling.
    """
    return _ffi_api.NormalizeControlFlowForSchedule()  # type: ignore


def RestoreWhileLoops():
    """Rewrite ``synthetic_while``-tagged for loops back into ``while(true)``."""
    return _ffi_api.RestoreWhileLoops()  # type: ignore


def AnnotateMultiBufferEligible():
    """Annotate each For loop with the set of buffers that can be multi-buffered."""
    return _ffi_api.AnnotateMultiBufferEligible()  # type: ignore


def NormalizeNoConflictHints():
    """Consume ``T.assume_no_conflict`` markers into ``tl.no_conflict`` For annotations."""
    return _ffi_api.NormalizeNoConflictHints()  # type: ignore


def AutoSchedule():
    """Auto schedule for high-level operations"""
    return _ffi_api.AutoSchedule()  # type: ignore


def AscendSimdVFLowerParallel():
    """Lower T.Parallel inside SIMD_VF blocks to MicroAPI register-level vector ops."""
    return _ffi_api.AscendSimdVFLowerParallel()  # type: ignore


def MergeUBAllocations(align_bytes: int = 16, disable_reuse: bool = False):
    """MergeUBAllocations

    Merge multiple Ascend Unified Buffer (and L0/L1) allocations into one,
    using synchronization-graph-based liveness analysis for UB reuse.

    Buffer reuse is always performed unless ``disable_reuse`` is set.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeUBAllocations(align_bytes, disable_reuse)  # type: ignore


def MarkScalarDcacheBypass():
    """Rewrite scalar global buffer accesses to use dcache bypass intrinsics
    for buffers that have writes (scalar BufferStore or MTE copy). Pure-read
    buffers keep normal BufferLoad (dcache path).
    """
    return _ffi_api.MarkScalarDcacheBypass()  # type: ignore


__all__ = [
    "AnnotateMultiBufferEligible",
    "AscendSimdVFLowerParallel",
    "AutoSchedule",
    "HoistSimdPairs",
    "MarkScalarDcacheBypass",
    "MergeUBAllocations",
    "NormalizeControlFlowForSchedule",
    "NormalizeMixedKernelSid",
    "NormalizeNoConflictHints",
    "RestoreWhileLoops",
]
