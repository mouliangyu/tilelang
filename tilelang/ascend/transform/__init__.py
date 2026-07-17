"""Ascend-only TIR transform passes."""
# pylint: disable=invalid-name

from tilelang.transform import _ffi_api

from . import z3_scheduler  # noqa: F401
from .insert_nd2nz import InsertNd2Nz  # noqa: F401
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


def RewriteFp4ToFp4x2():
    """Retype genuine fp4 (float4_e2m1fn, lanes=1) storage into the 1-byte
    packed-pair form (float4_e2m1fnx2, lanes=2), halving fp4 element counts /
    offsets / indices so pointer arithmetic addresses bytes correctly. fp4
    views over non-fp4 storage are left untouched.
    """
    return _ffi_api.RewriteFp4ToFp4x2()  # type: ignore


def RewriteFlagToBuf():
    """Split each hard_event's synchronization between the 8-slot flag namespace
    and the shared 32-slot get_buf/rls_buf mutex pool to minimize wasted
    flag_ids. When a hard_event's sync points total more than 8 event_ids, a 0/1
    knapsack (capacity 8) keeps the subset of sync-point blocks that fills the
    flag slots best (renumbered into [0,8)); the rest spill to the mutex pool as
    get_buf/rls_buf pairs. Hard_events fitting within 8 slots are untouched, and
    a kernel with no overflow is a no-op.

    Must run after MergeUBAllocations (which uses set_flag/wait_flag as its
    liveness-graph anchors).
    """
    return _ffi_api.RewriteFlagToBuf()  # type: ignore


__all__ = [
    "AnnotateMultiBufferEligible",
    "AscendSimdVFLowerParallel",
    "AutoSchedule",
    "IfConditionExtract",
    "InsertNd2Nz",
    "MarkScalarDcacheBypass",
    "MergeUBAllocations",
    "NormalizeControlFlowForSchedule",
    "NormalizeMixedKernelSid",
    "NormalizeNoConflictHints",
    "RestoreWhileLoops",
    "RewriteFp4ToFp4x2",
    "RewriteFlagToBuf",
]
