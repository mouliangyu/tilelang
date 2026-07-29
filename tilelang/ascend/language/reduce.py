"""Ascend SIMT warp-level reduction primitives."""

from __future__ import annotations

from tvm import tirx


def ascend_reduce_add(val):
    """Ascend SIMT warp-level reduce sum.

    Reduces a scalar value across all 32 threads in a warp using
    hardware tree reduction. Returns the sum on all threads.

    Generates: asc_reduce_add(val)

    Parameters
    ----------
    val : PrimExpr
        Scalar float value to reduce.

    Returns
    -------
    PrimExpr
        The reduced sum (same value on all threads in the warp).
    """
    return tirx.call_intrin("float32", tirx.op.Op.get("tl.ascend_reduce_add"), val)


def ascend_reduce_max(val):
    """Ascend SIMT warp-level reduce max.

    Generates: asc_reduce_max(val)
    """
    return tirx.call_intrin("float32", tirx.op.Op.get("tl.ascend_reduce_max"), val)


def ascend_reduce_min(val):
    """Ascend SIMT warp-level reduce min.

    Generates: asc_reduce_min(val)
    """
    return tirx.call_intrin("float32", tirx.op.Op.get("tl.ascend_reduce_min"), val)


__all__ = [
    "ascend_reduce_add",
    "ascend_reduce_max",
    "ascend_reduce_min",
]
