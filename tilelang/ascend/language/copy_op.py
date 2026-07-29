"""Ascend dual-destination copy (one source fanned out to both AIV sub-cores)."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy
from tilelang.utils.language import to_buffer_region
from tvm import tirx

__all__ = ["dual_copy"]


def _get_shape(buf: BufferLikeType) -> list:
    """Extract shape list from Buffer, BufferLoad, or BufferRegion."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, tirx.BufferLoad):
        return list(buf.buffer.shape)
    if isinstance(buf, tirx.BufferRegion):
        return [r.extent for r in buf.region]
    raise TypeError(f"Cannot get shape from {type(buf)}")


def _get_allocation_shape(buf: BufferLikeType) -> list:
    """Extract the underlying allocation shape rather than region extents."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, (tirx.BufferLoad, tirx.BufferRegion)):
        return list(buf.buffer.shape)
    raise TypeError(f"Cannot get allocation shape from {type(buf)}")


def _is_half(full_dim, half_dim) -> bool:
    """Check whether half_dim is full_dim // 2."""
    if isinstance(full_dim, (int, tirx.IntImm)) and isinstance(half_dim, (int, tirx.IntImm)):
        full_val = full_dim if isinstance(full_dim, int) else full_dim.value
        half_val = half_dim if isinstance(half_dim, int) else half_dim.value
        return half_val == full_val // 2
    try:
        from tvm.ir import structural_equal

        return structural_equal(half_dim * 2, full_dim)
    except Exception:
        return False


def dual_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    l2_cache_ctrl: int | str = 0,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions with dualDst (M-split or N-split across 2 AIVs).

    The split direction is auto-inferred from the shape mismatch between src and dst.
    Supports L0C→UB and UB→L1 transfers. For UB→L1 copies, ND→NZ format conversion
    is automatically handled by the InsertNd2Nz pass.

    - ``src[M, N]`` → ``dst[M/2, N]``: M-split (``dual_dst_ctl=0b01``)
      AIV0 receives rows ``[0, M/2)``, AIV1 receives rows ``[M/2, M)``
    - ``src[M, N]`` → ``dst[M, N/2]``: N-split (``dual_dst_ctl=0b10``)
      AIV0 receives cols ``[0, N/2)``, AIV1 receives cols ``[N/2, N)``

    Args:
        src: Source buffer (L0C ``__cc__`` or UB ``__ubuf__``), shape ``[M, N]``
        dst: Destination buffer (UB ``__ubuf__`` or L1 ``__cbuf__``), shape ``[M/2, N]`` or ``[M, N/2]``
        unit_flag_ctrl: Unit flag control (0=manual sync, 3=pipelined with mad).
            ``None`` omits the annotation and lowers as 0.
        l2_cache_ctrl (int | str): Ascend L2 cache control policy for the UB→GM
            store path. Accepts an integer or case-insensitive string name
            (e.g. ``"notalloc_clean"``). Defaults to 0 (``"normal_fv"``).
            Ascend only; ignored on other backends.

    Raises:
        ValueError: If the split direction cannot be inferred from shapes.
    """
    src_shape = _get_shape(src)
    dst_shape = _get_shape(dst)

    if len(src_shape) < 2 or len(dst_shape) < 2:
        raise ValueError("dual_copy requires at least 2D buffers")

    src_m, src_n = src_shape[-2], src_shape[-1]
    dst_m, dst_n = dst_shape[-2], dst_shape[-1]

    m_halved = _is_half(src_m, dst_m) or _is_half(dst_m, src_m)
    n_halved = _is_half(src_n, dst_n) or _is_half(dst_n, src_n)

    # Runtime tail regions do not retain a structural 2:1 relationship with
    # the fixed half-sized destination. Infer the split from the underlying
    # allocations in that case, while preserving the dynamic region extents
    # for the actual copy.
    alloc_src_shape = _get_allocation_shape(src)
    alloc_dst_shape = _get_allocation_shape(dst)
    alloc_src_m, alloc_src_n = alloc_src_shape[-2], alloc_src_shape[-1]
    alloc_dst_m, alloc_dst_n = alloc_dst_shape[-2], alloc_dst_shape[-1]
    if m_halved == n_halved:
        m_halved = _is_half(alloc_src_m, alloc_dst_m) or _is_half(alloc_dst_m, alloc_src_m)
        n_halved = _is_half(alloc_src_n, alloc_dst_n) or _is_half(alloc_dst_n, alloc_src_n)

    if m_halved and not n_halved:
        dual_dst_ctl = 1
    elif n_halved and not m_halved:
        dual_dst_ctl = 2
    else:
        raise ValueError(
            f"Cannot infer dual_copy split direction: src shape {src_shape} vs dst shape {dst_shape}. "
            "Exactly one trailing dimension must be halved."
        )

    # Convert both to buffer regions to bypass the shape equality check in copy()
    # This allows src[M,N] to be copied to dst[M/2,N] or dst[M,N/2]
    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)

    # WARNING: dav-950 does NOT support quant_pre (type conversion) with
    # fixpipe (cc->ub) dual-destination copies. If src and dst dtypes differ, the hardware may
    # silently produce incorrect results.
    src_dtype = src_region.buffer.dtype
    dst_dtype = dst_region.buffer.dtype
    if src_dtype != dst_dtype and src_region.buffer.scope() == "shared.l0c":
        raise ValueError(
            f"dual_copy: src dtype ({src_dtype}) != dst dtype ({dst_dtype}). "
            "Type conversion during cc->ub dual-destination copy is NOT supported on dav-950. "
            "Use a separate copy or cast inside a VF block instead."
        )

    annotations = {"dual_dst_ctl": dual_dst_ctl}
    if _is_half(dst_m, src_m) or _is_half(dst_n, src_n) or _is_half(alloc_dst_m, alloc_src_m) or _is_half(alloc_dst_n, alloc_src_n):
        annotations["double"] = True

    return copy(
        src_region,
        dst_region,
        annotations=annotations,
        unit_flag_ctrl=unit_flag_ctrl,
        l2_cache_ctrl=l2_cache_ctrl,
    )
