"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations
from typing import Literal, Any

from tilelang._typing import BufferLikeType
from tilelang.utils.language import (
    to_buffer_region,
    legalize_pairwise_extents,
)
from tilelang.utils.deprecated import deprecated
from tilelang.language.utils import get_extent, buffer_region_to_tile_region
import tvm
from tvm import ir, tirx


def _normalize_copy_regions_with_extents(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    list[tirx.PrimExpr] | None,
    list[tirx.PrimExpr] | None,
]:
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tirx.Buffer) and isinstance(dst, tirx.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    src_is_scalar_load = src_extent is None and isinstance(src, tirx.BufferLoad)
    dst_is_scalar_load = dst_extent is None and isinstance(dst, tirx.BufferLoad)

    # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
    # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
    if src_is_scalar_load and dst_is_scalar_load:
        return src, dst, None, None

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    # Treat missing extent as length-matched ones for convenience. This provides limited
    # broadcasting-like syntactic sugar, but does not implement general broadcasting support.
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Align and broadcast extents from the right (tail) side.
    # This is majorly for supporting some syntactic sugar, not the whole broadcasting ability of copy op.
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    # Use legalized extents for src and dst respectively.
    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)
    return src, dst, src_extent, dst_extent


def _normalize_copy_regions(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
]:
    src, dst, _, _ = _normalize_copy_regions_with_extents(src, dst)
    return src, dst


# Short-name → int mapping for L2 cache control.
_L2_CACHE_CTRL_MAP = {
    # ── LD_L2CacheType (Load / GM→L1) ──
    "NORMAL_FV": 0,
    "NORMAL_LV": 1,
    "NORMAL_PERS": 2,
    "NORMAL_PREF": 3,
    "NOTALLOC_KEEP": 4,
    "NOTALLOC_CLEAN": 5,
    "NOTALLOC_DROP": 6,
    "IDS_FV": 8,
    "IDS_LV": 9,
    "IDS_PERS": 10,
    "IDS_PREF": 11,
    "EXCLUSIV_FV": 12,
    "EXCLUSIV_LV": 13,
    "EXCLUSIV_PERS": 14,
    "EXCLUSIV_PREF": 15,
    "INVALID": 16,
    # ── ST_L2CacheType (Store / UB→GM) — names that differ from LD ──
    "NORMAL_RED": 3,
    "NOTALLOC_CI": 4,
    "NOTALLOC_PW": 5,
    "NOTALLOC_PI": 6,
    "NOTALLOC_RED": 7,
    "WBH_FV": 8,
    "WBH_LV": 9,
    "WBH_PERS": 10,
    "WBH_RED": 11,
    "WTS_FV": 12,
    "WTS_LV": 13,
    "WTS_PERS": 14,
    "WTS_RED": 15,
}


def _normalize_l2_cache_ctrl(value: int | str | None) -> int | None:
    """Convert a string l2_cache_ctrl name to its integer value.

    Matching is case-insensitive and suffix-based:
    ``"NORMAL_FV"``, ``"normal_fv"``, and ``"L2_CACHE_HINT_NORMAL_FV"``
    all map to ``0``.  Integers pass through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        key = value.upper()
        # Exact match
        if key in _L2_CACHE_CTRL_MAP:
            return _L2_CACHE_CTRL_MAP[key]
        # Suffix match (e.g. "L2_CACHE_HINT_NORMAL_FV" or "_NORMAL_FV")
        for map_key, map_val in _L2_CACHE_CTRL_MAP.items():
            if key.endswith(map_key):
                return map_val
        raise ValueError(f"Unknown l2_cache_ctrl string {value!r}. Valid suffixes: {sorted(_L2_CACHE_CTRL_MAP.keys())}")
    raise TypeError(f"l2_cache_ctrl must be int, str, or None, got {type(value)}")


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
    transpose: bool = False,
    nd2nz: bool = False,
    l2_cache_ctrl: int | str | None = None,
    scale_factor: bool = False,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions.

    Args:
        src (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Source memory region
        dst (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        disable_tma (bool, keyword-only): Whether to disable TMA acceleration. Defaults to False.
        eviction_policy (Optional[str], keyword-only): Cache eviction policy. Defaults to None.
        prefer_instruction (Optional[str], keyword-only): Backend-specific preferred lowering
            instruction category. For CUDA, recognized values include "tma", "cp_async", and
            "sync". For "tma", T.copy keeps synchronous copy semantics; global -> shared copies
            lower through TMA with an automatically allocated barrier and wait when constraints
            are satisfied.
        annotations (Optional[dict], keyword-only): Additional annotations dict. If provided,
            coalesced_width, disable_tma, eviction_policy, and prefer_instruction can also
            be specified here.
            Values in annotations take precedence over individual arguments.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy
            (only valid for normal SIMT copy; incompatible with TMA/LDSM/STSM/TMem). When provided,
            it is attached to the outermost parallel loop generated by this copy.
        transpose (bool, keyword-only): Ascend GM-to-L1 transpose hint for dn2nz layout.
        l2_cache_ctrl (Optional[int | str], keyword-only): Ascend L2 cache control
            policy for GM↔UB, GM↔L1, and UB↔GM DMA paths. Accepts an integer (hardware
            L2Ctrl value) or a case-insensitive string name.
            Common values: ``0`` / ``"normal_fv"`` (default), ``4`` / ``"notalloc_keep"``
            (bypass L2 to stream past), ``5`` / ``"notalloc_clean"`` (bypass L2 to
            avoid pollution). Defaults to None (hardware default 0).
            Ascend only; ignored on other backends.
        scale_factor (bool, keyword-only): Whether the copied data is an Ascend
            scale-factor tile. The marker changes GM→L1 NZ stride setup for
            scale-factor data. Defaults to False.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tirx.Call: A handle to the copy operation

    Range handling notes:
    - Accepts `Buffer`/`BufferRegion`/`BufferLoad` on either side. Extents are
      derived as follows: `Buffer -> shape`, `BufferRegion -> [r.extent]`,
      `BufferLoad -> extents from its inferred/encoded region`.
    - Normally, we require the extents of both sides to be the same. If they
      differ, the copy instruction follows an internal rule to select one side
      as the base range and create iteration space. This may generate unexpected
      code. And if some dimensions are 1, unexpected errors may happen.
    - Small Optimization: If both `src` and `dst` are scalar `BufferLoad` without
      region extents, lowers to a direct store: `dst[...] = src[...]`.
    - Syntactic Sugar: TileLang supports passing the head address of a buffer to represent
      the whole buffer if there are no ambiguity. For example, T.copy(A, A_shared[i, j]).
      To support this, we need some special shape checking. But remember currently we don't
      support something like "broadcast".
    - The finalized extents are encoded with `tl.region` via `to_buffer_region`
      and passed through to the backend; low-level loop construction and any
      scope-specific decisions happen during lowering.
    """
    src_orig, dst_orig = src, dst
    src, dst, src_extent, dst_extent = _normalize_copy_regions_with_extents(src, dst)
    if isinstance(src, tirx.BufferLoad) and isinstance(dst, tirx.BufferLoad):
        return tirx.BufferStore(dst.buffer, src, dst.indices)

    # Build annotations dict
    ann = annotations.copy() if annotations else {}

    # Individual arguments take lower precedence than annotations
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]
    if "prefer_instruction" not in ann and prefer_instruction is not None:
        ann["prefer_instruction"] = prefer_instruction
    if isinstance(ann.get("prefer_instruction"), str):
        ann["prefer_instruction"] = tirx.StringImm(ann["prefer_instruction"])

    # Parallel loop layout hint (Fragment). Mirrors T.Parallel(loop_layout=...)
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    # Ascend GM→L1: use dn2nz (transpose N/D mapping) instead of nd2nz
    if transpose and "transpose" not in ann:
        ann["transpose"] = tirx.IntImm("int32", 1)

    # Ascend UB→L1: fused ND→NZ scatter (+ optional cast) before raw DMA
    if nd2nz and "nd2nz" not in ann:
        ann["nd2nz"] = tirx.IntImm("int32", 1)

    # Ascend DMA: L2 cache control for GM↔L1 and UB↔GM paths.
    if l2_cache_ctrl is not None and "l2_cache_ctrl" not in ann:
        ann["l2_cache_ctrl"] = l2_cache_ctrl
    if scale_factor and "scale_factor" not in ann:
        ann["scale_factor"] = tirx.IntImm("int32", 1)
    if "l2_cache_ctrl" in ann:
        ann["l2_cache_ctrl"] = _normalize_l2_cache_ctrl(ann["l2_cache_ctrl"])

    # Ascend nd2nz: bypass tl.tileop.copy entirely and emit scatter/post_copy
    # intrinsics as separate TIR statements so later passes can schedule and
    # inspect both steps.
    nd2nz_active = bool(ann.get("nd2nz", 0))
    if nd2nz_active:
        assert src_extent is not None and dst_extent is not None
        return _emit_nd2nz_seq(src, dst, src_orig, dst_orig, src_extent, dst_extent, ann)

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def _emit_nd2nz_seq(
    src: tirx.PrimExpr,
    dst: tirx.PrimExpr,
    src_orig: BufferLikeType,
    dst_orig: BufferLikeType,
    src_extent: list,
    dst_extent: list,
    ann: dict,
) -> tirx.PrimExpr:
    """Emit the Ascend ND→NZ copy as a scatter + post_copy intrinsic pair.

    Emits a SimdVF scatter (ND → NZ reorder in UB tmp) followed by a
    2D copy_ubuf_to_cbuf (UB → L1).  When the tile is only one D-group wide
    (cols * elem_bytes == 32), NZ == ND and the scatter is skipped —
    the post_copy reads the raw ND source directly.

    Dual (``ann['double']``): src is the half-size per-AIV tile and dst is the
    full-size L1 buffer. The post_copy is emitted with two extra trailing args
    (``dual_split_dim``, ``dual_half_extent``) so AutoSchedule's
    ``warpgroup_partition`` can rewrite the dst access_ptr's BufferLoad indices
    to add ``sid * half_extent`` along the split axis. After that rewrite the
    extras are stripped and codegen sees the standard 6-arg form.
    """
    import tvm.script.ir_builder.tir as tb_tir
    from tilelang.language.builtin import access_ptr as _access_ptr
    from tilelang.language.allocate import alloc_shared as _alloc_shared

    is_dual = bool(ann.get("double"))
    rows_src, cols_src = src_extent[-2], src_extent[-1]
    rows_dst, cols_dst = dst_extent[-2], dst_extent[-1]
    rows = rows_src if is_dual else rows_dst
    cols = cols_src if is_dual else cols_dst

    src_buf = src.args[0].buffer
    dst_buf = dst.args[0].buffer
    full_rows = dst_buf.shape[-2]

    def _to_dtype_str(dtype) -> str:
        dtype_str = str(dtype)
        if "bfloat16" in dtype_str:
            return "bfloat16_t"
        if "float16" in dtype_str:
            return "half"
        if dtype_str in ("float32", "float"):
            return "float"
        raise ValueError(f"Unsupported nd2nz dtype: {dtype_str}")

    src_ptr = _access_ptr(src_orig, "r")
    dst_ptr = _access_ptr(dst_orig, "w")

    # Single D-group + same dtype: NZ == ND, scatter is a no-op → skip it.
    # If dtypes differ the scatter is still needed for the cast.
    _elem_bytes = dst_buf.dtype.bits // 8
    if isinstance(cols, int):
        cols_i = cols
    elif isinstance(cols, tirx.IntImm):
        cols_i = int(cols.value)
    else:
        cols_i = None
    single_dgroup = cols_i is not None and cols_i * _elem_bytes == 32
    same_dtype = src_buf.dtype == dst_buf.dtype

    src_dtype_str = _to_dtype_str(src_buf.dtype)
    dst_dtype_str = _to_dtype_str(dst_buf.dtype)

    scatter_op = tirx.op.Op.get("tl.ascend_nd2nz_scatter")
    post_copy_op = tirx.op.Op.get("tl.ascend_nd2nz_post_copy")

    dual_extras: list = []
    if is_dual:
        dual_dst_ctl_raw = ann.get("dual_dst_ctl", 0)
        dual_dst_ctl = int(dual_dst_ctl_raw.value) if isinstance(dual_dst_ctl_raw, tirx.IntImm) else int(dual_dst_ctl_raw)
        if dual_dst_ctl not in (1, 2):
            raise ValueError(f"dual_copy(nd2nz=True) requires dual_dst_ctl in {{1, 2}}, got {dual_dst_ctl}")
        split_dim_neg2 = 0 if dual_dst_ctl == 1 else 1
        half_extent = src_extent[-2 + split_dim_neg2]
        dual_extras = [tirx.IntImm("int32", split_dim_neg2), half_extent]

    if single_dgroup and same_dtype:
        return tirx.call_intrin(
            "handle",
            post_copy_op,
            dst_ptr,
            src_ptr,
            rows,
            cols,
            full_rows,
            dst_dtype_str,
            *dual_extras,
        )

    # Multi-fractal: allocate a UB scratch buffer for the NZ-laid intermediate,
    # emit the scatter into the active IRBuilder frame, and return the
    # post_copy Call so the eager-builder auto-evaluates it.
    # NOTE: Pad one more row to avoid bank conflict. Swizzle is sadly impossible
    #       on Ascend 950 given no vscatterb instruction support.
    tmp_buf = _alloc_shared(((rows + 1) * cols,), dst_buf.dtype, "shared.dyn")

    scatter = tirx.call_intrin(
        "handle",
        scatter_op,
        src_ptr,
        _access_ptr(tmp_buf, "w"),
        rows,
        cols,
        dst_dtype_str,
        src_dtype_str,
    )
    tb_tir.evaluate(scatter)

    return tirx.call_intrin(
        "handle",
        post_copy_op,
        dst_ptr,
        _access_ptr(tmp_buf, "r"),
        rows,
        cols,
        full_rows,
        dst_dtype_str,
        *dual_extras,
    )


def copy_cluster(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    dst_block: int | tirx.PrimExpr | None = None,
    cluster_mask: int | None = None,
    remote_barrier: tirx.BufferLoad | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Cluster-aware copy for TMA multicast or SM-to-SM shared-memory copy.

    Args:
        src: Source memory region.
        dst: Destination memory region.
        dst_block: Destination CTA rank in the cluster for SM-to-SM copy.
        cluster_mask: Bitmask of CTAs that participate in TMA multicast.
        remote_barrier: Shared-memory mbarrier for asynchronous SM-to-SM copy
            completion signalling.  The destination CTA should wait on its
            local copy of this barrier.
        eviction_policy: Cache eviction hint passed to the TMA instruction.
            Only relevant for the TMA multicast path (``cluster_mask`` set).
        coalesced_width: Vectorization width (in elements) for the SIMT loop
            used on the SM-to-SM fallback path (``dst_block`` set, no fast
            bulk-async route available).
        loop_layout: Parallel loop layout hint (Fragment) for the SIMT loop on
            the SM-to-SM fallback path. Incompatible with the TMA multicast
            path (``cluster_mask`` set).

    Returns:
        tirx.Call: A handle to the copy operation.
    """
    src, dst = _normalize_copy_regions(src, dst)

    ann: dict = {}
    if dst_block is not None:
        ann["dst_block"] = dst_block
    if cluster_mask is not None:
        ann["cluster_mask"] = cluster_mask
    if remote_barrier is not None:
        ann["barrier"] = remote_barrier
    if eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]
    if coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None:
        ann["parallel_loop_layout"] = loop_layout

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Asynchronous copy primitive lowered through cp.async.

    This operator is intended for explicitly asynchronous global->shared copy.
    The backend enforces cp.async constraints and emits:
      `ptx_cp_async(...)` + `ptx_commit_group()`.
    No wait is auto-inserted for `T.async_copy`; synchronization is explicit.

    Args:
        src (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Source memory region
        dst (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        annotations (Optional[dict], keyword-only): Additional annotations dict.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy loop.

    Returns:
        tirx.Call: A handle to the async copy operation
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src, tirx.BufferLoad) and isinstance(dst, tirx.BufferLoad):
        return tirx.BufferStore(dst.buffer, src, dst.indices)

    ann = annotations.copy() if annotations else {}
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.async_copy"),
        src,
        dst,
        annotations=ann if ann else None,
    )


def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier=None,
    leader_scope_threads: int | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """TMA copy with user-managed synchronization.

    For **loads** (global -> shared): issues expect_tx + tma_load (no wait).
    Unlike T.copy() which emits a full synchronous TMA sequence (arrive + load + wait),
    T.tma_copy() emits only the producer part (expect_tx + tma_load).
    The user manages synchronization explicitly via T.barrier_arrive() and
    T.mbarrier_wait_parity(). ``barrier`` is required for loads.

    For **stores** (shared -> global): issues tma_store + tma_store_arrive (no wait).
    Unlike T.copy() which emits tma_store + tma_store_arrive + tma_store_wait,
    T.tma_copy() omits the wait so the user can batch multiple stores before
    calling T.tma_store_wait() explicitly. ``barrier`` is not needed for stores.
    FP4 unpacked shared-memory storage is load-only for TMA: packed global
    ``float4_e2m1fn`` may be loaded into unpacked shared
    ``float4_e2m1_unpacked``, but the reverse TMA store is not supported.

    Args:
        src: Source memory region (global or shared)
        dst: Destination memory region (shared or global)
        barrier: Mbarrier (from T.alloc_barrier()) for TMA load synchronization.
            Required for loads (global -> shared). Not needed for stores.
            The TMA load will arrive at this barrier with expected byte count.
            The user must wait on the same barrier via T.mbarrier_wait_parity().
        leader_scope_threads: Number of threads in each TMA leader-election scope
            (e.g., 32 for per-warp). Defaults to the thread extend in the current context if not specified.
        eviction_policy: Cache eviction policy. Defaults to None.
        annotations: Additional annotations dict. Values in annotations take
            precedence over individual arguments.

    Returns:
        tirx.Call: A handle to the tma_copy operation
    """
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tirx.Buffer) and isinstance(dst, tirx.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    ann = annotations.copy() if annotations else {}

    if barrier is not None:
        from .builtin import _mbar_to_buffer_load

        ann["barrier"] = _mbar_to_buffer_load(barrier)

    if leader_scope_threads is not None:
        if not isinstance(leader_scope_threads, int) or leader_scope_threads <= 0:
            raise ValueError(f"leader_scope_threads must be a positive int, got {leader_scope_threads}")
        if leader_scope_threads % 32 != 0:
            raise ValueError(f"leader_scope_threads must be a multiple of warp size (32), got {leader_scope_threads}")
        if "leader_scope_threads" not in ann:
            ann["leader_scope_threads"] = leader_scope_threads

    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.tma_copy"), src, dst, annotations=ann if ann else None)


_TMA_SUPPORTED_DTYPES = frozenset(
    {
        "uint8",
        "uint16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
    }
)


def tma_gather4(
    src: tirx.Buffer,
    dst: tirx.Buffer,
    col: tirx.PrimExpr,
    rows,
    *,
    barrier,
    swizzle=None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
):
    """Issue a TMA tile::gather4 load (sm_100a, Blackwell).

    Loads four arbitrary rows of a 2D global tensor ``src`` into a 2D shared
    tile ``dst`` of shape ``(4, K_box)``. The CUtensorMap descriptor (dtype +
    swizzle) is built by the compiler from buffer + layout info.

    Caller must wrap this with ``T.shuffle_elect`` and pair it with
    ``T.mbarrier_expect_tx`` (use :func:`tma_gather4_bytes`) before, and
    ``barrier_arrive`` / ``mbarrier_wait_parity`` after.

    The ``swizzle`` kwarg is deprecated; mark the shared tile via
    ``T.annotate_layout`` for non-default swizzle.
    """
    if not isinstance(src, tirx.Buffer):
        raise TypeError("tma_gather4 src must be a tirx.Buffer (global)")
    if not isinstance(dst, tirx.Buffer):
        raise TypeError("tma_gather4 dst must be a tirx.Buffer (shared)")
    if src.scope() != "global":
        raise ValueError(f"tma_gather4 src must be a global buffer, got scope={src.scope()}")
    if dst.scope() not in ("shared", "shared.dyn"):
        raise ValueError(f"tma_gather4 dst must be a shared buffer, got scope={dst.scope()}")
    if len(src.shape) != 2:
        raise ValueError(f"tma_gather4 expects rank-2 global buffer, got {len(src.shape)}")
    if len(dst.shape) != 2:
        raise ValueError(f"tma_gather4 expects rank-2 shared buffer (4 x K_box), got {len(dst.shape)}")
    if src.dtype != dst.dtype:
        raise ValueError(f"tma_gather4 dtype mismatch: src={src.dtype}, dst={dst.dtype}")
    if not (isinstance(dst.shape[0], int) and dst.shape[0] == 4) and not (hasattr(dst.shape[0], "value") and int(dst.shape[0].value) == 4):
        raise ValueError(f"tma_gather4 shared tile leading dim must be 4, got {dst.shape[0]}")
    if src.strides:
        inner = src.strides[1]
        if not ((isinstance(inner, int) and inner == 1) or (hasattr(inner, "value") and int(inner.value) == 1)):
            raise ValueError(f"tma_gather4 requires unit innermost global stride, got {inner}")
    rows = list(rows)
    if len(rows) != 4:
        raise ValueError(f"tma_gather4 expects exactly 4 row indices, got {len(rows)}")
    if swizzle not in (None, "none", 0):
        import warnings

        warnings.warn(
            f"tma_gather4 swizzle={swizzle!r} is deprecated; use T.annotate_layout.",
            DeprecationWarning,
            stacklevel=2,
        )

    from .builtin import _mbar_to_buffer_load

    bar_load = _mbar_to_buffer_load(barrier)

    eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
    ep = 0 if eviction_policy is None else eviction_policy_map[eviction_policy]

    # Matching (4, K_box) extents satisfy CopyNode's shape check; the actual
    # access pattern lives in the gather4_rows / gather4_col annotations.
    K_box = dst.shape[1]
    src_region = to_buffer_region(src, access_type="r", extents=[4, K_box])
    dst_region = to_buffer_region(dst, access_type="w", extents=[4, K_box])

    ann = {
        "is_gather4": True,
        "gather4_rows": rows,
        "gather4_col": col,
        "barrier": bar_load,
        "eviction_policy": ep,
    }
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.copy"),
        src_region,
        dst_region,
        annotations=ann,
    )


def tma_gather4_bytes(K_box, dtype: str) -> int:
    """Transaction byte count for a 4-row gather4 of width ``K_box``. Pass
    to ``T.mbarrier_expect_tx`` immediately before ``T.tma_gather4``.
    """
    if dtype not in _TMA_SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    dt = tvm.DataType(dtype)
    if dt.is_float4_e2m1_unpacked():
        elem_bits = 4
    else:
        elem_bits = dt.bits
    return (4 * K_box * elem_bits + 7) // 8


def tma_scatter4(
    src: tirx.Buffer,
    dst: tirx.Buffer,
    col: tirx.PrimExpr,
    rows,
    *,
    swizzle=None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
):
    """Issue a TMA tile::scatter4 store (sm_100a, Blackwell).

    Stores a 2D shared tile of shape ``(4, K_box)`` to four arbitrary rows of
    a 2D global tensor ``dst``. Caller is responsible for ``tma_store_arrive``
    / ``tma_store_wait`` and the ``T.shuffle_elect`` guard. See
    :func:`tma_gather4` for descriptor / swizzle inference details.
    """
    if not isinstance(src, tirx.Buffer):
        raise TypeError("tma_scatter4 src must be a tirx.Buffer (shared)")
    if not isinstance(dst, tirx.Buffer):
        raise TypeError("tma_scatter4 dst must be a tirx.Buffer (global)")
    if src.scope() not in ("shared", "shared.dyn"):
        raise ValueError(f"tma_scatter4 src must be a shared buffer, got scope={src.scope()}")
    if dst.scope() != "global":
        raise ValueError(f"tma_scatter4 dst must be a global buffer, got scope={dst.scope()}")
    if len(src.shape) != 2:
        raise ValueError(f"tma_scatter4 expects rank-2 shared buffer (4 x K_box), got {len(src.shape)}")
    if len(dst.shape) != 2:
        raise ValueError(f"tma_scatter4 expects rank-2 global buffer, got {len(dst.shape)}")
    if src.dtype != dst.dtype:
        raise ValueError(f"tma_scatter4 dtype mismatch: src={src.dtype}, dst={dst.dtype}")
    if not (isinstance(src.shape[0], int) and src.shape[0] == 4) and not (hasattr(src.shape[0], "value") and int(src.shape[0].value) == 4):
        raise ValueError(f"tma_scatter4 shared tile leading dim must be 4, got {src.shape[0]}")
    if dst.strides:
        inner = dst.strides[1]
        if not ((isinstance(inner, int) and inner == 1) or (hasattr(inner, "value") and int(inner.value) == 1)):
            raise ValueError(f"tma_scatter4 requires unit innermost global stride, got {inner}")
    rows = list(rows)
    if len(rows) != 4:
        raise ValueError(f"tma_scatter4 expects exactly 4 row indices, got {len(rows)}")
    if swizzle not in (None, "none", 0):
        import warnings

        warnings.warn(
            f"tma_scatter4 swizzle={swizzle!r} is deprecated; use T.annotate_layout.",
            DeprecationWarning,
            stacklevel=2,
        )

    eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
    ep = 0 if eviction_policy is None else eviction_policy_map[eviction_policy]

    K_box = src.shape[1]
    src_region = to_buffer_region(src, access_type="r", extents=[4, K_box])
    dst_region = to_buffer_region(dst, access_type="w", extents=[4, K_box])

    ann = {
        "is_scatter4": True,
        "gather4_rows": rows,
        "gather4_col": col,
        "eviction_policy": ep,
    }
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.copy"),
        src_region,
        dst_region,
        annotations=ann,
    )


def transpose(
    src: BufferLikeType,
    dst: BufferLikeType,
) -> tirx.PrimExpr:
    """Transpose a 2D buffer in shared memory: dst[j, i] = src[i, j].

    Both src and dst should be shared memory buffers.
    If src has shape (M, N), dst should have shape (N, M).

    Args:
        src: Source buffer or region of shape (..., M, N).
        dst: Destination buffer or region of shape (..., N, M).

    Returns:
        tirx.Call: A handle to the transpose operation.
    """
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent is not None, "Cannot deduce extent for transpose src."
    assert dst_extent is not None, "Cannot deduce extent for transpose dst."
    assert len(src_extent) >= 2, "Transpose requires at least 2D buffers."
    assert len(dst_extent) >= 2, "Transpose requires at least 2D buffers."

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src = buffer_region_to_tile_region(src_region, "r", list(src_extent))
    dst = buffer_region_to_tile_region(dst_region, "w", list(dst_extent))

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.transpose"),
        src,
        dst,
    )


def _get_shape(buf: BufferLikeType) -> list:
    """Extract shape list from Buffer, BufferLoad, or BufferRegion."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, tirx.BufferLoad):
        return list(buf.buffer.shape)
    if isinstance(buf, tirx.BufferRegion):
        return [r.extent for r in buf.region]
    raise TypeError(f"Cannot get shape from {type(buf)}")


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
    unit_flag_ctl: int = 0,
    nd2nz: bool = False,
    l2_cache_ctrl: int | str = 0,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions with dualDst (M-split or N-split across 2 AIVs).

    The split direction is auto-inferred from the shape mismatch between src and dst.
    Supports L0C→UB and UB→L1 transfers.  For UB→L1 copies, ``nd2nz=True`` enables
    dense-to-fractal (ND→NZ) layout conversion via SimdVF scatter.

    - ``src[M, N]`` → ``dst[M/2, N]``: M-split (``dual_dst_ctl=0b01``)
      AIV0 receives rows ``[0, M/2)``, AIV1 receives rows ``[M/2, M)``
    - ``src[M, N]`` → ``dst[M, N/2]``: N-split (``dual_dst_ctl=0b10``)
      AIV0 receives cols ``[0, N/2)``, AIV1 receives cols ``[N/2, N)``

    Args:
        src: Source buffer (L0C ``__cc__`` or UB ``__ubuf__``), shape ``[M, N]``
        dst: Destination buffer (UB ``__ubuf__`` or L1 ``__cbuf__``), shape ``[M/2, N]`` or ``[M, N/2]``
        unit_flag_ctl: Unit flag control (0=manual sync, 3=pipelined with mad)
        nd2nz: If True, fuse ND→NZ scatter for UB→L1 copy (Ascend only).
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
    annotations = {
        "dual_dst_ctl": dual_dst_ctl,
        "unit_flag_ctl": unit_flag_ctl,
        "l2_cache_ctrl": l2_cache_ctrl,
    }
    if nd2nz:
        annotations["nd2nz"] = 1
    if _is_half(dst_m, src_m) or _is_half(dst_n, src_n):
        annotations["double"] = True

    return copy(src_region, dst_region, annotations=annotations)


def im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tirx.PrimExpr,
    c_step: tirx.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tirx.PrimExpr:
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tirx.Buffer): Input image buffer
        col (tirx.Buffer): Output column buffer
        nhw_step (tirx.PrimExpr): Step size for batch and spatial dimensions
        c_step (tirx.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tirx.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img)
    col_region = to_buffer_region(col)
    img_extents = [r.extent for r in img_region.region]
    col_extents = [r.extent for r in col_region.region]
    img_region = buffer_region_to_tile_region(img_region, "r", img_extents)
    col_region = buffer_region_to_tile_region(col_region, "w", col_extents)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )


@deprecated("T.c2d_im2col", "T.im2col", "0.14.0")
def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tirx.PrimExpr,
    c_step: tirx.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
) -> tirx.PrimExpr:
    """Deprecated alias for :func:`im2col`.

    Deprecated:
        Use :func:`im2col` instead. This alias is scheduled for removal in
        TileLang 0.14.0.
    """
    return im2col(
        img,
        col,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )
