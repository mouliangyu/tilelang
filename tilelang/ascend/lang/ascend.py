"""Ascend-specific language primitives for TileLang."""

from __future__ import annotations

from tvm import tirx


def ascend_pipe_barrier(pipe_t: str) -> None:
    """Insert an AscendC pipeline barrier.

    Generates: AscendC::PipeBarrier<pipe_t::XXX>();

    Parameters
    ----------
    pipe_t : str
        Pipeline type string, e.g., "PIPE_ALL", "PIPE_V", "PIPE_M",
        "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_S".

    Example
    -------
    >>> T.ascend_pipe_barrier("PIPE_ALL")
    """
    if not isinstance(pipe_t, str):
        raise TypeError(f"pipe_t must be a string, got {type(pipe_t)}")
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_pipe_barrier"), pipe_t)


def ascend_set_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC SetFlag for pipeline synchronization.

    Generates: AscendC::SetFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_set_flag("S_MTE3", event_id)
    """
    if not isinstance(hard_event, str):
        raise TypeError(f"hard_event must be a string, got {type(hard_event)}")
    if isinstance(event_id, int):
        event_id = tirx.IntImm("int32", event_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_set_flag"), hard_event, event_id)


def ascend_wait_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC WaitFlag for pipeline synchronization.

    Generates: AscendC::WaitFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_wait_flag("S_MTE3", event_id)
    """
    if not isinstance(hard_event, str):
        raise TypeError(f"hard_event must be a string, got {type(hard_event)}")
    if isinstance(event_id, int):
        event_id = tirx.IntImm("int32", event_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_wait_flag"), hard_event, event_id)


def ascend_threadfence() -> None:
    """Insert an AscendC thread fence.

    Generates: asc_threadfence();

    Example
    -------
    >>> T.ascend_threadfence()
    """
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_threadfence"))


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


def ascend_cross_core_set_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreSetFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreSetFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
    """
    if not isinstance(pipe, str):
        raise TypeError(f"pipe must be a string, got {type(pipe)}")
    if isinstance(mode_id, int):
        mode_id = tirx.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tirx.IntImm("int32", flag_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_cross_core_set_flag"), mode_id, pipe, flag_id)


def ascend_cross_core_wait_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreWaitFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreWaitFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
    """
    if not isinstance(pipe, str):
        raise TypeError(f"pipe must be a string, got {type(pipe)}")
    if isinstance(mode_id, int):
        mode_id = tirx.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tirx.IntImm("int32", flag_id)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_cross_core_wait_flag"), mode_id, pipe, flag_id)


def ascend_copy_gm_to_ubuf(
    dst,
    src,
    sid,
    n_burst,
    burst_len,
    left_padding,
    right_padding,
    data_select,
    l2_cache_ctl,
    burst_src_stride,
    burst_dst_stride,
) -> None:
    """Insert an Ascend DMA copy from GM to UBuf.

    Generates: copy_gm_to_ubuf_align_v2(dst, src, sid, nBurst, burstLen,
    leftPadding, rightPadding, dataSelect, l2CacheCtl, burstSrcStride,
    burstDstStride);

    Parameters
    ----------
    dst : PrimExpr
        Destination UBuf pointer (tvm_access_ptr or similar).
    src : PrimExpr
        Source GM pointer (tvm_access_ptr or similar).
    sid : int or PrimExpr
        Sub-block ID (usually 0).
    n_burst : int or PrimExpr
        Number of bursts.
    burst_len : int or PrimExpr
        Length of each burst in bytes.
    left_padding : int or PrimExpr
        Left padding count.
    right_padding : int or PrimExpr
        Right padding count.
    data_select : bool, int, or PrimExpr
        v2 data select / constant padding control.
    l2_cache_ctl : int or PrimExpr
        L2 cache control.
    burst_src_stride : int or PrimExpr
        Source stride between burst starts in bytes.
    burst_dst_stride : int or PrimExpr
        Destination stride between burst starts in bytes.
    """
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            n_burst,
            burst_len,
            left_padding,
            right_padding,
            data_select,
            l2_cache_ctl,
            burst_src_stride,
            burst_dst_stride,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_gm_to_ubuf"), *args)


def ascend_copy_ubuf_to_gm(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride) -> None:
    """Insert an Ascend DMA copy from UBuf to GM.

    Generates: copy_ubuf_to_gm_align_v2(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride);

    Parameters
    ----------
    dst : PrimExpr
        Destination GM pointer.
    src : PrimExpr
        Source UBuf pointer.
    sid : int or PrimExpr
        Sub-block ID (usually 0).
    burst_num : int or PrimExpr
        Number of bursts.
    burst_len : int or PrimExpr
        Length of each burst in bytes.
    l2_cache_ctl : int or PrimExpr
        L2 cache control.
    burst_dst_stride : int or PrimExpr
        Destination stride between bursts.
    burst_src_stride : int or PrimExpr
        Source stride between bursts.
    """
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            burst_num,
            burst_len,
            l2_cache_ctl,
            burst_dst_stride,
            burst_src_stride,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_ubuf_to_gm"), *args)


def _to_int32(v):
    return tirx.IntImm("int32", v) if isinstance(v, int) else v


def ascend_copy_gm_to_cbuf(
    dst, src, sid, loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en, dst_n_value=None
) -> None:
    """Ascend DMA copy from GM to CBuf (L1).

    Generates: copy_gm_to_cbuf_multi_nd2nz(dst, src, sid,
    loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en);
    """
    if dst_n_value is None:
        dst_n_value = n_value
    args = [dst, src] + [
        _to_int32(a) for a in [sid, loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en, dst_n_value]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_gm_to_cbuf"), *args)


def ascend_load_cbuf_to_ca(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CA (L0A).

    Generates: load_cbuf_to_ca(dst, src, mStartPosition,
    kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
    """
    args = [dst, src] + [_to_int32(a) for a in [m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose]]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_load_cbuf_to_ca"), *args)


def ascend_load_cbuf_to_cb(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CB (L0B).

    Generates: load_cbuf_to_cb(dst, src, mStartPosition,
    kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
    """
    args = [dst, src] + [_to_int32(a) for a in [m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose]]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_load_cbuf_to_cb"), *args)


def ascend_copy_matrix_cc_to_ub(
    dst,
    src,
    sid,
    n_size,
    m_size,
    loop_dst_stride,
    loop_src_stride,
    dual_dst_ctl,
    sub_blockid,
    clip_relu_pre,
    unit_flag_ctl,
    quant_pre,
    relu_pre,
    split_en,
    NZ2ND_en,
    quant_post,
    relu_post,
    clip_relu_post,
    loop_enhance_en,
    eltwise_op,
    eltwise_antq_en,
    loop_enhance_merge_en,
    C0_pad_en=0,
    wino_post_en=0,
    broadcast_en=0,
    NZ2DN_en=0,
) -> None:
    """Ascend copy matrix from CC (L0C) to UBuf.

    Generates: copy_matrix_cc_to_ub(dst, src, sid, n_size, m_size,
    loop_dst_stride, loop_src_stride, l2, clip_relu_pre, unit_flag_ctl,
    quant_pre, relu_pre, split_en, NZ2ND_en, quant_post, relu_post,
    clip_relu_post, loop_enhance_en, eltwise_op, eltwise_antq_en,
    loop_enhance_merge_en);
    """
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            n_size,
            m_size,
            loop_dst_stride,
            loop_src_stride,
            dual_dst_ctl,
            sub_blockid,
            clip_relu_pre,
            unit_flag_ctl,
            quant_pre,
            relu_pre,
            split_en,
            NZ2ND_en,
            quant_post,
            relu_post,
            clip_relu_post,
            loop_enhance_en,
            eltwise_op,
            eltwise_antq_en,
            loop_enhance_merge_en,
            C0_pad_en,
            wino_post_en,
            broadcast_en,
            NZ2DN_en,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_matrix_cc_to_ub"), *args)


def ascend_copy_ubuf_to_cbuf(dst, src, sub_blockid, burst_num, burst_len, src_gap, dst_gap) -> None:
    """Ascend DMA copy from UBuf to CBuf (L1).

    Generates: copy_ubuf_to_cbuf(dst, src, sub_blockid, burst_num,
    burst_len, src_gap, dst_gap);
    """
    from tvm.tirx import BufferLoad
    from tilelang.language.builtin import access_ptr

    if isinstance(dst, BufferLoad):
        dst = access_ptr(dst, "w")
    if isinstance(src, BufferLoad):
        src = access_ptr(src, "r")
    args = [dst, src] + [_to_int32(a) for a in [sub_blockid, burst_num, burst_len, src_gap, dst_gap]]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_ubuf_to_cbuf"), *args)


def ascend_copy_matrix_cc_to_gm(
    dst,
    src,
    sid,
    n_size,
    m_size,
    loop_dst_stride,
    loop_src_stride,
    l2,
    clip_relu_pre,
    unit_flag_ctl,
    quant_pre,
    relu_pre,
    split_en,
    NZ2ND_en,
    quant_post,
    relu_post,
    clip_relu_post,
    loop_enhance_en,
    eltwise_op,
    eltwise_antq_en,
    loop_enhance_merge_en,
    extra0=0,
    extra1=0,
    extra2=0,
    extra3=0,
) -> None:
    args = [dst, src] + [
        _to_int32(a)
        for a in [
            sid,
            n_size,
            m_size,
            loop_dst_stride,
            loop_src_stride,
            l2,
            clip_relu_pre,
            unit_flag_ctl,
            quant_pre,
            relu_pre,
            split_en,
            NZ2ND_en,
            quant_post,
            relu_post,
            clip_relu_post,
            loop_enhance_en,
            eltwise_op,
            eltwise_antq_en,
            loop_enhance_merge_en,
            extra0,
            extra1,
            extra2,
            extra3,
        ]
    ]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_copy_matrix_cc_to_gm"), *args)


def ascend_mad(dst, src_a, src_b, m, k, n, bias_addr=0, transpose_a=0, transpose_b=0, init_flag=1) -> None:
    args = [dst, src_a, src_b] + [_to_int32(a) for a in [m, k, n, bias_addr, transpose_a, transpose_b, init_flag]]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_mad"), *args)


def ascend_nd2nz_post_copy(dst, src, rows, cols, full_rows, dst_dtype: str) -> None:
    """Ascend ND->NZ post-copy from UB scratch to L1 with NZ pointer correction."""
    from tvm.tirx import BufferLoad
    from tilelang.language.builtin import access_ptr

    if isinstance(dst, BufferLoad):
        dst = access_ptr(dst, "w")
    if isinstance(src, BufferLoad):
        src = access_ptr(src, "r")
    args = [dst, src, _to_int32(rows), _to_int32(cols), _to_int32(full_rows), dst_dtype]
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_nd2nz_post_copy"), *args)


def ascend_get_buf(pipe, buf_id, mode=False):
    """Acquire a pipe buffer for double-buffer management.

    Generates: get_buf(pipe, buf_id, mode);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
    buf_id : int or PrimExpr
        Buffer slot index, typically (sk & 1) for double-buffering.
    mode : bool
        Optional mode flag (default False).

    Example
    -------
    >>> T.ascend_get_buf("PIPE_MTE1", sk & 1)
    """
    if isinstance(buf_id, int):
        buf_id = tirx.IntImm("int32", buf_id)
    mode_val = tirx.IntImm("int32", 1 if mode else 0)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_get_buf"), pipe, buf_id, mode_val)


def ascend_rls_buf(pipe, buf_id, mode=False):
    """Release a pipe buffer for double-buffer management.

    Generates: rls_buf(pipe, buf_id, mode);

    Parameters
    ----------
    pipe : str
        Pipeline type, e.g., "PIPE_MTE1", "PIPE_M".
    buf_id : int or PrimExpr
        Buffer slot index, typically (sk & 1) for double-buffering.
    mode : bool
        Optional mode flag (default False).

    Example
    -------
    >>> T.ascend_rls_buf("PIPE_M", sk & 1)
    """
    if isinstance(buf_id, int):
        buf_id = tirx.IntImm("int32", buf_id)
    mode_val = tirx.IntImm("int32", 1 if mode else 0)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.ascend_rls_buf"), pipe, buf_id, mode_val)


def set_hf32_mode(mode=None):
    """Set HF32 mode for Ascend Cube fp32 matmul.

    Controls whether fp32 inputs to the Cube unit are truncated to
    HF32 (19-bit mantissa) before multiplication, trading precision for
    throughput (~2x).  Ignored for non-fp32 input types.

    Generates: AscendC::SetHF32Mode(HF32Mode::DISABLE/ENABLE) and
               AscendC::SetHF32TransMode(HF32TransMode::NEAREST_ZERO/NEAREST_EVEN).

    Parameters
    ----------
    mode : None | "nearest_zero" | "nearest_even"
        - None (default): disable HF32, full fp32 precision.
        - "nearest_zero": enable HF32, round toward zero.
        - "nearest_even": enable HF32, round to nearest even.

    Example
    -------
    >>> T.set_hf32_mode("nearest_even")   # enable HF32
    >>> T.set_hf32_mode(None)             # restore full fp32
    """
    _HF32_MODE_INT = {None: 0, "nearest_zero": 1, "nearest_even": 2}
    if mode not in _HF32_MODE_INT:
        raise ValueError(f"hf32 mode must be None, 'nearest_zero', or 'nearest_even', got {mode!r}")
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.ascend_set_hf32_mode"),
        tirx.IntImm("int32", _HF32_MODE_INT[mode]),
    )


__all__ = [
    "ascend_pipe_barrier",
    "ascend_set_flag",
    "ascend_wait_flag",
    "ascend_threadfence",
    "ascend_reduce_add",
    "ascend_reduce_max",
    "ascend_reduce_min",
    "ascend_cross_core_set_flag",
    "ascend_cross_core_wait_flag",
    "ascend_copy_gm_to_ubuf",
    "ascend_copy_ubuf_to_gm",
    "ascend_copy_gm_to_cbuf",
    "ascend_load_cbuf_to_ca",
    "ascend_load_cbuf_to_cb",
    "ascend_copy_matrix_cc_to_ub",
    "ascend_copy_ubuf_to_cbuf",
    "ascend_copy_matrix_cc_to_gm",
    "ascend_mad",
    "ascend_nd2nz_post_copy",
    "ascend_get_buf",
    "ascend_rls_buf",
    "set_hf32_mode",
]
