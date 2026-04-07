"""Ascend-specific language primitives for TileLang."""

from __future__ import annotations

from tvm import tir


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
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_pipe_barrier"), pipe_t)


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
        event_id = tir.IntImm("int32", event_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_set_flag"), hard_event, event_id)


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
        event_id = tir.IntImm("int32", event_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_wait_flag"), hard_event, event_id)


def ascend_threadfence() -> None:
    """Insert an AscendC thread fence.

    Generates: asc_threadfence();

    Example
    -------
    >>> T.ascend_threadfence()
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_threadfence"))


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
        mode_id = tir.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tir.IntImm("int32", flag_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_cross_core_set_flag"), mode_id, pipe, flag_id)


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
        mode_id = tir.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tir.IntImm("int32", flag_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_cross_core_wait_flag"), mode_id, pipe, flag_id)


def ascend_copy_gm_to_ubuf(dst, src, sid, n_burst, len_burst, src_stride, dst_stride) -> None:
    """Insert an Ascend DMA copy from GM to UBuf.

    Generates: copy_gm_to_ubuf(dst, src, sid, nBurst, lenBurst, srcStride, dstStride);

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
    len_burst : int or PrimExpr
        Length of each burst in bytes.
    src_stride : int or PrimExpr
        Source stride between bursts.
    dst_stride : int or PrimExpr
        Destination stride between bursts.
    """
    if isinstance(sid, int):
        sid = tir.IntImm("int32", sid)
    if isinstance(n_burst, int):
        n_burst = tir.IntImm("int32", n_burst)
    if isinstance(len_burst, int):
        len_burst = tir.IntImm("int32", len_burst)
    if isinstance(src_stride, int):
        src_stride = tir.IntImm("int32", src_stride)
    if isinstance(dst_stride, int):
        dst_stride = tir.IntImm("int32", dst_stride)
    return tir.call_intrin(
        "void",
        tir.op.Op.get("tl.ascend_copy_gm_to_ubuf"),
        dst,
        src,
        sid,
        n_burst,
        len_burst,
        src_stride,
        dst_stride,
    )


def ascend_copy_ubuf_to_gm(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride) -> None:
    """Insert an Ascend DMA copy from UBuf to GM.

    Generates: bisheng::cce::copy_ubuf_to_gm_align_v2(dst, src, sid, burst_num, burst_len, l2_cache_ctl, burst_dst_stride, burst_src_stride);

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
    if isinstance(sid, int):
        sid = tir.IntImm("int32", sid)
    if isinstance(burst_num, int):
        burst_num = tir.IntImm("int32", burst_num)
    if isinstance(burst_len, int):
        burst_len = tir.IntImm("int32", burst_len)
    if isinstance(l2_cache_ctl, int):
        l2_cache_ctl = tir.IntImm("int32", l2_cache_ctl)
    if isinstance(burst_dst_stride, int):
        burst_dst_stride = tir.IntImm("int32", burst_dst_stride)
    if isinstance(burst_src_stride, int):
        burst_src_stride = tir.IntImm("int32", burst_src_stride)
    return tir.call_intrin(
        "void",
        tir.op.Op.get("tl.ascend_copy_ubuf_to_gm"),
        dst,
        src,
        sid,
        burst_num,
        burst_len,
        l2_cache_ctl,
        burst_dst_stride,
        burst_src_stride,
    )


def _to_int32(v):
    return tir.IntImm("int32", v) if isinstance(v, int) else v


def ascend_copy_gm_to_cbuf(dst, src, sid, loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en) -> None:
    """Ascend DMA copy from GM to CBuf (L1).

    Generates: bisheng::cce::copy_gm_to_cbuf_multi_nd2nz(dst, src, sid,
    loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en);
    """
    args = [dst, src] + [_to_int32(a) for a in [sid, loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_copy_gm_to_cbuf"), *args)


def ascend_load_cbuf_to_ca(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CA (L0A).

    Generates: bisheng::cce::load_cbuf_to_ca(dst, src, mStartPosition,
    kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
    """
    args = [dst, src] + [_to_int32(a) for a in [m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_load_cbuf_to_ca"), *args)


def ascend_load_cbuf_to_cb(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CB (L0B).

    Generates: bisheng::cce::load_cbuf_to_cb(dst, src, mStartPosition,
    kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
    """
    args = [dst, src] + [_to_int32(a) for a in [m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_load_cbuf_to_cb"), *args)


def ascend_copy_matrix_cc_to_ub(
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
        ]
    ]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_copy_matrix_cc_to_ub"), *args)


def ascend_copy_ubuf_to_cbuf(dst, src, sub_blockid, burst_num, burst_len, src_gap, dst_gap) -> None:
    """Ascend DMA copy from UBuf to CBuf (L1).

    Generates: copy_ubuf_to_cbuf(dst, src, sub_blockid, burst_num,
    burst_len, src_gap, dst_gap);
    """
    args = [dst, src] + [_to_int32(a) for a in [sub_blockid, burst_num, burst_len, src_gap, dst_gap]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_copy_ubuf_to_cbuf"), *args)
