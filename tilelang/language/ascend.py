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
    return tir.call_intrin("float32", tir.op.Op.get("tl.ascend_reduce_add"), val)


def ascend_reduce_max(val):
    """Ascend SIMT warp-level reduce max.

    Generates: asc_reduce_max(val)
    """
    return tir.call_intrin("float32", tir.op.Op.get("tl.ascend_reduce_max"), val)


def ascend_reduce_min(val):
    """Ascend SIMT warp-level reduce min.

    Generates: asc_reduce_min(val)
    """
    return tir.call_intrin("float32", tir.op.Op.get("tl.ascend_reduce_min"), val)


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

    Generates: copy_gm_to_cbuf_multi_nd2nz(dst, src, sid,
    loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en);
    """
    args = [dst, src] + [_to_int32(a) for a in [sid, loop1_src_stride, l2_cache_ctrl, n_value, d_value, loop4_src_stride, smallc0_en]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_copy_gm_to_cbuf"), *args)


def ascend_load_cbuf_to_ca(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CA (L0A).

    Generates: load_cbuf_to_ca(dst, src, mStartPosition,
    kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
    """
    args = [dst, src] + [_to_int32(a) for a in [m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose]]
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_load_cbuf_to_ca"), *args)


def ascend_load_cbuf_to_cb(dst, src, m_start, k_start, m_step, k_step, src_stride, dst_stride, transpose) -> None:
    """Ascend load from CBuf (L1) to CB (L0B).

    Generates: load_cbuf_to_cb(dst, src, mStartPosition,
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


def ascend_simd_add(dst, src0, src1, mask) -> None:
    """Ascend SimdVF MicroAPI element-wise addition.

    Low-level op emitted by AscendLowerParallelToVector pass.
    Generates: MicroAPI::Add(dst, src0, src1, mask)
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_add"), dst, src0, src1, mask)


def ascend_simd_sub(dst, src0, src1, mask) -> None:
    """Ascend SimdVF MicroAPI element-wise subtraction.

    Low-level op emitted by AscendLowerParallelToVector pass.
    Generates: MicroAPI::Sub(dst, src0, src1, mask)
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_sub"), dst, src0, src1, mask)


def ascend_simd_mul(dst, src0, src1, mask) -> None:
    """Ascend SimdVF MicroAPI element-wise multiplication.

    Low-level op emitted by AscendLowerParallelToVector pass.
    Generates: MicroAPI::Mul(dst, src0, src1, mask)
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_mul"), dst, src0, src1, mask)


def ascend_simd_div(dst, src0, src1, mask) -> None:
    """MicroAPI::Div(dst, src0, src1, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_div"), dst, src0, src1, mask)


def ascend_simd_max(dst, src0, src1, mask) -> None:
    """MicroAPI::Max(dst, src0, src1, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_max"), dst, src0, src1, mask)


def ascend_simd_min(dst, src0, src1, mask) -> None:
    """MicroAPI::Min(dst, src0, src1, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_min"), dst, src0, src1, mask)


def ascend_simd_reduce_max(dst, src, mask) -> None:
    """MicroAPI::Reduce<ReduceType::MAX>(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_reduce_max"), dst, src, mask)


def ascend_simd_reduce_sum(dst, src, mask) -> None:
    """MicroAPI::Reduce<ReduceType::SUM>(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_reduce_sum"), dst, src, mask)


def ascend_simd_exp(dst, src, mask) -> None:
    """MicroAPI::Exp(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_exp"), dst, src, mask)


def ascend_simd_ln(dst, src, mask) -> None:
    """MicroAPI::Ln(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_ln"), dst, src, mask)


def ascend_simd_sqrt(dst, src, mask) -> None:
    """MicroAPI::Sqrt(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_sqrt"), dst, src, mask)


def ascend_simd_rsqrt(dst, src, mask) -> None:
    """MicroAPI::Rsqrt(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_rsqrt"), dst, src, mask)


def ascend_simd_abs(dst, src, mask) -> None:
    """MicroAPI::Abs(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_abs"), dst, src, mask)


def ascend_simd_neg(dst, src, mask) -> None:
    """MicroAPI::Neg(dst, src, mask)"""
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_neg"), dst, src, mask)


def ascend_simd_duplicate(dst, src, mask) -> None:
    """Ascend SimdVF MicroAPI Duplicate: broadcast lowest element of src to all lanes.

    Low-level op emitted by AscendLowerParallelToVector pass.
    Generates: AscendC::MicroAPI::Duplicate<dtype, LOWEST>(dst, src, mask)

    Parameters
    ----------
    dst : Var
        Destination fragment register (data field).
    src : Var
        Source fragment register whose lowest lane is broadcast.
    mask : PrimExpr
        Mask register controlling active lanes.
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_duplicate"), dst, src, mask)


def ascend_simd_dup(dst, scalar) -> None:
    """Ascend SimdVF MicroAPI Duplicate (scalar): broadcast scalar to all lanes.

    Generates: AscendC::MicroAPI::Duplicate(dst, scalar)

    Parameters
    ----------
    dst : Var
        Destination fragment register (data field).
    scalar : PrimExpr
        Scalar value to broadcast to all lanes.
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_dup"), dst, scalar)


def ascend_simd_cast(dst, src, mask, to_dtype="bfloat16_t", from_dtype="float") -> None:
    """Ascend SimdVF MicroAPI Cast: type convert between fragment registers.

    Generates: AscendC::MicroAPI::Cast<to_dtype, from_dtype, castTrait>(dst, src, mask)

    Parameters
    ----------
    dst : Var
        Destination fragment register.
    src : Var
        Source fragment register.
    mask : PrimExpr
        Mask register controlling active lanes.
    to_dtype : str
        Destination C++ type (e.g. "bfloat16_t").
    from_dtype : str
        Source C++ type (e.g. "float").
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_cast"), to_dtype, from_dtype, dst, src, mask)


def ascend_simd_create_addr_reg(*args, dtype="float"):
    """MicroAPI::CreateAddrReg<dtype>(idx0, stride0, idx1, stride1, ...)

    Returns an opaque AddrReg handle for multi-dimensional addressing.
    """
    int_args = []
    for a in args:
        if isinstance(a, int):
            int_args.append(tir.IntImm("int32", a))
        else:
            int_args.append(a)
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_simd_create_addr_reg"), dtype, *int_args)


def ascend_simd_load(dst_frag, src_ptr, addr_reg=None, dtype="float") -> None:
    """Ascend SimdVF MicroAPI LoadAlign: UB memory → RegTensor.

    Generates:
        3-arg: AscendC::MicroAPI::LoadAlign<dtype>(dst, src)
        4-arg: AscendC::MicroAPI::LoadAlign<dtype>(dst, src, addr_reg)
    """
    if addr_reg is not None:
        return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_load"), dtype, dst_frag, src_ptr, addr_reg)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_load"), dtype, dst_frag, src_ptr)


def ascend_simd_load_single(dst_frag, src_ptr, dtype="float") -> None:
    """Ascend SimdVF MicroAPI LoadAlign with DIST_BRC: load single scalar and broadcast to all lanes.

    The load_dist (DIST_BRC_B8/B16/B32) is automatically inferred from dtype bit width:
        float/int32/uint32 → DIST_BRC_B32,  half/int16/uint16 → DIST_BRC_B16,  int8/uint8 → DIST_BRC_B8

    Generates: AscendC::MicroAPI::LoadAlign<dtype, MicroAPI::LoadDist::DIST_BRC_*>(dst, src)

    Parameters
    ----------
    dst_frag : Var
        Destination fragment register (data field).
    src_ptr : PrimExpr
        Source UB pointer (address_of a scalar element).
    dtype : str
        C++ type name for template parameter (e.g. "float", "half_t", "int8_t").
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_load_single"), dtype, dst_frag, src_ptr)


def ascend_simd_store(dst_ptr, src_frag, mask, addr_reg=None, dtype="float") -> None:
    """Ascend SimdVF MicroAPI StoreAlign: RegTensor → UB memory.

    Generates:
        4-arg: AscendC::MicroAPI::StoreAlign<dtype>(dst, src, mask)
        5-arg: AscendC::MicroAPI::StoreAlign<dtype>(dst, src, addr_reg, mask)
    """
    if addr_reg is not None:
        return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_store"), dtype, dst_ptr, src_frag, addr_reg, mask)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_store"), dtype, dst_ptr, src_frag, mask)


def ascend_simd_store_pack(dst_ptr, src_frag, mask, dtype="bfloat16_t") -> None:
    """Ascend SimdVF MicroAPI StoreAlign with DIST_PACK_B32.

    Generates: AscendC::MicroAPI::StoreAlign<dtype, StoreDist::DIST_PACK_B32>(dst, src, mask)

    Used after f32→bf16 cast to compact bf16 values from 32-bit slots into contiguous 16-bit memory.
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_simd_store_pack"), dtype, dst_ptr, src_frag, mask)


def ascend_simd_update_mask(count, dtype="float"):
    """Ascend SimdVF MicroAPI UpdateMask: create mask from element count.

    Generates: AscendC::MicroAPI::UpdateMask<dtype>(count)

    Parameters
    ----------
    count : int or PrimExpr
        A handle returned by ascend_init_mask.
    dtype : str
        C++ type name for template parameter (e.g. "float", "half_t").

    Returns
    -------
    PrimExpr
        Opaque mask value (MaskReg).
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_simd_update_mask"), dtype, count)


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
        buf_id = tir.IntImm("int32", buf_id)
    mode_val = tir.IntImm("int32", 1 if mode else 0)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_get_buf"), pipe, buf_id, mode_val)


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
        buf_id = tir.IntImm("int32", buf_id)
    mode_val = tir.IntImm("int32", 1 if mode else 0)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_rls_buf"), pipe, buf_id, mode_val)


def ascend_init_mask(count):
    """Declare a uint32_t mask count variable (should be placed outside loops).

    Generates: uint32_t var = count;

    The returned handle can be passed as the count argument to
    ascend_simd_update_mask so that the count variable is not
    redeclared inside a loop body.

    Parameters
    ----------
    count : int or PrimExpr
        Number of active elements.

    Returns
    -------
    PrimExpr
        Opaque handle representing the uint32_t count variable.
    """
    if isinstance(count, int):
        count = tir.IntImm("int32", count)
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_init_mask"), count)


def ascend_simd_create_mask(dtype="float"):
    """Create a full mask register with all lanes active.

    Generates: AscendC::MicroAPI::CreateMask<dtype>()

    Parameters
    ----------
    dtype : str
        The data type string for the mask (e.g. "float", "half_t").

    Returns
    -------
    PrimExpr
        Opaque handle representing the MaskReg.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_simd_create_mask"), dtype)


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
    return tir.call_intrin(
        "void",
        tir.op.Op.get("tl.ascend_set_hf32_mode"),
        tir.IntImm("int32", _HF32_MODE_INT[mode]),
    )
