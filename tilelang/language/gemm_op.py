"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_offset,
    prim_expr_equal,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)
from tilelang.env import env as _env


def _gemm_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
    block_scaled: bool = False,
    sfa: BufferLikeType | None = None,
    sfb: BufferLikeType | None = None,
    sf_k_within_chunk: int = 0,
) -> tir.PrimExpr:
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key.
    """

    def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    mbar = legalize_arguments(mbar) if mbar is not None else None
    sfa = legalize_arguments(sfa) if sfa is not None else None
    sfb = legalize_arguments(sfb) if sfb is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    if mbar is not None:
        assert isinstance(mbar, (tir.Buffer, tir.BufferLoad)), (
            f"mbar for tcgen5mma must be a tir.Buffer or tir.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")
    C_coords = [r.min for r in C_region.region]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "w" if isinstance(clear_accum, bool) and clear_accum else "rw", [r for r in C_shape])
    # When mbar is None, pass a placeholder constant (0).
    # The C++ side checks if arg 16 is a BufferLoadNode before using it,
    # so a non-BufferLoad value will be correctly ignored.
    mbar_arg = mbar if mbar is not None else tir.const(0, dtype="int32")
    if sfa is not None:
        sfa_region = to_buffer_region(sfa, access_type="r")
        sfa_arg = buffer_region_to_tile_region(sfa_region, "r", retrieve_shape(sfa_region))
    else:
        sfa_arg = tir.const(0, dtype="int32")
    if sfb is not None:
        sfb_region = to_buffer_region(sfb, access_type="r")
        sfb_arg = buffer_region_to_tile_region(sfb_region, "r", retrieve_shape(sfb_region))
    else:
        sfb_arg = tir.const(0, dtype="int32")
    block_scaled_arg = tir.IntImm("int32", 1 if block_scaled else 0)
    sf_k_within_chunk_arg = tir.IntImm("int32", sf_k_within_chunk) if isinstance(sf_k_within_chunk, int) else sf_k_within_chunk
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_key),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
        mbar_arg,
        C_coords[0],
        C_coords[1],
        block_scaled_arg,
        sfa_arg,
        sfb_arg,
        sf_k_within_chunk_arg,
    )


# Public wrappers
def gemm_v1(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v1: use op tl.gemm."""
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# experimental currently, for fast compilation
def gemm_v2(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v2: use op tl.gemm_py."""
    return _gemm_impl(
        "tl.tileop.gemm_py",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# Default to v2; allow forcing v1 via environment variable
# gemm = gemm_v1 if _env.use_gemm_v1() else gemm_v2


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """TileLang GEMM operator.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        k_pack (int): Numbers of packed matrix cores, for ROCm only. Defaults to 1.
        wg_wait (int): Int identifier of the warpgroup MMA batch to wait on.. Defaults to 0.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell. Defaults to None.

    Returns:
        tir.Call: A handle to the GEMM operation.
    """

    impl = gemm_v1 if _env.use_gemm_v1() else gemm_v2
    return impl(A, B, C, transpose_A, transpose_B, policy, clear_accum, k_pack, wg_wait, mbar)


def blockscaled_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    sfa: BufferLikeType,
    sfb: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
    sf_k_within_chunk: int = 0,
) -> tir.PrimExpr:
    """Block-scaled (MXFP8) GEMM operator.

    Args:
        A (BufferLikeType): Input buffer A (FP8 data).
        B (BufferLikeType): Input buffer B (FP8 data).
        C (BufferLikeType): Output buffer C.
        sfa (BufferLikeType): Scale factor buffer for A.
        sfb (BufferLikeType): Scale factor buffer for B.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Always True on Ascend.
        clear_accum (bool): Whether to clear the accumulator.
        sf_k_within_chunk (int): Which SF chunk within the loaded chunk group. Defaults to 0.

    Returns:
        tir.Call: A handle to the block-scaled GEMM operation.
    """
    op_key = "tl.tileop.gemm" if _env.use_gemm_v1() else "tl.tileop.gemm_py"
    return _gemm_impl(
        op_key,
        A,
        B,
        C,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        policy=GemmWarpPolicy.Square,
        clear_accum=clear_accum,
        k_pack=1,
        wg_wait=0,
        mbar=None,
        block_scaled=True,
        sfa=sfa,
        sfb=sfb,
        sf_k_within_chunk=sf_k_within_chunk,
    )
