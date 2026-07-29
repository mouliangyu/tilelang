"""Ascend block-scaled (MXFP8) GEMM."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.tileop.base import GemmWarpPolicy
from tvm import tirx

__all__ = ["blockscaled_gemm"]


def blockscaled_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    sfa: BufferLikeType | None = None,
    sfb: BufferLikeType | None = None,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
    unit_flag_ctrl: int = 0,
) -> tirx.PrimExpr:
    """Ascend block-scaled MXFP8 GEMM.

    Scale buffers are required for L1 A/B inputs. For L0 A/B inputs the scales
    are expected to have been loaded by the preceding ``T.copy(..., scale=...)``.
    """

    ann = {"unit_flag_ctrl": unit_flag_ctrl, "blockscaled": 1}
    return _gemm_impl(
        "tl.tileop.gemm",
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
        sfa=sfa,
        sfb=sfb,
        annotations=ann,
    )
