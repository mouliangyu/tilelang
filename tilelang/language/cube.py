"""Cube (AIC) execution region for Ascend NPU."""

from __future__ import annotations

from tvm.ffi import register_object
from tvm.script.ir_builder.tir.frame import TIRFrame
from tilelang import _ffi_api


@register_object("tl.CubeFrame")
class CubeFrame(TIRFrame):
    """Frame for Cube (AIC) execution region."""

    pass


def Cube() -> CubeFrame:
    """Construct a Cube (AIC) execution region frame.

    Returns
    -------
    res : CubeFrame
    """
    return _ffi_api.Cube()
