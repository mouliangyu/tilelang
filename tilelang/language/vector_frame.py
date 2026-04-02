"""Vector (AIV) execution region for Ascend NPU."""

from __future__ import annotations

from tvm.ffi import register_object
from tvm.script.ir_builder.tir.frame import TIRFrame
from tilelang import _ffi_api


@register_object("tl.VectorFrame")
class VectorFrame(TIRFrame):
    """Frame for Vector (AIV) execution region.

    In Mix mode, provides a subblock ID variable via `as` binding:
        with T.Vector(vector=2) as sid:
            # sid = get_subblockid()
    """

    def __enter__(self):
        super().__enter__()
        return self.sid_var


def Vector(vector: int = 2) -> VectorFrame:
    """Construct a Vector (AIV) execution region frame.

    Parameters
    ----------
    vector : int
        Number of AIV cores (1 or 2). Default 2.

    Returns
    -------
    res : VectorFrame
    """
    return _ffi_api.Vector(vector)
