from __future__ import annotations

from tvm import tirx
from tvm.tirx import BufferStore, PrimFunc, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")


@tirx.functor.visitor
class _VFLocalVarCheckVisitor(PyStmtExprVisitor):
    """Check that local.var buffers defined outside a VF block are not modified inside it."""

    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False
        self.vf_alloc_vars: set[tirx.Var] = set()

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        if op.name_hint in _VF_BLOCK_NAMES:
            saved_inside = self.inside_vf
            saved_vars = self.vf_alloc_vars
            self.inside_vf = True
            self.vf_alloc_vars = set()
            self.visit_stmt(op.body)
            self.inside_vf = saved_inside
            self.vf_alloc_vars = saved_vars
        else:
            super().visit_sblock_(op)

    def visit_alloc_buffer_(self, op: tirx.AllocBuffer) -> None:
        if self.inside_vf and op.buffer.scope() == "local.var":
            self.vf_alloc_vars.add(op.buffer.data)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if self.inside_vf and op.buffer.scope() == "local.var" and op.buffer.data not in self.vf_alloc_vars:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"local.var buffer '{op.buffer}' is modified "
                "inside a SIMD_VF/SIMT_VF block but is defined outside the VF block. "
                "This is not allowed because the variable would need pointer-type capture."
            )


def VFLocalVarChecker():
    """Check that local.var buffers are not modified across VF block boundaries.

    VF blocks are identified by SBlock ``name_hint`` (``SIMD_VF`` /
    ``SIMT_VF``).  Any ``AllocBuffer(local.var)`` encountered inside the
    VF body (including inside nested For loops) is tracked as VF-local.
    BufferStore to a local.var buffer not tracked as VF-local is an error.

    Returns:
        A ``prim_func_pass`` that performs the validation.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFLocalVarCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
