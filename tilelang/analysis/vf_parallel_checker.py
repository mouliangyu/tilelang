from __future__ import annotations
from tvm import tir
from tvm.tir import For, PrimFunc, PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")


@tir.functor.visitor
class _VFParallelCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False

    def visit_block_(self, op: tir.Block) -> None:
        if op.name_hint in _VF_BLOCK_NAMES:
            was_inside = self.inside_vf
            self.inside_vf = True
            self.visit_stmt(op.body)
            self.inside_vf = was_inside
        else:
            super().visit_block_(op)

    def visit_for_(self, op: For) -> None:
        if op.kind == tir.ForKind.PARALLEL and not self.inside_vf:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.Parallel loop '{op.loop_var.name}' must be inside a VF block "
                "(T.SimdVF() or T.SimtVF()). "
                "Parallel loops outside VF blocks are not supported on Ascend NPU."
            )
        super().visit_for_(op)


def VFParallelChecker():
    """Check that all T.Parallel loops are inside SIMD_VF or SIMT_VF blocks.

    On Ascend NPU, T.Parallel loops must be nested within T.SimdVF() or
    T.SimtVF() blocks, because they are lowered to vector instructions
    only within those scopes.  This pass raises a ``ValueError`` if any
    ``ForKind.kParallel`` loop is found outside a VF block.

    Returns:
        A ``prim_func_pass`` that performs the validation.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFParallelCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
