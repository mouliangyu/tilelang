from __future__ import annotations

from tvm import tirx
from tvm.tirx import BufferLoad, BufferStore, For, PrimFunc, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass

_VF_BLOCK_NAMES = ("SIMD_VF", "SIMT_VF")


@tirx.functor.visitor
class _VFCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False
        self.inside_simd_vf = False

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        if op.name_hint in _VF_BLOCK_NAMES:
            was_inside = self.inside_vf
            was_simd = self.inside_simd_vf
            self.inside_vf = True
            self.inside_simd_vf = op.name_hint == "SIMD_VF"
            self.visit_stmt(op.body)
            self.inside_vf = was_inside
            self.inside_simd_vf = was_simd
        else:
            super().visit_sblock_(op)

    def visit_for_(self, op: For) -> None:
        if op.kind == tirx.ForKind.PARALLEL and not self.inside_vf:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.Parallel loop '{op.loop_var.name}' must be inside a VF block "
                "(T.SimdVF() or T.SimtVF()). "
                "Parallel loops outside VF blocks are not supported on Ascend NPU."
            )
        super().visit_for_(op)

    def _check_copy(self, call: tirx.Call) -> None:
        src_region = call.args[0]
        dst_region = call.args[1]
        if not isinstance(src_region, tirx.Call) or not isinstance(dst_region, tirx.Call):
            return
        src_buf = src_region.args[0].buffer
        dst_buf = dst_region.args[0].buffer

        if not self.inside_vf and src_buf.dtype != dst_buf.dtype:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() from dtype '{src_buf.dtype}' to dtype '{dst_buf.dtype}' "
                "outside a VF block is not allowed. "
                "DMA copies cannot perform type casting — perform the cast "
                "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
            )

        if self.inside_simd_vf and (src_buf.scope() == "global" or dst_buf.scope() == "global"):
            raise ValueError(
                "[TileLang Semantic Check] "
                f"T.copy() involving global buffer '{src_buf.name}' -> '{dst_buf.name}' "
                "inside a SIMD_VF block is not allowed. "
                "SIMD_VF blocks cannot access global memory — move the copy "
                "outside the SIMD_VF block."
            )

    def visit_evaluate_(self, op: tirx.Evaluate) -> None:
        call = op.value
        if isinstance(call, tirx.Call) and call.op == tirx.op.Op.get("tl.tileop.copy"):
            self._check_copy(call)
        super().visit_evaluate_(op)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if not self.inside_vf:
            value = op.value
            if isinstance(value, BufferLoad) and value.buffer.dtype != op.buffer.dtype:
                raise ValueError(
                    "[TileLang Semantic Check] "
                    f"Buffer store from dtype '{value.buffer.dtype}' to dtype '{op.buffer.dtype}' "
                    "outside a VF block implies a type cast. "
                    "DMA copies cannot perform type casting — perform the cast "
                    "inside a VF block (T.SimdVF() or T.SimtVF()) instead."
                )

        if self.inside_simd_vf and op.buffer.scope() == "global":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"BufferStore to global buffer '{op.buffer.name}' inside a SIMD_VF block "
                "is not allowed. SIMD_VF blocks cannot access global memory — move the "
                "store outside the SIMD_VF block."
            )
        super().visit_buffer_store_(op)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        if self.inside_simd_vf and op.buffer.scope() == "global":
            raise ValueError(
                "[TileLang Semantic Check] "
                f"BufferLoad from global buffer '{op.buffer.name}' inside a SIMD_VF block "
                "is not allowed. SIMD_VF blocks cannot access global memory — move the "
                "load outside the SIMD_VF block."
            )


def VFChecker():
    """Check VF block usage rules:
    - All T.Parallel loops must be inside SIMD_VF or SIMT_VF blocks.
    - Copy operations outside VF blocks must have matching source and destination dtypes.
    - SIMD_VF blocks must not access global memory.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
