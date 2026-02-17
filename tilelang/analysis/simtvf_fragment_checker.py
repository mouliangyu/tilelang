from __future__ import annotations

from tvm import tir
from tvm.tir import Buffer, BufferLoad, BufferStore, For, PrimFunc, PyStmtExprVisitor
from tvm.tir.transform import prim_func_pass


def _is_fragment_buffer(buf: Buffer) -> bool:
    return buf.scope().startswith("local.fragment")


def _is_simtvf_loop(op: For) -> bool:
    return "tl.simt_vf_threads" in op.annotations


@tir.functor.visitor
class _SimtVFFragmentNarrowVisitor(PyStmtExprVisitor):
    """Check all local.fragment buffers are defined and accessed inside SimtVF."""

    def __init__(self) -> None:
        super().__init__()
        self._simtvf_depth = 0

    def _in_simtvf(self) -> bool:
        return self._simtvf_depth > 0

    def _check_fragment_in_simtvf(self, buf: Buffer, action: str) -> None:
        if not _is_fragment_buffer(buf):
            return
        if self._in_simtvf():
            return
        raise ValueError(
            "[Tilelang Semantic Check] "
            f"Buffer {buf.name} in scope {buf.scope()} is {action} outside any T.SimtVF region. "
            "On Ascend path, all local.fragment buffers must be narrowed and defined inside T.SimtVF."
        )

    def visit_for_(self, op: For) -> None:
        is_simtvf = _is_simtvf_loop(op)
        if is_simtvf:
            self._simtvf_depth += 1
        self.visit_stmt(op.body)
        if is_simtvf:
            self._simtvf_depth -= 1

    def visit_block_(self, op: tir.Block) -> None:
        for buf in op.alloc_buffers:
            self._check_fragment_in_simtvf(buf, action="defined")
        for match_buffer in op.match_buffers:
            self._check_fragment_in_simtvf(match_buffer.buffer, action="defined")
        super().visit_block_(op)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        self._check_fragment_in_simtvf(op.buffer, action="accessed")
        super().visit_buffer_load_(op)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        self._check_fragment_in_simtvf(op.buffer, action="accessed")
        super().visit_buffer_store_(op)


def SimtVFFragmentNarrowChecker():
    """Validate local.fragment buffers are narrowed into SimtVF region."""

    def pass_fn(func: PrimFunc, mod, ctx):
        _SimtVFFragmentNarrowVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
