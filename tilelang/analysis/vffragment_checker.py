from __future__ import annotations

from collections import defaultdict

from tvm import tir
from tvm.tir import Buffer, BufferLoad, BufferStore, For, PrimFunc, PyStmtExprVisitor, Var
from tvm.tir.transform import prim_func_pass


def _is_fragment_layout(layout) -> bool:
    cls = layout.__class__
    return cls.__name__ == "Fragment" and cls.__module__.startswith("tilelang.layout")


def _is_vf_block(op: tir.Block) -> bool:
    return op.name_hint == "SIMT_VF" or op.name_hint == "SIMD_VF"


@tir.functor.visitor
class _VFFragmentNarrowVisitor(PyStmtExprVisitor):
    """Check all local.fragment buffers are defined and accessed inside VF (SimtVF or SimdVF)."""

    def __init__(self) -> None:
        super().__init__()
        self._vf_depth = False
        self._fragment_data_var_depth = defaultdict(int)

    def _in_simtvf(self) -> bool:
        return self._vf_depth

    def _mark_fragment_buffer(self, data: Var) -> None:
        self._fragment_data_var_depth[data] += 1

    def _unmark_fragment_buffer(self, data: Var) -> None:
        depth = self._fragment_data_var_depth.get(data, 0)
        if depth <= 1:
            self._fragment_data_var_depth.pop(data, None)
        else:
            self._fragment_data_var_depth[data] = depth - 1

    def _is_known_fragment_buffer(self, buf: Buffer) -> bool:
        return bool(self._fragment_data_var_depth.get(buf.data, 0))

    def _is_fragment_buffer(self, buf: Buffer) -> bool:
        if buf.scope().startswith("local.fragment"):
            return True
        return self._is_known_fragment_buffer(buf)

    def _check_fragment_in_simtvf(self, buf: Buffer, action: str) -> None:
        if not self._is_fragment_buffer(buf):
            return
        if self._in_simtvf():
            return
        raise ValueError(
            "[Tilelang Semantic Check] "
            f"Buffer {buf.name} in scope {buf.scope()} is {action} outside any VF region. "
            "On Ascend path, all local.fragment buffers must be defined inside T.SimtVF or T.SimdVF."
        )

    def _extract_fragment_layout_buffers(self, op: tir.Block) -> list[Buffer]:
        if op.annotations is None or "layout_map" not in op.annotations:
            return []
        layout_map = op.annotations["layout_map"]
        fragment_buffers: list[Buffer] = []
        for buf, layout in layout_map.items():
            if _is_fragment_layout(layout):
                fragment_buffers.append(buf)
        return fragment_buffers

    def visit_for_(self, op: For) -> None:
        self.visit_stmt(op.body)

    def visit_block_(self, op: tir.Block) -> None:
        is_simtvf = _is_vf_block(op)
        if is_simtvf:
            self._vf_depth = True

        fragment_layout_buffers = self._extract_fragment_layout_buffers(op)
        for buf in fragment_layout_buffers:
            self._mark_fragment_buffer(buf.data)

        for buf in op.alloc_buffers:
            self._check_fragment_in_simtvf(buf, action="defined")
        for match_buffer in op.match_buffers:
            self._check_fragment_in_simtvf(match_buffer.buffer, action="defined")
        super().visit_block_(op)

        for buf in reversed(fragment_layout_buffers):
            self._unmark_fragment_buffer(buf.data)

        if is_simtvf:
            self._vf_depth = False

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        self._check_fragment_in_simtvf(op.buffer, action="accessed")
        super().visit_buffer_load_(op)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        self._check_fragment_in_simtvf(op.buffer, action="accessed")
        super().visit_buffer_store_(op)


def VFFragmentNarrowChecker():
    """Validate local.fragment buffers are narrowed into VF region (SimtVF or SimdVF)."""

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFFragmentNarrowVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)

