from __future__ import annotations
from tvm import tirx
from tvm.tirx import PyStmtExprVisitor, PrimFunc, BufferStore
from tvm.tirx.transform import prim_func_pass


_VF_ATTR_KEYS = ("tl.simdvf_scope", "tl.simtvf_scope")


def _is_direct_vf_wrapper(stmt: tirx.Stmt) -> bool:
    """Check if *stmt* consists entirely of ``Allocate``/``DeclBuffer``
    wrappers around a VF ``AttrStmt``, with no intervening non-allocation
    statements."""
    s = stmt
    while isinstance(s, (tirx.Allocate, tirx.DeclBuffer)):
        if hasattr(s, "body"):
            s = s.body
        else:
            return False
    return isinstance(s, tirx.AttrStmt) and s.attr_key in _VF_ATTR_KEYS


@tirx.functor.visitor
class _VFLocalVarCheckVisitor(PyStmtExprVisitor):
    """Check that local.var buffers defined outside a VF block are not
    modified inside it.

    After ``LowerOpaqueBlock``, VF-local buffers appear as
    ``Allocate``/``DeclBuffer`` nodes that directly wrap the VF
    ``AttrStmt`` (with no other statements in between).  This visitor
    detects those boundaries and enters the VF scope at that level, so
    that all VF-local ``local.var`` buffers are correctly counted.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inside_vf = False
        self.vf_declared_vars: set[tirx.Var] = set()

    def visit_decl_buffer_(self, op: tirx.DeclBuffer) -> None:
        if op.buffer.scope() == "local.var" and _is_direct_vf_wrapper(op.body):
            # This DeclBuffer is part of an alloc chain that directly wraps a
            # VF AttrStmt. Only the outermost such DeclBuffer should open a new
            # VF scope; inner ones must extend the existing scope so that all
            # buffers in the chain are tracked as VF-local.
            if not self.inside_vf:
                was_inside = self.inside_vf
                saved_vars = self.vf_declared_vars

                self.inside_vf = True
                self.vf_declared_vars = set()
                self.vf_declared_vars.add(op.buffer.data)

                self.visit_stmt(op.body)

                self.inside_vf = was_inside
                self.vf_declared_vars = saved_vars
            else:
                self.vf_declared_vars.add(op.buffer.data)
                self.visit_stmt(op.body)
            return

        if self.inside_vf and op.buffer.scope() == "local.var":
            self.vf_declared_vars.add(op.buffer.data)

        self.visit_stmt(op.body)

    def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
        if not self.inside_vf and op.attr_key in _VF_ATTR_KEYS:
            # VF block with no alloc_buffers — enter scope at the AttrStmt.
            was_inside = self.inside_vf
            saved_vars = self.vf_declared_vars

            self.inside_vf = True
            self.vf_declared_vars = set()

            self.visit_stmt(op.body)

            self.inside_vf = was_inside
            self.vf_declared_vars = saved_vars
        else:
            self.visit_stmt(op.body)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if self.inside_vf and op.buffer.scope() == "local.var" and op.buffer.data not in self.vf_declared_vars:
            raise ValueError(
                "[TileLang Semantic Check] "
                f"local.var buffer '{op.buffer}' is modified "
                "inside a SIMD_VF/SIMT_VF block but is defined outside the VF block. "
                "This is not allowed because the variable would need pointer-type capture."
            )


def VFLocalVarChecker():
    """Check that local.var buffers are not modified across VF block boundaries.

    After ``LowerOpaqueBlock``, VF blocks are identified by ``AttrStmt``
    nodes with ``tl.simdvf_scope`` / ``tl.simtvf_scope``.  VF-local
    ``local.var`` buffers appear as ``Allocate``/``DeclBuffer`` nodes
    that directly wrap those ``AttrStmt`` nodes.

    This pass ensures that any ``BufferStore`` to a ``local.var`` buffer
    inside a VF block targets a buffer that was declared inside that
    same VF block.

    Returns:
        A ``prim_func_pass`` that performs the validation.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _VFLocalVarCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
