"""Normalize MixedKernel sid after autoschedule.

When T.MixedKernel is used, the kernel frame injects an outer cthread binding
(AttrStmt(sid_outer, thread_extent/cthread, 2, ...)) wrapping the entire kernel
body.  After autoschedule, the VECTOR block does not yet have its own cthread.
This pass strips the outer cthread and inserts a new cthread binding wrapping
the VECTOR block body, reusing the outer sid var so that user-space references
to sid resolve correctly.
"""

from tvm.tir import IterVar, AttrStmt, Block
from tvm.tir.stmt_functor import ir_transform
from tvm.tir.transform import prim_func_pass


def NormalizeMixedKernelSid():
    """Normalize MixedKernel sid after autoschedule.

    If an outer cthread AttrStmt from T.MixedKernel is found outside
    tilelang_root, strip it and insert a new cthread binding inside the
    VECTOR block using the same sid var.  No-op for regular T.Kernel.
    """

    def pass_fn(func, mod, ctx):
        cthread_stack = []
        inside_vector = 0

        def pre_visit(stmt):
            nonlocal inside_vector
            if isinstance(stmt, Block) and stmt.name_hint == "VECTOR":
                inside_vector += 1
            if isinstance(stmt, AttrStmt) and stmt.attr_key == "thread_extent" and not inside_vector:
                iv = stmt.node
                if iv.thread_tag == "cthread":
                    cthread_stack.append((iv.var, iv.dom, iv.iter_type, iv.thread_tag))
            return None

        def post_visit(stmt):
            nonlocal inside_vector
            if isinstance(stmt, Block) and stmt.name_hint == "VECTOR":
                inside_vector -= 1
                if cthread_stack:
                    var, dom, iter_type, tag = cthread_stack[-1]
                    new_iv = IterVar(dom, var, iter_type, tag)
                    wrapped_body = AttrStmt(new_iv, "thread_extent", dom.extent, stmt.body)
                    return Block(
                        iter_vars=stmt.iter_vars,
                        reads=stmt.reads,
                        writes=stmt.writes,
                        name_hint=stmt.name_hint,
                        body=wrapped_body,
                        init=stmt.init,
                        alloc_buffers=stmt.alloc_buffers,
                        annotations=stmt.annotations,
                    )

            if isinstance(stmt, AttrStmt) and stmt.attr_key == "thread_extent":
                iv = stmt.node
                if iv.thread_tag == "cthread" and cthread_stack and cthread_stack[-1][0].same_as(iv.var):
                    cthread_stack.pop()
                    return stmt.body

            return stmt

        new_body = ir_transform(func.body, pre_visit, post_visit)
        if new_body == func.body:
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
