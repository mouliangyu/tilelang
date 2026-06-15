"""Normalize MixedKernel sid after autoschedule.

When T.MixedKernel is used, the kernel frame injects an outer cthread binding
(AttrStmt(sid_outer, thread_extent/cthread, 2, ...)) wrapping the entire kernel
body.  After autoschedule, the VECTOR block does not yet have its own cthread.
This pass strips the outer cthread and inserts a new cthread binding wrapping
the VECTOR block body, reusing the outer sid var so that user-space references
to sid resolve correctly.
"""

from tvm.tirx import IterVar, AttrStmt, Bind, Evaluate, SBlock, SeqStmt
from tvm.tirx.analysis import undefined_vars
from tvm.tirx.stmt_functor import ir_transform
from tvm.tirx.transform import prim_func_pass


def NormalizeMixedKernelSid():
    """Normalize MixedKernel sid after autoschedule.

    If an outer cthread AttrStmt from T.MixedKernel is found outside
    tilelang_root, strip it and insert a new cthread binding inside the
    VECTOR block using the same sid var.  No-op for regular T.Kernel.
    """

    def pass_fn(func, mod, ctx):
        cthread_stack = []
        cthread_vars = []
        inside_vector = 0

        def _same_var(lhs, rhs):
            return lhs.same_as(rhs)

        def _contains_any_var(node, vars_):
            return any(any(_same_var(undefined, var) for var in vars_) for undefined in undefined_vars(node))

        def _is_used_later(var, stmts):
            if not stmts:
                return False
            tail = stmts[0] if len(stmts) == 1 else SeqStmt(stmts)
            return any(_same_var(undefined, var) for undefined in undefined_vars(tail))

        def drop_unused_cthread_binds(stmt):
            if not cthread_vars:
                return stmt

            def post_drop(node):
                if not isinstance(node, SeqStmt):
                    return node

                new_seq = []
                seq = list(node.seq)
                for i, child in enumerate(seq):
                    if isinstance(child, Bind):
                        depends_on_cthread = _contains_any_var(child.value, cthread_vars)
                        used_later = _is_used_later(child.var, seq[i + 1 :])
                        if depends_on_cthread and not used_later:
                            continue
                    new_seq.append(child)

                if len(new_seq) == len(seq):
                    return node
                if not new_seq:
                    return Evaluate(0)
                if len(new_seq) == 1:
                    return new_seq[0]
                return SeqStmt(new_seq)

            return ir_transform(stmt, None, post_drop)

        def pre_visit(stmt):
            nonlocal inside_vector
            if isinstance(stmt, SBlock) and stmt.name_hint == "VECTOR":
                inside_vector += 1
            if isinstance(stmt, AttrStmt) and stmt.attr_key == "thread_extent" and not inside_vector:
                iv = stmt.node
                if iv.thread_tag == "cthread":
                    cthread_stack.append((iv.var, iv.dom, iv.iter_type, iv.thread_tag))
                    if not any(iv.var.same_as(var) for var in cthread_vars):
                        cthread_vars.append(iv.var)
            return None

        def post_visit(stmt):
            nonlocal inside_vector
            if isinstance(stmt, SBlock) and stmt.name_hint == "VECTOR":
                inside_vector -= 1
                if cthread_stack:
                    var, dom, iter_type, tag = cthread_stack[-1]
                    new_iv = IterVar(dom, var, iter_type, tag)
                    wrapped_body = AttrStmt(new_iv, "thread_extent", dom.extent, stmt.body)
                    return SBlock(
                        stmt.iter_vars,
                        stmt.reads,
                        stmt.writes,
                        stmt.name_hint,
                        wrapped_body,
                        stmt.init,
                        stmt.alloc_buffers,
                        stmt.match_buffers,
                        stmt.annotations,
                    )

            if isinstance(stmt, AttrStmt) and stmt.attr_key == "thread_extent":
                iv = stmt.node
                if iv.thread_tag == "cthread" and cthread_stack and cthread_stack[-1][0].same_as(iv.var):
                    cthread_stack.pop()
                    return stmt.body

            return stmt

        new_body = ir_transform(func.body, pre_visit, post_visit)
        new_body = drop_unused_cthread_binds(new_body)
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
