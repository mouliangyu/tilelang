"""Hoist duplicate vintlv/vdintlv calls inside pair_get into shared LetStmt bindings.

::

    # Before:
    let a = pair_get(vintlv(x, y), 0) in
      let b = pair_get(vintlv(x, y), 1) in body

    # After:
    let _p = vintlv(x, y) in
      let a = pair_get(_p, 0) in
        let b = pair_get(_p, 1) in body
"""

from collections import defaultdict

from tvm import tir
from tvm.tir import (
    Call,
    LetStmt,
    PyStmtExprMutator,
    Var,
)
from tvm.tir.transform import prim_func_pass


def _scan_pair_uses(body):
    """Collect all pair_get(vintlv/vdintlv, N) expressions and group by the
    pair call node, returning {call_ptr: [(index, var_name, let_node), ...]}."""
    pairs = defaultdict(list)

    def visitor(node):
        if isinstance(node, LetStmt):
            value = node.value
            if isinstance(value, Call) and value.op.name == "tl.simd.pair_get":
                pair_arg = value.args[0]
                if isinstance(pair_arg, Call) and pair_arg.op.name in (
                    "tl.simd.vintlv",
                    "tl.simd.vdintlv",
                ):
                    idx = int(value.args[1])
                    pairs[pair_arg].append((idx, node.var, node))

    tir.stmt_functor.post_order_visit(body, visitor)
    return pairs


@tir.functor.mutator
class HoistSimdPairsMutator(PyStmtExprMutator):
    def __init__(self, hoist_map):
        super().__init__()
        self._hoist_map = hoist_map  # {orig_pair_call: (pair_var, first_idx)}

    def visit_let_stmt_(self, op: LetStmt):
        new_body = self.visit_stmt(op.body)

        value = op.value
        if isinstance(value, Call) and value.op.name == "tl.simd.pair_get":
            pair_arg = value.args[0]
            if isinstance(pair_arg, Call) and pair_arg.op.name in (
                "tl.simd.vintlv",
                "tl.simd.vdintlv",
            ):
                info = self._hoist_map.get(pair_arg)
                if info is not None:
                    pair_var, first_idx = info
                    idx = int(value.args[1])
                    # Replace pair arg with Var
                    new_value = Call(value.dtype, value.op, [pair_var, value.args[1]])
                    if idx == first_idx:
                        # This is the first occurrence — insert pair LetStmt
                        inner = LetStmt(op.var, new_value, new_body)
                        return LetStmt(pair_var, pair_arg, inner)
                    else:
                        # Subsequent occurrence — just replace
                        if new_body.same_as(op.body) and new_value.same_as(value):
                            return op
                        return LetStmt(op.var, new_value, new_body)

        new_value = self.visit_expr(value)
        if new_body.same_as(op.body) and new_value.same_as(value):
            return op
        return LetStmt(op.var, new_value, new_body)


def HoistSimdPairs():
    """Hoist duplicate vintlv/vdintlv calls in pair_get into shared LetStmt bindings."""

    def pass_fn(func, mod, ctx):
        # Phase 1: scan for duplicate pair calls
        pair_uses = _scan_pair_uses(func.body)
        hoist_map = {}
        counter = 0
        for pair_call, occurrences in pair_uses.items():
            if len(occurrences) >= 2:
                pair_var = Var(f"vintlv_{counter}", pair_call.dtype)
                first_idx = min(idx for idx, _, _ in occurrences)
                hoist_map[pair_call] = (pair_var, first_idx)
                counter += 1
        if not hoist_map:
            return func
        # Phase 2: rewrite
        mutator = HoistSimdPairsMutator(hoist_map)
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
