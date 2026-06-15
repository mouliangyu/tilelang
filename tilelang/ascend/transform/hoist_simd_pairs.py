"""Hoist duplicate vintlv/vdintlv pair calls used by pair_get.

::

    # Before:
    vsts(pair_get(vdintlv(pair_get(vintlv(x, y), 0),
                         pair_get(vintlv(x, y), 1)), 0))

    # After:
    vintlv_0 = vintlv(x, y)
    vintlv_1 = vdintlv(pair_get(vintlv_0, 0), pair_get(vintlv_0, 1))
    vsts(pair_get(vintlv_1, 0))
"""

from collections import defaultdict

from tvm import tirx
from tvm.ir import structural_hash
from tvm.tirx import Call, Evaluate, PyStmtExprMutator, SeqStmt, Var
from tvm.tirx.stmt import Bind
from tvm.tirx.transform import prim_func_pass


def _op_name(call_or_op):
    op = getattr(call_or_op, "op", call_or_op)
    return getattr(op, "name", None)


def _is_pair_call(value):
    return isinstance(value, Call) and _op_name(value) in (
        "tl.simd.vintlv",
        "tl.simd.vdintlv",
    )


def _pair_key(value):
    return (structural_hash(value), str(value))


def _scan_pair_uses(body):
    pairs = defaultdict(list)

    def visitor(node):
        if isinstance(node, Call) and _op_name(node) == "tl.simd.pair_get":
            pair_arg = node.args[0]
            if _is_pair_call(pair_arg):
                idx = int(node.args[1])
                pairs[_pair_key(pair_arg)].append((idx, node, pair_arg))

    tirx.stmt_functor.post_order_visit(body, visitor)
    return pairs


@tirx.functor.mutator
class HoistSimdPairsMutator(PyStmtExprMutator):
    def __init__(self, hoist_map):
        super().__init__()
        self._hoist_map = hoist_map
        self._inserted = set()
        self._pending = []

    def _flush_stmt(self, stmt):
        if not self._pending:
            return stmt
        seq = [*self._pending, stmt]
        self._pending = []
        return SeqStmt(seq)

    def visit_seq_stmt_(self, op: SeqStmt):
        seq = []
        for stmt in op.seq:
            new_stmt = self.visit_stmt(stmt)
            if isinstance(new_stmt, SeqStmt):
                seq.extend(new_stmt.seq)
            else:
                seq.append(new_stmt)
        return SeqStmt(seq)

    def visit_bind_(self, op: Bind):
        new_value = self.visit_expr(op.value)
        stmt = op if new_value.same_as(op.value) else Bind(op.var, new_value)
        return self._flush_stmt(stmt)

    def visit_evaluate_(self, op: Evaluate):
        new_value = self.visit_expr(op.value)
        stmt = op if new_value.same_as(op.value) else Evaluate(new_value)
        return self._flush_stmt(stmt)

    def visit_call_(self, op: Call):
        new_args = [self.visit_expr(arg) for arg in op.args]
        if _op_name(op) == "tl.simd.pair_get":
            pair_arg = op.args[0]
            key = _pair_key(pair_arg) if _is_pair_call(pair_arg) else None
            info = self._hoist_map.get(key)
            if info is not None:
                pair_var = info
                if key not in self._inserted:
                    self._pending.append(Bind(pair_var, new_args[0]))
                    self._inserted.add(key)
                new_args[0] = pair_var
        return Call(op.dtype, op.op, new_args)


def HoistSimdPairs():
    """Hoist duplicate vintlv/vdintlv calls referenced through pair_get."""

    def pass_fn(func, mod, ctx):
        pair_uses = _scan_pair_uses(func.body)
        hoist_map = {}
        counter = 0
        for key, occurrences in pair_uses.items():
            if len(occurrences) >= 2:
                pair_call = occurrences[0][2]
                hoist_map[key] = Var(f"vintlv_{counter}", pair_call.dtype)
                counter += 1
        if not hoist_map:
            return func
        mutator = HoistSimdPairsMutator(hoist_map)
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
