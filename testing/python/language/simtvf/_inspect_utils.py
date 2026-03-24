from __future__ import annotations

from tilelang import tvm


def collect_simtvf_blocks(func: tvm.tir.PrimFunc) -> list[tvm.tir.Block]:
    blocks: list[tvm.tir.Block] = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Block) and stmt.name_hint == "SIMT_VF":
            blocks.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return blocks


def print_stage(title: str, mod: tvm.IRModule, func_name: str) -> None:
    print(f"\n===== {title} =====")
    print(mod.script())
    func = mod[func_name]
    blocks = collect_simtvf_blocks(func)
    print(f"[check] simtvf(block has SIMT_VF name_hint) count = {len(blocks)}")
    for idx, block in enumerate(blocks):
        print(f"  block[{idx}] name_hint: {block.name_hint}")


def collect_thread_extent_tags(stmt: tvm.tir.Stmt) -> list[str]:
    tags: list[str] = []

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "thread_extent":
            iv = node.node
            if isinstance(iv, tvm.tir.IterVar):
                tags.append(iv.thread_tag)

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    return tags


def collect_thread_extent_tags_outside_simtvf(stmt: tvm.tir.Stmt) -> list[str]:
    tags: list[str] = []

    def _walk(node: tvm.tir.Stmt, simtvf_depth: int) -> None:
        if isinstance(node, tvm.tir.Block):
            is_simtvf = node.name_hint == "SIMT_VF"
            _walk(node.body, simtvf_depth + (1 if is_simtvf else 0))
            return
        if isinstance(node, tvm.tir.For):
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tir.AttrStmt):
            if simtvf_depth == 0 and node.attr_key == "thread_extent" and isinstance(node.node, tvm.tir.IterVar):
                tags.append(node.node.thread_tag)
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tir.SeqStmt):
            for s in node.seq:
                _walk(s, simtvf_depth)
            return
        if isinstance(node, tvm.tir.BlockRealize):
            _walk(node.block.body, simtvf_depth)
            return
        if isinstance(node, tvm.tir.LetStmt):
            _walk(node.body, simtvf_depth)
            return
        if isinstance(node, tvm.tir.IfThenElse):
            _walk(node.then_case, simtvf_depth)
            if node.else_case is not None:
                _walk(node.else_case, simtvf_depth)
            return
        if isinstance(node, tvm.tir.AssertStmt):
            _walk(node.body, simtvf_depth)
            return

    _walk(stmt, 0)
    return tags


def contains_parallel_for(stmt: tvm.tir.Stmt) -> bool:
    found = False

    def _visit(node):
        nonlocal found
        if isinstance(node, tvm.tir.For) and node.kind == tvm.tir.ForKind.PARALLEL:
            found = True

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    return found


def find_thread_predicate_hoist_violations(func: tvm.tir.PrimFunc) -> list[tvm.tir.PrimExpr]:
    violations: list[tvm.tir.PrimExpr] = []

    def _is_thread_predicate(expr: tvm.tir.PrimExpr) -> bool:
        found = False

        def _visit(node):
            nonlocal found
            if isinstance(node, tvm.tir.Var) and node.name in ("tx", "ty", "tz", "simtvf_tx"):
                found = True

        tvm.tir.stmt_functor.post_order_visit(expr, _visit)
        return found

    def _walk_stmt(stmt: tvm.tir.Stmt, simtvf_depth: int) -> None:
        if isinstance(stmt, tvm.tir.Block):
            is_simtvf = stmt.name_hint == "SIMT_VF"
            _walk_stmt(stmt.body, simtvf_depth + (1 if is_simtvf else 0))
            return
        if isinstance(stmt, tvm.tir.For):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.IfThenElse):
            if simtvf_depth == 0 and _is_thread_predicate(stmt.condition):
                violations.append(stmt.condition)
            _walk_stmt(stmt.then_case, simtvf_depth)
            if stmt.else_case is not None:
                _walk_stmt(stmt.else_case, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.SeqStmt):
            for s in stmt.seq:
                _walk_stmt(s, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.LetStmt):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.AttrStmt):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.AssertStmt):
            _walk_stmt(stmt.body, simtvf_depth)
            return
        if isinstance(stmt, tvm.tir.BlockRealize):
            _walk_stmt(stmt.block.body, simtvf_depth)
            return

    _walk_stmt(func.body, 0)
    return violations
