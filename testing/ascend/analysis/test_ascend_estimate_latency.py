import pytest

import tilelang.ascend.transform as ascend_transform
import tilelang.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit


def _make_program(latency=None, ii=None):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(latency=latency, ii=ii):
                T.copy(A, temp)
            T.copy(temp, B)

    return main


def _make_guarded_program():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2) as bx:
            temp = T.alloc_shared((16,), "float32")
            if bx == 0:
                T.copy(A, temp)
            else:
                T.copy(temp, B)

    return main


def _bind_target(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    return tirx.transform.BindTarget(determine_target("ascend"))(mod)


def _materialize_schedule_units(mod):
    return ascend_transform.MaterializeScheduleUnits()(mod)


def _collect_task_metadata(mod):
    metadata = {"latency": [], "ii": [], "core_mask": []}

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_task":
            for key in metadata:
                if key in node.node:
                    metadata[key].append(int(node.node[key]))

    post_order_visit(mod["main"].body, visit)
    return metadata


def _collect_schedule_units(mod):
    units = []

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            units.append(node)

    post_order_visit(mod["main"].body, visit)
    return units


def _duplicate_one_task_schedule_unit(mod):
    duplicated = False

    def rewrite(node):
        nonlocal duplicated
        if duplicated or not isinstance(node, tirx.AttrStmt) or node.attr_key != "tl.schedule_unit":
            return None
        if not isinstance(node.body, tirx.AttrStmt) or node.body.attr_key != "tl.ascend_task":
            return None
        duplicated = True
        return tirx.AttrStmt(
            node.node,
            node.attr_key,
            node.value,
            tirx.SeqStmt([node.body, node.body]),
        )

    func = mod["main"]
    body = ir_transform(func.body, None, rewrite)
    assert duplicated
    return tvm.IRModule({"main": func.with_body(body)})


def test_estimate_latency_annotates_all_materialized_tasks():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program())))
    metadata = _collect_task_metadata(mod)

    assert len(metadata["latency"]) == 2
    assert len(metadata["ii"]) == 2
    assert all(latency >= 0 for latency in metadata["latency"])
    assert all(ii > 0 for ii in metadata["ii"])
    assert _collect_schedule_units(mod)


def test_frontend_task_hints_feed_shared_task_metadata():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program(latency=17, ii=5))))
    metadata = _collect_task_metadata(mod)

    assert 17 in metadata["latency"]
    assert 5 in metadata["ii"]


def test_materialize_schedule_units_builds_trivial_guarded_schedule():
    mod = _materialize_schedule_units(_bind_target(_make_guarded_program()))
    schedule_units = _collect_schedule_units(mod)
    task_metadata = _collect_task_metadata(mod)

    assert schedule_units
    assert all(int(unit.node["stage"]) == -1 for unit in schedule_units)
    assert all(len(unit.node) == 1 for unit in schedule_units)
    assert sum(isinstance(unit.body, tirx.IfThenElse) for unit in schedule_units) >= 2
    assert task_metadata["latency"] == []
    assert task_metadata["ii"] == []


def test_auto_schedule_requires_estimate_latency():
    mod = _materialize_schedule_units(_bind_target(_make_program()))
    with pytest.raises(tvm.error.InternalError, match="EstimateLatency"):
        ascend_transform.AutoSchedule()(mod)


def test_auto_schedule_requires_materialized_schedule_units():
    mod = _bind_target(_make_program())
    with pytest.raises(tvm.error.InternalError, match="MaterializeScheduleUnits"):
        ascend_transform.AutoSchedule()(mod)


def test_auto_schedule_rejects_multiple_tasks_in_one_schedule_unit():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program())))
    malformed = _duplicate_one_task_schedule_unit(mod)

    with pytest.raises(tvm.error.InternalError, match="exactly one outer T.Task/T.PerCoreTask"):
        ascend_transform.AutoSchedule()(malformed)


def test_task_cost_arguments_are_validated():
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_program(latency=-1)
    with pytest.raises(ValueError, match="positive integer"):
        _make_program(ii=0)


if __name__ == "__main__":
    pytest.main([__file__])
