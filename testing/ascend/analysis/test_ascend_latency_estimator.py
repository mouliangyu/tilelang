import pytest

import tilelang.ascend.transform as ascend_transform
import tilelang.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _make_program(latency=None, ii=None):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(latency=latency, ii=ii):
                T.copy(A, temp)
            T.copy(temp, B)

    return main


def _bind_target(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    return tirx.transform.BindTarget(determine_target("ascend"))(mod)


def _normalize_tasks(mod):
    return ascend_transform.NormalizeTask()(mod)


def _collect_task_annotations(mod):
    annotations = {"latency": [], "ii": [], "pipe": []}

    def visit(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_task":
            for key in annotations:
                if key in node.node:
                    annotations[key].append(int(node.node[key]))

    post_order_visit(mod["main"].body, visit)
    return annotations


def test_normalize_task_annotates_pipe():
    annotations = _collect_task_annotations(_normalize_tasks(_bind_target(_make_program())))

    assert annotations["latency"] == []
    assert annotations["ii"] == []
    assert len(annotations["pipe"]) == 2
    assert all(pipe > 0 for pipe in annotations["pipe"])
    assert len(set(annotations["pipe"])) == 2


def test_latency_estimator_annotates_all_normalized_tasks():
    mod = ascend_transform.LatencyEstimator()(_normalize_tasks(_bind_target(_make_program())))
    annotations = _collect_task_annotations(mod)

    assert len(annotations["latency"]) == 2
    assert len(annotations["ii"]) == 2
    assert len(annotations["pipe"]) == 2
    assert all(latency >= 0 for latency in annotations["latency"])
    assert all(ii > 0 for ii in annotations["ii"])
    assert all(pipe > 0 for pipe in annotations["pipe"])
    assert len(set(annotations["pipe"])) == 2


def test_auto_schedule_requires_latency_estimator():
    mod = _normalize_tasks(_bind_target(_make_program()))
    with pytest.raises(ValueError, match="LatencyEstimator"):
        ascend_transform.AutoSchedule()(mod)


def test_task_cost_arguments_are_validated():
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_program(latency=-1)
    with pytest.raises(ValueError, match="positive integer"):
        _make_program(ii=0)


if __name__ == "__main__":
    pytest.main([__file__])
