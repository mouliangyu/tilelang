import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.utils.target import determine_target

from testing.python.language.simtvf._inspect_utils import (
    collect_simtvf_blocks,
    collect_thread_extent_tags_outside_simtvf,
    contains_parallel_for,
    find_thread_predicate_hoist_violations,
    print_stage,
)


def main():
    target = determine_target("ascend", return_object=True)
    with target:

        @T.prim_func
        def simtvf_demo(
            A: T.Buffer((16,), "float32"),
            B: T.Buffer((16,), "float32"),
        ):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    mod = tvm.IRModule({"simtvf_demo": simtvf_demo})

    print_stage("Raw PrimFunc", mod, "simtvf_demo")
    raw_simtvf_blocks = collect_simtvf_blocks(mod["simtvf_demo"])
    assert raw_simtvf_blocks, "raw stage should contain at least one SimtVF block"
    assert all(contains_parallel_for(block.body) for block in raw_simtvf_blocks), "SimtVF region should wrap Parallel loops in raw stage"
    raw_tags = collect_thread_extent_tags_outside_simtvf(mod["simtvf_demo"].body)
    assert "blockIdx.x" in raw_tags, "Raw stage should keep blockIdx launch binding"
    assert not any(tag.startswith("threadIdx.") for tag in raw_tags), "Raw stage should not have outer threadIdx launch bindings on ascend"

    with target:
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)
        print_stage("After LowerAndLegalize", lowered, "simtvf_demo")
        lowered_blocks = collect_simtvf_blocks(lowered["simtvf_demo"])
        assert lowered_blocks, "LowerAndLegalize should keep SimtVF annotation"
        assert all(block.name_hint == "SIMT_VF" for block in lowered_blocks), "LowerAndLegalize should preserve SIMT_VF name_hint"

        optimized = OptimizeForTarget(lowered, target)
        print_stage("After OptimizeForTarget", optimized, "simtvf_demo")
        optimized_blocks = collect_simtvf_blocks(optimized["simtvf_demo"])
        assert optimized_blocks, "OptimizeForTarget should keep SimtVF annotation"
        assert all(block.name_hint == "SIMT_VF" for block in optimized_blocks), "OptimizeForTarget should preserve SIMT_VF name_hint"
        violations = find_thread_predicate_hoist_violations(optimized["simtvf_demo"])
        assert not violations, f"thread predicate should not be hoisted outside SimtVF on ascend, got: {violations}"
        opt_func = optimized["simtvf_demo"]
        opt_tags = collect_thread_extent_tags_outside_simtvf(opt_func.body)
        assert "blockIdx.x" in opt_tags, "OptimizeForTarget should keep blockIdx launch binding"
        assert not any(tag.startswith("threadIdx.") for tag in opt_tags), (
            "OptimizeForTarget should not have outer threadIdx launch bindings on ascend"
        )
        assert opt_func.attrs is None or "thread_extent" not in opt_func.attrs, (
            "OptimizeForTarget should not leave func_attr thread_extent on ascend"
        )

    print("\n[ok] Ascend SimtVF structure/annotations and hoist boundary checks passed.")


if __name__ == "__main__":
    main()
