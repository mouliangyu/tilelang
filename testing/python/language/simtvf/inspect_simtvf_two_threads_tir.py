import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.utils.target import determine_target

from testing.python.language.simtvf._inspect_utils import (
    collect_simtvf_loops,
    collect_thread_extent_tags,
    collect_thread_extent_tags_outside_simtvf,
    print_stage,
)


def main():
    target = determine_target("ascend", return_object=True)
    with target:

        @T.prim_func
        def simtvf_two_threads_demo(
            A: T.Buffer((256,), "float32"),
            B: T.Buffer((256,), "float32"),
            C: T.Buffer((256,), "float32"),
        ):
            with T.Kernel(1):
                with T.SimtVF(threads=128):
                    for i in T.Parallel(256):
                        B[i] = A[i] + T.float32(1)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(256):
                        C[i] = A[i] + T.float32(2)

    mod = tvm.IRModule({"simtvf_two_threads_demo": simtvf_two_threads_demo})

    print_stage("Raw PrimFunc", mod, "simtvf_two_threads_demo")
    raw_loops = collect_simtvf_loops(mod["simtvf_two_threads_demo"])
    assert len(raw_loops) == 2, "raw stage should contain exactly two SimtVF loops"
    raw_tags = collect_thread_extent_tags_outside_simtvf(mod["simtvf_two_threads_demo"].body)
    assert "blockIdx.x" in raw_tags, "Raw stage should keep blockIdx launch binding"
    assert not any(tag.startswith("threadIdx.") for tag in raw_tags), (
        "Raw stage should not have outer threadIdx launch bindings on ascend"
    )

    with target:
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)
        print_stage("After LowerAndLegalize", lowered, "simtvf_two_threads_demo")
        lowered_loops = collect_simtvf_loops(lowered["simtvf_two_threads_demo"])
        assert len(lowered_loops) == 2, "LowerAndLegalize should keep two SimtVF loops"

        optimized = OptimizeForTarget(lowered, target)
        print_stage("After OptimizeForTarget", optimized, "simtvf_two_threads_demo")
        optimized_loops = collect_simtvf_loops(optimized["simtvf_two_threads_demo"])
        assert len(optimized_loops) == 2, "OptimizeForTarget should keep two SimtVF loops"

        threads = [int(loop.annotations["tl.simt_vf_threads"]) for loop in optimized_loops]
        assert sorted(threads) == [128, 256], (
            f"expected SimtVF thread annotations [128, 256], got {threads}"
        )
        opt_func = optimized["simtvf_two_threads_demo"]
        opt_tags = collect_thread_extent_tags_outside_simtvf(opt_func.body)
        assert "blockIdx.x" in opt_tags, "OptimizeForTarget should keep blockIdx launch binding"
        assert (
            opt_func.attrs is None or "thread_extent" not in opt_func.attrs
        ), "OptimizeForTarget should not leave func_attr thread_extent on ascend"
        for loop in optimized_loops:
            loop_tags = collect_thread_extent_tags(loop.body)
            assert "threadIdx.x" in loop_tags, (
                "Each SimtVF region should contain local threadIdx.x thread_extent"
            )

    print("\n[ok] Dual-SimtVF (128/256) IR inspection finished.")


if __name__ == "__main__":
    main()
