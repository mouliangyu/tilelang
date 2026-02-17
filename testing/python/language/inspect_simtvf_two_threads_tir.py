import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO_ROOT / ".cache"))
os.environ.setdefault("TILELANG_CACHE_DIR", str(_REPO_ROOT / ".tilelang_cache"))

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.utils.target import determine_target


@T.prim_func
def simtvf_two_threads_demo(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
    C: T.Buffer((256,), "float32"),
):
    with T.Kernel(1, threads=256):
        with T.SimtVF(threads=128):
            for i in T.Parallel(256):
                B[i] = A[i] + T.float32(1)
        with T.SimtVF(threads=256):
            for i in T.Parallel(256):
                C[i] = A[i] + T.float32(2)


def _collect_simtvf_loops(func: tvm.tir.PrimFunc):
    loops = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.For):
            annotations = stmt.annotations
            if annotations is not None and "tl.simt_vf" in annotations:
                loops.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return loops


def _print_stage(title: str, mod: tvm.IRModule):
    print(f"\n===== {title} =====")
    print(mod.script())
    func = mod["simtvf_two_threads_demo"]
    loops = _collect_simtvf_loops(func)
    print(f"[check] tl.simt_vf loop count = {len(loops)}")
    for idx, loop in enumerate(loops):
        print(
            f"  loop[{idx}] annotations: "
            f"tl.simt_vf={loop.annotations['tl.simt_vf']}, "
            f"tl.simt_vf_threads={loop.annotations['tl.simt_vf_threads']}"
        )


def main():
    mod = tvm.IRModule({"simtvf_two_threads_demo": simtvf_two_threads_demo})
    target = determine_target("ascend", return_object=True)

    _print_stage("Raw PrimFunc", mod)
    raw_loops = _collect_simtvf_loops(mod["simtvf_two_threads_demo"])
    assert len(raw_loops) == 2, "raw stage should contain exactly two SimtVF loops"

    with target:
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)
        _print_stage("After LowerAndLegalize", lowered)
        lowered_loops = _collect_simtvf_loops(lowered["simtvf_two_threads_demo"])
        assert len(lowered_loops) == 2, "LowerAndLegalize should keep two SimtVF loops"

        optimized = OptimizeForTarget(lowered, target)
        _print_stage("After OptimizeForTarget", optimized)
        optimized_loops = _collect_simtvf_loops(optimized["simtvf_two_threads_demo"])
        assert len(optimized_loops) == 2, "OptimizeForTarget should keep two SimtVF loops"

        threads = [int(loop.annotations["tl.simt_vf_threads"]) for loop in optimized_loops]
        assert sorted(threads) == [128, 256], (
            f"expected SimtVF thread annotations [128, 256], got {threads}"
        )

    print("\n[ok] Dual-SimtVF (128/256) IR inspection finished.")


if __name__ == "__main__":
    main()
