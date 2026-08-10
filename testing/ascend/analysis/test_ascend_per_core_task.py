import re

import pytest
import tilelang.language as T
import tilelang.testing

from tilelang.engine.lower import lower


TILE = 256
FLAG = 3


def _core_branch(source: str, token: str) -> str:
    position = source.index(token)
    aic = source.rfind("if ASCEND_IS_AIC", 0, position)
    aiv = source.rfind("if ASCEND_IS_AIV", 0, position)
    assert max(aic, aiv) >= 0
    return "AIC" if aic > aiv else "AIV"


def _make_per_core_mte2_program(to_l1: bool):
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        X: T.Tensor((64,), "float32"),
        C: T.Tensor((TILE, TILE), "float32"),
        Y: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            x_ub = T.alloc_shared((64,), "float32")
            if to_l1:
                with T.PerCoreTask():
                    with T.Task():
                        T.copy(A, a_l1)
                    T.ascend_sync_inter_arrive("PIPE_MTE2", FLAG)
                    T.ascend_sync_inter_wait("PIPE_MTE2", FLAG)
            else:
                T.copy(A, a_l1)
            T.copy(B, b_l1)
            if not to_l1:
                with T.PerCoreTask():
                    with T.Task():
                        T.copy(X, x_ub)
                    T.ascend_sync_inter_arrive("PIPE_MTE2", FLAG)
                    T.ascend_sync_inter_wait("PIPE_MTE2", FLAG)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)
            if to_l1:
                T.copy(X, x_ub)
            T.copy(x_ub, Y)

    return main


def test_per_core_mte2_uses_copy_destination_core():
    marker = f"AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>({FLAG});"
    for to_l1, expected in ((True, "AIC"), (False, "AIV")):
        source = lower(_make_per_core_mte2_program(to_l1), target="ascend").kernel_source
        assert _core_branch(source, marker) == expected


def _make_per_core_candidate_program(explicit: bool, leave_second_unmarked: bool = False):
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
    ):
        with T.Kernel(2) as bx:
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            with T.PerCoreTask():
                if bx == 0:
                    if explicit:
                        with T.Task():
                            T.copy(A, a_l1)
                    else:
                        T.copy(A, a_l1)
                if bx != 0:
                    if explicit and not leave_second_unmarked:
                        with T.Task():
                            T.copy(B, a_l1)
                    else:
                        T.copy(B, a_l1)
            T.copy(a_l1, a_l0)

    return main


def _assert_per_core_mte2_mte1_sync(source: str):
    set_flag = "AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>"
    wait_flag = "AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>"
    assert source.count(set_flag) == 2
    assert source.count(wait_flag) == 1


def test_per_core_explicit_tasks_drive_sync_migration():
    source = lower(_make_per_core_candidate_program(explicit=True), target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def test_per_core_implicit_candidate_inference_remains_supported():
    source = lower(_make_per_core_candidate_program(explicit=False), target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def test_per_core_mixes_explicit_and_inferred_tasks():
    program = _make_per_core_candidate_program(explicit=True, leave_second_unmarked=True)
    source = lower(program, target="ascend").kernel_source
    _assert_per_core_mte2_mte1_sync(source)


def _make_per_core_dual_copy_store_program():
    @T.prim_func
    def main(C: T.Tensor((TILE, TILE), "float32")):
        with T.Kernel(1):
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.dual_copy(accum, temp)
            with T.PerCoreTask(), T.Task():
                T.dual_copy(temp, C)

    return main


def test_per_core_task_rewrites_standalone_ub_to_gm_dual_copy():
    source = lower(_make_per_core_dual_copy_store_program(), target="ascend").kernel_source
    store = re.search(r"copy_ubuf_to_gm_align_v2\(\(__gm__ void\*\)\(([^,]+)\),", source)
    assert store is not None
    assert "sid" in store.group(1)


def test_per_core_task_rejects_nesting_inside_task():
    @T.prim_func
    def main(
        A: T.Tensor((16,), "float32"),
        B: T.Tensor((16,), "float32"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((16,), "float32")
            with T.Task(), T.PerCoreTask():
                T.copy(A, temp)
            T.copy(temp, B)

    with pytest.raises(ValueError, match=r"T\.PerCoreTask cannot be nested inside T\.Task"):
        lower(main, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
