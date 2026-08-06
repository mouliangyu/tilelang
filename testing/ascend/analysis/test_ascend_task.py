import pytest
import tilelang.language as T
import tilelang.testing

from tilelang.engine.lower import lower


TILE = 256
FLAG = 3


def test_task_rejects_cross_pipe_body():
    @T.prim_func
    def main(A: T.Tensor((TILE, TILE), "bfloat16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(a_l1, a_l0)

    with pytest.raises(ValueError, match="exactly one Ascend hardware pipe"):
        lower(main, target="ascend")


def test_task_rejects_conflicting_warpgroup_affinity():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_ub = T.alloc_shared((TILE, TILE), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(B, b_ub)

    with pytest.raises(ValueError, match="both AIC and AIV HBM paths"):
        lower(main, target="ascend")


def test_task_rejects_dual_copy_in_composed_body():
    @T.prim_func
    def main():
        with T.Kernel(1):
            accum0 = T.alloc_l0c((TILE, TILE), "float32")
            accum1 = T.alloc_l0c((TILE, TILE), "float32")
            temp0 = T.alloc_shared((TILE // 2, TILE), "float32")
            temp1 = T.alloc_shared((TILE // 2, TILE), "float32")
            with T.Task():
                T.dual_copy(accum0, temp0)
                T.dual_copy(accum1, temp1)

    with pytest.raises(ValueError, match="dual_copy after it is combined"):
        lower(main, target="ascend")


def test_task_allows_standalone_cross_core_sync():
    @T.prim_func
    def main():
        with T.Kernel(1), T.Task():
            T.ascend_sync_inter_wait("PIPE_V", FLAG)

    source = lower(main, target="ascend").kernel_source
    assert f"AscendC::CrossCoreWaitFlag<0, PIPE_V>({FLAG});" in source


def test_task_rejects_multiple_cross_core_syncs():
    @T.prim_func
    def main():
        with T.Kernel(1), T.Task():
            T.ascend_sync_inter_arrive("PIPE_V", FLAG)
            T.ascend_sync_inter_wait("PIPE_V", FLAG)

    with pytest.raises(ValueError, match="must be the task's only statement"):
        lower(main, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
