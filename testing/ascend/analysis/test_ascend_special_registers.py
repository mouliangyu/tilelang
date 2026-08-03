import re

import pytest
import tilelang.language as T
import tilelang.testing
import tvm
from tilelang.engine.lower import lower


TILE = 256
FLAG = 3
LOOP_TRIPS = 2


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


def test_per_core_mte2_uses_copy_destination_warpgroup():
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


def _make_loop_nested_wait_program():
    @T.prim_func
    def main(
        values: T.Tensor((64, 64), "float32"),
        exchange: T.Tensor((64, 64), "float32"),
        independent: T.Tensor((64, 64), "float32"),
        output: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(64) as bx:
            write_ub = T.alloc_shared((64,), "float32")
            loop_ub = T.alloc_shared((64,), "float32")
            read_ub = T.alloc_shared((64,), "float32")
            T.copy(values[bx, :], write_ub)
            T.copy(write_ub, exchange[bx, :])
            iteration = T.alloc_var("int32")
            iteration = 0
            while iteration < LOOP_TRIPS:
                T.copy(exchange[bx, :], loop_ub)
                T.ascend_sync_inter_wait("PIPE_MTE2", FLAG)
                iteration = iteration + 1
            T.copy(independent[bx, :], read_ub)
            T.copy(read_ub, output[bx, :])

    return main


def test_loop_nested_wait_orders_following_pipe_user():
    source = lower(_make_loop_nested_wait_program(), target="ascend").kernel_source
    assert "while (1)" in source
    wait = source.index(f"AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>({FLAG});")
    mte2_loads = [match.start() for match in re.finditer("copy_gm_to_ubuf", source)]
    assert len(mte2_loads) == 3
    assert wait < mte2_loads[-1]


def _make_loop_nested_pad_writer_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        X: T.Tensor((1, 30), "float32"),
        C: T.Tensor((TILE, TILE), "float32"),
        Y: T.Tensor((1, 32), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            x_ub = T.alloc_shared((1, 32), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)
            iteration = T.alloc_var("int32")
            iteration = 0
            while iteration < LOOP_TRIPS:
                with T.Task():
                    T.ascend_set_copy_pad_value(-1.0, dtype="float32")
                iteration = iteration + 1
            T.copy(X, x_ub[:, :30], data_select=True)
            T.copy(x_ub, Y)

    return main


def test_loop_nested_pad_writer_follows_external_reader_warpgroup():
    source = lower(_make_loop_nested_pad_writer_program(), target="ascend").kernel_source
    writer = source.index("set_mov_pad_val")
    reader = source.index("copy_gm_to_ubuf_align_v2")
    assert source.rfind("while (1)", 0, writer) >= 0
    assert _core_branch(source, "set_mov_pad_val") == "AIV"
    assert writer < reader


def _make_sparse_explicit_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            with T.PerCoreTask():
                T.dual_copy(accum, temp)
                T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                T.ascend_sync_inter_wait("PIPE_FIX", 15)
            T.dual_copy(temp, C)

    return main


def test_sparse_explicit_cross_core_flag_allocation():
    source = lower(_make_sparse_explicit_cross_core_flag_program(), target="ascend").kernel_source
    assert "AscendC::CrossCoreSetFlag<0, PIPE_FIX>(15);" in source
    generated_slots = {int(value) % 16 for value in re.findall(r"CrossCore(?:Set|Wait)Flag<4, [^>]+>\((\d+)\)", source)}
    assert generated_slots == {0}


def _make_bounded_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            for outer in T.Serial(2):
                for inner in T.Serial(2):
                    flag_id = T.bind(outer * 2 + inner)
                    T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                    T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_bounded_dynamic_cross_core_flag_allocation():
    with tvm.transform.PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        source = lower(_make_bounded_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    generated_slots = {int(value) % 16 for value in re.findall(r"CrossCore(?:Set|Wait)Flag<4, [^>]+>\((\d+)\)", source)}
    assert generated_slots == {4}


def _make_guarded_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            for flag_id in T.Serial(32):
                if flag_id < 4:
                    T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                    T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_guarded_dynamic_cross_core_flag_allocation():
    source = lower(_make_guarded_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    generated_slots = {int(value) % 16 for value in re.findall(r"CrossCore(?:Set|Wait)Flag<4, [^>]+>\((\d+)\)", source)}
    assert generated_slots == {4}


def _make_assumed_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
        flag_id: T.int32,
    ):
        with T.Kernel(1):
            T.assume(flag_id >= 0)
            T.assume(flag_id < 4)
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
            T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_assumed_dynamic_cross_core_flag_allocation():
    source = lower(_make_assumed_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    generated_slots = {int(value) % 16 for value in re.findall(r"CrossCore(?:Set|Wait)Flag<4, [^>]+>\((\d+)\)", source)}
    assert generated_slots == {4}


def _make_unbounded_dynamic_cross_core_flag_program():
    flag_extent = T.dynamic("flag_extent")

    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
        Marker: T.Tensor((flag_extent,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            with T.PerCoreTask():
                T.dual_copy(accum, temp)
                T.ascend_sync_inter_arrive("PIPE_FIX", flag_extent)
                T.ascend_sync_inter_wait("PIPE_FIX", flag_extent)
            T.dual_copy(temp, C)

    return main


def test_unbounded_dynamic_cross_core_flag_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="integer range cannot be bounded"):
        lower(_make_unbounded_dynamic_cross_core_flag_program(), target="ascend")


def _make_nested_loop_break_program():
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
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            i = T.alloc_var("int32")
            i = 0
            while i < 2:
                T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
                T.copy(accum, C)
                for _ in T.Serial(2):
                    T.copy(X, x_ub)
                    T.copy(x_ub, Y)
                i = i + 1

    return main


def test_nested_loop_break_is_emitted_in_every_warpgroup():
    source = lower(_make_nested_loop_break_program(), target="ascend").kernel_source
    aic_start = source.index("if ASCEND_IS_AIC")
    aiv_start = source.index("if ASCEND_IS_AIV")
    assert "break;" in source[aic_start:aiv_start]
    assert "break;" in source[aiv_start:]


if __name__ == "__main__":
    tilelang.testing.main()
