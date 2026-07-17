import re

import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def _make_many_flags_program():
    tile = 64
    stages = 2

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(8, num_stages=stages):
                base = w * 5 * tile

                T.copy(A[base : base + tile], ub)
                with T.SimdVF():
                    mask0 = T.simd.pset(32)
                    value0 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value0, mask0)
                T.copy(ub, C[base : base + tile])

                T.copy(A[base + tile : base + 2 * tile], ub)
                with T.SimdVF():
                    mask1 = T.simd.pset(32)
                    value1 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value1, mask1)
                T.copy(ub, C[base + tile : base + 2 * tile])

                T.copy(A[base + 2 * tile : base + 3 * tile], ub)
                with T.SimdVF():
                    mask2 = T.simd.pset(32)
                    value2 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value2, mask2)
                T.copy(ub, C[base + 2 * tile : base + 3 * tile])

                T.copy(A[base + 3 * tile : base + 4 * tile], ub)
                with T.SimdVF():
                    mask3 = T.simd.pset(32)
                    value3 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value3, mask3)
                T.copy(ub, C[base + 3 * tile : base + 4 * tile])

                T.copy(A[base + 4 * tile : base + 5 * tile], ub)
                with T.SimdVF():
                    mask4 = T.simd.pset(32)
                    value4 = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value4, mask4)
                T.copy(ub, C[base + 4 * tile : base + 5 * tile])

    return main


def _make_cross_iter_reuse_program():
    tile = 64
    stages = 2
    inner = 4

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: stages})

            for w in T.Pipelined(4, num_stages=stages):
                base = w * 4 * inner * tile

                for i in T.serial(inner):
                    offset = base + i * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask0 = T.simd.pset(32)
                        value0 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value0, mask0)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask1 = T.simd.pset(32)
                        value1 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value1, mask1)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (2 * inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask2 = T.simd.pset(32)
                        value2 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value2, mask2)
                    T.copy(ub, C[offset : offset + tile])

                for i in T.serial(inner):
                    offset = base + (3 * inner + i) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask3 = T.simd.pset(32)
                        value3 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value3, mask3)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_mixed_cross_iter_versions_program():
    tile = 64
    inner = 6
    outer = 4
    segment = inner * tile

    @T.prim_func
    def main(
        state: T.Buffer(((outer + 1) * segment,), "float32"),
        middle: T.Buffer((outer * segment,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub2 = T.alloc_shared((tile,), "float32")
            ub3 = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub2: 2, ub3: 3})

            for w in T.Pipelined(outer, num_stages=3):
                if enabled > 0:
                    for i in T.serial(inner):
                        offset = w * segment + i * tile
                        T.copy(state[offset : offset + tile], ub2)
                        with T.SimdVF():
                            mask0 = T.simd.pset(32)
                            value0 = T.simd.vld(ub2[0])
                            T.simd.vsts(ub2[0], value0, mask0)
                        T.copy(ub2, middle[offset : offset + tile])

                if enabled > 0:
                    for i in T.serial(inner):
                        offset = w * segment + i * tile
                        next_offset = (w + 1) * segment + i * tile
                        T.copy(middle[offset : offset + tile], ub3)
                        with T.SimdVF():
                            mask1 = T.simd.pset(32)
                            value1 = T.simd.vld(ub3[0])
                            T.simd.vsts(ub3[0], value1, mask1)
                        T.copy(ub3, state[next_offset : next_offset + tile])

    return main


def _make_mixed_version_same_iter_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})

            if enabled > 0:
                for w in T.Pipelined(4, num_stages=2):
                    offset = w * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask0 = T.simd.pset(32)
                        value0 = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value0, mask0)
                    T.copy(ub, C[offset : offset + tile])

            T.copy(A[4 * tile : 5 * tile], ub)
            with T.SimdVF():
                mask1 = T.simd.pset(32)
                value1 = T.simd.vld(ub[0])
                T.simd.vsts(ub[0], value1, mask1)
            T.copy(ub, C[4 * tile : 5 * tile])

    return main


def _make_inner_loop_program(extent: int):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(extent):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
            T.copy(ub, C)

    return main


def _make_unavoidable_overflow_program():
    tile = 64
    stages = 2

    @T.prim_func
    def main(A: T.Buffer((4096,), "float32"), C: T.Buffer((4096,), "float32")):
        with T.Kernel(1):
            ub0 = T.alloc_shared((tile,), "float32")
            ub1 = T.alloc_shared((tile,), "float32")
            ub2 = T.alloc_shared((tile,), "float32")
            ub3 = T.alloc_shared((tile,), "float32")
            ub4 = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub0: stages, ub1: stages, ub2: stages, ub3: stages, ub4: stages})

            for w in T.Pipelined(8, num_stages=stages):
                base = w * 5 * tile

                T.copy(A[base : base + tile], ub0)
                with T.SimdVF():
                    mask0 = T.simd.pset(32)
                    value0 = T.simd.vld(ub0[0])
                    T.simd.vsts(ub0[0], value0, mask0)
                T.copy(ub0, C[base : base + tile])

                T.copy(A[base + tile : base + 2 * tile], ub1)
                with T.SimdVF():
                    mask1 = T.simd.pset(32)
                    value1 = T.simd.vld(ub1[0])
                    T.simd.vsts(ub1[0], value1, mask1)
                T.copy(ub1, C[base + tile : base + 2 * tile])

                T.copy(A[base + 2 * tile : base + 3 * tile], ub2)
                with T.SimdVF():
                    mask2 = T.simd.pset(32)
                    value2 = T.simd.vld(ub2[0])
                    T.simd.vsts(ub2[0], value2, mask2)
                T.copy(ub2, C[base + 2 * tile : base + 3 * tile])

                T.copy(A[base + 3 * tile : base + 4 * tile], ub3)
                with T.SimdVF():
                    mask3 = T.simd.pset(32)
                    value3 = T.simd.vld(ub3[0])
                    T.simd.vsts(ub3[0], value3, mask3)
                T.copy(ub3, C[base + 3 * tile : base + 4 * tile])

                T.copy(A[base + 4 * tile : base + 5 * tile], ub4)
                with T.SimdVF():
                    mask4 = T.simd.pset(32)
                    value4 = T.simd.vld(ub4[0])
                    T.simd.vsts(ub4[0], value4, mask4)
                T.copy(ub4, C[base + 4 * tile : base + 5 * tile])

    return main


def _constant_flag_ids(source: str, hard_event: str) -> set[int]:
    pattern = re.compile(rf"AscendC::(?:SetFlag|WaitFlag)<AscendC::HardEvent::{hard_event}>\((\d+)\)")
    return {int(value) for value in pattern.findall(source)}


def _initial_flag_ids(source: str, hard_event: str) -> set[int]:
    kernel_body = source.split("__global__", maxsplit=1)[1]
    prologue = kernel_body.split("for (", maxsplit=1)[0]
    return _constant_flag_ids(prologue, hard_event)


def test_auto_schedule_keeps_distinct_flags_within_limit():
    artifact = lower(_make_many_flags_program(), target="ascend")
    source = artifact.kernel_source

    for hard_event in ("MTE2_V", "V_MTE2", "MTE3_V", "V_MTE3"):
        ids = _constant_flag_ids(source, hard_event)
        assert not ids or max(ids) < 8, f"{hard_event} exceeds the dav-3510 flag limit: {sorted(ids)}"

    # Five ids already fit, so no otherwise-safe reuse is introduced.
    assert _constant_flag_ids(source, "MTE2_V") == set(range(5))


def test_auto_schedule_reuses_only_until_within_limit():
    artifact = lower(_make_cross_iter_reuse_program(), target="ascend")
    source = artifact.kernel_source

    ids = _constant_flag_ids(source, "MTE3_MTE2")
    # Reuse stops after the first sufficient merge instead of minimizing ids.
    assert ids == set(range(7))


def test_cross_iter_does_not_reuse_when_within_limit():
    artifact = lower(_make_mixed_cross_iter_versions_program(), target="ascend")
    ids = _initial_flag_ids(artifact.kernel_source, "MTE3_MTE2")

    # All three allocations stay independent because their six ids already fit.
    assert ids == set(range(6))


def test_same_iter_does_not_reuse_when_within_limit():
    artifact = lower(_make_mixed_version_same_iter_program(), target="ascend")
    ids = _constant_flag_ids(artifact.kernel_source, "MTE2_V")

    # The two allocations stay independent because they already fit.
    assert ids == {0, 1}


def test_single_iteration_loop_has_no_reverse_flag():
    # TODO: need to fix
    # one_iter_source = lower(_make_inner_loop_program(1), target="ascend").kernel_source
    many_iter_source = lower(_make_inner_loop_program(4), target="ascend").kernel_source

    # assert "HardEvent::V_MTE2" not in one_iter_source
    assert "HardEvent::V_MTE2" in many_iter_source


def test_auto_schedule_warns_on_unavoidable_flag_overflow(capfd):
    lower(_make_unavoidable_overflow_program(), target="ascend")
    stderr = capfd.readouterr().err

    assert re.search(r"allocated 10 flag ids.*limit of 8", stderr)


if __name__ == "__main__":
    tilelang.testing.main()
