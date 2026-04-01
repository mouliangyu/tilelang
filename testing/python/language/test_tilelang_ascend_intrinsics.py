import tilelang.language as T
from tilelang.engine.lower import lower


def test_ascend_pipe_barrier():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_pipe_barrier("PIPE_ALL")
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::PipeBarrier<pipe_t::PIPE_ALL>();" in source


def test_ascend_set_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0)" in source
    assert "AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0)" in source


def test_ascend_threadfence():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_threadfence()
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_threadfence();" in source


def test_ascend_cross_core_set_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8)" in source


def test_ascend_cross_core_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(8)" in source


if __name__ == "__main__":
    test_ascend_pipe_barrier()
    test_ascend_set_wait_flag()
    test_ascend_threadfence()
    test_ascend_cross_core_set_flag()
    test_ascend_cross_core_wait_flag()
    print("[ok] All ascend intrinsic tests passed.")
