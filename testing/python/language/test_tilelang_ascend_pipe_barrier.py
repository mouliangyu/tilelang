import tilelang.language as T
from tilelang.engine.lower import lower


def test_ascend_pipe_barrier_pipe_all():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_pipe_barrier("PIPE_ALL")
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::PipeBarrier<pipe_t::PIPE_ALL>();" in source, f"Expected AscendC::PipeBarrier<pipe_t::PIPE_ALL>(); in:\n{source}"


def test_ascend_pipe_barrier_pipe_v():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_pipe_barrier("PIPE_V")
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::PipeBarrier<pipe_t::PIPE_V>();" in source, f"Expected AscendC::PipeBarrier<pipe_t::PIPE_V>(); in:\n{source}"


if __name__ == "__main__":
    test_ascend_pipe_barrier_pipe_all()
    test_ascend_pipe_barrier_pipe_v()
    print("[ok] All ascend_pipe_barrier tests passed.")
