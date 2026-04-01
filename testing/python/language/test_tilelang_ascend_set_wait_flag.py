import tilelang.language as T
from tilelang.engine.lower import lower


def test_ascend_set_flag_literal():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("MTE2_MTE1", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0)" in source, (
        f"Expected AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0) in:\n{source}"
    )


def test_ascend_wait_flag_literal():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_wait_flag("MTE2_MTE1", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0)" in source, (
        f"Expected AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0) in:\n{source}"
    )


def test_ascend_set_flag_different_event():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("S_MTE3", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0)" in source, (
        f"Expected AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0) in:\n{source}"
    )


def test_ascend_wait_flag_different_event():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_wait_flag("V_MTE2", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0)" in source, (
        f"Expected AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0) in:\n{source}"
    )


def test_ascend_set_wait_flag_combined():
    """Test SetFlag + WaitFlag in the same kernel (typical usage pattern)."""

    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0)" in source, f"Expected SetFlag in:\n{source}"
    assert "AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0)" in source, f"Expected WaitFlag in:\n{source}"


if __name__ == "__main__":
    test_ascend_set_flag_literal()
    test_ascend_wait_flag_literal()
    test_ascend_set_flag_different_event()
    test_ascend_wait_flag_different_event()
    test_ascend_set_wait_flag_combined()
    print("[ok] All ascend_set_wait_flag tests passed.")
