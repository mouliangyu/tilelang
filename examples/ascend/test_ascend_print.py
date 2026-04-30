"""Test T.print() on Ascend target."""
import tilelang
import tilelang.language as T


def test_prim_expr_with_msg():
    """T.print(expr, msg=...) → AscendC::printf with blockIdx."""
    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(32) as bx:
            T.print(bx + 42, msg="block_val")

    kernel = tilelang.compile(program, target="ascend",
                              pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "AscendC::printf" in source
    assert "block_val" in source
    assert "debug_print_var" not in source
    print("PASS: test_prim_expr_with_msg")


def test_shared_print():
    """T.print(buffer) → AscendC::DumpTensor."""
    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(64) as bx:
            shared_buf = T.alloc_shared([16, 16], T.float32)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(16, 16):
                    shared_buf[i, j] = bx            
            T.print(shared_buf)

    kernel = tilelang.compile(program, target="ascend",
                                pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "AscendC::DumpTensor" in source
    assert "debug_print_buffer_value" not in source
    print("PASS: test_shared_print")


def test_global_print():
    """T.print(buffer) → AscendC::DumpTensor."""
    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(16) as bx:
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    Q[bx, i] = bx            
            if bx == 0:
                T.print(Q)

    kernel = tilelang.compile(program, target="ascend",
                                pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "AscendC::DumpTensor" in source
    assert "debug_print_buffer_value" not in source
    print("PASS: test_global_print")


def test_float_var():
    """T.print(float_expr) → AscendC::printf with %f."""
    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(32) as bx:
            val = bx * 3.14
            T.print(val, msg="float_val")

    kernel = tilelang.compile(program, target="ascend",
                              pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "AscendC::printf" in source
    assert "%f" in source
    print("PASS: test_float_var")


if __name__ == "__main__":
    test_prim_expr_with_msg()
    test_shared_print()
    test_global_print()
    test_float_var()
    print("\nAll tests passed!")
