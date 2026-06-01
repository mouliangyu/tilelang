import torch
import tilelang
import tilelang.language as T


THREADS = 128


@tilelang.jit(
    out_idx=[1],
)
def cast_f32_to_fp8_kernel(n: int):
    @T.prim_func
    def main(
        X: T.Tensor((n,), T.float32),
        Y: T.Tensor((n,), T.float8_e4m3fn),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), T.float32)
            y_ub = T.alloc_shared((n,), T.float8_e4m3fn)

            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                x_local = T.alloc_fragment((n,), T.float32)
                y_local = T.alloc_fragment((n,), T.float8_e4m3fn)

                T.copy(x_ub, x_local)
                T.copy(x_local, y_local)
                T.copy(y_local, y_ub)
            T.copy(y_ub, Y)

    return main


def run_case(num_float32_per_thread: int) -> bool:
    n = THREADS * num_float32_per_thread
    print(f"\n=== num_float32_per_thread={num_float32_per_thread}, n={n} ===")

    kernel = cast_f32_to_fp8_kernel(n)
    source = kernel.get_kernel_source()

    print(source)

    x = torch.linspace(-448.0, 448.0, n, device="npu", dtype=torch.float32)
    y = kernel(x)
    torch.npu.synchronize()

    ref = x.to(torch.float8_e4m3fn)
    y_f32 = y.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    ok = torch.equal(y_f32, ref_f32)

    if not ok:
        mismatch = torch.nonzero(y_f32 != ref_f32).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        print(f"first mismatch index: {first}")
        print(f"input f32: {x[first : first + 8].cpu()}")
        print(f"tilelang f32: {y_f32[first : first + 8].cpu()}")
        print(f"torch f32: {ref_f32[first : first + 8].cpu()}")
        exit(1)
    return ok


@tilelang.jit(
    out_idx=[1],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
)
def cast_fp8_to_f32_kernel(n: int):
    @T.prim_func
    def main(
        X: T.Tensor((n,), T.float8_e4m3fn),
        Y: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n,), T.float8_e4m3fn)
            y_ub = T.alloc_shared((n,), T.float32)

            T.copy(X, x_ub)
            T.ascend_set_flag("MTE2_V", 0)

            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimtVF(threads=THREADS):
                x_local = T.alloc_fragment((n,), T.float8_e4m3fn)
                y_local = T.alloc_fragment((n,), T.float32)

                T.copy(x_ub, x_local)
                T.copy(x_local, y_local)
                T.copy(y_local, y_ub)
            T.ascend_set_flag("V_MTE3", 0)

            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y_ub, Y)

    return main


def run_case_fp8_to_f32(num_fp8_per_thread: int) -> bool:
    n = THREADS * num_fp8_per_thread
    print(f"\n=== fp8->f32: num_fp8_per_thread={num_fp8_per_thread}, n={n} ===")

    kernel = cast_fp8_to_f32_kernel(n)
    source = kernel.get_kernel_source()

    print(source)

    x_f32 = torch.linspace(-448.0, 448.0, n, device="npu", dtype=torch.float32)
    x = x_f32.to(torch.float8_e4m3fn)
    y = kernel(x)
    torch.npu.synchronize()

    ref = x.to(torch.float32)
    ok = torch.equal(y, ref)

    if not ok:
        mismatch = torch.nonzero(y != ref).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        print(f"first mismatch index: {first}")
        print(f"input f32: {x_f32[first : first + 8].cpu()}")
        print(f"tilelang f32: {y[first : first + 8].cpu()}")
        print(f"torch f32: {ref[first : first + 8].cpu()}")
        exit(1)
    return ok


def main():
    results = {num_float32_per_thread: run_case(num_float32_per_thread) for num_float32_per_thread in (16, 8, 4, 2, 1)}
    print("\n=== f32->fp8 summary ===")
    for num_float32_per_thread, ok in results.items():
        print(f"num_float32_per_thread={num_float32_per_thread}: {'PASS' if ok else 'FAIL'}")

    results_fp8_to_f32 = {num_fp8_per_thread: run_case_fp8_to_f32(num_fp8_per_thread) for num_fp8_per_thread in (16, 8, 4, 2, 1)}
    print("\n=== fp8->f32 summary ===")
    for num_fp8_per_thread, ok in results_fp8_to_f32.items():
        print(f"num_fp8_per_thread={num_fp8_per_thread}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
