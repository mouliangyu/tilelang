import tilelang
import tilelang.language as T


def vector_add(N):
    NUM_BLOCKS = 64  # AI cores
    NUM_THREADS = 2048  # SIMT threads per core
    TILE_ELEMS = NUM_THREADS * 4  # 8192 floats = 32 KB per tile

    TOTAL_TILES = N // (TILE_ELEMS * NUM_BLOCKS)  # tiles per core
    NUM_STAGES = 2

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            temp1 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )
            temp2 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )
            temp3 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )

            for f in range(NUM_STAGES):
                T.ascend_set_flag("MTE3_V", f)
                T.ascend_set_flag("V_MTE2", f)

            for iter in range(TOTAL_TILES):
                begin = (iter * NUM_BLOCKS + bx) * TILE_ELEMS
                end = (iter * NUM_BLOCKS + bx + 1) * TILE_ELEMS
                stage = iter % NUM_STAGES

                T.ascend_wait_flag("V_MTE2", stage)
                T.copy(A[begin:end], temp1[stage, :])
                T.copy(B[begin:end], temp2[stage, :])
                T.ascend_set_flag("MTE2_V", stage)

                T.ascend_wait_flag("MTE2_V", stage)
                T.ascend_wait_flag("MTE3_V", stage)
                with T.SimtVF(threads=NUM_THREADS):
                    for i in T.Parallel(TILE_ELEMS):
                        temp3[stage, i] = temp1[stage, i] + temp2[stage, i]
                T.ascend_set_flag("V_MTE3", stage)
                T.ascend_set_flag("V_MTE2", stage)

                T.ascend_wait_flag("V_MTE3", stage)
                T.copy(temp3[stage, :], C[begin:end])
                T.ascend_set_flag("MTE3_V", stage)

            for f in range(NUM_STAGES):
                T.ascend_wait_flag("MTE3_V", f)
                T.ascend_wait_flag("V_MTE2", f)

    return main


def ref_program(a, b):
    """Reference implementation: C = A + B"""
    return a + b


NUM_REPEATS = 10

if __name__ == "__main__":
    import torch

    N = 2**30
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling vector_add kernel (N={N})...")
    program = vector_add(N)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    # Verify
    expected = ref_program(a, b)
    if not torch.equal(c, expected):
        max_diff = torch.max(torch.abs(c - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")

    # Benchmark
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(NUM_REPEATS):
        kernel(a, b)
    t1.record()
    torch.npu.synchronize()

    ms = t0.elapsed_time(t1)
    total_gb = N * 4 * 3 * NUM_REPEATS / 1e9
    print(f"{ms:.1f} ms  |  {ms / NUM_REPEATS:.1f} ms/iter  |  {total_gb / (ms / 1e3):.1f} GB/s")
