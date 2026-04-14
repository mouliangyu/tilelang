import tilelang
import tilelang.language as T


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192):
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    TILE_K_SUB = 64
    SUB_K = TILE_K // TILE_K_SUB
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    TILES_PER_CORE = OUT_TILES // NUM_BLOCKS
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        W: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            x_l0 = T.alloc_l0a((NUM_STAGES, TILE_M, TILE_K_SUB), "bfloat16")
            w_l0 = T.alloc_l0b((NUM_STAGES, TILE_N, TILE_K_SUB), "bfloat16")
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((NUM_STAGES, TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((NUM_STAGES, TILE_N, TILE_K), "bfloat16")
            for f in T.Serial(NUM_STAGES):
                T.ascend_set_flag("M_MTE1", f + NUM_STAGES)
                T.ascend_set_flag("MTE1_MTE2", f)
            T.ascend_set_flag("FIX_M", 0)
            for out_tile in T.Serial(TILES_PER_CORE):
                tile_idx = out_tile * NUM_BLOCKS + bx
                m_tile, n_tile = aswt_swizzle(tile_idx)

                for kt in T.Serial(K_TILES):
                    f = kt % NUM_STAGES
                    T.ascend_wait_flag("MTE1_MTE2", f)

                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1[f, :, :])
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1[f, :, :])
                    T.ascend_set_flag("MTE2_MTE1", f)

                    T.ascend_wait_flag("MTE2_MTE1", f)
                    for sk in T.Serial(SUB_K):
                        sf = sk % NUM_STAGES
                        T.ascend_wait_flag("M_MTE1", sf + NUM_STAGES)
                        T.copy(x_l1[f, :, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], x_l0[sf, :, :])
                        T.copy(w_l1[f, :, sk * TILE_K_SUB : (sk + 1) * TILE_K_SUB], w_l0[sf, :, :])
                        T.ascend_set_flag("MTE1_M", sf + 2)
                        T.ascend_wait_flag("MTE1_M", sf + 2)
                        if kt == 0 and sk == 0:
                            T.ascend_wait_flag("FIX_M", 0)
                        T.gemm(x_l0[sf, :, :], w_l0[sf, :, :], res, transpose_B=True, clear_accum=(kt == 0 and sk == 0))
                        T.ascend_set_flag("M_MTE1", sf + 2)

                    T.ascend_set_flag("MTE1_MTE2", f)

                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])
                T.ascend_set_flag("FIX_M", 0)
            for f in T.Serial(NUM_STAGES):
                T.ascend_wait_flag("M_MTE1", f + NUM_STAGES)
                T.ascend_wait_flag("MTE1_MTE2", f)
            T.ascend_wait_flag("FIX_M", 0)

    return main


def ref_program(x, w):
    """Reference implementation: C = A @ B"""
    return x.float() @ w.float().T


NUM_REPEATS = 500
M_DIM = 8192
K_DIM = 8192
N_DIM = 8192

if __name__ == "__main__":
    import torch

    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print("Compiling gemm kernel...")
    program = gemm(M_DIM, K_DIM, N_DIM)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Create test data
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(x, w)
    torch.npu.synchronize()
    print("Done")

    # Verify
    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  {'PASS' if max_diff < 1e-2 else 'FAIL'}  rel={max_diff:.2e}")
    if not max_diff < 1e-2:
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")

    # Benchmark
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(NUM_REPEATS):
        kernel(x, w)
    t1.record()
    torch.npu.synchronize()

    ms = t0.elapsed_time(t1)
    flops = 2.0 * M_DIM * N_DIM * K_DIM * NUM_REPEATS
    ref = flops / (ms / 1e3) / 1e12
    print(f"{ms / 1e3:.1f}s total  |  {ms / NUM_REPEATS:.3f} ms/iter  |  {ref:.1f} TFLOPS")
