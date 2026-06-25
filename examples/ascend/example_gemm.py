import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, dtype="bfloat16", out_dtype="float32", MIXED=None, hf32=None, enable_unit_flag=True):
    """Auto-scheduled GEMM.

    dtype: 'bfloat16' or 'float32'.
    out_dtype: 'bfloat16' or 'float32'.
    MIXED: True/False/None (auto: bf16→mixed, fp32→cube-only).
    hf32: None (off) / 'nearest_zero' / 'nearest_even' (fp32 only).
    """
    NUM_BLOCKS = 32
    is_fp32 = dtype == "float32"
    if MIXED is None:
        MIXED = not is_fp32 and out_dtype != "bfloat16"  # default: bf16→mixed, fp32/bf16_out→cube-only
    TILE_M = 256
    TILE_N = 256
    TILE_K = 128 if is_fp32 else 256  # fp32: L1 512KB, 2*(256+256)*128*4=512KB fits
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW
    if enable_unit_flag:
        UF_2 = 2
        UF_3 = 3
    else:
        UF_2 = 0
        UF_3 = 0

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
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        C: T.Buffer((M_DIM, N_DIM), out_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            if is_fp32:
                T.set_hf32_mode(hf32)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0), unit_flag_ctrl=T.Select(kt == K_TILES - 1, UF_3, UF_2))
                if MIXED:
                    T.dual_copy(res, temp, unit_flag_ctrl=UF_3)
                    T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])
                else:
                    T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N], unit_flag_ctrl=UF_3)

    return main


import torch


def ref_program(x, w, out_dtype="float32"):
    if out_dtype == "bfloat16":
        return (x @ w.T).to(torch.bfloat16)
    return x.float() @ w.float().T


def run_regression_perf(M=8192, K=8192, N=8192, dtype="bfloat16", hf32=None):
    """Compile + benchmark an auto-scheduled GEMM, returning latency in ms (msprof)."""
    import torch

    device = torch.device("npu")
    program = gemm(M, K, N, dtype=dtype, hf32=hf32)
    kernel = tilelang.compile(program, out_idx=-1)

    x = torch.randn(M, K, device=device).to(dtype=getattr(torch, dtype))
    w = torch.randn(N, K, device=device).to(dtype=getattr(torch, dtype))
    kernel(x, w)
    torch.npu.synchronize()

    def run_kernel():
        return kernel(x, w)

    num_repeats = 100 if dtype == "bfloat16" else 50
    prof = do_bench(run_kernel, backend="msprof_detail", _n_warmup=30, _n_repeat=num_repeats)
    flops = 2.0 * M * N * K
    print(f"    [{dtype}] {prof.dur_us:.2f} us/iter  |  {prof.tflops(flops):.1f} TFLOPS")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    device = torch.device("npu")
    M, K, N = 8192, 8192, 8192

    test_pairs = [
        ("float8_e4m3fn", 1e-1, "float32"),
        ("bfloat16", 1e-2, "float32"),
        ("float32", 5e-3, "float32"),
        ("bfloat16", 1e-2, "bfloat16"),
    ]

    for dt, thresh, out_dt in test_pairs:
        print(f"\n=== {dt} GEMM ===")
        program = gemm(M, K, N, dtype=dt, out_dtype=out_dt)
        kernel = tilelang.compile(program, out_idx=-1)
        print("Compilation succeeded!")

        x = torch.randn(M, K, device=device).to(dtype=getattr(torch, dt))
        w = torch.randn(N, K, device=device).to(dtype=getattr(torch, dt))

        c = kernel(x, w)
        torch.npu.synchronize()

        expected = ref_program(x, w, out_dtype=out_dt)
        max_diff = torch.max(torch.abs(c - expected)).item()
        print(f"  {'PASS' if max_diff < thresh else 'FAIL'}  max_diff={max_diff:.2e}")

        # Quick benchmark
        NUM_REPEATS = 100 if dt == "bfloat16" else 50

        def run_kernel(kernel=kernel, x=x, w=w):
            return kernel(x, w)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
        flops = 2.0 * M * N * K
        print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")

    # HF32 benchmark
    print("\n=== float32 HF32 GEMM ===")
    program = gemm(M, K, N, dtype="float32", hf32="nearest_even")
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")
    x = torch.randn(M, K, dtype=torch.float32, device=device)
    w = torch.randn(N, K, dtype=torch.float32, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  max_diff={max_diff:.2e} (HF32 rel_err ~3e-4)")
    NUM_REPEATS = 50

    def run_kernel():
        return kernel(x, w)

    latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    flops = 2.0 * M * N * K
    print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
