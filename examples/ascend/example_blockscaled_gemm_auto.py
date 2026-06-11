import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, dtype="bfloat16", MIXED=None, hf32=None):
    """Auto-scheduled GEMM.

    dtype: 'bfloat16' or 'float32'.
    MIXED: True/False/None (auto: bf16→mixed, fp32→cube-only).
    hf32: None (off) / 'nearest_zero' / 'nearest_even' (fp32 only).
    """
    NUM_BLOCKS = 32
    is_fp32 = dtype == "float32"
    if MIXED is None:
        MIXED = not is_fp32  # default: bf16→mixed, fp32→cube-only
    TILE_M = 256
    TILE_N = 256
    TILE_K = 128 if is_fp32 else 256  # fp32: L1 512KB, 2*(256+256)*128*4=512KB fits

    SF_DIV = 64  # one SF pair (int16) covers 64 K-elements
    SF_K = K_DIM // SF_DIV  # SF pairs over the full K
    TILE_SF_K = TILE_K // SF_DIV  # SF pairs per K-tile (L1 SF tile width)
    SF_LOAD_CHUNK_SIZE = 1  # load one SF pair per 1 K-tiles; adjust as needed for better performance

    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
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
        X: T.Buffer((M_DIM, K_DIM), dtype),
        W: T.Buffer((N_DIM, K_DIM), dtype),
        SFX: T.Buffer((SF_K, M_DIM), "uint16"),
        SFW: T.Buffer((SF_K, N_DIM), "uint16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            if is_fp32:
                T.set_hf32_mode(hf32)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            xsf_l1 = T.alloc_l1((SF_LOAD_CHUNK_SIZE * TILE_SF_K, TILE_M), "uint16")
            wsf_l1 = T.alloc_l1((SF_LOAD_CHUNK_SIZE * TILE_SF_K, TILE_N), "uint16")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            sf_int = SF_LOAD_CHUNK_SIZE

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)

                    if kt % sf_int == 0:
                        T.copy(
                            SFX[kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K, m_tile * TILE_M : (m_tile + 1) * TILE_M], xsf_l1, scale_factor=1
                        )
                        T.copy(
                            SFW[kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K, n_tile * TILE_N : (n_tile + 1) * TILE_N], wsf_l1, scale_factor=1
                        )

                    T.blockscaled_gemm(
                        x_l1,
                        w_l1,
                        res,
                        sfa=xsf_l1,
                        sfb=wsf_l1,
                        sf_k_within_chunk=(kt % sf_int),
                        transpose_B=True,
                        clear_accum=(kt == 0),
                    )
                if MIXED:
                    T.dual_copy(res, temp)
                    T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])
                else:
                    T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])

    return main


def ref_program(x, w, sfx, sfw):
    """Reference MXFP8 block-scaled matmul: D = sum_g SFX[m,g]*SFW[n,g]*(X@W^T)_g.

    x   : [M, K] float8_e4m3
    w   : [N, K] float8_e4m3
    sfx : [M, K/32] uint8 e8m0  (per 32-K group scale for X)
    sfw : [N, K/32] uint8 e8m0  (per 32-K group scale for W)
    """
    import torch

    K = x.shape[1]
    xf = x.float()
    wf = w.float()
    # e8m0 -> float scale = 2^(v - 127); each value covers 32 K-elements.
    sx = torch.pow(2.0, sfx.float() - 127.0).repeat_interleave(32, dim=1)[:, :K]  # [M, K]
    sw = torch.pow(2.0, sfw.float() - 127.0).repeat_interleave(32, dim=1)[:, :K]  # [N, K]
    return (xf * sx) @ (wf * sw).T


if __name__ == "__main__":
    import torch

    device = torch.device("npu")
    M, K, N = 8192, 8192, 8192

    for dt, thresh in [("float8_e4m3fn", 1e-1)]:
        print(f"\n=== {dt} GEMM ===")
        program = gemm(M, K, N, dtype=dt)
        kernel = tilelang.compile(program, out_idx=-1)

        print(kernel.get_kernel_source())
        print("Compilation succeeded!")

        x = (torch.randn(M, K, device=device) * 0.1).to(getattr(torch, dt))
        w = (torch.randn(N, K, device=device) * 0.1).to(getattr(torch, dt))

        sf_k_e8m0 = K // 32  # e8m0 values along K
        sfx_e8m0 = torch.randint(120, 135, (M, sf_k_e8m0), dtype=torch.uint8, device=device)
        sfw_e8m0 = torch.randint(120, 135, (N, sf_k_e8m0), dtype=torch.uint8, device=device)

        sfx_pairs = sfx_e8m0.view(torch.uint16)  # [M, K/64]
        sfw_pairs = sfw_e8m0.view(torch.uint16)  # [N, K/64]
        sfx = sfx_pairs.T.contiguous()  # [K/64, M]  (MN-major)
        sfw = sfw_pairs.T.contiguous()  # [K/64, N]

        c = kernel(x, w, sfx, sfw)
        torch.npu.synchronize()

        expected = ref_program(x, w, sfx_e8m0, sfw_e8m0)
        denom = expected.abs().mean().clamp_min(1e-6)
        max_diff = (torch.abs(c - expected).mean() / denom).item()

        print(f"  {'PASS' if max_diff < thresh else 'FAIL'}  max_diff={max_diff:.2e}")

        # Quick benchmark
        NUM_REPEATS = 100 if dt == "bfloat16" else 50

        def run_kernel(kernel=kernel, x=x, w=w, sfx=sfx, sfw=sfw):
            return kernel(x, w, sfx, sfw)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
        flops = 2.0 * M * N * K
        print(f"  {latency_ms:.3f} ms/iter  |  {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")
