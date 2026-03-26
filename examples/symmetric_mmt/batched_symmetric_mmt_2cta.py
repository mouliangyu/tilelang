import torch
import tilelang
import tilelang.language as T


# ── Shapes from 1T+ models (DeepSeek V3 1.2T / V33 1.6T) ──────────────────
# Notation: (B=num_heads_or_batch, M=seq_len, K=head_dim_or_rank)
#
# DSV3  1.2T : hidden=7168, heads=128, q_head_dim=192, v_head_dim=128, kv_lora_rank=512
# DSV33 1.6T : hidden=7168, heads=128, q_head_dim=512, v_head_dim=512, kv_lora_rank=448
#
# Y = X @ X^T  →  X:(B,M,K)  →  Y:(B,M,M)
LARGE_MODEL_SHAPES = [
    # ── Gram / covariance: X(B,M,K) @ X^T → Y(B,M,M) ───────────────
    # K = hidden_size or intermediate_size from 1T+ models
    #
    # DSV3  1.2T : hidden=7168,  intermediate≈18432 (MoE expert)
    # DSV33 1.6T : hidden=7168,  intermediate≈24576
    # Dense 1T+  : hidden=12288~16384
    #                                                     Y mem (fp16)
    # ── K = hidden_size = 7168 (DSV3/V33) ────────────────────────────
    (  1,  4096,  7168),  # B=1, medium seq                0.03 GB
    (  1,  8192,  7168),  # B=1, long seq                  0.13 GB
    (  4,  4096,  7168),  # B=4, medium seq                0.13 GB
    (  4,  8192,  7168),  # B=4, long seq                  0.50 GB

    # ── K = intermediate (MoE expert dim) ────────────────────────────
    (  1,  4096, 18432),  # DSV3 expert ffn dim            0.03 GB
    (  4,  4096, 18432),  # batch=4                        0.13 GB

    # ── K = 12288~16384 (dense 1T model hidden) ─────────────────────
    (  1,  8192, 12288),  # dense 1T, long seq             0.13 GB
    (  4,  8192, 12288),  # batch=4                        0.50 GB

    # ── Large dense GEMM stress test ─────────────────────────────────
    (  4, 16384, 16384),  # 4×16K×16K                      2.0+8.0 GB
]

DEFAULT_BENCHMARK_SHAPES = [(1, 4096, 2048), (2, 4096, 4096), (4, 4096, 4096)]


def ref_program(X):
    """Reference: Y = X @ X.transpose(-1, -2)"""
    return X @ X.transpose(-1, -2)


def check_sm100():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (10, 0):
        device_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"tcgen05 batched 2cta ws requires SM100/Blackwell, got {device_name} with capability {(major, minor)}."
        )


def format_config(block_M, block_N, block_K, num_stages, threads):
    return (
        f"block_M={block_M}, block_N={block_N}, "
        f"block_K={block_K}, stages={num_stages}, "
        f"threads={threads}"
    )


def tflops(B, M, K, latency_ms):
    return 2 * B * M * M * K / (latency_ms / 1e3) / 1e12


def get_default_correctness_shapes(block_M=128, block_N=256, block_K=64):
    base_M = 2 * max(block_M, block_N)
    base_K = block_K
    return [
        (1, base_M, base_K),
        (2, 2 * base_M, 2 * base_K),
        (4, 4 * base_M, 4 * base_K),
    ]


def make_batched_mmt_tcgen05_ws_2cta_kernel(
    B,
    M,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    threads,
    dtype,
    accum_dtype,
):
    if B <= 0:
        raise ValueError("B must be positive for the batched 2cta ws kernel.")
    if threads < 64:
        raise ValueError("threads must be at least 64 for the batched 2cta ws kernel.")
    if block_N % 2 != 0:
        raise ValueError("block_N must be even for the batched 2cta ws kernel.")
    if block_N % block_M != 0:
        raise ValueError("block_N must be a multiple of block_M for the batched symmetry-aware 2cta kernel.")

    num_col_subtiles = block_N // block_M
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(M, block_N)

    @T.prim_func
    def batched_mmt_tcgen05_ws_2cta_kernel(
        X: T.Tensor((B, M, K), dtype),
        Y: T.Tensor((B, M, M), dtype),
    ):
        with T.Kernel(
            m_blocks,
            n_blocks,
            B,
            threads=threads,
            cluster_dims=2,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_N // 2, block_K), dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_shared = T.alloc_shared((block_M, block_M), dtype)
            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            k_iters = T.ceildiv(K, block_K)
            macro_col_tile_start = by * num_col_subtiles
            macro_col_tile_end = macro_col_tile_start + num_col_subtiles - 1

            T.assume(cta_id < 2)
            T.use_swizzle(16)

            if bx > macro_col_tile_end:
                if tx < 32:
                    T.deallocate_tmem(C_tmem)
                T.thread_return()

            if tx < 32:
                for ko in T.serial(k_iters):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.mbarrier_wait_parity(consumed[stage], phase ^ 1)
                    T.tma_copy(
                        X[
                            bz,
                            bx * block_M : (bx + 1) * block_M,
                            ko * block_K : (ko + 1) * block_K,
                        ],
                        A_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        X[
                            bz,
                            (by * block_N + cta_id * (block_N // 2)) : (
                                by * block_N + (cta_id + 1) * (block_N // 2)
                            ),
                            ko * block_K : (ko + 1) * block_K,
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.mbarrier_arrive(loaded[stage], 0)
            elif cta_id == 0 and tx < 64:
                for ko in T.serial(k_iters):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.mbarrier_wait_parity(loaded[stage], phase)
                    T.tcgen05_gemm(
                        A_shared[stage, :, :],
                        B_shared[stage, :, :],
                        C_tmem,
                        transpose_A=False,
                        transpose_B=True,
                        mbar=consumed[stage],
                        clear_accum=ko == 0,
                        use_2cta=True,
                    )
                T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)

            Y_shared_T = T.alloc_shared((block_M, block_M), dtype)

            for sub_tile in T.unroll(num_col_subtiles):
                col_tile = macro_col_tile_start + sub_tile
                col_offset = sub_tile * block_M

                if bx <= col_tile:
                    T.copy(C_local[:, col_offset : col_offset + block_M], Y_shared)
                    T.copy(
                        Y_shared,
                        Y[bz, bx * block_M, col_tile * block_M],
                    )

                if bx < col_tile:
                    T.transpose(Y_shared, Y_shared_T)
                    T.copy(
                        Y_shared_T,
                        Y[bz, col_tile * block_M, bx * block_M],
                    )

    return batched_mmt_tcgen05_ws_2cta_kernel


@tilelang.jit(out_idx=[-1])
def batched_matmul_transpose_tcgen05_ws_2cta(
    B,
    M,
    K,
    block_M=128,
    block_N=256,
    block_K=64,
    num_stages=6,
    threads=128,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    Batched warp-specialized 2CTA TileLang TCGEN05 kernel that computes Y = X @ X.T.
    """
    return make_batched_mmt_tcgen05_ws_2cta_kernel(
        B=B,
        M=M,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def bench_ms(fn, n_iter=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def run_correctness_suite(
    shapes=None,
    block_M=128,
    block_N=256,
    block_K=64,
    num_stages=6,
    threads=128,
    atol=1e-2,
    rtol=1e-2,
    seed=0,
):
    shapes = (
        get_default_correctness_shapes(block_M=block_M, block_N=block_N, block_K=block_K)
        if shapes is None
        else shapes
    )
    results = []

    print("=" * 104)
    print(
        "Correctness check for batched tcgen05 ws 2cta with "
        f"{format_config(block_M, block_N, block_K, num_stages, threads)}"
    )
    print(f"{'B':>4} {'M':>6} {'K':>6} | {'Max Err':>10} {'Sym Err':>10} | {'Status'}")
    print("=" * 104)

    for idx, (B, M, K) in enumerate(shapes):
        torch.manual_seed(seed + idx)
        x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        kernel = batched_matmul_transpose_tcgen05_ws_2cta(
            B,
            M,
            K,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
        )

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()
        symmetry_err = (y - y.transpose(-1, -2)).abs().max().item()

        try:
            torch.testing.assert_close(y, ref_y, atol=atol, rtol=rtol)
        except AssertionError as err:
            print(f"{B:>4} {M:>6} {K:>6} | {max_err:>10.4f} {symmetry_err:>10.4f} | FAIL")
            raise AssertionError(
                "Correctness check failed for "
                f"B={B}, M={M}, K={K} with {format_config(block_M, block_N, block_K, num_stages, threads)}."
            ) from err

        print(f"{B:>4} {M:>6} {K:>6} | {max_err:>10.4f} {symmetry_err:>10.4f} | PASS")
        results.append(
            {
                "B": B,
                "M": M,
                "K": K,
                "max_err": max_err,
                "symmetry_err": symmetry_err,
            }
        )

    print("=" * 104)
    return results


def run_default_benchmark_suite(shapes=None, n_iter=100):
    shapes = DEFAULT_BENCHMARK_SHAPES if shapes is None else shapes

    print("=" * 112)
    print(
        f"{'B':>4} {'M':>6} {'K':>6} | {'TCGEN05-WS-2CTA(ms)':>20} {'PyTorch(ms)':>12} | {'2CTA/PyTorch':>14} | {'Max Err'}"
    )
    print("=" * 112)

    for B, M, K in shapes:
        x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        kernel = batched_matmul_transpose_tcgen05_ws_2cta(B, M, K)

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()

        ws_2cta_ms = bench_ms(lambda: kernel(x), n_iter=n_iter)
        pytorch_ms = bench_ms(lambda: x @ x.transpose(-1, -2), n_iter=n_iter)

        print(
            f"{B:>4} {M:>6} {K:>6} | {ws_2cta_ms:>20.4f} {pytorch_ms:>12.4f} | "
            f"{ws_2cta_ms / pytorch_ms:>13.2f}x | {max_err:.4f}"
        )

    print("=" * 112)


def run_large_shape_benchmark(
    B=4,
    M=16384,
    K=16384,
    block_M=128,
    block_N=256,
    block_K=64,
    num_stages=6,
    threads=128,
    bench_warmup=5,
    bench_iter=20,
    benchmark_torch=True,
):
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)

    kernel = batched_matmul_transpose_tcgen05_ws_2cta(
        B,
        M,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
    )
    y = kernel(x)
    ref_y = ref_program(x)
    max_err = (y - ref_y).abs().max().item()

    kernel_ms = bench_ms(lambda: kernel(x), n_iter=bench_iter, warmup=bench_warmup)

    torch_ms = None
    if benchmark_torch:
        torch_ms = bench_ms(lambda: x @ x.transpose(-1, -2), n_iter=bench_iter, warmup=bench_warmup)

    print("=" * 116)
    print(
        f"Large-shape batched benchmark on B={B}, M=N={M}, K={K} with "
        f"{format_config(block_M, block_N, block_K, num_stages, threads)}"
    )
    print("=" * 116)
    print(
        f"{'Kernel':>16} | {'Latency(ms)':>12} | {'TFLOPS':>10}"
        + (f" | {'Vs Torch':>9}" if torch_ms is not None else "")
        + f" | {'Max Err':>8}"
    )
    print("=" * 116)
    print(
        f"{'TCGEN05-WS-2CTA':>16} | {kernel_ms:>12.4f} | {tflops(B, M, K, kernel_ms):>10.2f}"
        + (f" | {kernel_ms / torch_ms:>8.2f}x" if torch_ms is not None else "")
        + f" | {max_err:>8.4f}"
    )
    if torch_ms is not None:
        print(
            f"{'PyTorch':>16} | {torch_ms:>12.4f} | {tflops(B, M, K, torch_ms):>10.2f} | {'1.00x':>9} | {'-':>8}"
        )
    print("=" * 116)

    return {
        "kernel_ms": kernel_ms,
        "kernel_tflops": tflops(B, M, K, kernel_ms),
        "torch_ms": torch_ms,
        "torch_tflops": tflops(B, M, K, torch_ms) if torch_ms is not None else None,
        "max_err": max_err,
    }


def make_batched_mmt_strided_store_kernel(
    B, M, K, block_M, block_N, block_K, num_stages, threads, dtype, accum_dtype,
):
    """Baseline kernel: uses strided (non-coalesced) global store for the
    lower-triangle symmetric write-back instead of T.transpose + T.copy."""
    if B <= 0:
        raise ValueError("B must be positive.")
    if threads < 64:
        raise ValueError("threads must be at least 64.")
    if block_N % 2 != 0:
        raise ValueError("block_N must be even.")
    if block_N % block_M != 0:
        raise ValueError("block_N must be a multiple of block_M.")

    num_col_subtiles = block_N // block_M
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(M, block_N)

    @T.prim_func
    def kernel(
        X: T.Tensor((B, M, K), dtype),
        Y: T.Tensor((B, M, M), dtype),
    ):
        with T.Kernel(
            m_blocks, n_blocks, B,
            threads=threads, cluster_dims=2,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_N // 2, block_K), dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_shared = T.alloc_shared((block_M, block_M), dtype)
            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            k_iters = T.ceildiv(K, block_K)
            macro_col_tile_start = by * num_col_subtiles
            macro_col_tile_end = macro_col_tile_start + num_col_subtiles - 1

            T.assume(cta_id < 2)
            T.use_swizzle(16)

            if bx > macro_col_tile_end:
                if tx < 32:
                    T.deallocate_tmem(C_tmem)
                T.thread_return()

            if tx < 32:
                for ko in T.serial(k_iters):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.mbarrier_wait_parity(consumed[stage], phase ^ 1)
                    T.tma_copy(
                        X[bz, bx * block_M:(bx + 1) * block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared[stage, :, :], barrier=loaded[stage],
                    )
                    T.tma_copy(
                        X[bz,
                          (by * block_N + cta_id * (block_N // 2)):
                          (by * block_N + (cta_id + 1) * (block_N // 2)),
                          ko * block_K:(ko + 1) * block_K],
                        B_shared[stage, :, :], barrier=loaded[stage],
                    )
                    T.mbarrier_arrive(loaded[stage], 0)
            elif cta_id == 0 and tx < 64:
                for ko in T.serial(k_iters):
                    stage = ko % num_stages
                    phase = (ko // num_stages) & 1
                    T.mbarrier_wait_parity(loaded[stage], phase)
                    T.tcgen05_gemm(
                        A_shared[stage, :, :], B_shared[stage, :, :], C_tmem,
                        transpose_A=False, transpose_B=True,
                        mbar=consumed[stage], clear_accum=ko == 0,
                        use_2cta=True,
                    )
                T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)

            for sub_tile in T.unroll(num_col_subtiles):
                col_tile = macro_col_tile_start + sub_tile
                col_offset = sub_tile * block_M

                if bx <= col_tile:
                    T.copy(C_local[:, col_offset:col_offset + block_M], Y_shared)
                    T.copy(Y_shared, Y[bz, bx * block_M, col_tile * block_M])

                # ── old approach: strided (non-coalesced) global store ──
                if bx < col_tile:
                    for i, j in T.Parallel(block_M, block_M):
                        Y[bz, col_tile * block_M + j, bx * block_M + i] = Y_shared[i, j]

    return kernel


@tilelang.jit(out_idx=[-1])
def batched_mmt_strided_store(
    B, M, K,
    block_M=128, block_N=256, block_K=64,
    num_stages=6, threads=128,
    dtype=T.float16, accum_dtype=T.float32,
):
    """Baseline: strided global store for lower-triangle write-back."""
    return make_batched_mmt_strided_store_kernel(
        B, M, K, block_M, block_N, block_K,
        num_stages, threads, dtype, accum_dtype,
    )


def gb_of_shape(B, M, K, dtype_bytes=2):
    """Estimate total GPU memory (input + output) in GB."""
    x_bytes = B * M * K * dtype_bytes
    y_bytes = B * M * M * dtype_bytes
    return (x_bytes + y_bytes) / (1 << 30)


def run_1t_benchmark(
    shapes=None,
    block_M=128,
    block_N=256,
    block_K=64,
    num_stages=6,
    threads=128,
    n_iter=50,
    warmup=10,
):
    """Benchmark 1T+ model shapes: T.transpose kernel vs PyTorch."""
    shapes = LARGE_MODEL_SHAPES if shapes is None else shapes
    kw = dict(block_M=block_M, block_N=block_N, block_K=block_K,
              num_stages=num_stages, threads=threads)

    sep = "=" * 105
    print(sep)
    print(
        f"  1T+ Model Benchmark  |  {format_config(block_M, block_N, block_K, num_stages, threads)}"
    )
    print(sep)
    hdr = (
        f"{'B':>5} {'M':>6} {'K':>5} {'Mem':>6} | "
        f"{'Kernel(ms)':>12} {'PyTorch(ms)':>12} | "
        f"{'vs Torch':>9} | "
        f"{'TFLOPS(K)':>10} {'TFLOPS(P)':>10} | "
        f"{'Err':>7}"
    )
    print(hdr)
    print(sep)

    for B, M, K in shapes:
        mem_gb = gb_of_shape(B, M, K)
        try:
            x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)

            k_new = batched_matmul_transpose_tcgen05_ws_2cta(B, M, K, **kw)
            y_new = k_new(x)

            ref_y = ref_program(x)
            max_err = (y_new - ref_y).abs().max().item()

            new_ms = bench_ms(lambda: k_new(x), n_iter=n_iter, warmup=warmup)
            pyt_ms = bench_ms(lambda: x @ x.transpose(-1, -2), n_iter=n_iter, warmup=warmup)

            vs_torch = pyt_ms / new_ms
            tf_new = tflops(B, M, K, new_ms)
            tf_pyt = tflops(B, M, K, pyt_ms)

            print(
                f"{B:>5} {M:>6} {K:>5} {mem_gb:>5.1f}G | "
                f"{new_ms:>12.4f} {pyt_ms:>12.4f} | "
                f"{vs_torch:>8.2f}x | "
                f"{tf_new:>10.1f} {tf_pyt:>10.1f} | "
                f"{max_err:>7.4f}"
            )

            del x, y_new, ref_y
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(
                f"{B:>5} {M:>6} {K:>5} {mem_gb:>5.1f}G |  {'OOM':^12} {'':^12} |"
            )
            torch.cuda.empty_cache()

    print(sep)


if __name__ == "__main__":
    check_sm100()
    run_correctness_suite()
    print()
    run_1t_benchmark()
