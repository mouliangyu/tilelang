import torch
import tilelang
import tilelang.language as T


def ref_program(X):
    """Reference: Y = X @ X.T"""
    return X @ X.T


def check_sm100():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (10, 0):
        device_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"tcgen05 ws requires SM100/Blackwell, got {device_name} with capability {(major, minor)}."
        )


def get_default_correctness_sizes(block_M=128, block_K=64):
    base_M = 2 * block_M
    return [
        (base_M, block_K),
        (2 * base_M, 2 * block_K),
        (4 * base_M, 4 * block_K),
    ]


@tilelang.jit(out_idx=[-1])
def matmul_transpose_tcgen05_ws(
    M,
    K,
    block_M=128,
    block_K=64,
    num_stages=6,
    threads=128,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    Warp-specialized TileLang TCGEN05 kernel that computes Y = X @ X.T.
    """
    if threads < 64:
        raise ValueError("threads must be at least 64 for the 1CTA ws kernel.")

    @T.prim_func
    def mmt_tcgen05_ws_kernel(
        X: T.Tensor((M, K), dtype),
        Y: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            C_tmem = T.alloc_tmem([block_M, block_M], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_M), accum_dtype)
            Y_shared = T.alloc_shared((block_M, block_M), dtype)
            loaded = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            k_iters = T.ceildiv(K, block_K)

            T.use_swizzle(16)

            if by > bx:
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
                            by * block_M : (by + 1) * block_M,
                            ko * block_K : (ko + 1) * block_K,
                        ],
                        A_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        X[
                            bx * block_M : (bx + 1) * block_M,
                            ko * block_K : (ko + 1) * block_K,
                        ],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.mbarrier_arrive(loaded[stage])
            elif tx < 64:
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
                    )
                T.tcgen05_mma_arrive(tmem_full)

            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, Y_shared)
            T.copy(Y_shared, Y[by * block_M, bx * block_M])

            if by < bx:
                for i, j in T.Parallel(block_M, block_M):
                    Y[bx * block_M + j, by * block_M + i] = Y_shared[i, j]

    return mmt_tcgen05_ws_kernel


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
    sizes=None,
    block_M=128,
    block_K=64,
    num_stages=4,
    threads=128,
    atol=1e-2,
    rtol=1e-2,
    seed=0,
):
    sizes = get_default_correctness_sizes(block_M=block_M, block_K=block_K) if sizes is None else sizes

    print("=" * 88)
    print(
        f"Correctness check for tcgen05 ws symmetry kernel with "
        f"block_M={block_M}, block_K={block_K}, stages={num_stages}, threads={threads}"
    )
    print(f"{'M':>6} {'K':>6} | {'Max Err':>10} {'Sym Err':>10} | {'Status'}")
    print("=" * 88)

    for idx, (M, K) in enumerate(sizes):
        torch.manual_seed(seed + idx)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        kernel = matmul_transpose_tcgen05_ws(
            M,
            K,
            block_M=block_M,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
        )

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()
        symmetry_err = (y - y.T).abs().max().item()
        torch.testing.assert_close(y, ref_y, atol=atol, rtol=rtol)

        print(f"{M:>6} {K:>6} | {max_err:>10.4f} {symmetry_err:>10.4f} | PASS")

    print("=" * 88)


if __name__ == "__main__":
    check_sm100()
    run_correctness_suite()

    # sizes = [(4096, 2048), (4096, 4096), (8192, 4096)]
    sizes = [(16384, 16384)]
    n_iter = 100

    print("=" * 96)
    print(
        f"{'M':>6} {'K':>6} | {'TCGEN05-WS(ms)':>15} {'PyTorch(ms)':>12} | {'WS/PyTorch':>12} | {'Max Err'}"
    )
    print("=" * 96)

    for M, K in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        kernel = matmul_transpose_tcgen05_ws(M, K)

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()

        ws_ms = bench_ms(lambda: kernel(x), n_iter=n_iter)
        pytorch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter)

        print(
            f"{M:>6} {K:>6} | {ws_ms:>15.4f} {pytorch_ms:>12.4f} | "
            f"{ws_ms / pytorch_ms:>11.2f}x | {max_err:.4f}"
        )

    print("=" * 96)
