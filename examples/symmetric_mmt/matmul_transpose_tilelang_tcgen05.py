import torch
import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


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
            f"tcgen05 requires SM100/Blackwell, got {device_name} with capability {(major, minor)}."
        )


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def matmul_transpose_tcgen05(
    M,
    K,
    block_M=128,
    block_K=128,
    num_stages=2,
    threads=256,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    TileLang TCGEN05 kernel that computes Y = X @ X.T on Blackwell/SM100.
    """

    @T.prim_func
    def mmt_tcgen05_kernel(
        X: T.Tensor((M, K), dtype),
        Y: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_K), dtype)
            C_tmem = T.alloc_tmem([block_M, block_M], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_M), accum_dtype)
            Y_shared = T.alloc_shared((block_M, block_M), dtype)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(X[bx * block_M, ko * block_K], B_shared)
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    transpose_A=False,
                    transpose_B=True,
                    mbar=mbar,
                    clear_accum=ko == 0,
                )
                T.mbarrier_wait_parity(mbar, ko % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, Y_shared)
            T.copy(Y_shared, Y[by * block_M, bx * block_M])

    return mmt_tcgen05_kernel


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


if __name__ == "__main__":
    check_sm100()

    sizes = [(4096, 2048), (4096, 4096), (8192, 4096)]
    n_iter = 100

    print("=" * 96)
    print(
        f"{'M':>6} {'K':>6} | {'TCGEN05(ms)':>12} {'PyTorch(ms)':>12} | {'TCGEN05/PyTorch':>16} | {'Max Err'}"
    )
    print("=" * 96)

    for M, K in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        kernel = matmul_transpose_tcgen05(M, K)

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()

        tcgen05_ms = bench_ms(lambda: kernel(x), n_iter=n_iter)
        pytorch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter)

        print(
            f"{M:>6} {K:>6} | {tcgen05_ms:>12.4f} {pytorch_ms:>12.4f} | "
            f"{tcgen05_ms / pytorch_ms:>15.2f}x | {max_err:.4f}"
        )

    print("=" * 96)
