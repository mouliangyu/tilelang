import itertools
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
import torch


def ref_program(X):
    """Reference: Y = X @ X.T"""
    return X @ X.T


def get_configs():
    """Generate autotune search space, similar to the Triton version."""
    block_M = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 2, 3, 4]
    threads = [128, 256]
    enable_rasterization = [True, False]

    return [
        {
            "block_M": c[0],
            "block_K": c[1],
            "num_stages": c[2],
            "threads": c[3],
            "enable_rasterization": c[4],
        }
        for c in itertools.product(
            block_M, block_K, num_stages, threads, enable_rasterization,
        )
    ]


def matmul_transpose_autotune(M, K, dtype=T.float16, accum_dtype=T.float32):
    """Autotune the matmul_transpose kernel and return the best kernel."""

    def kernel(
        block_M=None,
        block_K=None,
        num_stages=None,
        threads=None,
        enable_rasterization=None,
    ):
        @T.prim_func
        def mmt_kernel(
            X: T.Tensor((M, K), dtype),
            Y: T.Tensor((M, M), dtype),
        ):
            with T.Kernel(
                T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=threads
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_M, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_M), accum_dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.clear(C_local)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(X[by * block_M, ko * block_K], A_shared)
                    T.copy(X[bx * block_M, ko * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, Y[by * block_M, bx * block_M])

        return mmt_kernel

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Normal,
            ref_prog=ref_program,
            skip_check=False,
        )
    )
    return autotuner.run(warmup=3, rep=20)


@tilelang.jit(out_idx=[-1])
def matmul_transpose(
    M,
    K,
    block_M=128,
    block_K=64,
    num_stages=3,
    threads=128,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    TileLang kernel that computes Y = X @ X.T
    Equivalent to the Triton mmt_kernel in matmul_transpose_triton.py
    """

    @T.prim_func
    def mmt_kernel(
        X: T.Tensor((M, K), dtype),
        Y: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_M), accum_dtype)

            T.use_swizzle(panel_size=10, enable=True)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(X[bx * block_M, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, Y[by * block_M, bx * block_M])

    return mmt_kernel


if __name__ == "__main__":
    import sys
    from matmul_transpose_triton import matmul_transpose as triton_mmt

    sizes = [(4096, 2048), (4096, 4096), (8192, 4096)]
    N_ITER = 100

    print("=" * 120)
    print(f"{'M':>6} {'K':>6} | {'Triton(ms)':>11} {'TL-auto(ms)':>12} {'PyTorch(ms)':>12} | "
          f"{'TL/Triton':>10} {'TL/PyTorch':>11} | {'Best Config'}")
    print("=" * 120)

    for M, K in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        ref_y = x @ x.T

        # ===== TileLang Autotune =====
        result = matmul_transpose_autotune(M, K)
        best_config = result.config
        kernel = result.kernel

        tl_y = kernel(x)
        tl_err = (tl_y - ref_y).abs().max().item()

        # ===== Benchmark =====
        # Warmup
        triton_mmt(x)
        kernel(x)
        x @ x.T
        for _ in range(10):
            triton_mmt(x)
            kernel(x)
            _ = x @ x.T
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        # Triton
        s.record()
        for _ in range(N_ITER):
            triton_mmt(x)
        e.record()
        torch.cuda.synchronize()
        triton_ms = s.elapsed_time(e) / N_ITER

        # TileLang (autotuned)
        s.record()
        for _ in range(N_ITER):
            kernel(x)
        e.record()
        torch.cuda.synchronize()
        tl_ms = s.elapsed_time(e) / N_ITER

        # PyTorch
        s.record()
        for _ in range(N_ITER):
            _ = x @ x.T
        e.record()
        torch.cuda.synchronize()
        ref_ms = s.elapsed_time(e) / N_ITER

        cfg_str = (f"block_M={best_config['block_M']}, block_K={best_config['block_K']}, "
                   f"stages={best_config['num_stages']}, threads={best_config['threads']}, "
                   f"raster={best_config['enable_rasterization']}")
        print(f"{M:>6} {K:>6} | {triton_ms:>11.4f} {tl_ms:>12.4f} {ref_ms:>12.4f} | "
              f"{tl_ms/triton_ms:>9.2f}x {tl_ms/ref_ms:>10.2f}x | {cfg_str}  (err={tl_err:.4f})")

    print("=" * 120)
    print()
    print("NOTE: Triton uses upper-triangle symmetry optimization (~50% less compute).")
    print("      TileLang computes the full M x M output.")
