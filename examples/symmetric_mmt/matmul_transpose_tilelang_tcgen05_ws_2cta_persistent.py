import argparse
import itertools

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.arch import driver


DEFAULT_CONFIG = {
    "block_M": 128,
    "block_N": 256,
    "store_block_N": 64,
    "block_K": 64,
    "num_stages": 6,
    "threads": 256,
}

DEFAULT_BENCHMARK_SIZES = [(4096, 2048), (4096, 4096), (8192, 4096)]


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
            f"tcgen05 2cta persistent requires SM100/Blackwell, got {device_name} with capability {(major, minor)}."
        )


def format_config(config):
    return (
        f"block_M={config['block_M']}, block_N={config['block_N']}, "
        f"store_block_N={config['store_block_N']}, block_K={config['block_K']}, "
        f"stages={config['num_stages']}, threads={config['threads']}"
    )


def get_dtype_nbytes(dtype):
    if dtype in (T.float16, T.bfloat16):
        return 2
    if dtype in (T.float32, T.int32, T.uint32):
        return 4
    raise ValueError(f"Unsupported dtype for shared-memory estimation: {dtype}")


def estimate_shared_memory_bytes(block_M, block_N, store_block_N, block_K, num_stages, dtype_nbytes=2):
    return (
        num_stages * block_M * block_K * dtype_nbytes
        + num_stages * (block_N // 2) * block_K * dtype_nbytes
        + block_M * store_block_N * dtype_nbytes
    )


def get_shared_memory_limit_bytes():
    if not torch.cuda.is_available():
        return None
    device_props = torch.cuda.get_device_properties(0)
    return getattr(device_props, "shared_memory_per_block_optin", device_props.shared_memory_per_block)


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


def tflops(M, K, latency_ms):
    return 2 * M * M * K / (latency_ms / 1e3) / 1e12


def get_autotune_configs(M=None):
    configs = []
    block_M_values = [64, 128]
    block_N_values = [128, 256]
    store_block_N_values = [64, 128, 256]
    block_K_values = [32, 64]
    num_stages_values = [4, 6, 8]
    thread_values = [256]
    smem_limit = get_shared_memory_limit_bytes()

    for block_M, block_N, store_block_N, block_K, num_stages, threads in itertools.product(
        block_M_values,
        block_N_values,
        store_block_N_values,
        block_K_values,
        num_stages_values,
        thread_values,
    ):
        if store_block_N > block_N or block_N % store_block_N != 0:
            continue
        if M is not None and ((M + block_M - 1) // block_M) % 2 != 0:
            continue
        if smem_limit is not None:
            smem_bytes = estimate_shared_memory_bytes(
                block_M=block_M,
                block_N=block_N,
                store_block_N=store_block_N,
                block_K=block_K,
                num_stages=num_stages,
            )
            if smem_bytes > smem_limit:
                continue
        configs.append(
            {
                "block_M": block_M,
                "block_N": block_N,
                "store_block_N": store_block_N,
                "block_K": block_K,
                "num_stages": num_stages,
                "threads": threads,
            }
        )

    return configs


def make_mmt_tcgen05_ws_2cta_persistent_kernel(
    M,
    K,
    block_M,
    block_N,
    store_block_N,
    block_K,
    num_stages,
    threads,
    dtype,
    accum_dtype,
):
    smem_limit = get_shared_memory_limit_bytes()
    shared_bytes = estimate_shared_memory_bytes(
        block_M=block_M,
        block_N=block_N,
        store_block_N=store_block_N,
        block_K=block_K,
        num_stages=num_stages,
        dtype_nbytes=get_dtype_nbytes(dtype),
    )

    if threads < 256:
        raise ValueError("threads must be at least 256 for the persistent 2cta kernel.")
    if block_N % 2 != 0:
        raise ValueError("block_N must be even for the persistent 2cta kernel.")
    if store_block_N <= 0 or block_N % store_block_N != 0:
        raise ValueError("store_block_N must be a positive divisor of block_N.")
    if T.ceildiv(M, block_M) % 2 != 0:
        raise ValueError("persistent 2cta kernel requires an even number of row tiles.")
    if smem_limit is not None and shared_bytes > smem_limit:
        raise ValueError(
            f"shared memory requirement {shared_bytes} exceeds device limit {smem_limit}."
        )

    sm_num = driver.get_num_sms()
    if sm_num < 2:
        raise ValueError("persistent 2cta kernel requires at least 2 SMs.")
    num_clusters = sm_num // 2
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(M, block_N)
    k_blocks = T.ceildiv(K, block_K)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)

    @T.prim_func
    def mmt_tcgen05_ws_2cta_persistent_kernel(
        X: T.Tensor((M, K), dtype),
        Y: T.Tensor((M, M), dtype),
    ):
        with T.Kernel(sm_num, threads=threads, cluster_dims=2) as (block_id):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_N // 2, block_K), dtype)
            C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_shared = T.alloc_shared((block_M, store_block_N), dtype)
            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_cluster_barrier([1] * 2)
            tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            cluster_id = block_id // 2

            T.assume(cta_id < 2)
            T.use_swizzle(16)

            if tx < 32:
                for w in T.serial(waves):
                    tile_id = num_clusters * w + cluster_id
                    by_cluster = tile_id // n_blocks
                    by = by_cluster * 2 + cta_id
                    bx = tile_id % n_blocks

                    if by < m_blocks and bx < n_blocks:
                        for ko in T.serial(k_blocks):
                            phase = w * k_blocks + ko
                            stage = phase % num_stages
                            parity = (phase // num_stages) & 1

                            T.mbarrier_wait_parity(consumed[stage], parity ^ 1)
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
                                    (bx * block_N + cta_id * (block_N // 2)) : (
                                        bx * block_N + (cta_id + 1) * (block_N // 2)
                                    ),
                                    ko * block_K : (ko + 1) * block_K,
                                ],
                                B_shared[stage, :, :],
                                barrier=loaded[stage],
                            )
                            T.mbarrier_arrive(loaded[stage], 0)

            elif cta_id == 0 and tx < 64:
                for w in T.serial(waves):
                    tile_id = num_clusters * w + cluster_id
                    by_cluster = tile_id // n_blocks
                    by = by_cluster * 2 + cta_id
                    bx = tile_id % n_blocks

                    if by < m_blocks and bx < n_blocks:
                        T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                        for ko in T.serial(k_blocks):
                            phase = w * k_blocks + ko
                            stage = phase % num_stages
                            parity = (phase // num_stages) & 1

                            T.mbarrier_wait_parity(loaded[stage], parity)
                            if (w & 1) == 0:
                                T.tcgen05_gemm(
                                    A_shared[stage, :, :],
                                    B_shared[stage, :, :],
                                    C_tmem_0,
                                    transpose_A=False,
                                    transpose_B=True,
                                    mbar=consumed[stage],
                                    clear_accum=ko == 0,
                                    use_2cta=True,
                                )
                            else:
                                T.tcgen05_gemm(
                                    A_shared[stage, :, :],
                                    B_shared[stage, :, :],
                                    C_tmem_1,
                                    transpose_A=False,
                                    transpose_B=True,
                                    mbar=consumed[stage],
                                    clear_accum=ko == 0,
                                    use_2cta=True,
                                )
                        T.tcgen05_mma_arrive(tmem_full[w & 1], arrive_2cta=True)

            elif 128 <= tx < 256:
                for w in T.serial(waves):
                    tile_id = num_clusters * w + cluster_id
                    by_cluster = tile_id // n_blocks
                    by = by_cluster * 2 + cta_id
                    bx = tile_id % n_blocks

                    if by < m_blocks and bx < n_blocks:
                        T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                        T.sync_threads(1, 128)
                        if (w & 1) == 0:
                            T.copy(C_tmem_0, C_local)
                        else:
                            T.copy(C_tmem_1, C_local)
                        T.mbarrier_arrive(tmem_empty[w & 1], 0)

                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(
                                C_local[:, i * store_block_N : (i + 1) * store_block_N],
                                Y_shared,
                            )
                            T.copy(
                                Y_shared,
                                Y[
                                    by * block_M,
                                    bx * block_N + i * store_block_N,
                                ],
                            )

    return mmt_tcgen05_ws_2cta_persistent_kernel


@tilelang.jit(out_idx=[-1])
def matmul_transpose_tcgen05_ws_2cta_persistent(
    M,
    K,
    block_M=DEFAULT_CONFIG["block_M"],
    block_N=DEFAULT_CONFIG["block_N"],
    store_block_N=DEFAULT_CONFIG["store_block_N"],
    block_K=DEFAULT_CONFIG["block_K"],
    num_stages=DEFAULT_CONFIG["num_stages"],
    threads=DEFAULT_CONFIG["threads"],
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    Persistent warp-specialized 2CTA TCGEN05 kernel for Y = X @ X.T.

    The cluster persists across multiple output tiles. Both CTAs in a cluster
    cooperate on TMA loads, while the leader CTA also performs the MMA and
    epilogue for each persistent wave.
    """
    return make_mmt_tcgen05_ws_2cta_persistent_kernel(
        M=M,
        K=K,
        block_M=block_M,
        block_N=block_N,
        store_block_N=store_block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )


def matmul_transpose_tcgen05_ws_2cta_persistent_autotune(
    M,
    K,
    dtype=T.float16,
    accum_dtype=T.float32,
    configs=None,
    warmup=3,
    rep=10,
    skip_check=True,
):
    configs = get_autotune_configs(M=M) if configs is None else configs

    def kernel(
        block_M=None,
        block_N=None,
        store_block_N=None,
        block_K=None,
        num_stages=None,
        threads=None,
    ):
        return make_mmt_tcgen05_ws_2cta_persistent_kernel(
            M=M,
            K=K,
            block_M=block_M,
            block_N=block_N,
            store_block_N=store_block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype,
            accum_dtype=accum_dtype,
        )

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Normal,
            ref_prog=None if skip_check else ref_program,
            skip_check=skip_check,
        )
    )
    return autotuner.run(warmup=warmup, rep=rep)


def run_default_benchmark_suite(sizes=None, n_iter=100):
    sizes = DEFAULT_BENCHMARK_SIZES if sizes is None else sizes

    print("=" * 114)
    print(
        f"{'M':>6} {'K':>6} | {'TCGEN05-WS-2CTA-PERSIST(ms)':>29} {'PyTorch(ms)':>12} | {'Persistent/PyTorch':>20} | {'Max Err'}"
    )
    print("=" * 114)

    for M, K in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        kernel = matmul_transpose_tcgen05_ws_2cta_persistent(M, K)

        y = kernel(x)
        ref_y = ref_program(x)
        max_err = (y - ref_y).abs().max().item()

        persistent_ms = bench_ms(lambda: kernel(x), n_iter=n_iter)
        pytorch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter)

        print(
            f"{M:>6} {K:>6} | {persistent_ms:>29.4f} {pytorch_ms:>12.4f} | "
            f"{persistent_ms / pytorch_ms:>19.2f}x | {max_err:.4f}"
        )

    print("=" * 114)


def run_large_shape_benchmark(
    M=16384,
    K=16384,
    n_iter=20,
    warmup=5,
    use_autotune=False,
    tune_warmup=3,
    tune_rep=10,
    benchmark_torch=True,
):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    baseline_kernel = matmul_transpose_tcgen05_ws_2cta_persistent(M, K, **DEFAULT_CONFIG)
    baseline_ms = bench_ms(lambda: baseline_kernel(x), n_iter=n_iter, warmup=warmup)
    baseline_ms = bench_ms(lambda: baseline_kernel(x), n_iter=n_iter, warmup=warmup)
    baseline_ms = bench_ms(lambda: baseline_kernel(x), n_iter=n_iter, warmup=warmup)
    baseline_ms = bench_ms(lambda: baseline_kernel(x), n_iter=n_iter, warmup=warmup)

    best_config = DEFAULT_CONFIG.copy()
    autotune_latency_ms = None
    tuned_ms = None

    if use_autotune:
        result = matmul_transpose_tcgen05_ws_2cta_persistent_autotune(
            M=M,
            K=K,
            warmup=tune_warmup,
            rep=tune_rep,
            skip_check=True,
        )
        best_config = result.config
        autotune_latency_ms = result.latency
        tuned_kernel = matmul_transpose_tcgen05_ws_2cta_persistent(M, K, **best_config)
        tuned_ms = bench_ms(lambda: tuned_kernel(x), n_iter=n_iter, warmup=warmup)
        tuned_ms = bench_ms(lambda: tuned_kernel(x), n_iter=n_iter, warmup=warmup)
        tuned_ms = bench_ms(lambda: tuned_kernel(x), n_iter=n_iter, warmup=warmup)
        tuned_ms = bench_ms(lambda: tuned_kernel(x), n_iter=n_iter, warmup=warmup)

    torch_ms = None
    if benchmark_torch:
        torch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter, warmup=warmup)
        torch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter, warmup=warmup)
        torch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter, warmup=warmup)
        torch_ms = bench_ms(lambda: x @ x.T, n_iter=n_iter, warmup=warmup)

    print("=" * 114)
    print(f"Persistent 2CTA benchmark for M=N={M}, K={K}")
    print(f"Baseline config: {format_config(DEFAULT_CONFIG)}")
    if use_autotune:
        print(f"Search space size: {len(get_autotune_configs(M=M))}")
        print(f"Autotuner best config: {format_config(best_config)}")
        print(f"Autotuner reported latency: {autotune_latency_ms:.4f} ms")
    print("=" * 114)

    print(
        f"{'Kernel':>16} | {'Latency(ms)':>12} | {'TFLOPS':>10} | {'Vs Baseline':>11}"
        + (f" | {'Vs Torch':>9}" if torch_ms is not None else "")
    )
    print("=" * 114)
    print(
        f"{'Baseline':>16} | {baseline_ms:>12.4f} | {tflops(M, K, baseline_ms):>10.2f} | {'1.00x':>11}"
        + (f" | {baseline_ms / torch_ms:>8.2f}x" if torch_ms is not None else "")
    )
    if tuned_ms is not None:
        print(
            f"{'Autotuned':>16} | {tuned_ms:>12.4f} | {tflops(M, K, tuned_ms):>10.2f} | {baseline_ms / tuned_ms:>10.2f}x"
            + (f" | {tuned_ms / torch_ms:>8.2f}x" if torch_ms is not None else "")
        )
    if torch_ms is not None:
        print(
            f"{'PyTorch':>16} | {torch_ms:>12.4f} | {tflops(M, K, torch_ms):>10.2f} | {baseline_ms / torch_ms:>10.2f}x | {'1.00x':>9}"
        )
    print("=" * 114)

    return {
        "baseline_ms": baseline_ms,
        "baseline_tflops": tflops(M, K, baseline_ms),
        "autotune_latency_ms": autotune_latency_ms,
        "tuned_ms": tuned_ms,
        "tuned_tflops": tflops(M, K, tuned_ms) if tuned_ms is not None else None,
        "torch_ms": torch_ms,
        "torch_tflops": tflops(M, K, torch_ms) if torch_ms is not None else None,
        "best_config": best_config if use_autotune else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="store_true", help="Run the original small-shape benchmark suite.")
    parser.add_argument("--autotune", action="store_true", help="Run autotune before benchmarking.")
    parser.add_argument("--M", type=int, default=16384, help="Matrix dimension M (and N for X @ X.T).")
    parser.add_argument("--K", type=int, default=16384, help="Reduction dimension K.")
    parser.add_argument("--bench-iter", type=int, default=20, help="Timing repetitions.")
    parser.add_argument("--bench-warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--tune-warmup", type=int, default=3, help="Autotuner warmup iterations per config.")
    parser.add_argument("--tune-rep", type=int, default=10, help="Autotuner timing repetitions per config.")
    parser.add_argument("--skip-torch", action="store_true", help="Skip the PyTorch comparison benchmark.")
    return parser.parse_args()


if __name__ == "__main__":
    check_sm100()

    args = parse_args()
    if args.suite:
        run_default_benchmark_suite()
    else:
        run_large_shape_benchmark(
            M=args.M,
            K=args.K,
            n_iter=args.bench_iter,
            warmup=args.bench_warmup,
            use_autotune=args.autotune,
            tune_warmup=args.tune_warmup,
            tune_rep=args.tune_rep,
            benchmark_torch=not args.skip_torch,
        )
