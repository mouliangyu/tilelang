import tilelang
import tilelang.language as T

from tilelang.profiler import do_bench


def atomic_add_gm_float(N, num_blocks=8, threads=256):
    """Atomically add 1.0 into a GM float counter from each thread.

    Total expected = num_blocks * threads * 1.0.
    """

    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(num_blocks), T.SimtVF(threads=threads):
            for _ in T.Parallel(threads):
                T.atomic_add(counter[0], 1.0)

    return main


def run_regression_perf(num_blocks=64, threads=2048):
    import torch

    device = torch.device("npu")
    bench_kernel = tilelang.compile(atomic_add_gm_float(1, num_blocks, threads))
    counter = torch.zeros(1, dtype=torch.float32, device=device)

    def run():
        bench_kernel(counter)

    run()
    torch.npu.synchronize()
    latency_ms = do_bench(run, backend="msprof", _n_warmup=10, _n_repeat=50)
    print(f"  {latency_ms:.3f} ms/iter")
    return latency_ms


if __name__ == "__main__":
    import torch

    device = torch.device("npu")

    print("=== float32 GM atomic_add ===")
    num_blocks, threads = 8, 256
    program = atomic_add_gm_float(1, num_blocks, threads)
    kernel = tilelang.compile(program, out_idx=-1)
    print(kernel.get_kernel_source())

    counter = torch.zeros(1, dtype=torch.float32, device=device)
    kernel(counter)
    torch.npu.synchronize()
    expected = float(num_blocks * threads)
    actual = counter.item()
    print(f"  Expected: {expected}, Got: {actual}")
    assert abs(actual - expected) < 0.5, f"Mismatch: {actual} != {expected}"
    print("  PASS")

    # Benchmark
    print("\n=== Benchmark ===")
    bench_kernel = tilelang.compile(atomic_add_gm_float(1, 64, 2048), out_idx=-1)
    counter = torch.zeros(1, dtype=torch.float32, device=device)

    def run():
        bench_kernel(counter)

    latency_ms = do_bench(run, backend="msprof", _n_warmup=10, _n_repeat=50)
    print(f"  {latency_ms:.3f} ms/iter")
