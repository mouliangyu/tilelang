import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def vector_add(N):
    NUM_BLOCKS = 1
    LANES = 64
    TILE_ELEMS = LANES
    if N % (TILE_ELEMS * NUM_BLOCKS) != 0:
        raise ValueError(f"N must be a multiple of {TILE_ELEMS * NUM_BLOCKS}, got {N}")
    TOTAL_TILES = N // (TILE_ELEMS * NUM_BLOCKS)
    NUM_STAGES = 2

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            a_ub = T.alloc_shared((TILE_ELEMS,), "float32")
            b_ub = T.alloc_shared((TILE_ELEMS,), "float32")
            c_ub = T.alloc_shared((TILE_ELEMS,), "float32")
            T.annotate_buffer_versions({a_ub: NUM_STAGES, b_ub: NUM_STAGES, c_ub: NUM_STAGES})

            for tile in T.Pipelined(TOTAL_TILES, num_stages=NUM_STAGES):
                begin = (tile * NUM_BLOCKS + bx) * TILE_ELEMS
                end = begin + TILE_ELEMS

                T.copy(A[begin:end], a_ub)
                T.copy(B[begin:end], b_ub)
                with T.SimdVF():
                    mask = T.vmi.create_mask(LANES, size=LANES)
                    for i in range(TILE_ELEMS // LANES):
                        a_vec = T.vmi.vload(a_ub[i * LANES], size=LANES)
                        b_vec = T.vmi.vload(b_ub[i * LANES], size=LANES)
                        c_vec = T.vmi.vadd(a_vec, b_vec, mask)
                        T.vmi.vstore(c_vec, c_ub[i * LANES], mask)
                T.copy(c_ub, C[begin:end])

    return main


def ref_program(a, b):
    return a.cpu() + b.cpu()


NUM_REPEATS = 10
DEFAULT_N = 64


def simulator_safe_randn(shape, *, dtype, device):
    """Generate random inputs without using NPU-side normal_."""
    import torch

    return torch.randn(shape, dtype=dtype, device="cpu").to(device)


def run_regression_perf(N=DEFAULT_N):
    import torch

    device = torch.device("npu")
    program = vector_add(N)
    kernel = tilelang.compile(program, target="pto", out_idx=-1)

    a = simulator_safe_randn(N, dtype=torch.float32, device=device)
    b = simulator_safe_randn(N, dtype=torch.float32, device=device)
    kernel(a, b)
    torch.npu.synchronize()

    prof = do_bench(lambda: kernel(a, b), backend="msprof_detail", _n_warmup=30, _n_repeat=NUM_REPEATS)
    nbytes = N * a.element_size() * 3
    print(f"    [N={N}] {prof.dur_us:.2f} us/iter  |  {prof.gbps(nbytes):.1f} GB/s")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Run the VMI vector-add example.")
    parser.add_argument("--bench", action="store_true", help="Run the msprof performance benchmark.")
    args = parser.parse_args()

    N = DEFAULT_N
    dtype = torch.float32
    device = torch.device("npu")

    print(f"Compiling VMI vector_add kernel (N={N})...")
    program = vector_add(N)
    kernel = tilelang.compile(program, target="pto", out_idx=-1)
    print("Compilation succeeded!")

    print("\n--- Generated PTO Source ---")
    print(kernel.get_kernel_source())

    a = simulator_safe_randn(N, dtype=dtype, device=device)
    b = simulator_safe_randn(N, dtype=dtype, device=device)

    print("Running kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    expected = ref_program(a, b)
    c_cpu = c.cpu()
    if not torch.equal(c_cpu, expected):
        max_diff = torch.max(torch.abs(c_cpu - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")

    if args.bench:
        latency_ms = run_regression_perf(N)
        print(f"{latency_ms:.1f} ms/iter")
