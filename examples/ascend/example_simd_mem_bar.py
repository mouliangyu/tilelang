"""Demonstrate T.simd.mem_bar — SIMD memory barrier for Ascend vector operations.

mem_bar(VST_VLD) ensures all pending vector stores complete before subsequent
vector loads begin. This is useful when you need to enforce ordering between
read and write operations targeting overlapping UBuf regions.

Note: T.SimdVF() can only access UBuf (shared memory), not GM.
"""

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


def mem_bar_demo(N):
    """Compute C = A * (A + B) with explicit mem_bar between vector ops.

    mem_bar(VST_VLD) is inserted to guarantee the store to temp_acc is visible
    before the subsequent read of temp_acc for the multiply step.
    """
    TILE = 8192

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1):
            # UBuf buffers (SimdVF only touches these)
            ubuf_a = T.alloc_shared((TILE,), "float32")
            ubuf_b = T.alloc_shared((TILE,), "float32")
            ubuf_acc = T.alloc_shared((TILE,), "float32")
            ubuf_out = T.alloc_shared((TILE,), "float32")

            T.ascend_set_flag("MTE3_V", 0)
            T.ascend_set_flag("V_MTE2", 0)

            for tile in range(N // TILE):
                begin = tile * TILE
                end = (tile + 1) * TILE

                # DMA: GM -> UBuf
                T.ascend_wait_flag("V_MTE2", 0)
                T.copy(A[begin:end], ubuf_a)
                T.copy(B[begin:end], ubuf_b)
                T.ascend_set_flag("MTE2_V", 0)

                # Vector compute on UBuf only
                T.ascend_wait_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE3_V", 0)
                with T.SimdVF():
                    # Step 1: ubuf_acc = ubuf_a + ubuf_b
                    for i in T.Parallel(TILE):
                        ubuf_acc[i] = ubuf_a[i] + ubuf_b[i]
                    # mem_bar ensures the stores to ubuf_acc are visible
                    # before the next step reads ubuf_acc
                    T.simd.mem_bar("VST_VLD")
                    # Step 2: ubuf_out = ubuf_a * ubuf_acc
                    for i in T.Parallel(TILE):
                        ubuf_out[i] = ubuf_a[i] * ubuf_acc[i]
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_set_flag("V_MTE2", 0)

                # DMA: UBuf -> GM
                T.ascend_wait_flag("V_MTE3", 0)
                T.copy(ubuf_out, C[begin:end])
                T.ascend_set_flag("MTE3_V", 0)

            T.ascend_wait_flag("MTE3_V", 0)
            T.ascend_wait_flag("V_MTE2", 0)

    return main


def ref_program(a, b):
    return a * (a + b)


if __name__ == "__main__":
    import torch

    N = 2**20
    NUM_REPEATS = 10

    device = torch.device("npu")

    print(f"Compiling mem_bar_demo kernel (N={N})...")
    program = mem_bar_demo(N)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    print("Compilation succeeded!")

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    torch.manual_seed(42)
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)

    print("\nRunning kernel...")
    c = kernel(a, b)
    torch.npu.synchronize()

    expected = ref_program(a, b)
    max_diff = torch.max(torch.abs(c - expected)).item()
    status = "PASS" if max_diff == 0 else f"FAIL (max diff={max_diff:.6f})"
    print(f"Verification: {status}")

    latency_ms = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    print(f"{latency_ms:.1f} ms/iter")
