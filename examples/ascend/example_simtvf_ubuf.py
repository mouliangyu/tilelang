"""Example: SimtVF with __ubuf__ shared memory on Ascend NPU.

This example demonstrates using T.alloc_shared to allocate Unified Buffer (UB)
memory for intermediate results shared between two SimtVF regions.

The kernel computes: D = A * B + C

VF1: computes A * B → stores to UB (temp_ub)
VF2: reads UB + C → stores to D
"""

import tilelang
import tilelang.language as T


def ubuf_two_vf(N):
    """Two-VF kernel with shared UB buffer.

    The alloc_shared must be placed OUTSIDE the SimtVF regions.
    The shared buffer is allocated in the Unified Buffer (on-chip memory).
    """

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
        D: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _:
            # alloc_shared OUTSIDE SimtVF regions
            temp_ub = T.alloc_shared((N,), "float32")

            # VF1: compute A * B → temp_ub
            with T.SimtVF(threads=128):
                for i in T.Parallel(N):
                    temp_ub[i] = A[i] * B[i]

            # VF2: compute temp_ub + C → D
            with T.SimtVF(threads=128):
                for i in T.Parallel(N):
                    D[i] = temp_ub[i] + C[i]

    return main


def ref_program(a, b, c):
    """Reference implementation: D = A * B + C"""
    return a * b + c


if __name__ == "__main__":
    import torch

    N = 256
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling ubuf_two_vf kernel (N={N})...")
    program = ubuf_two_vf(N)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    print("Compilation succeeded!")

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)
    c = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    d = kernel(a, b, c)
    torch.npu.synchronize()

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Verify
    expected = ref_program(a, b, c)
    if not torch.equal(d, expected):
        max_diff = torch.max(torch.abs(d - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")
