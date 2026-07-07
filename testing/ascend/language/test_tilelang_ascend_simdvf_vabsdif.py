"""pytest test for T.simd.vabsdif — fused abs-sub on Ascend SimdVF."""

import torch
import tilelang
import tilelang.language as T

import pytest


def vabsdif_kernel(N, dtype):
    NUM_BLOCKS = 64
    NUM_THREADS = 2048
    TILE = NUM_THREADS * 4
    TOTAL_TILES = N // (TILE * NUM_BLOCKS)
    NUM_STAGES = 2

    @T.prim_func
    def main(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((N,), dtype),
        C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            tA = T.alloc_shared((TILE,), dtype)
            tB = T.alloc_shared((TILE,), dtype)
            tC = T.alloc_shared((TILE,), dtype)
            T.annotate_buffer_versions({tA: NUM_STAGES, tB: NUM_STAGES, tC: NUM_STAGES})

            for iter in T.Pipelined(TOTAL_TILES, num_stages=NUM_STAGES):
                begin = (iter * NUM_BLOCKS + bx) * TILE
                end = (iter * NUM_BLOCKS + bx + 1) * TILE

                T.copy(A[begin:end], tA)
                T.copy(B[begin:end], tB)
                with T.SimdVF():
                    elem_width = 32 if dtype == "float32" else 16
                    mask = T.simd.pset(elem_width)
                    vec_len = 64 if dtype == "float32" else 128
                    for i in range(TILE // vec_len):
                        r0 = T.simd.vld(tA[i * vec_len])
                        r1 = T.simd.vld(tB[i * vec_len])
                        r2 = T.simd.vabsdif(r0, r1, mask)
                        T.simd.vsts(tC[i * vec_len], r2, mask)
                T.copy(tC, C[begin:end])

    return main


def _test_dtype(N, dtype):
    kernel = tilelang.compile(vabsdif_kernel(N, dtype), out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=getattr(torch, dtype), device=device)
    b = torch.randn(N, dtype=getattr(torch, dtype), device=device)
    c = kernel(a, b)
    c_ref = torch.abs(a - b)
    if not torch.equal(c, c_ref):
        raise AssertionError(f"vabsdif {dtype} mismatch")


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_simdvf_vabsdif(dtype):
    _test_dtype(N=2**20, dtype=dtype)


if __name__ == "__main__":
    for dt in ("float32", "float16"):
        test_simdvf_vabsdif(dt)
        print(f"PASS: test_simdvf_vabsdif({dt})")
