"""Test for T.simd.vcpadd on Ascend SimdVF."""

import torch

import tilelang
import tilelang.language as T


def vcpadd_kernel():
    @T.prim_func
    def main(A: T.Buffer((64,), "float32"), B: T.Buffer((32,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((32,), "float32")

            T.copy(A, a_ub)
            with T.SimdVF():
                low_half = T.simd.pset(32, "PAT_VL32")
                src = T.simd.vld(a_ub[0])
                result = T.simd.vcpadd(src)
                T.simd.vsts(b_ub[0], result, low_half, extent=32)
            T.copy(b_ub, B)

    return main


def test_simdvf_vcpadd():
    kernel = tilelang.compile(vcpadd_kernel(), out_idx=-1)
    device = torch.device("npu")
    a = torch.arange(64, dtype=torch.float32, device=device)

    result = kernel(a)

    torch.testing.assert_close(result, a.reshape(32, 2).sum(dim=1), rtol=0, atol=0)


if __name__ == "__main__":
    test_simdvf_vcpadd()
