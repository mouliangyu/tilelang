"""Tests for T.simd.vxor and T.simd.vnot on Ascend SimdVF."""

import torch

import tilelang
import tilelang.language as T


def bitwise_kernel(n):
    @T.prim_func
    def main(
        A: T.Buffer((n,), "int32"),
        B: T.Buffer((n,), "int32"),
        Xor: T.Buffer((n,), "int32"),
        Not: T.Buffer((n,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            b_ub = T.alloc_shared((n,), "int32")
            xor_ub = T.alloc_shared((n,), "int32")
            not_ub = T.alloc_shared((n,), "int32")

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(b_ub[0])
                T.simd.vsts(xor_ub[0], T.simd.vxor(a, b), mask)
                T.simd.vsts(not_ub[0], T.simd.vnot(a), mask)
            T.copy(xor_ub, Xor)
            T.copy(not_ub, Not)

    return main


def test_simdvf_vxor_vnot():
    n = 64
    kernel = tilelang.compile(bitwise_kernel(n), out_idx=[2, 3])
    device = torch.device("npu")
    a = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device=device)
    b = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device=device)

    xor, not_ = kernel(a, b)

    torch.testing.assert_close(xor, torch.bitwise_xor(a, b), rtol=0, atol=0)
    torch.testing.assert_close(not_, torch.bitwise_not(a), rtol=0, atol=0)


if __name__ == "__main__":
    test_simdvf_vxor_vnot()
