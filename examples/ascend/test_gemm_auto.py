"""pytest test for example_gemm_auto.py — bf16 + fp32 auto GEMM."""

import torch
import tilelang
from example_gemm_auto import gemm, ref_program


def _test(dtype, thresh):
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(gemm(M, K, N, dtype=dtype), out_idx=-1)
    device = torch.device("npu")
    x = torch.randn(M, K, dtype=getattr(torch, dtype), device=device)
    w = torch.randn(N, K, dtype=getattr(torch, dtype), device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < thresh, f"{dtype} max_diff={max_diff:.2e}"


def test_gemm_auto_bf16():
    _test("bfloat16", 1e-2)


def test_gemm_auto_fp32():
    _test("float32", 5e-3)


def test_gemm_auto_fp32_hf32():
    M, K, N = 8192, 8192, 8192
    device = torch.device("npu")
    torch.manual_seed(42)
    for hf32 in ["nearest_zero", "nearest_even"]:
        kernel = tilelang.compile(gemm(M, K, N, dtype="float32", hf32=hf32), out_idx=-1)
        x = torch.randn(M, K, dtype=torch.float32, device=device)
        w = torch.randn(N, K, dtype=torch.float32, device=device)
        c = kernel(x, w)
        torch.npu.synchronize()
        expected = ref_program(x, w)
        max_diff = (c - expected).abs().max().item()
        assert max_diff < 2e-1, f"{hf32} max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_gemm_auto_bf16()
    print("PASS: test_gemm_auto_bf16")
    test_gemm_auto_fp32()
    print("PASS: test_gemm_auto_fp32")
