"""pytest test for example_gemm.py — bf16 + fp32 auto GEMM."""

import torch
import tilelang
import pytest
from example_gemm import gemm, ref_program


def _test(dtype, thresh, out_dtype="float32"):
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(gemm(M, K, N, dtype=dtype, out_dtype=out_dtype), out_idx=-1)
    device = torch.device("npu")
    x = torch.randn(M, K, device=device).to(dtype=getattr(torch, dtype))
    w = torch.randn(N, K, device=device).to(dtype=getattr(torch, dtype))
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w, out_dtype=out_dtype)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < thresh, f"{dtype} max_diff={max_diff:.2e}"


def _test_pto(dtype, thresh, out_dtype="float32"):
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(
        gemm(M, K, N, dtype=dtype, out_dtype=out_dtype, MIXED=False),
        target="pto",
        out_idx=-1,
    )
    device = torch.device("npu")
    x = torch.randn(M, K, device=device).to(dtype=getattr(torch, dtype))
    w = torch.randn(N, K, device=device).to(dtype=getattr(torch, dtype))
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w, out_dtype=out_dtype)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < thresh, f"{dtype} max_diff={max_diff:.2e}"


def test_gemm_auto_fp8():
    _test("float8_e4m3fn", 1e-1)


def test_gemm_auto_bf16():
    _test("bfloat16", 1e-2)
    _test("bfloat16", 1e-2, "bfloat16")


@pytest.mark.pto
def test_gemm_auto_bf16_pto():
    _test_pto("bfloat16", 1e-2)


@pytest.mark.pto
def test_gemm_auto_fp8_pto():
    _test_pto("float8_e4m3fn", 1e-1)


def test_gemm_auto_fp32():
    _test("float32", 5e-3)


def _test_acc(out_dtype, thresh):
    """C += A @ B^T via store-mode atomic; C is an in/out buffer."""
    M, K, N = 8192, 8192, 8192
    device = torch.device("npu")
    kernel = tilelang.compile(gemm(M, K, N, dtype="bfloat16", out_dtype=out_dtype, acc=True))
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    c0 = torch.randn(M, N, device=device, dtype=getattr(torch, out_dtype))
    expected = ref_program(x, w, out_dtype=out_dtype, c=c0)
    c = c0.clone()
    kernel(x, w, c)
    torch.npu.synchronize()
    # Relative error: bf16 output rounds each element, so a max-abs bound is
    # dominated by rounding of large K=8192 accumulations.
    rel = (c.float() - expected.float()).abs().mean().item() / expected.float().abs().mean().clamp_min(1e-6).item()
    assert rel < thresh, f"acc out={out_dtype} rel={rel:.2e}"


def test_gemm_auto_acc():
    _test_acc("float32", 1e-2)
    _test_acc("bfloat16", 1e-2)


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
    test_gemm_auto_fp8()
    print("PASS: test_gemm_auto_fp8")
    test_gemm_auto_bf16()
    print("PASS: test_gemm_auto_bf16")
    test_gemm_auto_fp32()
    print("PASS: test_gemm_auto_fp32")
    test_gemm_auto_bf16_pto()
    print("PASS: test_gemm_auto_bf16_pto")
    test_gemm_auto_fp8_pto()
    print("PASS: test_gemm_auto_fp8_pto")
    test_gemm_auto_acc()
    print("PASS: test_gemm_auto_acc")
