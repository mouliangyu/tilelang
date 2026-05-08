"""pytest test for example_gemm_auto.py — auto-scheduled Mix GEMM (W is N x K)."""

import torch
import tilelang

from example_gemm_auto import gemm, ref_program


def test_gemm_auto():
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(gemm(M, K, N), out_idx=-1)
    device = torch.device("npu")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_gemm_auto()
    print("PASS: test_gemm_auto")
