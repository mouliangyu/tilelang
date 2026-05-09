"""pytest test for example_gemm_mix.py — Mix kernel (AIC + AIV) GEMM (W is N x K)."""

import torch
import tilelang

from example_gemm_mix import gemm, ref_program


def test_gemm_mix():
    M, K, N = 8192, 8192, 8192
    kernel = tilelang.compile(
        gemm(M, K, N),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_gemm_mix()
    print("PASS: test_gemm_mix")
