"""Correctness tests for the auto-scheduled Ascend Split-K GEMM."""

import torch
import tilelang

from example_gemm_splitk import gemm_splitk, ref_program


def _test_gemm_splitk(deterministic):
    M, K, N, split_k = 512, 8192, 512, 8
    device = torch.device("npu")
    torch.manual_seed(42)

    kernel = tilelang.compile(
        gemm_splitk(M, K, N, split_k, deterministic=deterministic),
        out_idx=-1,
    )
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    result = kernel(x, w)
    torch.npu.synchronize()

    torch.testing.assert_close(result, ref_program(x, w), rtol=1e-2, atol=1e-2)
    if deterministic:
        repeated = kernel(x, w)
        torch.npu.synchronize()
        assert torch.equal(result, repeated)


def test_gemm_splitk():
    _test_gemm_splitk(deterministic=False)


def test_gemm_splitk_deterministic():
    _test_gemm_splitk(deterministic=True)


if __name__ == "__main__":
    test_gemm_splitk()
    test_gemm_splitk_deterministic()
