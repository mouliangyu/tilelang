"""pytest test for block-scaled GEMM with mxFP8."""

import torch
import tilelang
from example_blockscaled_gemm_auto import gemm, ref_program


def _make_sf_data(M, K, N, device):
    sf_k_e8m0 = K // 32
    sfx_e8m0 = torch.randint(120, 135, (M, sf_k_e8m0), dtype=torch.uint8, device=device)
    sfw_e8m0 = torch.randint(120, 135, (N, sf_k_e8m0), dtype=torch.uint8, device=device)
    sfx_pairs = sfx_e8m0.view(torch.uint16)
    sfw_pairs = sfw_e8m0.view(torch.uint16)
    sfx = sfx_pairs.T.contiguous()
    sfw = sfw_pairs.T.contiguous()
    return sfx, sfw, sfx_e8m0, sfw_e8m0


def _test(dtype, thresh, M=8192, K=8192, N=8192, seed=42):
    torch.manual_seed(seed)
    kernel = tilelang.compile(gemm(M, K, N, dtype=dtype), out_idx=-1)
    device = torch.device("npu")
    x = (torch.randn(M, K, device=device) * 0.1).to(getattr(torch, dtype))
    w = (torch.randn(N, K, device=device) * 0.1).to(getattr(torch, dtype))
    sfx, sfw, sfx_e8m0, sfw_e8m0 = _make_sf_data(M, K, N, device)
    c = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program(x.cpu(), w.cpu(), sfx_e8m0.cpu(), sfw_e8m0.cpu()).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (c - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"{dtype} rel_err={rel_err.item():.2e}"
    return rel_err.item()


def test_blockscaled_gemm_fp8_8192():
    err = _test("float8_e4m3fn", 1e-2)
    print(f"test_blockscaled_gemm_fp8_8192: PASS  rel_err={err:.2e}")


def test_blockscaled_gemm_fp8_4096():
    err = _test("float8_e4m3fn", 1e-2, M=4096, N=4096, K=4096, seed=123)
    print(f"test_blockscaled_gemm_fp8_4096: PASS  rel_err={err:.2e}")


if __name__ == "__main__":
    test_blockscaled_gemm_fp8_8192()
    test_blockscaled_gemm_fp8_4096()
