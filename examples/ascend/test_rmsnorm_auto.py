"""pytest test for example_rmsnorm_auto.py — Pipelined-loop RMSNorm.

Mirrors the example's __main__ which sweeps d ∈ {4096, 5120, 7168} at batch=4096.
"""

import pytest
import torch
import tilelang

from example_rmsnorm_auto import ref_program, rms_norm_fwd


@pytest.mark.parametrize("d", [4096, 5120, 7168])
def test_rmsnorm_auto(d):
    batch = 4096
    kernel = tilelang.compile(rms_norm_fwd(batch, d, "float32"), out_idx=[1, 3])
    device = torch.device("npu")
    x = torch.randn(batch, d, dtype=torch.float32, device=device)
    weight = torch.randn(d, dtype=torch.float32, device=device)
    eps = 1e-6

    y_flat, _rstd = kernel(x.view(-1).contiguous(), weight, eps)
    torch.npu.synchronize()

    y = y_flat.view(batch, d)
    expected = ref_program(x, weight, eps)
    max_diff = (y.float() - expected).abs().max().item()
    assert max_diff < 1e-3, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    for d in [4096, 5120, 7168]:
        test_rmsnorm_auto(d)
        print(f"PASS: test_rmsnorm_auto d={d}")
