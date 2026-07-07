"""pytest test for example_simdvf_topk_gate.py — TopK-gate via SimdVF selection loop."""

import pytest
import torch
import tilelang
import tilelang.testing

from example_simdvf_topk_gate import topk_gate, ref_program


@pytest.mark.parametrize(
    ("num_experts", "num_topk"),
    [
        (256, 8),
        (128, 6),
        # num_experts=161 -> row is 644B, not 32B-aligned: exercises the GM->UB
        # T.copy(..., pad_value=-inf) right-pad path in the kernel.
        (161, 8),
    ],
)
def test_topk_gate(num_experts, num_topk, num_tokens=4096):
    kernel = tilelang.compile(topk_gate(num_experts, num_topk), out_idx=-1)
    device = torch.device("npu")
    torch.manual_seed(42)
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)

    out = kernel(scores)
    torch.npu.synchronize()

    expected = ref_program(scores, num_topk)
    assert torch.equal(out, expected), (
        f"mismatch for experts={num_experts} topk={num_topk}: {(out != expected).any(dim=1).sum().item()} bad rows"
    )


if __name__ == "__main__":
    tilelang.testing.main()
