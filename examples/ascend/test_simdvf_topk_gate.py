"""pytest test for example_simdvf_topk_gate.py — TopK-gate via SimdVF selection loop."""

import torch
import tilelang

from example_simdvf_topk_gate import topk_gate, ref_program


def _run(num_experts: int, num_topk: int, num_tokens: int = 4096):
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


def test_topk_gate_256_8():
    _run(256, 8)


def test_topk_gate_128_6():
    _run(128, 6)


def test_topk_gate_161_8():
    _run(161, 8)


if __name__ == "__main__":
    test_topk_gate_256_8()
    test_topk_gate_128_6()
    test_topk_gate_161_8()
    print("PASS: test_simdvf_topk_gate")
