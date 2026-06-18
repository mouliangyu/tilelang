"""pytest test for example_simdvf_scalar_topk_scalar_write.py — scalar GM write topk."""

import torch

from example_simdvf_scalar_topk_scalar_write import make_kernel, ref_program

NUM_EXPERTS = 128
NUM_TOPK = 8
NUM_CORES = 32


def test_simdvf_scalar_topk_scalar_write():
    kernel = make_kernel()
    device = torch.device("npu")
    n = 4096
    torch.manual_seed(42)
    logits = torch.randn(n, NUM_EXPERTS, dtype=torch.float32, device=device)
    out = kernel(logits)
    torch.npu.synchronize()
    assert torch.equal(out, ref_program(logits, NUM_TOPK))


if __name__ == "__main__":
    test_simdvf_scalar_topk_scalar_write()
    print("PASS: test_simdvf_scalar_topk_scalar_write")
