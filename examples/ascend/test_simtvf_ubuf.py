"""pytest test for example_simtvf_ubuf.py — D = A * B + C across two SimtVF regions."""

import torch
import tilelang

from example_simtvf_ubuf import ref_program, ubuf_two_vf


def test_simtvf_ubuf():
    N = 256
    kernel = tilelang.compile(ubuf_two_vf(N), target="ascend", out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = torch.randn(N, dtype=torch.float32, device=device)
    d = kernel(a, b, c)
    torch.npu.synchronize()
    assert torch.equal(d, ref_program(a, b, c))


if __name__ == "__main__":
    test_simtvf_ubuf()
    print("PASS: test_simtvf_ubuf")
