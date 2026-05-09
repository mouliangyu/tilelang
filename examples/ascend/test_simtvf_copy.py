"""pytest test for example_simtvf_copy.py."""

import torch
import tilelang

from example_simtvf_copy import ref_program, ubuf_multi


def test_simtvf_copy():
    N = 1024
    kernel = tilelang.compile(ubuf_multi(N), out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = torch.randn(N, dtype=torch.float32, device=device)
    d = torch.randn(N, dtype=torch.float32, device=device)
    e = kernel(a, b, c, d)
    torch.npu.synchronize()
    assert torch.equal(e, ref_program(a, b, c, d))


if __name__ == "__main__":
    test_simtvf_copy()
    print("PASS: test_simtvf_copy")
