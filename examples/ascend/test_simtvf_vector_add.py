"""pytest test for example_simtvf_vector_add.py."""

import torch
import tilelang

from example_simtvf_vector_add import ref_program, vector_add


def test_simtvf_vector_add():
    N = 1024
    kernel = tilelang.compile(vector_add(N), target="ascend", out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simtvf_vector_add()
    print("PASS: test_simtvf_vector_add")
