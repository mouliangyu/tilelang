"""pytest test for example_simdvf_vecadd_lower_pto.py."""

import torch
import tilelang

from example_simdvf_vecadd_lower_pto import ref_program, vector_add


def test_simdvf_vecadd_lower_pto():
    n = 2**30
    kernel = tilelang.compile(vector_add(n), target="pto", out_idx=-1)

    device = torch.device("npu")
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simdvf_vecadd_lower_pto()
    print("PASS: test_simdvf_vecadd_lower_pto")
