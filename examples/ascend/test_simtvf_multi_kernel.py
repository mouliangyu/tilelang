"""pytest test for example_simtvf_multi_kernel.py.

Verifies two distinct SimtVF prim funcs in one file compile and run correctly —
guards the VF-helper namespacing (``<global_symbol>_simt_vf_<idx>``) that keeps
multiple kernels from colliding on a bare ``simt_vf_0``.
"""

import pytest
import torch
import tilelang

from example_simtvf_multi_kernel import (
    ref_add,
    ref_scale,
    vector_add,
    vector_scale,
)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_simtvf_multi_kernel(target):
    N = 1024
    SCALE = 3.0
    device = torch.device("npu")

    add_kernel = tilelang.compile(vector_add(N), target=target, out_idx=-1)
    scale_kernel = tilelang.compile(vector_scale(N, SCALE), target=target, out_idx=-1)

    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)

    c_add = add_kernel(a, b)
    c_scale = scale_kernel(a)
    torch.npu.synchronize()

    assert torch.equal(c_add, ref_add(a, b))
    assert torch.equal(c_scale, ref_scale(a, SCALE))


if __name__ == "__main__":
    test_simtvf_multi_kernel(target="ascend")
    print("PASS: test_simtvf_multi_kernel (ascend)")
    test_simtvf_multi_kernel(target="pto")
    print("PASS: test_simtvf_multi_kernel (pto)")
