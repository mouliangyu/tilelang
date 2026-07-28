"""pytest test for example_vmi_vecadd.py — PTO VMI vector add end-to-end."""

from __future__ import annotations

import pytest
import torch
import tilelang

from example_vmi_vecadd import DEFAULT_N, ref_program, simulator_safe_randn, vector_add


pytest.importorskip("torch_npu")


@pytest.mark.pto
def test_vmi_vecadd_pto():
    kernel = tilelang.compile(vector_add(DEFAULT_N), target="pto", out_idx=-1)
    device = torch.device("npu")
    a = simulator_safe_randn(DEFAULT_N, dtype=torch.float32, device=device)
    b = simulator_safe_randn(DEFAULT_N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()

    expected = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_vmi_vecadd_pto()
    print("PASS: test_vmi_vecadd_pto")
