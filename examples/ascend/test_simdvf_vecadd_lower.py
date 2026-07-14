"""pytest test for example_simdvf_vecadd_lower.py.

This variant exercises the SimdVF kernel with the Z3 auto-scheduler enabled
(the default), which requires AutoSchedule's MemoryAccessDetector to recognize
the simd vld/vsts buffer accesses (now emitted via tl.access_ptr).
"""

import pytest
import tilelang
import torch

from example_simdvf_vecadd_lower import ref_program, vector_add


def _run_vector_add(target):
    n = 2**30
    kernel = tilelang.compile(vector_add(n), target=target, out_idx=-1)

    device = torch.device("npu")
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(c, ref_program(a, b))


def test_simdvf_vecadd_lower_auto():
    _run_vector_add("ascend")


@pytest.mark.pto
def test_simdvf_vecadd_lower_pto():
    _run_vector_add("pto")


if __name__ == "__main__":
    test_simdvf_vecadd_lower_auto()
    print("PASS: test_simdvf_vecadd_lower_auto")
    test_simdvf_vecadd_lower_pto()
    print("PASS: test_simdvf_vecadd_lower_pto")
