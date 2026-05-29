"""pytest test for example_simdvf_vecadd_lower_auto.py.

This variant exercises the SimdVF kernel with the Z3 auto-scheduler enabled
(the default), which requires AutoSchedule's MemoryAccessDetector to recognize
the simd vld/vsts buffer accesses (now emitted via tl.access_ptr).
"""

import torch
import tilelang

from example_simdvf_vecadd_lower_auto import ref_program, vector_add


def test_simdvf_vecadd_lower_auto():
    N = 2**30
    kernel = tilelang.compile(vector_add(N), out_idx=-1)
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simdvf_vecadd_lower_auto()
    print("PASS: test_simdvf_vecadd_lower_auto")
