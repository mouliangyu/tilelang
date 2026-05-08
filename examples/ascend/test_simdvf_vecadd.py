"""pytest test for example_simdvf_vecadd.py.

Note: the SimdVF kernel is currently incompatible with the Z3 auto-scheduler
(reports 'no feasible II found'); we compile with auto-schedule disabled.
"""

import torch
import tilelang

from example_simdvf_vecadd import ref_program, vector_add


def test_simdvf_vecadd():
    N = 2**30
    kernel = tilelang.compile(
        vector_add(N),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simdvf_vecadd()
    print("PASS: test_simdvf_vecadd")
