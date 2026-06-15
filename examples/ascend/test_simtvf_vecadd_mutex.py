"""pytest test for example_simtvf_vecadd_mutex.py — mutex-based get_buf/rls_buf sync."""

import torch
import tilelang

from example_simtvf_vecadd_mutex import ref_program, vector_add


def test_simtvf_vecadd_mutex():
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
    test_simtvf_vecadd_mutex()
    print("PASS: test_simtvf_vecadd_mutex")
