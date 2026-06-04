"""pytest test for example_simtvf_vecadd.py."""

import pytest
import torch
import tilelang

from example_simtvf_vecadd import ref_program, vector_add


@pytest.mark.parametrize(
    ("tl_dtype", "torch_dtype"),
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_simtvf_vecadd(tl_dtype, torch_dtype):
    N = 2**30
    kernel = tilelang.compile(
        vector_add(N, tl_dtype),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    device = torch.device("npu")
    a = torch.randn(N, dtype=torch_dtype, device=device)
    b = torch.randn(N, dtype=torch_dtype, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()
    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_simtvf_vecadd("float32", torch.float32)
    test_simtvf_vecadd("bfloat16", torch.bfloat16)
    print("PASS: test_simtvf_vecadd")
