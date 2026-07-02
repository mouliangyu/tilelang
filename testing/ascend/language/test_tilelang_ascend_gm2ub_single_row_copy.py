"""Regression test for single-row GM->UBuf lowering."""

import re

import torch
import tilelang
import tilelang.testing
from tilelang import language as T


def single_row_copy(n: int):
    @T.prim_func
    def kernel(
        a: T.Buffer((n,), "float32"),
        out: T.Buffer((n,), "float32"),
    ):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((n,), "float32")
            T.copy(a[:n], temp)
            T.copy(temp, out[:n])

    return kernel


def test_gm2ub_single_row_copy():
    n = 1024  # 4096 bytes => lenBurst should be 128
    kernel = tilelang.compile(
        single_row_copy(n),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*4096,\s*0,\s*0,\s*0,\s*0,\s*4096,\s*4096\);",
        source,
    )
    bad = re.search(r"copy_gm_to_ubuf\(", source)
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    a = torch.arange(n, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a)


if __name__ == "__main__":
    tilelang.testing.main()
