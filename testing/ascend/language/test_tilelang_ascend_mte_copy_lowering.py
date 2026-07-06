"""Regression tests for Ascend MTE (GM<->UBuf) copy-arg lowering.

Both cases pin the exact ``copy_*_align_v2`` burst arguments emitted by codegen
and validate the data end-to-end on NPU:

* ``test_gm2ub_single_row_copy`` — a contiguous 1D GM->UBuf load must be a single
  burst of the full byte length.
* ``test_ub2gm_partial_col_copy`` — a 2D ``buf[:, :cols]`` UBuf->GM store must be
  issued as ``n_rows`` strided bursts that skip the UBuf row pitch. A previous
  heuristic collapsed any copy with ``row_bytes < 32`` into a single contiguous
  burst, reading one padded row and silently dropping the remaining rows.
"""

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


ROWS = 4
WIDTH = 64
COLS = 4


def partial_col_copy():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS, WIDTH), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, WIDTH), "int32")
            T.copy(src, buf)
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_ub2gm_partial_col_copy():
    kernel = tilelang.compile(
        partial_col_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    # 4 bursts of 4 int32 (16B), source stride = row pitch (64*4 = 256B),
    # dest stride = contiguous 16B. cacheMode for a store defaults to 4.
    good = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*4,\s*16,\s*256\);",
        source,
    )
    # The buggy collapse emitted a single 64B contiguous burst.
    bad = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*1,\s*64,\s*4,\s*64,\s*64\);",
        source,
    )
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * WIDTH, dtype=torch.int32, device=device).reshape(ROWS, WIDTH)
    out = kernel(src)
    torch.npu.synchronize()

    expect = src[:, :COLS].reshape(-1).contiguous()
    torch.testing.assert_close(out, expect)


if __name__ == "__main__":
    tilelang.testing.main()
