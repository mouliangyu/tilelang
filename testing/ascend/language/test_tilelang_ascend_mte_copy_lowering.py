"""Regression tests for Ascend MTE (GM<->UBuf) copy-arg lowering.

The cases pin the exact ``copy_*_align_v2`` burst arguments emitted by codegen
and validate supported layout combinations end-to-end on NPU:

* ``test_gm2ub_single_row_copy`` — a contiguous 1D GM->UBuf load must be a single
  burst of the full byte length.
* ``test_ub2gm_partial_col_copy`` — a 2D ``buf[:, :cols]`` UBuf->GM store must be
  issued as ``n_rows`` strided bursts that skip the UBuf row pitch. A previous
  heuristic collapsed any copy with ``row_bytes < 32`` into a single contiguous
  burst, reading one padded row and silently dropping the remaining rows.
* ``test_gm2ub_1d_to_2d`` — a contiguous source is split at the destination's
  physical row boundary.
* ``test_gm2ub_2d_to_2d`` — two strided sides with matching row boundaries keep
  their independent row strides.
* ``test_gm2ub_dynamic_rows_middle_dim`` — a runtime-sized strided source chooses
  the physical source row boundary instead of relying on symbolic size ordering.
* ``test_gm2ub_incompatible_2d_rejected`` — two different physical row boundaries
  are rejected until outer-loop lowering is implemented.
* ``test_gm2ub2gm_fp4_copy`` — a packed ``float4_e2m1fn`` GM->UBuf->GM copy
  must use 4-bit-based byte counts (``N`` logical elements => ``N/2`` bytes).
"""

import re

import pytest
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


def gm2ub_1d_to_2d():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS * COLS,), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, WIDTH), "int32")
            T.copy(src, buf[:, :COLS])
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_gm2ub_1d_to_2d():
    kernel = tilelang.compile(
        gm2ub_1d_to_2d(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*0,\s*0,\s*0,\s*0,\s*16,\s*256\);",
        source,
    )
    assert good is not None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * COLS, dtype=torch.int32, device=device)
    out = kernel(src)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src)


MATCHED_SRC_WIDTH = 8
MATCHED_DST_WIDTH = 16


def gm2ub_2d_to_2d():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS, MATCHED_SRC_WIDTH), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, MATCHED_DST_WIDTH), "int32")
            T.copy(src[:, :COLS], buf[:, :COLS])
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_gm2ub_2d_to_2d():
    kernel = tilelang.compile(
        gm2ub_2d_to_2d(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*0,\s*0,\s*0,\s*0,\s*32,\s*64\);",
        source,
    )
    assert good is not None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * MATCHED_SRC_WIDTH, dtype=torch.int32, device=device).reshape(ROWS, MATCHED_SRC_WIDTH)
    out = kernel(src)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src[:, :COLS].reshape(-1).contiguous())


DYNAMIC_ROWS = 4
OVERLAP = 2
DYNAMIC_WIDTH = 512


def dynamic_rows_middle_dim_copy():
    @T.prim_func
    def kernel(
        src: T.Buffer((DYNAMIC_ROWS, OVERLAP, DYNAMIC_WIDTH), "float32"),
        rows: T.Buffer((1,), "int32"),
        out: T.Buffer((DYNAMIC_ROWS, DYNAMIC_WIDTH), "float32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((DYNAMIC_ROWS, DYNAMIC_WIDTH), "float32")
            n = T.int32(rows[0])
            if n > 0:
                T.copy(src[0:n, 0, :], buf[0:n, :])
            T.copy(buf, out)

    return kernel


def test_gm2ub_dynamic_rows_middle_dim():
    kernel = tilelang.compile(
        dynamic_rows_middle_dim_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*n,\s*2048,\s*0,\s*0,\s*0,\s*0,\s*4096,\s*2048\);",
        source,
    )
    bad = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*\(n \* 2048\),\s*0,\s*0,\s*0,\s*0,",
        source,
    )
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    src = torch.arange(DYNAMIC_ROWS * OVERLAP * DYNAMIC_WIDTH, dtype=torch.float32, device=device).reshape(
        DYNAMIC_ROWS, OVERLAP, DYNAMIC_WIDTH
    )
    rows = torch.tensor([DYNAMIC_ROWS], dtype=torch.int32, device=device)
    out = kernel(src, rows)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src[:, 0, :])


INCOMPATIBLE_A = 2
INCOMPATIBLE_B = 3
INCOMPATIBLE_C = 4
INCOMPATIBLE_SRC_A_STRIDE = INCOMPATIBLE_B * INCOMPATIBLE_C + 2
INCOMPATIBLE_DST_C_PAD = 1


def gm2ub_incompatible_2d():
    @T.prim_func
    def kernel(
        src: T.StridedTensor(
            shape=[INCOMPATIBLE_A, INCOMPATIBLE_B, INCOMPATIBLE_C],
            strides=[INCOMPATIBLE_SRC_A_STRIDE, INCOMPATIBLE_C, 1],
            dtype="float32",
        ),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared(
                (INCOMPATIBLE_A, INCOMPATIBLE_B, INCOMPATIBLE_C + INCOMPATIBLE_DST_C_PAD),
                "float32",
            )
            T.copy(src, buf[:, :, :INCOMPATIBLE_C])

    return kernel


def test_gm2ub_incompatible_2d_rejected():
    with pytest.raises(Exception, match="incompatible 2D row boundaries require outer-loop lowering"):
        tilelang.compile(
            gm2ub_incompatible_2d(),
            target="ascend",
        )


FP4_N = 1024


def fp4_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((FP4_N,), T.float4_e2m1fn),
        dst: T.Tensor((FP4_N,), T.float4_e2m1fn),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((FP4_N,), T.float4_e2m1fn)
            T.copy(src, buf)
            T.copy(buf, dst)

    return kernel


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_gm2ub2gm_fp4_copy():
    kernel = tilelang.compile(
        fp4_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good_gm2ub = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*512,\s*0,\s*0,\s*0,\s*0,\s*512,\s*512\);",
        source,
    )
    good_ub2gm = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*1,\s*512,\s*4,\s*512,\s*512\);",
        source,
    )
    assert good_gm2ub is not None, source
    assert good_ub2gm is not None, source

    device = torch.device("npu")
    nbytes = FP4_N // 2
    src_bytes = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cpu").to(device)
    src_fp4 = src_bytes.view(torch.float4_e2m1fn_x2)
    out = kernel(src_fp4)
    torch.npu.synchronize()

    assert out.shape == (nbytes,)
    assert out.dtype == torch.float4_e2m1fn_x2
    out_bytes = out.view(torch.uint8)
    torch.testing.assert_close(out_bytes.cpu(), src_bytes.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
