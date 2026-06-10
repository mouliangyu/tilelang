"""Test Ascend reduce with explicit fragment layout + generated-code verification."""

import pytest
import torch
import tilelang
import tilelang.language as T

DTYPE_MAP = {"float32": torch.float32, "int32": torch.int32, "int64": torch.int64}
REDUCER = {"sum": "SumOp", "max": "MaxOp", "min": "MinOp"}


def make_reduce_kernel(extent, scale, dtype, op):
    threads = extent * scale

    @T.prim_func
    def kernel(
        A: T.Tensor((scale, extent), dtype),
        B: T.Tensor((scale,), dtype),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=threads):
            A_local = T.alloc_fragment((scale, extent), dtype)
            B_local = T.alloc_fragment((scale,), dtype)
            T.copy(A, A_local)
            T.annotate_layout(
                {
                    A_local: T.Fragment(
                        (scale, extent),
                        forward_fn=lambda g, e: (e * scale + g, 0),
                    )
                }
            )
            getattr(T, f"reduce_{op}")(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return kernel


def ref_program(A, op):
    return {"sum": A.sum(dim=1), "max": A.max(dim=1).values, "min": A.min(dim=1).values}[op]


def _test_one(extent, scale, dtype, op):
    td = DTYPE_MAP[dtype]
    threads = extent * scale
    expected = f"AscendAllReduce<tl::{REDUCER[op]}, {threads}, {scale}, 0>::run"

    kernel = tilelang.compile(
        make_reduce_kernel(extent, scale, dtype, op),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    assert expected in kernel.get_kernel_source(), f"Missing {expected}"

    device = torch.device("npu")
    a = (
        torch.randn(scale, extent, dtype=td, device=device)
        if td.is_floating_point
        else torch.randint(-100, 100, (scale, extent), dtype=td, device=device)
    )
    b = kernel(a)
    torch.npu.synchronize()

    max_diff = (b.float() - ref_program(a, op).float()).abs().max().item()
    assert max_diff < 1e-3, f"extent={extent} scale={scale} dtype={dtype} op={op}: max_diff={max_diff:.2e}"


@pytest.mark.parametrize(
    "extent,scale,dtype,op",
    [
        # float32  — hw_reduce + ShflXor
        (32, 1, "float32", "sum"),
        (8, 4, "float32", "sum"),
        (16, 2, "float32", "sum"),
        (8, 4, "float32", "max"),
        (8, 4, "float32", "min"),
        (64, 1, "float32", "sum"),
        (16, 4, "float32", "sum"),
        (6, 5, "float32", "sum"),
        (7, 3, "float32", "sum"),
        # int32 — hw_reduce + ShflXor
        (32, 1, "int32", "sum"),
        (8, 4, "int32", "sum"),
        (8, 4, "int32", "max"),
        (8, 4, "int32", "min"),
        (64, 1, "int32", "sum"),
        (6, 5, "int32", "sum"),
        # float16: commented out — bisheng ICE (Peephole segfault)
        # (32, 1, "float16", "sum"), (6, 5, "float16", "sum"), (8, 4, "float16", "max"),
        # bfloat16: commented out — bisheng ICE (Peephole segfault)
        # (32, 1, "bfloat16", "sum"), (8, 4, "bfloat16", "min"),
        # int64 — ShflXor only, no hw_reduce
        (32, 1, "int64", "sum"),
        (8, 4, "int64", "sum"),
    ],
)
def test_reduce(extent, scale, dtype, op):
    _test_one(extent, scale, dtype, op)


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
