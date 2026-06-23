"""Example: T.make_tensor from a pointer table on Ascend NPU.

The kernel receives a table of int64 device addresses and reconstructs a tensor
view onto one of them with T.make_tensor. The resulting buffer is backed by a
LetStmt-bound pointer (reinterpret of the address) carrying a `global`
(`__gm__`) storage scope, which the Ascend codegen must materialize as a scoped
pointer rather than a scope-less `void*`.
"""

import tilelang
import tilelang.language as T


def make_tensor_copy(n):
    """Copy src (addressed via a pointer table) to out, staging through UB."""

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1) as _:
            src = T.make_tensor(src_ptrs[0], (n,), "float32")
            buf = T.alloc_shared(n, "float32")
            T.copy(src[0], buf)
            T.copy(buf, out)

    return main


def make_tensor_copy_both(n):
    """Both src and out are addressed via pointer tables, staging through UB.

    Exercises two T.make_tensor pointers in one kernel: each is backed by its
    own LetStmt-bound `data` var, and the first copy uses src before out is
    even bound. The scheduler must keep each pointer binding ahead of its use.
    """

    @T.prim_func
    def main(
        src_ptrs: T.Tensor((1,), T.ptr),
        out_ptrs: T.Tensor((1,), T.ptr),
    ):
        with T.Kernel(1) as _:
            src = T.make_tensor(src_ptrs[0], (n,), "float32")
            out = T.make_tensor(out_ptrs[0], (n,), "float32")
            buf = T.alloc_shared(n, "float32")
            T.copy(src[0], buf)
            T.copy(buf, out[0])

    return main


def make_ptr_table(tensors):
    import torch

    device = tensors[0].device
    return torch.tensor([t.data_ptr() for t in tensors], device=device, dtype=torch.int64)


if __name__ == "__main__":
    import torch

    n = 64
    device = torch.device("npu")

    print(f"Compiling make_tensor_copy kernel (n={n})...")
    kernel = tilelang.compile(make_tensor_copy(n), target="ascend")
    print("Compilation succeeded!")

    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    src_ptrs = make_ptr_table([src])

    kernel(src_ptrs, out)
    torch.npu.synchronize()

    if not torch.equal(out, src):
        max_diff = torch.max(torch.abs(out - src)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("Verification passed!")

    print(f"Compiling make_tensor_copy_both kernel (n={n})...")
    kernel_both = tilelang.compile(make_tensor_copy_both(n), target="ascend")
    print("Compilation succeeded!")

    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)

    kernel_both(make_ptr_table([src]), make_ptr_table([out]))
    torch.npu.synchronize()

    if not torch.equal(out, src):
        max_diff = torch.max(torch.abs(out - src)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("Verification passed!")
