"""pytest test for example_make_tensor_ptr.py."""

import torch
import tilelang

from example_make_tensor_ptr import make_ptr_table, make_tensor_copy, make_tensor_copy_both


def test_make_tensor_ptr():
    n = 64
    kernel = tilelang.compile(make_tensor_copy(n), target="ascend")
    device = torch.device("npu")
    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    src_ptrs = make_ptr_table([src])
    kernel(src_ptrs, out)
    torch.npu.synchronize()
    assert torch.equal(out, src)


def test_make_tensor_ptr_both():
    n = 64
    kernel = tilelang.compile(make_tensor_copy_both(n), target="ascend")
    device = torch.device("npu")
    src = torch.randn(n, dtype=torch.float32, device=device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    kernel(make_ptr_table([src]), make_ptr_table([out]))
    torch.npu.synchronize()
    assert torch.equal(out, src)


if __name__ == "__main__":
    test_make_tensor_ptr()
    print("PASS: test_make_tensor_ptr")
    test_make_tensor_ptr_both()
    print("PASS: test_make_tensor_ptr_both")
