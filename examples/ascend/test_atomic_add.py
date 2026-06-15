"""pytest test for example_atomic_add.py."""

import pytest
import torch
import tilelang

from example_atomic_add import atomic_add_gm_float


@pytest.mark.parametrize(
    ("num_blocks", "threads"),
    [
        (8, 256),
        (4, 128),
    ],
)
def test_atomic_add_gm_float(num_blocks, threads):
    kernel = tilelang.compile(atomic_add_gm_float(1, num_blocks, threads))
    device = torch.device("npu")
    counter = torch.zeros(1, dtype=torch.float32, device=device)
    kernel(counter)
    torch.npu.synchronize()
    expected = float(num_blocks * threads)
    assert abs(counter.item() - expected) < 0.5, f"{counter.item()} != {expected}"


if __name__ == "__main__":
    test_atomic_add_gm_float(8, 256)
    print("PASS: atomic_add GM float32")
