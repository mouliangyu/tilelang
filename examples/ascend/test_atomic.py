"""pytest tests for example_atomic.py."""

import pytest
import torch
import tilelang

from example_atomic import atomic_add_gm_float, atomic_max_gm_float, atomic_min_gm_float


@pytest.mark.parametrize(
    ("num_blocks", "threads"),
    [
        (8, 256),
        (4, 128),
    ],
)
@pytest.mark.parametrize(
    ("op_name", "program_factory", "initial"),
    [
        pytest.param("atomic_add", atomic_add_gm_float, 0.0, id="add"),
        pytest.param("atomic_max", atomic_max_gm_float, -float("inf"), id="max"),
        pytest.param("atomic_min", atomic_min_gm_float, float("inf"), id="min"),
    ],
)
def test_atomic_gm_float(op_name, program_factory, initial, num_blocks, threads):
    kernel = tilelang.compile(program_factory(1, num_blocks, threads))
    device = torch.device("npu")
    counter = torch.full((1,), initial, dtype=torch.float32, device=device)
    kernel(counter)
    torch.npu.synchronize()
    num_updates = num_blocks * threads
    expected = {
        "atomic_add": float(num_updates),
        "atomic_max": float(num_updates - 1),
        "atomic_min": 0.0,
    }[op_name]
    assert abs(counter.item() - expected) < 0.5, f"{op_name}: {counter.item()} != {expected}"


if __name__ == "__main__":
    tilelang.testing.main()
