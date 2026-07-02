"""
Test: call the nd2nz scatter intrinsic directly (no T.copy, no Cube).

Does:  UB ND (32, 64) float -> UB NZ (33, 64) bfloat16 via scatter.
Then writes the NZ result back to GM and prints it.
"""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import tirx


def _to_dtype_str(s) -> str:
    if s is torch.bfloat16:
        return "bfloat16_t"
    if s is torch.float32:
        return "float"
    raise ValueError(f"Unsupported nd2nz dtype: {s}")


def nd2nz_scatter_test(sd: torch.dtype, dd: torch.dtype, rows: int, cols: int):
    """Scatter a float tile from ND layout to bfloat16 NZ layout in UB.

    ND layout:  (rows, cols) dense
    NZ layout:  (rows + 1, cols) with padding row to avoid bank conflict
    """

    @T.prim_func
    def main(
        input_gm: T.Buffer((rows, cols), sd),
        output_gm: T.Buffer((rows + 1, cols), dd),
    ):
        with T.Kernel(1):
            with T.Cube():
                pass
            with T.Vector() as sid:
                if sid == 0:
                    # UB buffers (allocated outside SimtVF so they survive across barriers)
                    src_ub = T.alloc_shared((rows, cols), sd)  # ND layout
                    dst_ub = T.alloc_shared((rows + 1, cols), dd)  # NZ layout
                    T.copy(input_gm, src_ub)
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.ascend_nd2nz_scatter"),
                        T.access_ptr(src_ub, "r"),
                        T.access_ptr(dst_ub, "w"),
                        rows,
                        cols,
                        _to_dtype_str(dd),  # dst dtype
                        _to_dtype_str(sd),  # src dtype
                    )
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.copy(dst_ub, output_gm)

    return main


ROWS = 32
COLS = 128


@pytest.mark.parametrize(
    "sd,dd",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_nd2nz_scatter(sd, dd):
    # Prepare input: a recognizable pattern
    input_gm = torch.arange(ROWS * COLS, dtype=sd, device="npu").reshape(ROWS, COLS) + 1.0

    program = nd2nz_scatter_test(sd, dd, ROWS, COLS)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )

    output_gm = kernel(input_gm)
    torch.npu.synchronize()

    frac_len = 32 // output_gm.element_size()
    input_cast = input_gm.to(dtype=dd)
    output_back = output_gm.view(COLS // frac_len, ROWS + 1, frac_len)[:, :-1, :].permute(1, 0, 2).reshape(ROWS, COLS)
    assert torch.equal(input_cast, output_back), f"sd={sd} dd={dd}"


if __name__ == "__main__":
    tilelang.testing.main()
