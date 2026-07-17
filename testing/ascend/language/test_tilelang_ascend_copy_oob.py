"""Runtime tests for automatic OOB tail clamping in Ascend T.copy."""

import torch
import tilelang
import tilelang.testing
from tilelang import language as T


HIDDEN = 576
BLOCK_M = 128
BLOCK_K = 128
GEMM_TILE_M = 128
GEMM_TILE_N = 128
GEMM_TILE_K = 128


@tilelang.jit()
def make_1d_dynamic_copy(hidden: int, block_k: int = 128):
    @T.prim_func
    def copy_1d(
        x: T.Tensor[(hidden,), T.float32],
        out: T.Tensor[(hidden,), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(hidden, block_k)) as pid:
            x_ub = T.alloc_shared((block_k,), "float32")
            offset = pid * block_k

            T.copy(x[offset : offset + block_k], x_ub)
            T.copy(x_ub, out[offset : offset + block_k])

    return copy_1d


@tilelang.jit()
def make_2d_dynamic_inner_copy(hidden: int, block_m: int = 128, block_k: int = 128):
    @T.prim_func
    def copy_2d(
        x: T.Tensor[(block_m, hidden), T.float32],
        out: T.Tensor[(block_m, hidden), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(hidden, block_k)) as pid:
            x_ub = T.alloc_shared((block_m, block_k), "float32")
            offset = pid * block_k

            T.copy(x[0:block_m, offset : offset + block_k], x_ub)
            T.copy(x_ub, out[0:block_m, offset : offset + block_k])

    return copy_2d


def make_bf16_gemm_oob_kernel(m: int, n: int, k: int):
    m_tiles = (m + GEMM_TILE_M - 1) // GEMM_TILE_M
    n_tiles = (n + GEMM_TILE_N - 1) // GEMM_TILE_N
    k_tiles = (k + GEMM_TILE_K - 1) // GEMM_TILE_K

    @T.prim_func
    def kernel(
        A: T.Buffer((m, k), "bfloat16"),
        B: T.Buffer((n, k), "bfloat16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(m_tiles * n_tiles) as bx:
            m_tile = bx // n_tiles
            n_tile = bx % n_tiles
            a_l1 = T.alloc_l1((GEMM_TILE_M, GEMM_TILE_K), "bfloat16")
            b_l1 = T.alloc_l1((GEMM_TILE_N, GEMM_TILE_K), "bfloat16")
            a_l0 = T.alloc_l0a((GEMM_TILE_M, GEMM_TILE_K), "bfloat16")
            b_l0 = T.alloc_l0b((GEMM_TILE_N, GEMM_TILE_K), "bfloat16")
            acc = T.alloc_l0c((GEMM_TILE_M, GEMM_TILE_N), "float32")

            for kt in T.serial(k_tiles):
                # Edge requests use full tiles; lowering clamps each dimension
                # to the remaining logical GM region.
                T.copy(
                    A[
                        m_tile * GEMM_TILE_M : (m_tile + 1) * GEMM_TILE_M,
                        kt * GEMM_TILE_K : (kt + 1) * GEMM_TILE_K,
                    ],
                    a_l1,
                )
                T.copy(
                    B[
                        n_tile * GEMM_TILE_N : (n_tile + 1) * GEMM_TILE_N,
                        kt * GEMM_TILE_K : (kt + 1) * GEMM_TILE_K,
                    ],
                    b_l1,
                )
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=(kt == 0))

            T.copy(
                acc,
                C[
                    m_tile * GEMM_TILE_M : (m_tile + 1) * GEMM_TILE_M,
                    n_tile * GEMM_TILE_N : (n_tile + 1) * GEMM_TILE_N,
                ],
            )

    return kernel


def test_copy_oob_1d_tail() -> None:
    device = "npu"

    x_1d = torch.arange(HIDDEN, dtype=torch.float32, device=device)
    out_1d = torch.empty_like(x_1d)
    make_1d_dynamic_copy(HIDDEN, BLOCK_K)(x_1d, out_1d)
    torch.npu.synchronize()
    torch.testing.assert_close(out_1d, x_1d)


def test_copy_oob_2d_inner_tail() -> None:
    device = "npu"

    x_2d = torch.arange(BLOCK_M * HIDDEN, dtype=torch.float32, device=device).reshape(BLOCK_M, HIDDEN)
    out_2d = torch.empty_like(x_2d)
    make_2d_dynamic_inner_copy(HIDDEN, BLOCK_M, BLOCK_K)(x_2d, out_2d)
    torch.npu.synchronize()
    torch.testing.assert_close(out_2d, x_2d)


def test_bf16_gemm_oob_multi_axis_tail() -> None:
    m, n, k = 250, 250, 250
    torch.manual_seed(42)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="npu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="npu")

    kernel = tilelang.compile(
        make_bf16_gemm_oob_kernel(m, n, k),
        out_idx=-1,
        target="ascend",
    )

    actual = kernel(a, b)
    torch.npu.synchronize()

    expected = a.float() @ b.float().T
    assert actual.shape == (m, n)

    for m_tile in range((m + GEMM_TILE_M - 1) // GEMM_TILE_M):
        for n_tile in range((n + GEMM_TILE_N - 1) // GEMM_TILE_N):
            m_slice = slice(m_tile * GEMM_TILE_M, min((m_tile + 1) * GEMM_TILE_M, m))
            n_slice = slice(n_tile * GEMM_TILE_N, min((n_tile + 1) * GEMM_TILE_N, n))
            torch.testing.assert_close(
                actual[m_slice, n_slice],
                expected[m_slice, n_slice],
                rtol=1e-2,
                atol=1e-2,
            )


if __name__ == "__main__":
    tilelang.testing.main()
