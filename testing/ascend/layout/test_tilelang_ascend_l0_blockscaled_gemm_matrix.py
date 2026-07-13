from dataclasses import dataclass

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@dataclass(frozen=True)
class Case:
    M: int
    K: int
    N: int
    scale_dtype: str
    scale_packed: bool
    rel_tol: float


def make_l0_blockscaled_gemm_kernel(
    M: int,
    K: int,
    N: int,
    scale_dtype: str,
    scale_packed: bool,
):
    dtype = "float8_e4m3fn"
    sf_div = 64 if scale_packed else 32
    sf_k = K // sf_div
    sf_k_e8m0 = K // 32

    @T.prim_func
    def gemm_kernel(
        X: T.Buffer((M, K), dtype),
        W: T.Buffer((N, K), dtype),
        SFX: T.Buffer((sf_k, M) if scale_packed else (M, sf_k_e8m0), scale_dtype),
        SFW: T.Buffer((sf_k, N) if scale_packed else (N, sf_k_e8m0), scale_dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            xsf_l1 = T.alloc_l1((M, sf_k), scale_dtype)
            wsf_l1 = T.alloc_l1((N, sf_k), scale_dtype)
            x_l0a = T.alloc_l0a((M, K), dtype)
            w_l0b = T.alloc_l0b((N, K), dtype)

            T.copy(X, x_l1)
            T.copy(W, w_l1)
            if scale_packed:
                T.copy(SFX, xsf_l1, transpose=True)
                T.copy(SFW, wsf_l1, transpose=True)
            else:
                T.copy(SFX, xsf_l1)
                T.copy(SFW, wsf_l1)
            T.copy(x_l1, x_l0a, scale=xsf_l1)
            T.copy(w_l1, w_l0b, scale=wsf_l1)
            T.blockscaled_gemm(x_l0a, w_l0b, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return gemm_kernel


def make_inputs(case: Case):
    dtype = torch.float8_e4m3fn
    x = (torch.randn(case.M, case.K, device="npu") * 0.1).to(dtype)
    w = (torch.randn(case.N, case.K, device="npu") * 0.1).to(dtype)

    sf_k_e8m0 = case.K // 32
    sfx_e8m0 = torch.randint(124, 131, (case.M, sf_k_e8m0), dtype=torch.uint8, device="npu")
    sfw_e8m0 = torch.randint(124, 131, (case.N, sf_k_e8m0), dtype=torch.uint8, device="npu")

    if case.scale_packed:
        sfx = sfx_e8m0.view(torch.uint16).T.contiguous()
        sfw = sfw_e8m0.view(torch.uint16).T.contiguous()
    else:
        scale_torch_dtype = getattr(torch, case.scale_dtype)
        sfx = sfx_e8m0.view(scale_torch_dtype)
        sfw = sfw_e8m0.view(scale_torch_dtype)
    return x, w, sfx, sfw, sfx_e8m0, sfw_e8m0


def ref_program(x, w, sfx_e8m0, sfw_e8m0):
    k = x.shape[1]
    sx = torch.pow(2.0, sfx_e8m0.float() - 127.0).repeat_interleave(32, dim=1)[:, :k]
    sw = torch.pow(2.0, sfw_e8m0.float() - 127.0).repeat_interleave(32, dim=1)[:, :k]
    return (x.float() * sx) @ (w.float() * sw).T


def run_case(case: Case):
    program = make_l0_blockscaled_gemm_kernel(
        case.M,
        case.K,
        case.N,
        case.scale_dtype,
        case.scale_packed,
    )
    kernel = tilelang.compile(program, out_idx=-1)

    x, w, sfx, sfw, sfx_e8m0, sfw_e8m0 = make_inputs(case)
    actual = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()

    expected = ref_program(x, w, sfx_e8m0, sfw_e8m0)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel = ((actual - expected).abs().mean() / denom).item()
    if rel < case.rel_tol:
        return

    diff = (actual - expected).abs()
    row, col = (int(v.item()) for v in torch.nonzero(diff == diff.max())[0])
    pytest.fail(
        f"shape=({case.M}, {case.K}, {case.N}) scale={case.scale_dtype} "
        f"packed={case.scale_packed} rel={rel:.4e} "
        f"actual[{row},{col}]={actual[row, col].item():.6g} "
        f"expected[{row},{col}]={expected[row, col].item():.6g}"
    )


def default_cases():
    return [
        Case(15, 128, 15, "uint16", True, 2e-1),
        Case(32, 128, 32, "uint16", True, 2e-1),
        Case(100, 128, 100, "uint16", True, 2e-1),
    ]


def case_id(case: Case) -> str:
    packed = "packed" if case.scale_packed else "unpacked"
    return f"{case.M}x{case.K}x{case.N}-{case.scale_dtype}-{packed}"


@pytest.mark.parametrize("case", default_cases(), ids=case_id)
def test_l0_blockscaled_gemm_matrix(case: Case):
    torch.manual_seed(42)
    run_case(case)


if __name__ == "__main__":
    tilelang.testing.main()
