"""FlashAttention forward kernel on Ascend NPU using Mix kernel (Cube + Vector).

Online-softmax FlashAttention: O = softmax(Q @ K^T / sqrt(d)) @ V

Uses SimdVF for the online softmax (row-max, exp, sum, rescale).
"""

import math

import tilelang
import tilelang.language as T
from tilelang.language.simd import alloc, pset, vld, vsts, vmul, vadd, vmax, vdupv, vcadd, vcmax, vexpdif, vcvt, vmuls


def flash_attention(D=128, SEQ_LEN=4096):
    BR = 128
    BC = 128
    NUM_KV_BLOCKS = SEQ_LEN // BC
    dtype = "bfloat16"
    accum_dtype = "float32"

    scale = math.sqrt(1.0 / D) * 1.4426950408889634

    @T.prim_func
    def main(
        Q: T.Buffer((SEQ_LEN, D), dtype),
        K: T.Buffer((SEQ_LEN, D), dtype),
        V: T.Buffer((SEQ_LEN, D), dtype),
        O: T.Buffer((SEQ_LEN, D), accum_dtype),
    ):
        with T.Kernel(SEQ_LEN // BR) as bx:
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            acc_s = T.alloc_l0c((BR, BC), accum_dtype)
            acc_o = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((BR // 2, BC), accum_dtype)
            P_ub = T.alloc_shared((BR // 2, BC), dtype)
            O_ub = T.alloc_shared((BR // 2, D), accum_dtype)
            O_tmp_ub = T.alloc_shared((BR // 2, D), accum_dtype)
            m_ub = T.alloc_shared((BR // 2,), accum_dtype)
            l_ub = T.alloc_shared((BR // 2,), accum_dtype)
            alpha_ub = T.alloc_shared((BR // 2,), accum_dtype)

            T.copy(Q[bx * BR : (bx + 1) * BR, 0:D], Q_shared)

            with T.SimdVF():
                T.fill(O_ub, 0.0)
                T.fill(l_ub, 0.0)
                T.fill(m_ub, -T.infinity(accum_dtype))

            for k in T.Pipelined(NUM_KV_BLOCKS, num_stages=2):
                T.copy(K[k * BC : (k + 1) * BC, 0:D], K_shared)
                T.copy(
                    V[k * BC : (k + 1) * BC, 0:D],
                    V_shared,
                    transpose=True,
                )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                T.dual_copy(acc_s, S_ub)

                scale_ln2 = scale * 0.6931471805599453
                ROWS = BR // 2
                VL = 64

                with T.SimdVF():
                    full = pset(32, "PAT_ALL")
                    one = pset(32, "PAT_VL1")

                    s00 = alloc("float32")
                    s01 = alloc("float32")
                    s10 = alloc("float32")
                    s11 = alloc("float32")
                    max0 = alloc("float32")
                    max1 = alloc("float32")
                    old0 = alloc("float32")
                    old1 = alloc("float32")
                    alpha0 = alloc("float32")
                    alpha1 = alloc("float32")
                    pReg = alloc("float32")
                    sum0 = alloc("float32")
                    sum1 = alloc("float32")
                    lReg = alloc("float32")
                    outReg = alloc("float32")
                    pBf16Reg = alloc("bfloat16")

                    for r in range(0, ROWS, 2):
                        # Load old max
                        vld(old0, m_ub[r], "BRC_B32")
                        vld(old1, m_ub[r + 1], "BRC_B32")
                        # Load S
                        vld(s00, S_ub[r, 0], "NORM")
                        vld(s01, S_ub[r, VL], "NORM")
                        vld(s10, S_ub[r + 1, 0], "NORM")
                        vld(s11, S_ub[r + 1, VL], "NORM")
                        # Scale
                        vmuls(s00, s00, T.float32(scale_ln2), full, "MODE_ZEROING")
                        vmuls(s01, s01, T.float32(scale_ln2), full, "MODE_ZEROING")
                        vmuls(s10, s10, T.float32(scale_ln2), full, "MODE_ZEROING")
                        vmuls(s11, s11, T.float32(scale_ln2), full, "MODE_ZEROING")
                        # Row max r
                        vmax(max0, old0, s00, full, "MODE_ZEROING")
                        vmax(max0, max0, s01, full, "MODE_ZEROING")
                        vcmax(max0, max0, full, "MODE_ZEROING")
                        vsts(m_ub[r], max0, one, "ONEPT_B32")
                        vdupv(max0, max0, full, "POS_LOWEST", "MODE_ZEROING")
                        # Row max r+1
                        vmax(max1, old1, s10, full, "MODE_ZEROING")
                        vmax(max1, max1, s11, full, "MODE_ZEROING")
                        vcmax(max1, max1, full, "MODE_ZEROING")
                        vsts(m_ub[r + 1], max1, one, "ONEPT_B32")
                        vdupv(max1, max1, full, "POS_LOWEST", "MODE_ZEROING")
                        # alpha = exp(old - max)
                        vexpdif(alpha0, old0, max0, full, 0)
                        vsts(alpha_ub[r], alpha0, one, "ONEPT_B32")
                        vexpdif(alpha1, old1, max1, full, 0)
                        vsts(alpha_ub[r + 1], alpha1, one, "ONEPT_B32")
                        # row r: P = exp(S-max), cast bf16, reduce sum
                        vexpdif(sum0, s00, max0, full, 0)
                        vcvt(pBf16Reg, sum0, full, 0, 0, 0, "MODE_ZEROING")
                        vsts(P_ub[r, 0], pBf16Reg, full, "PK_B32")
                        vexpdif(pReg, s01, max0, full, 0)
                        vcvt(pBf16Reg, pReg, full, 0, 0, 0, "MODE_ZEROING")
                        vsts(P_ub[r, VL], pBf16Reg, full, "PK_B32")
                        vadd(sum0, sum0, pReg, full, "MODE_ZEROING")
                        vcadd(sum0, sum0, full, "MODE_ZEROING")
                        # row r+1: P = exp(S-max), cast bf16, reduce sum
                        vexpdif(sum1, s10, max1, full, 0)
                        vcvt(pBf16Reg, sum1, full, 0, 0, 0, "MODE_ZEROING")
                        vsts(P_ub[r + 1, 0], pBf16Reg, full, "PK_B32")
                        vexpdif(pReg, s11, max1, full, 0)
                        vcvt(pBf16Reg, pReg, full, 0, 0, 0, "MODE_ZEROING")
                        vsts(P_ub[r + 1, VL], pBf16Reg, full, "PK_B32")
                        vadd(sum1, sum1, pReg, full, "MODE_ZEROING")
                        vcadd(sum1, sum1, full, "MODE_ZEROING")
                        # l[r] = l[r]*alpha + sum
                        vld(lReg, l_ub[r], "BRC_B32")
                        vmul(outReg, lReg, alpha0, full, "MODE_ZEROING")
                        vadd(outReg, outReg, sum0, one, "MODE_ZEROING")
                        vsts(l_ub[r], outReg, one, "ONEPT_B32")
                        vld(lReg, l_ub[r + 1], "BRC_B32")
                        vmul(outReg, lReg, alpha1, full, "MODE_ZEROING")
                        vadd(outReg, outReg, sum1, one, "MODE_ZEROING")
                        vsts(l_ub[r + 1], outReg, one, "ONEPT_B32")

                T.dual_copy(P_ub, P_shared, nd2nz=True)
                T.gemm(P_shared, V_shared, acc_o, transpose_B=True, clear_accum=True)
                T.dual_copy(acc_o, O_tmp_ub)

                with T.SimdVF():
                    for i, j in T.Parallel(BR // 2, D):
                        O_ub[i, j] = O_ub[i, j] * alpha_ub[i] + O_tmp_ub[i, j]

            with T.SimdVF():
                for i, j in T.Parallel(BR // 2, D):
                    O_ub[i, j] = O_ub[i, j] / l_ub[i]

            T.dual_copy(O_ub, O[bx * BR : (bx + 1) * BR, 0:D])

    return main


def ref_program(Q, K, V):
    import torch

    return (
        torch.nn.functional.scaled_dot_product_attention(
            Q.unsqueeze(0),
            K.unsqueeze(0),
            V.unsqueeze(0),
        )
        .squeeze(0)
        .to(torch.float32)
    )


if __name__ == "__main__":
    import torch

    D = 128
    SEQ_LEN = 4096

    device = torch.device("npu")
    print(f"Compiling flash_attention (D={D}, SEQ_LEN={SEQ_LEN})...")
    program = flash_attention(D, SEQ_LEN)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    K = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)
    V = torch.randn(SEQ_LEN, D, dtype=torch.bfloat16, device=device)

    O = kernel(Q, K, V)
    torch.npu.synchronize()

    expected = ref_program(Q, K, V)
    diff = (O.cpu().float() - expected.cpu()).abs().max().item()
    rel = diff / expected.cpu().abs().max().item()
    print(f"  rel={rel:.4f} {'PASS' if rel < 0.05 else 'FAIL'}")

    from bench import bench_msprof

    prof = bench_msprof(lambda: kernel(Q, K, V), kernel_names="main")
    flops = 4.0 * SEQ_LEN * SEQ_LEN * D
    print(f"  {prof.dur_us:6.1f} us  |  {prof.tflops(flops):5.1f} TFLOPS")
