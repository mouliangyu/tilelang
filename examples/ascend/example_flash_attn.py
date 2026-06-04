"""FlashAttention forward kernel on Ascend NPU using Mix kernel (Cube + Vector).

Online-softmax FlashAttention: O = softmax(Q @ K^T / sqrt(d)) @ V

Uses SimdVF for the online softmax (row-max, exp, sum, rescale).
"""

import math

import tilelang
import tilelang.language as T
from tilelang.language.simd import pset, vld, vsts, vmul, vadd, vmax, vdupv, vcadd, vcmax, vexpdif, vcvt, vmuls


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

                    for r in range(0, ROWS, 2):
                        # Load old max
                        old0 = vld(m_ub[r], "BRC_B32")
                        old1 = vld(m_ub[r + 1], "BRC_B32")
                        # Load S
                        s00 = vld(S_ub[r, 0], "NORM")
                        s01 = vld(S_ub[r, VL], "NORM")
                        s10 = vld(S_ub[r + 1, 0], "NORM")
                        s11 = vld(S_ub[r + 1, VL], "NORM")
                        # Scale
                        s00 = vmuls(s00, T.float32(scale_ln2), full)
                        s01 = vmuls(s01, T.float32(scale_ln2), full)
                        s10 = vmuls(s10, T.float32(scale_ln2), full)
                        s11 = vmuls(s11, T.float32(scale_ln2), full)
                        # Row max r
                        max0 = vmax(old0, s00, full)
                        max0 = vmax(max0, s01, full)
                        max0 = vcmax(max0, full)
                        vsts(m_ub[r], max0, one, "ONEPT_B32")
                        max0 = vdupv(max0, full, "POS_LOWEST")
                        # Row max r+1
                        max1 = vmax(old1, s10, full)
                        max1 = vmax(max1, s11, full)
                        max1 = vcmax(max1, full)
                        vsts(m_ub[r + 1], max1, one, "ONEPT_B32")
                        max1 = vdupv(max1, full, "POS_LOWEST")
                        # alpha = exp(old - max)
                        alpha0 = vexpdif(old0, max0, full, 0)
                        vsts(alpha_ub[r], alpha0, one, "ONEPT_B32")
                        alpha1 = vexpdif(old1, max1, full, 0)
                        vsts(alpha_ub[r + 1], alpha1, one, "ONEPT_B32")
                        # row r: P = exp(S-max), cast bf16, reduce sum
                        sum0 = vexpdif(s00, max0, full, 0)
                        pBf16Reg = vcvt(sum0, full, "bfloat16", 0, 0, 0)
                        vsts(P_ub[r, 0], pBf16Reg, full, "PK_B32")
                        pReg = vexpdif(s01, max0, full, 0)
                        pBf16Reg = vcvt(pReg, full, "bfloat16", 0, 0, 0)
                        vsts(P_ub[r, VL], pBf16Reg, full, "PK_B32")
                        sum0 = vadd(sum0, pReg, full)
                        sum0 = vcadd(sum0, full)
                        # row r+1: P = exp(S-max), cast bf16, reduce sum
                        sum1 = vexpdif(s10, max1, full, 0)
                        pBf16Reg = vcvt(sum1, full, "bfloat16", 0, 0, 0)
                        vsts(P_ub[r + 1, 0], pBf16Reg, full, "PK_B32")
                        pReg = vexpdif(s11, max1, full, 0)
                        pBf16Reg = vcvt(pReg, full, "bfloat16", 0, 0, 0)
                        vsts(P_ub[r + 1, VL], pBf16Reg, full, "PK_B32")
                        sum1 = vadd(sum1, pReg, full)
                        sum1 = vcadd(sum1, full)
                        # l[r] = l[r]*alpha + sum
                        lReg = vld(l_ub[r], "BRC_B32")
                        outReg = vmul(lReg, alpha0, full)
                        outReg = vadd(outReg, sum0, one)
                        vsts(l_ub[r], outReg, one, "ONEPT_B32")
                        lReg = vld(l_ub[r + 1], "BRC_B32")
                        outReg = vmul(lReg, alpha1, full)
                        outReg = vadd(outReg, sum1, one)
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
