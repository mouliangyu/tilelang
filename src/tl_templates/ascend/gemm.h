#pragma once

#include "kernel_operator.h"

/*!
 * \brief Ascend GEMM template for L1-scoped inputs with double-buffered L0.
 *
 * Encapsulates the sub-K pipeline: double-buffered L0A/L0B allocation,
 * L1->L0 DMA copies with SetFlag/WaitFlag synchronization, and mad calls.
 *
 * ## Event ID contract
 *
 * This template uses **M_MTE1 event IDs 0 and 1** internally for L0
 * double-buffer synchronization (sk & 1).  The caller MUST:
 *
 *   1. Pre-set these flags exactly once before the first call:
 *        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
 *        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
 *
 *   2. Drain them exactly once after the last call:
 *        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
 *        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
 *
 *   3. NOT use IDs 0/1 for any other M_MTE1 flags in the surrounding
 *      code.  If the caller also needs M_MTE1 flags for its own L1
 *      double-buffer (e.g. an explicit sub-K loop), use IDs >= 2.
 *
 * The flags are self-sustaining across consecutive calls — no per-call
 * re-initialization is needed.
 *
 * \tparam M           Tile M dimension
 * \tparam K           Full K dimension of the L1 tile
 * \tparam N           Tile N dimension
 * \tparam TILE_K_SUB  Sub-K chunk size that fits in L0A/L0B
 * \tparam TRANS_B     Whether B is transposed (must be true on dav-3510;
 *                     for NN matmul use dn2nz to transpose B during GM→L1)
 * \tparam InT         Input element type (bfloat16_t, half_t, etc.)
 * \tparam AccumT      Accumulator element type (deduced from cc_ptr)
 */
template <int M, int K, int N, int TILE_K_SUB, bool TRANS_B, typename InT,
          typename AccumT>
__aicore__ inline void
ascend_gemm_l1(__cc__ AccumT *cc_ptr, __cbuf__ InT *cbuf_a_ptr,
               __cbuf__ InT *cbuf_b_ptr, int clear_accum) {
  static_assert(K % TILE_K_SUB == 0, "K must be divisible by TILE_K_SUB");
  static_assert(TILE_K_SUB % 16 == 0, "TILE_K_SUB must be a multiple of 16");
  static_assert(M % 16 == 0, "M must be a multiple of 16");
  static_assert(N % 16 == 0, "N must be a multiple of 16");

  constexpr int SUB_K = K / TILE_K_SUB;
  constexpr int C0 = 16;
  constexpr int mStepA = M / C0;
  constexpr int mStepB = N / C0;
  constexpr int kStep = TILE_K_SUB / C0;
  constexpr int L0A_STAGE = M * TILE_K_SUB;
  constexpr int L0B_STAGE = N * TILE_K_SUB;

  __ca__ InT *l0a = (__ca__ InT *)0;
  __cb__ InT *l0b = (__cb__ InT *)0;

  for (int sk = 0; sk < SUB_K; ++sk) {
    // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((sk & 1));
    get_buf(PIPE_MTE1, sk & 1, false);
    bisheng::cce::load_cbuf_to_ca((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
                                  cbuf_a_ptr, 0, (sk * kStep), mStepA, kStep,
                                  mStepA, mStepA, 0);
    rls_buf(PIPE_MTE1, sk & 1, false);
    if constexpr (TRANS_B) {
      // NT case: W[N,K] in L1, transpose=false (ZZ)
      // rows=N, cols=K → sub-K along cols (kStart)
      bisheng::cce::load_cbuf_to_cb(
          (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr, 0,
          (sk * kStep), mStepB, kStep, mStepB, mStepB, 0);
    } else {
      // NN case: B[K,N] in L1, transpose=true (ZN)
      // rows=K, cols=N → sub-K along rows (mStart)
      constexpr int srcStrideNN = K / C0; // nBlk = K/16
      bisheng::cce::load_cbuf_to_cb(
          (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr,
          (sk * kStep), // mStartPosition: sub-K row offset
          0,            // kStartPosition: start from col 0
          kStep,        // mStep: TILE_K_SUB/16 K-row fractals
          mStepB,       // kStep: N/16 N-col fractals
          srcStrideNN,  // srcStride: K/16 (nBlk in L1)
          kStep,        // dstStride: mStep
          1);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((sk & 1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((sk & 1));
    get_buf(PIPE_M, sk & 1, false);
    bisheng::cce::mad(cc_ptr, (__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
                      (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), M,
                      TILE_K_SUB, N, 0, 0, 0, clear_accum && (sk == 0));
    rls_buf(PIPE_M, sk & 1, false);
    // AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((sk & 1));
  }
}
