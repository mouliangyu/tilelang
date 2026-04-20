#pragma once

#include "kernel_operator.h"

using namespace AscendC;

/*!
 * \brief ND→NZ scatter reorder in UB using vsstb (DATA_BLOCK_COPY).
 *
 * Reorders [ROWS, COLS] data from ND (row-major) to NZ (fractal) layout
 * within UB.  Optionally fuses a type cast (SrcT→DstT, e.g. float→bf16).
 *
 * Key trick: StoreAlign with DATA_BLOCK_COPY writes 16-element blocks
 * with blockStride in units of 32B blocks.  blockStride = ROWS means
 * the inter-block gap is ROWS * 16 elements = nBlk * C0², which is
 * exactly the NZ D-group stride.
 *
 * This function is __simd_vf__ (runs on AIV vector pipe).  The caller
 * must issue copy_ubuf_to_cbuf() separately after this returns.
 *
 * For single-fractal tiles (ROWS <= 16 && COLS <= 16), this is a no-op
 * since ND == NZ.  The caller should just use raw copy in that case.
 */
/*!
 * \tparam ROWS       Number of rows in this sub-tile (e.g. Br_HALF after
 * M-split)
 * \tparam COLS       Number of columns
 * \tparam FULL_ROWS  Number of rows in the FULL matrix that will be assembled
 * in L1. This determines the NZ D-group stride (= FULL_ROWS * 16 elements). For
 * single-writer (no M-split), FULL_ROWS == ROWS.
 * \tparam FUSE_CAST  Whether to fuse type cast
 * \tparam DstT       Destination element type
 * \tparam SrcT       Source element type
 */
template <int ROWS, int COLS, int FULL_ROWS, bool FUSE_CAST, typename DstT,
          typename SrcT>
__simd_vf__ inline void ascend_nd2nz_scatter(__ubuf__ SrcT *src_ub,
                                             __ubuf__ DstT *dst_nz_ub) {
  static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");
  static_assert(COLS % 16 == 0, "COLS must be a multiple of 16");
  static_assert(FULL_ROWS % 16 == 0, "FULL_ROWS must be a multiple of 16");
  static_assert(FULL_ROWS >= ROWS, "FULL_ROWS must be >= ROWS");

  constexpr int C0 = 16;
  // nBlk is based on FULL matrix rows, not sub-tile rows.
  // This ensures D-group stride matches the complete NZ layout in L1.
  constexpr int nBlk = FULL_ROWS / C0;
  constexpr int D_GROUP_ELEMS = nBlk * C0 * C0;

  if constexpr (FUSE_CAST && !std::is_same_v<SrcT, DstT>) {
    // Fused cast + scatter: SrcT (e.g. float) → DstT (e.g. bfloat16)
    constexpr int VL_SRC = 256 / sizeof(SrcT); // floats per register
    constexpr int COLS_PER_PASS = (COLS >= VL_SRC) ? VL_SRC : COLS;
    constexpr int NUM_PASSES = COLS / COLS_PER_PASS;
    constexpr int DBLK_PER_PASS = COLS_PER_PASS / C0;

    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    MicroAPI::MaskReg srcMask = MicroAPI::CreateMask<SrcT>();
    uint32_t dstMaskCount = COLS_PER_PASS;
    MicroAPI::MaskReg dstMask = MicroAPI::UpdateMask<DstT>(dstMaskCount);

    MicroAPI::RegTensor<SrcT> srcReg;
    MicroAPI::RegTensor<DstT> castReg;
    using PackUintT =
        typename std::conditional<sizeof(DstT) == 2, uint16_t, uint32_t>::type;
    using PackSrcUintT =
        typename std::conditional<sizeof(SrcT) == 4, uint32_t, uint16_t>::type;
    MicroAPI::RegTensor<PackUintT> packReg;

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      __ubuf__ DstT *dstPtr = dst_nz_ub + pass * DBLK_PER_PASS * D_GROUP_ELEMS;
      __ubuf__ SrcT *srcBase = src_ub + pass * COLS_PER_PASS;

      for (int r = 0; r < ROWS; ++r) {
        MicroAPI::LoadAlign(srcReg, srcBase + r * COLS);
        MicroAPI::Cast<DstT, SrcT, castTrait>(castReg, srcReg, srcMask);
        MicroAPI::Pack<PackUintT, PackSrcUintT, MicroAPI::HighLowPart::LOWEST>(
            packReg, *(MicroAPI::RegTensor<PackSrcUintT> *)&castReg);
        MicroAPI::StoreAlign<DstT, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                             MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, *(MicroAPI::RegTensor<DstT> *)&packReg, (uint32_t)FULL_ROWS,
            (uint32_t)1, dstMask);
      }
    }
  } else {
    // Same-type scatter: just reorder, no cast
    static_assert(std::is_same_v<SrcT, DstT>,
                  "FUSE_CAST=false requires SrcT==DstT");

    constexpr int VL_DST = 256 / sizeof(DstT);
    constexpr int COLS_PER_PASS = (COLS >= VL_DST) ? VL_DST : COLS;
    constexpr int NUM_PASSES = COLS / COLS_PER_PASS;
    constexpr int DBLK_PER_PASS = COLS_PER_PASS / C0;

    uint32_t maskCount = COLS_PER_PASS;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<DstT>(maskCount);
    MicroAPI::RegTensor<DstT> reg;

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      __ubuf__ DstT *dstPtr = dst_nz_ub + pass * DBLK_PER_PASS * D_GROUP_ELEMS;
      __ubuf__ DstT *srcBase = ((__ubuf__ DstT *)src_ub) + pass * COLS_PER_PASS;

      for (int r = 0; r < ROWS; ++r) {
        MicroAPI::LoadAlign(reg, srcBase + r * COLS);
        MicroAPI::StoreAlign<DstT, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                             MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, reg, (uint32_t)FULL_ROWS, (uint32_t)1, mask);
      }
    }
  }
}
