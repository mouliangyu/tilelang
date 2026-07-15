/*!
 * \file tl/ascend/op/ascend_mte_plan.h
 * \brief Utilities for planning Ascend MTE (Memory Transfer Engine) DMA copies.
 *
 * Provides a simple strided-layout abstraction (similar to CuTe's shape-stride
 * pairs) and a planner that coalesces contiguous dimensions and produces a 2D
 * copy descriptor suitable for MTE hardware.
 */

#ifndef TVM_TL_ASCEND_OP_ASCEND_MTE_PLAN_H_
#define TVM_TL_ASCEND_OP_ASCEND_MTE_PLAN_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

/// Convert a logical element count into the byte count used by MTE copy
/// descriptors. Sub-byte dtypes such as float4_e2m1fn are packed in Ascend
/// storage (1 byte / 2 logical elements), so the byte count is
/// ceil(elements * elem_bits / 8).
inline PrimExpr AscendMTEBytesFromElements(PrimExpr elements, int elem_bits) {
  if (elem_bits % 8 == 0) {
    return elements * make_const(elements.dtype(), elem_bits / 8);
  }
  return FloorDiv(elements * make_const(elements.dtype(), elem_bits) +
                      make_const(elements.dtype(), 7),
                  make_const(elements.dtype(), 8));
}

// ---------------------------------------------------------------------------
// StridedLayout: a flat sequence of (size, stride) modes, innermost first.
// ---------------------------------------------------------------------------

/// A single mode in the strided layout: size elements, separated by stride.
struct Mode {
  PrimExpr size;   ///< Number of elements along this dimension
  PrimExpr stride; ///< Stride in elements between consecutive positions
};

/// A simplified layout description: an ordered list of modes (innermost first)
/// plus a base offset in elements.
struct StridedLayout {
  std::vector<Mode> modes; ///< From innermost (stride smallest) to outermost
  PrimExpr offset; ///< Flat element offset to the first accessed element
  int elem_bits;   ///< Size of one element in bits

  /// Construct a StridedLayout from a Buffer and the access Range.
  ///
  /// If buf->strides is empty, row-major contiguous layout is assumed.
  /// The Range provides the sub-region being accessed:
  ///   - range[i]->min  contributes to the base offset
  ///   - range[i]->extent  becomes the mode size
  ///   - buf strides (explicit or row-major inferred) become mode strides
  ///
  /// Modes are stored innermost-first (i.e. modes[0] = last dimension).
  static StridedLayout
  FromBufferRange(const Buffer &buf, const Array<Range> &range, int elem_bits) {
    int ndim = static_cast<int>(range.size());
    auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };

    // Compute strides (elements). If buffer has explicit strides, use them;
    // otherwise assume row-major.
    std::vector<PrimExpr> strides(ndim);
    if (!buf->strides.empty()) {
      for (int i = 0; i < ndim; ++i) {
        strides[i] = buf->strides[i];
      }
    } else {
      strides[ndim - 1] = I(1);
      for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * buf->shape[i + 1];
      }
    }

    // Compute base offset = sum(range[i]->min * strides[i])
    PrimExpr offset = I(0);
    for (int i = 0; i < ndim; ++i) {
      offset = offset + range[i]->min * strides[i];
    }

    // Build modes: innermost first
    std::vector<Mode> modes(ndim);
    for (int i = 0; i < ndim; ++i) {
      // Map dimension (ndim-1-i) to modes[i] so modes[0] is innermost
      int dim = ndim - 1 - i;
      modes[i] = {range[dim]->extent, strides[dim]};
    }

    return StridedLayout{std::move(modes), offset, elem_bits};
  }

  /// Coalesce adjacent modes where the outer mode's stride equals the inner
  /// mode's size * stride (i.e. they are contiguous in memory).
  /// Also eliminates size-1 modes (they carry no data extent).
  ///
  /// Returns a new StridedLayout with fewer modes.
  StridedLayout Coalesce(arith::Analyzer *analyzer) const {
    if (modes.empty())
      return *this;

    // First, filter out size-1 modes (they don't contribute to data layout)
    std::vector<Mode> filtered;
    for (const auto &m : modes) {
      if (auto *imm = m.size.as<IntImmNode>()) {
        if (imm->value == 1)
          continue;
      }
      filtered.push_back(m);
    }
    // If all modes were size-1, the total is 1 element; return single mode.
    if (filtered.empty()) {
      auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
      return StridedLayout{{Mode{I(1), I(1)}}, offset, elem_bits};
    }

    // Then coalesce contiguous adjacent modes
    std::vector<Mode> result;
    result.push_back(filtered[0]);

    for (size_t i = 1; i < filtered.size(); ++i) {
      Mode &prev = result.back();
      const Mode &cur = filtered[i];
      // Check: cur.stride == prev.stride * prev.size
      // i.e. the outer dimension is contiguous with the inner
      PrimExpr expected_stride = prev.stride * prev.size;
      if (analyzer->CanProveEqual(cur.stride, expected_stride)) {
        // Merge: new size = prev.size * cur.size, keep prev.stride
        prev.size = prev.size * cur.size;
      } else {
        result.push_back(cur);
      }
    }

    return StridedLayout{std::move(result), offset, elem_bits};
  }
};

// ---------------------------------------------------------------------------
// MTECopy2D: the result of planning a 2D MTE copy.
// ---------------------------------------------------------------------------

/// Describes a 2D strided DMA copy (the common denominator of GM<->UBuf MTE).
struct MTECopy2D {
  PrimExpr n_rows;           ///< Number of rows to copy
  PrimExpr row_elems;        ///< Logical elements per row (contiguous burst)
  PrimExpr src_stride_elems; ///< Source: logical elements between rows
  PrimExpr dst_stride_elems; ///< Dest: logical elements between rows
  PrimExpr src_offset;       ///< Source base offset in elements
  PrimExpr dst_offset;       ///< Dest base offset in elements
};

/// Plan a 2D MTE copy from coalesced source and destination layouts.
///
/// The row size is determined by the strided side's physical row boundary.
/// A fully contiguous side may be split to match that boundary. Two strided
/// sides must expose both the same innermost row boundary and the same row
/// count; otherwise one 2D MTE descriptor cannot represent both address
/// sequences with constant row strides.
inline MTECopy2D PlanMTECopy(const StridedLayout &src_raw,
                             const StridedLayout &dst_raw,
                             arith::Analyzer *analyzer) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };

  // 1. Coalesce both sides independently
  StridedLayout src = src_raw.Coalesce(analyzer);
  StridedLayout dst = dst_raw.Coalesce(analyzer);

  // 2. A single MTE descriptor supports one contiguous row and one constant
  // inter-row stride on each side. Keep unsupported higher-dimensional
  // layouts out of the structural cases below.
  ICHECK(!src.modes.empty() && !dst.modes.empty())
      << "PlanMTECopy: empty coalesced layout.";
  ICHECK(src.modes.size() <= 2 && dst.modes.size() <= 2)
      << "PlanMTECopy: >2D strided copy not yet supported. " << "src has "
      << src.modes.size() << " modes, " << "dst has " << dst.modes.size()
      << " modes after coalesce.";
  ICHECK(analyzer->CanProveEqual(src.modes[0].stride, I(1)) &&
         analyzer->CanProveEqual(dst.modes[0].stride, I(1)))
      << "PlanMTECopy: innermost MTE modes must be contiguous, got src stride "
      << src.modes[0].stride << " and dst stride " << dst.modes[0].stride
      << ".";

  auto total_elems = [&](const StridedLayout &layout) -> PrimExpr {
    PrimExpr total = I(1);
    for (const auto &mode : layout.modes) {
      total = total * mode.size;
    }
    return analyzer->Simplify(total);
  };
  PrimExpr src_total = total_elems(src);
  PrimExpr dst_total = total_elems(dst);
  ICHECK(analyzer->CanProveEqual(src_total, dst_total))
      << "PlanMTECopy: source and destination regions must contain the same "
         "number of elements, got src="
      << src_total << " and dst=" << dst_total << ".";

  // 3. Select the row boundary from layout structure rather than symbolic
  // size ordering. A one-mode side is fully contiguous and can be split to
  // match a two-mode side. If both sides are strided, their physical row
  // boundaries and row counts must match exactly.
  bool src_1d = src.modes.size() == 1;
  bool dst_1d = dst.modes.size() == 1;
  PrimExpr row_elems;
  PrimExpr n_rows;
  PrimExpr src_stride_elems;
  PrimExpr dst_stride_elems;

  if (src_1d && dst_1d) {
    row_elems = src_total;
    n_rows = I(1);
    src_stride_elems = row_elems;
    dst_stride_elems = row_elems;
  } else if (!src_1d && dst_1d) {
    row_elems = analyzer->Simplify(src.modes[0].size);
    n_rows = analyzer->Simplify(src.modes[1].size);
    src_stride_elems = src.modes[1].stride;
    dst_stride_elems = row_elems;
  } else if (src_1d && !dst_1d) {
    row_elems = analyzer->Simplify(dst.modes[0].size);
    n_rows = analyzer->Simplify(dst.modes[1].size);
    src_stride_elems = row_elems;
    dst_stride_elems = dst.modes[1].stride;
  } else {
    ICHECK(analyzer->CanProveEqual(src.modes[0].size, dst.modes[0].size))
        << "PlanMTECopy: incompatible 2D row boundaries require outer-loop "
           "lowering, got src inner="
        << src.modes[0].size << " and dst inner=" << dst.modes[0].size << ".";
    ICHECK(analyzer->CanProveEqual(src.modes[1].size, dst.modes[1].size))
        << "PlanMTECopy: source and destination row counts differ, got src="
        << src.modes[1].size << " and dst=" << dst.modes[1].size << ".";
    row_elems = analyzer->Simplify(src.modes[0].size);
    n_rows = analyzer->Simplify(src.modes[1].size);
    src_stride_elems = src.modes[1].stride;
    dst_stride_elems = dst.modes[1].stride;
  }

  return MTECopy2D{
      n_rows,           row_elems,  src_stride_elems,
      dst_stride_elems, src.offset, dst.offset,
  };
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_ASCEND_MTE_PLAN_H_
