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
#include <tvm/tirx/op.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

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
  int elem_bytes;  ///< Size of one element in bytes

  /// Construct a StridedLayout from a Buffer and the access Range.
  ///
  /// If buf->strides is empty, row-major contiguous layout is assumed.
  /// The Range provides the sub-region being accessed:
  ///   - range[i]->min  contributes to the base offset
  ///   - range[i]->extent  becomes the mode size
  ///   - buf strides (explicit or row-major inferred) become mode strides
  ///
  /// Modes are stored innermost-first (i.e. modes[0] = last dimension).
  static StridedLayout FromBufferRange(const Buffer &buf,
                                       const Array<Range> &range,
                                       int elem_bytes) {
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

    return StridedLayout{std::move(modes), offset, elem_bytes};
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
      return StridedLayout{{Mode{I(1), I(1)}}, offset, elem_bytes};
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

    return StridedLayout{std::move(result), offset, elem_bytes};
  }
};

// ---------------------------------------------------------------------------
// MTECopy2D: the result of planning a 2D MTE copy.
// ---------------------------------------------------------------------------

/// Describes a 2D strided DMA copy (the common denominator of GM<->UBuf MTE).
struct MTECopy2D {
  PrimExpr n_rows;    ///< Number of rows to copy
  PrimExpr row_bytes; ///< Bytes per row (contiguous burst)
  PrimExpr
      src_stride_bytes; ///< Source: bytes between start of consecutive rows
  PrimExpr dst_stride_bytes; ///< Dest: bytes between start of consecutive rows
  PrimExpr src_offset;       ///< Source base offset in elements
  PrimExpr dst_offset;       ///< Dest base offset in elements

  /// Outer loop dimensions (modes[2:] after coalesce), empty if <= 2D.
  /// When non-empty, the caller must emit nested loops over these modes,
  /// adjusting src/dst pointers by mode.stride * elem_bytes per iteration.
  std::vector<Mode> outer_loops;
};

/// Plan a 2D MTE copy from coalesced source and destination layouts.
///
/// The row size is determined by the minimum contiguous segment of src and dst.
/// When the two sides have different innermost contiguous sizes, the larger
/// side is logically split so that both share the same row_bytes.
inline MTECopy2D PlanMTECopy(const StridedLayout &src_raw,
                             const StridedLayout &dst_raw,
                             arith::Analyzer *analyzer) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  int elem_bytes = src_raw.elem_bytes;

  // 1. Coalesce both sides independently
  StridedLayout src = src_raw.Coalesce(analyzer);
  StridedLayout dst = dst_raw.Coalesce(analyzer);

  // 2. Determine the common row size = min of innermost contiguous segments.
  PrimExpr src_inner = src.modes.empty() ? I(0) : src.modes[0].size;
  PrimExpr dst_inner = dst.modes.empty() ? I(0) : dst.modes[0].size;

  // Simplify for constant folding (e.g. 256*128 -> 32768)
  src_inner = analyzer->Simplify(src_inner);
  dst_inner = analyzer->Simplify(dst_inner);

  PrimExpr row_elems;
  bool src_is_smaller = false;
  bool dst_is_smaller = false;

  if (analyzer->CanProveEqual(src_inner, dst_inner)) {
    row_elems = src_inner;
  } else if (analyzer->CanProve(src_inner < dst_inner)) {
    row_elems = src_inner;
    src_is_smaller = true;
  } else if (analyzer->CanProve(dst_inner < src_inner)) {
    row_elems = dst_inner;
    dst_is_smaller = true;
  } else {
    // Cannot compare symbolically; use the smaller of the two mode sizes.
    // Fallback: assume dst (often the non-compact global side) is smaller.
    row_elems = dst_inner;
    dst_is_smaller = true;
  }

  PrimExpr row_bytes = row_elems * I(elem_bytes);

  // 3. Compute total elements to determine n_rows.
  //    total_elems is the same for src and dst (same copy region).
  PrimExpr total_elems = I(1);
  for (const auto &m : src.modes) {
    total_elems = total_elems * m.size;
  }
  total_elems = analyzer->Simplify(total_elems);
  PrimExpr n_rows = analyzer->Simplify(total_elems / row_elems);

  // 4. Determine strides for each side.
  //    - If a side's innermost mode == row_elems, stride comes from mode[1].
  //    - If a side's innermost mode > row_elems (was split), stride = row_elems
  //      (the split portion is contiguous, stride = row_elems * elem_bytes).
  //    - If a side's innermost mode == row_elems and modes.size() == 1,
  //      stride = row_bytes (contiguous, no gap).

  auto get_stride_bytes = [&](const StridedLayout &layout,
                              bool is_smaller_side) -> PrimExpr {
    if (layout.modes.size() == 1) {
      if (is_smaller_side) {
        // This side determined row_elems; it has only 1 mode, meaning
        // all data is contiguous. Stride = row_bytes (no gap).
        return row_bytes;
      } else {
        // Other side is larger but was coalesced to 1 mode; it is fully
        // contiguous. After splitting by row_elems, stride = row_bytes.
        return row_bytes;
      }
    }
    // modes.size() >= 2
    if (analyzer->CanProveEqual(layout.modes[0].size, row_elems)) {
      // Innermost mode matches exactly: stride from the next mode.
      return layout.modes[1].stride * I(elem_bytes);
    } else {
      // Innermost mode is larger than row_elems (we split it).
      // The split region is contiguous, so stride = row_bytes.
      return row_bytes;
    }
  };

  PrimExpr src_stride_bytes = get_stride_bytes(src, src_is_smaller);
  PrimExpr dst_stride_bytes = get_stride_bytes(dst, dst_is_smaller);

  // 5. Assert no >2D cases (not yet implemented).
  ICHECK(src.modes.size() <= 2 && dst.modes.size() <= 2)
      << "PlanMTECopy: >2D strided copy not yet supported. "
      << "src has " << src.modes.size() << " modes, "
      << "dst has " << dst.modes.size() << " modes after coalesce.";
  std::vector<Mode> outer_loops;

  return MTECopy2D{
      n_rows,     row_bytes,  src_stride_bytes,       dst_stride_bytes,
      src.offset, dst.offset, std::move(outer_loops),
  };
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_ASCEND_MTE_PLAN_H_
