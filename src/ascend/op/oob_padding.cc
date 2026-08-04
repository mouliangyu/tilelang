/*!
 * \file tl/ascend/op/oob_padding.cc
 * \brief Shared Ascend DMA copy OOB clamping / L1 padding helpers.
 */

#include "oob_padding.h"

#include "ascend_mte_plan.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

DMAPath GetDMAPath(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return DMAPath::kGMToUB;
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return DMAPath::kUBToGM;
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return DMAPath::kGMToL1;
  if (IsL1Buffer(src) && IsL0ABuffer(dst))
    return DMAPath::kL1ToL0A;
  if (IsL1Buffer(src) && IsL0BBuffer(dst))
    return DMAPath::kL1ToL0B;
  if (IsL0CBuffer(src) && IsSharedBuffer(dst))
    return DMAPath::kL0CToUB;
  if (IsSharedBuffer(src) && IsL1Buffer(dst))
    return DMAPath::kUBToL1;
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return DMAPath::kL0CToGM;
  return DMAPath::kNone;
}

Optional<Layout> FindLayoutForBuffer(const LayoutMap &layout_map,
                                     const Buffer &buffer) {
  if (layout_map.count(buffer)) {
    return layout_map[buffer];
  }
  for (const auto &kv : layout_map) {
    if (kv.first->data.same_as(buffer->data)) {
      return kv.second;
    }
  }
  return std::nullopt;
}

BoundedDMACopyRanges ClampDMACopyTail(const CopyNode &op,
                                      arith::Analyzer *analyzer) {
  BoundedDMACopyRanges result{op.src_range, op.dst_range, false};
  if (op.src_range.size() != op.src->shape.size() ||
      op.dst_range.size() != op.dst->shape.size()) {
    return result;
  }

  bool need_transpose = op.GetTranspose();
  DMAPath dma_path = GetDMAPath(op.src, op.dst);
  bool preserve_l0c_source_geometry = dma_path == DMAPath::kL0CToGM;
  std::vector<size_t> src_axes, dst_axes;
  std::vector<size_t> src_singleton_axes, dst_singleton_axes;

  if (dma_path == DMAPath::kGMToL1) {
    StridedLayout src_layout = NormalizeMTE2DLayout(
        op.src, op.src_range, analyzer, "Ascend GM->L1 OOB handling");
    StridedLayout dst_layout = NormalizeTrailingMTE2DLayout(
        op.dst, op.dst_range, analyzer, "Ascend GM->L1 OOB handling");
    size_t src_inner_axis = static_cast<size_t>(src_layout.modes[0].axis);
    size_t src_row_axis = static_cast<size_t>(src_layout.modes[1].axis);
    size_t dst_inner_axis = static_cast<size_t>(dst_layout.modes[0].axis);
    size_t dst_row_axis = static_cast<size_t>(dst_layout.modes[1].axis);
    src_axes = {src_row_axis, src_inner_axis};
    dst_axes = need_transpose
                   ? std::vector<size_t>{dst_inner_axis, dst_row_axis}
                   : std::vector<size_t>{dst_row_axis, dst_inner_axis};

    auto collect_other_axes = [](size_t ndim, size_t axis0, size_t axis1) {
      std::vector<size_t> axes;
      for (size_t i = 0; i < ndim; ++i) {
        if (i != axis0 && i != axis1) {
          axes.push_back(i);
        }
      }
      return axes;
    };
    src_singleton_axes =
        collect_other_axes(op.src_range.size(), src_row_axis, src_inner_axis);
    dst_singleton_axes =
        collect_other_axes(op.dst_range.size(), dst_row_axis, dst_inner_axis);
  } else {
    auto collect_non_singleton_axes = [](const Array<Range> &ranges) {
      std::vector<size_t> axes, singleton_axes;
      for (size_t i = 0; i < ranges.size(); ++i) {
        if (!is_one(ranges[i]->extent))
          axes.push_back(i);
        else
          singleton_axes.push_back(i);
      }
      return std::make_pair(axes, singleton_axes);
    };

    std::tie(src_axes, src_singleton_axes) =
        collect_non_singleton_axes(op.src_range);
    std::tie(dst_axes, dst_singleton_axes) =
        collect_non_singleton_axes(op.dst_range);
    if (src_axes.size() != dst_axes.size()) {
      return result;
    }
  }

  bool needs_clamp = false;
  for (size_t i = 0; i < src_axes.size(); ++i) {
    size_t src_axis = src_axes[i];
    size_t dst_axis = dst_axes[i];
    const Range &src = op.src_range[src_axis];
    const Range &dst = op.dst_range[dst_axis];
    if (!analyzer->CanProveEqual(src->extent, dst->extent) ||
        !analyzer->CanProve(src->min >= make_zero(src->min.dtype()),
                            arith::ProofStrength::kSymbolicBound) ||
        !analyzer->CanProve(dst->min >= make_zero(dst->min.dtype()),
                            arith::ProofStrength::kSymbolicBound)) {
      return result;
    }
    if (!analyzer->CanProve(src->min + src->extent <= op.src->shape[src_axis],
                            arith::ProofStrength::kSymbolicBound) ||
        !analyzer->CanProve(dst->min + dst->extent <= op.dst->shape[dst_axis],
                            arith::ProofStrength::kSymbolicBound)) {
      needs_clamp = true;
    }
  }

  auto analyze_singleton_axes = [&](const Array<Range> &ranges,
                                    const Array<PrimExpr> &shape,
                                    const std::vector<size_t> &singleton_axes) {
    for (auto &i : singleton_axes) {
      const Range &range = ranges[i];
      if (!analyzer->CanProve(range->min >= make_zero(range->min.dtype()) &&
                                  range->min + range->extent <= shape[i],
                              arith::ProofStrength::kSymbolicBound)) {
        needs_clamp = true;
      }
    }
  };
  analyze_singleton_axes(op.src_range, op.src->shape, src_singleton_axes);
  analyze_singleton_axes(op.dst_range, op.dst->shape, dst_singleton_axes);
  if (!needs_clamp) {
    return result;
  }

  Array<Range> src_range = op.src_range;
  Array<Range> dst_range = op.dst_range;

  for (size_t i = 0; i < src_axes.size(); ++i) {
    size_t src_axis = src_axes[i];
    size_t dst_axis = dst_axes[i];
    DataType dtype = op.src_range[src_axis]->extent.dtype();
    auto cast_to_dtype = [dtype](PrimExpr value) {
      return value.dtype() == dtype ? value : cast(dtype, value);
    };
    PrimExpr zero = make_zero(dtype);
    PrimExpr extent = cast_to_dtype(op.src_range[src_axis]->extent);
    PrimExpr src_remaining = cast_to_dtype(op.src->shape[src_axis]) -
                             cast_to_dtype(op.src_range[src_axis]->min);
    PrimExpr dst_remaining = cast_to_dtype(op.dst->shape[dst_axis]) -
                             cast_to_dtype(op.dst_range[dst_axis]->min);
    PrimExpr src_valid_extent = Min(extent, Max(src_remaining, zero));
    PrimExpr dst_valid_extent = Min(extent, Max(dst_remaining, zero));
    PrimExpr valid_extent = Min(src_valid_extent, dst_valid_extent);
    src_valid_extent = analyzer->Simplify(src_valid_extent);
    valid_extent = analyzer->Simplify(valid_extent);

    // A destination OOB tail only reduces the L0C->GM transfer count. It must
    // not rewrite the producer's L0C matrix geometry: lowering uses the source
    // region to recover the compact FIX source pitch. Keep the independently
    // in-bounds source extent while the destination carries valid_extent.
    PrimExpr src_transfer_extent =
        preserve_l0c_source_geometry ? src_valid_extent : valid_extent;
    src_range.Set(src_axis, Range::FromMinExtent(op.src_range[src_axis]->min,
                                                 src_transfer_extent));
    dst_range.Set(dst_axis, Range::FromMinExtent(op.dst_range[dst_axis]->min,
                                                 valid_extent));
  }

  // Keep fixed/singleton-axis validity in the rewritten regions as a semantic
  // 0/1 extent. LowerDMACopy consumes that information to build the final
  // in-bounds guard after AutoSchedule, then normalizes a possibly-empty unit
  // axis to extent 1 only for hardware DMA geometry.
  auto clamp_singleton_axes =
      [&](const Array<Range> &original, const Array<PrimExpr> &shape,
          const std::vector<size_t> &axes, Array<Range> *clamped) {
        for (size_t axis : axes) {
          const Range &range = original[axis];
          DataType dtype = range->extent.dtype();
          auto cast_to_dtype = [dtype](PrimExpr value) {
            return value.dtype() == dtype ? value : cast(dtype, value);
          };
          PrimExpr zero = make_zero(dtype);
          PrimExpr remaining =
              cast_to_dtype(shape[axis]) - cast_to_dtype(range->min);
          PrimExpr valid_extent =
              Min(cast_to_dtype(range->extent), Max(remaining, zero));
          clamped->Set(axis, Range::FromMinExtent(
                                 range->min, analyzer->Simplify(valid_extent)));
        }
      };
  clamp_singleton_axes(op.src_range, op.src->shape, src_singleton_axes,
                       &src_range);
  clamp_singleton_axes(op.dst_range, op.dst->shape, dst_singleton_axes,
                       &dst_range);

  result.src = std::move(src_range);
  result.dst = std::move(dst_range);
  result.clamped = true;
  return result;
}

std::pair<PrimExpr, int> MakeL1FillPattern(const CopyNode &op) {
  DataType dtype = op.dst->dtype;
  int bits = dtype.bits();
  ICHECK_EQ(dtype.lanes(), 1)
      << "Ascend GM->L1 padding requires a scalar destination dtype, got "
      << dtype;
  ICHECK(bits > 0 && bits <= 32 && (bits <= 16 || bits == 32))
      << "Ascend GM->L1 padding supports destination element widths up to "
         "16 bits or exactly 32 bits, got "
      << dtype;

  int fill_word_bits = bits <= 16 ? 16 : 32;
  Optional<PrimExpr> pad_value = op.GetPadValue();
  if (pad_value.defined()) {
    const auto *int_value = pad_value.value().as<IntImmNode>();
    const auto *float_value = pad_value.value().as<FloatImmNode>();
    // TODO(ascend): Support non-zero pad values by encoding them into the
    // 16/32-bit fill pattern required by create_cbuf_matrix.
    ICHECK((int_value && int_value->value == 0) ||
           (float_value && float_value->value == 0.0))
        << "Ascend GM->L1 pad_value only supports immediate zero, got "
        << pad_value.value();
  }
  return {make_const(DataType::UInt(32), 0), fill_word_bits};
}

Optional<Stmt> MakeL1ColPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &dst_ptr, const PrimExpr &dst_n,
                                const PrimExpr &valid_cols) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  size_t dst_ndim = op.dst_range.size();
  ICHECK_GE(dst_ndim, 2U);
  PrimExpr dst_rows = op.dst_range[dst_ndim - 2]->extent;
  PrimExpr dst_cols = op.dst_range[dst_ndim - 1]->extent;
  PrimExpr dst_row_min = op.dst_range[dst_ndim - 2]->min;
  PrimExpr dst_col_min = op.dst_range[dst_ndim - 1]->min;
  PrimExpr tail_cols = dst_cols - valid_cols;
  if (analyzer->CanProve(tail_cols <= I(0))) {
    return std::nullopt;
  }

  ICHECK(analyzer->CanProve(valid_cols <= dst_cols))
      << "Ascend GM->L1 padded copy requires destination col extent >= "
         "mapped source col extent, got source="
      << valid_cols << " and destination=" << dst_cols;
  ICHECK(analyzer->CanProveEqual(dst_row_min, I(0)) &&
         analyzer->CanProveEqual(dst_col_min, I(0)) &&
         analyzer->CanProveEqual(dst_rows, layout.rows))
      << "Ascend GM->L1 col padding requires a zero-based destination "
         "region covering the full row axis; got destination ranges "
      << op.dst_range;

  PrimExpr valid_c0_blocks = FloorDiv(valid_cols + layout.c0 - I(1), layout.c0);
  PrimExpr dst_c0_blocks = FloorDiv(dst_cols + layout.c0 - I(1), layout.c0);
  PrimExpr tail_c0_blocks = dst_c0_blocks - valid_c0_blocks;
  if (analyzer->CanProve(tail_c0_blocks <= I(0))) {
    return std::nullopt;
  }

  // Every NZ row within one C0 block occupies exactly 32 bytes, including
  // sub-byte element dtypes.
  PrimExpr tail_byte_offset = valid_c0_blocks * dst_n * I(32);
  auto [fill_pattern, fill_word_bits] = MakeL1FillPattern(op);
  return Evaluate(Call(DataType::Void(), ascend_fill_l1(),
                       {dst_ptr, tail_byte_offset, fill_pattern, tail_c0_blocks,
                        dst_n, I(0), I(fill_word_bits)}));
}

Optional<Stmt> MakeL1RowPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &dst_ptr, const PrimExpr &dst_n,
                                const PrimExpr &valid_rows) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  size_t dst_ndim = op.dst_range.size();
  ICHECK_GE(dst_ndim, 2U);
  PrimExpr dst_rows = op.dst_range[dst_ndim - 2]->extent;
  PrimExpr dst_cols = op.dst_range[dst_ndim - 1]->extent;
  PrimExpr dst_row_min = op.dst_range[dst_ndim - 2]->min;
  PrimExpr dst_col_min = op.dst_range[dst_ndim - 1]->min;
  PrimExpr tail_rows = dst_rows - valid_rows;
  if (analyzer->CanProve(tail_rows <= I(0))) {
    return std::nullopt;
  }

  ICHECK(analyzer->CanProve(valid_rows <= dst_rows))
      << "Ascend GM->L1 padded copy requires destination row extent >= "
         "mapped source row extent, got source="
      << valid_rows << " and destination=" << dst_rows;
  ICHECK(analyzer->CanProveEqual(dst_row_min, I(0)) &&
         analyzer->CanProveEqual(dst_col_min, I(0)) &&
         analyzer->CanProveEqual(dst_cols, layout.cols))
      << "Ascend GM->L1 row padding requires a zero-based destination "
         "region covering the full col axis; got destination ranges "
      << op.dst_range;

  // ND2NZ does not initialize missing rows inside a row16 fractal. Fill the
  // row tail in every C0 group with one strided create_cbuf_matrix call.
  PrimExpr dst_c0_blocks = FloorDiv(dst_cols + layout.c0 - I(1), layout.c0);
  PrimExpr tail_byte_offset = valid_rows * I(32);
  PrimExpr dst_gap = dst_n - tail_rows;
  auto [fill_pattern, fill_word_bits] = MakeL1FillPattern(op);
  return Evaluate(Call(DataType::Void(), ascend_fill_l1(),
                       {dst_ptr, tail_byte_offset, fill_pattern, dst_c0_blocks,
                        tail_rows, dst_gap, I(fill_word_bits)}));
}

} // namespace ascend
} // namespace tl
} // namespace tvm
