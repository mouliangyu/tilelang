/*!
 * \file tl/ascend/op/oob_padding.cc
 * \brief Shared Ascend DMA copy OOB clamping / L1 padding helpers.
 */

#include "oob_padding.h"

#include "op/builtin.h"
#include "op/utils.h"

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
  BoundedDMACopyRanges result{op.src_range, op.dst_range, std::nullopt};
  if (op.src_range.size() != op.src->shape.size() ||
      op.dst_range.size() != op.dst->shape.size()) {
    return result;
  }

  bool need_transpose = op.GetTranspose();

  size_t tail_transpose_axes = 0;
  if (need_transpose) {
    ICHECK(op.dst_range.size() >= 2 && op.src_range.size() >= 2)
        << "Ascend transposed copy requires >= 2D source and "
        << "destination, got src rank " << op.src_range.size() << ", dst rank "
        << op.dst_range.size() << ".";
    tail_transpose_axes = 2;
  }

  auto collect_non_singleton_axes = [&](const Array<Range> &ranges) {
    std::vector<size_t> axes, singleton_axes;
    for (size_t i = 0; i < ranges.size() - tail_transpose_axes; ++i) {
      if (!is_one(ranges[i]->extent))
        axes.push_back(i);
      else
        singleton_axes.push_back(i);
    }
    return std::make_pair(axes, singleton_axes);
  };

  auto [src_axes, src_singleton_axes] =
      collect_non_singleton_axes(op.src_range);
  auto [dst_axes, dst_singleton_axes] =
      collect_non_singleton_axes(op.dst_range);
  if (src_axes.size() != dst_axes.size()) {
    return result;
  }

  if (need_transpose) {
    // Only the trailing M/K pair may be non-singleton for a transposed copy;
    // any non-singleton leading (batch) axis is unsupported here.
    ICHECK(src_axes.empty() && dst_axes.empty())
        << "Ascend transposed copy only supports singleton leading axes, got "
        << src_axes.size() << " non-singleton src leading axes and "
        << dst_axes.size() << " non-singleton dst leading axes.";
    src_axes.insert(src_axes.end(),
                    {op.src_range.size() - 2, op.src_range.size() - 1});
    dst_axes.insert(dst_axes.end(),
                    {op.dst_range.size() - 1, op.dst_range.size() - 2});
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

  PrimExpr singleton_guard = const_true();
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

      DataType dtype = range->extent.dtype();
      auto cast_to_dtype = [dtype](PrimExpr value) {
        return value.dtype() == dtype ? value : cast(dtype, value);
      };
      PrimExpr zero = make_zero(dtype);
      PrimExpr remaining = cast_to_dtype(shape[i]) - cast_to_dtype(range->min);
      PrimExpr valid_extent =
          Min(cast_to_dtype(range->extent), Max(remaining, zero));
      singleton_guard =
          singleton_guard && (analyzer->Simplify(valid_extent) > zero);
    }
  };
  analyze_singleton_axes(op.src_range, op.src->shape, src_singleton_axes);
  analyze_singleton_axes(op.dst_range, op.dst->shape, dst_singleton_axes);
  if (!needs_clamp) {
    return result;
  }

  Array<Range> src_range = op.src_range;
  Array<Range> dst_range = op.dst_range;
  PrimExpr has_data = const_true();

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
    PrimExpr valid_extent =
        Min(extent, Min(Max(src_remaining, zero), Max(dst_remaining, zero)));
    valid_extent = analyzer->Simplify(valid_extent);

    // Keep a possibly empty unit axis coalescible by guarding it with
    // `has_data` instead of exposing a symbolic extent to PlanMTECopy.
    PrimExpr layout_extent = valid_extent;
    if (analyzer->CanProve(valid_extent <= make_const(dtype, 1),
                           arith::ProofStrength::kSymbolicBound)) {
      layout_extent = make_const(dtype, 1);
    }
    src_range.Set(src_axis, Range::FromMinExtent(op.src_range[src_axis]->min,
                                                 layout_extent));
    dst_range.Set(dst_axis, Range::FromMinExtent(op.dst_range[dst_axis]->min,
                                                 layout_extent));
    has_data = has_data && (valid_extent > zero);
  }

  has_data = has_data && singleton_guard;

  result.src = std::move(src_range);
  result.dst = std::move(dst_range);
  result.has_data = analyzer->Simplify(has_data);
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
