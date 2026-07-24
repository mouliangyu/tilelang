/*!
 * \file tl/ascend/op/oob_padding.h
 * \brief Shared helpers for Ascend DMA copy out-of-bounds (OOB) clamping and
 *        L1 padding fills.
 *
 * These were originally private helpers inside ascend/op/copy.cc. They are
 * exposed here so a pre-AutoSchedule pass can perform the same OOB clamp and
 * emit the padding fills as standalone statements (visible to scheduling),
 * while copy.cc's lowering keeps consuming the already-clamped ranges.
 */

#ifndef TVM_TL_ASCEND_OP_OOB_PADDING_H_
#define TVM_TL_ASCEND_OP_OOB_PADDING_H_

#include "op/copy.h"

#include "ascend/layout/ascend_layouts.h"
#include "layout/layout.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <utility>

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

/*! \brief Ascend DMA route selected from the source/destination scopes. */
enum class DMAPath {
  kNone,
  kGMToUB,
  kUBToGM,
  kGMToL1,
  kL1ToL0A,
  kL1ToL0B,
  kL0CToUB,
  kUBToL1,
  kL0CToGM,
};

/*!
 * \brief Classify a copy between the given buffer scopes.
 * \return The matching Ascend DMA route, or DMAPath::kNone when the copy is
 * not handled by an Ascend DMA instruction.
 */
DMAPath GetDMAPath(const Buffer &src, const Buffer &dst);

/*!
 * \brief Look up a buffer's inferred Layout in a layout map, matching either
 * the buffer object or its underlying data Var. Returns nullopt if absent.
 */
Optional<Layout> FindLayoutForBuffer(const LayoutMap &layout_map,
                                     const Buffer &buffer);

/*!
 * \brief Result of clamping a DMA copy's ranges to stay in bounds.
 *
 * `src` / `dst` are the (possibly shrunk) copy ranges. `has_data` is an
 * optional runtime guard predicate: when defined and not trivially true, the
 * copy must be guarded by `if (has_data)`; when it is provably false the copy
 * should be dropped entirely.
 */
struct BoundedDMACopyRanges {
  Array<Range> src;
  Array<Range> dst;
  Optional<PrimExpr> has_data;
};

/*!
 * \brief Clamp a copy's trailing tail to the valid in-bounds region.
 *
 * Pairs non-singleton axes between src and dst (swapping the trailing two for a
 * transposed copy), shrinks any axis whose range exceeds the buffer shape, and
 * builds a `has_data` guard for dynamically-empty tiles. Returns the original
 * ranges unchanged (has_data == nullopt) when no clamp is needed or the shape
 * structure is unsupported.
 */
BoundedDMACopyRanges ClampDMACopyTail(const CopyNode &op,
                                      arith::Analyzer *analyzer);

/*!
 * \brief Build the (fill_pattern, fill_word_bits) pair for an L1 padding fill.
 *
 * The pattern is the raw word written by create_cbuf_matrix; fill_word_bits is
 * 16 or 32 depending on the destination element width. Only an immediate-zero
 * pad value is currently supported.
 */
std::pair<PrimExpr, int> MakeL1FillPattern(const CopyNode &op);

/*!
 * \brief Emit an ascend_fill_l1 that zeros the column (C0-block) tail of a
 *        padded GM->L1 destination, or nullopt when there is no tail.
 *
 * \param dst_ptr Base pointer to the destination L1 buffer (byte offset is
 *                computed internally and passed as the fill's arg[1]).
 * \param dst_n   Destination NZ row count (outer1 * row_frac for major
 * layouts).
 * \param valid_cols In-bounds column extent actually filled by the copy.
 */
Optional<Stmt> MakeL1ColPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &dst_ptr, const PrimExpr &dst_n,
                                const PrimExpr &valid_cols);

/*!
 * \brief Emit an ascend_fill_l1 that zeros the row tail of a padded GM->L1
 *        destination, or nullopt when there is no tail.
 */
Optional<Stmt> MakeL1RowPadding(const CopyNode &op,
                                const AscendFractalLayoutInfo &layout,
                                arith::Analyzer *analyzer,
                                const PrimExpr &dst_ptr, const PrimExpr &dst_n,
                                const PrimExpr &valid_rows);

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_OOB_PADDING_H_
