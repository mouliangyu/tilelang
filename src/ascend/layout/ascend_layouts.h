/*!
 * \file ascend/layout/ascend_layouts.h
 * \brief Ascend fractal/NZ layouts for Cube operands and their analysis.
 *
 * These layouts and utilities are Ascend-specific and were split out of the
 * generic src/layout/ tree. They rely on the shared layout infrastructure in
 * "layout/layout.h" (Layout, tirx::Buffer, Range, ExpandLayout2D, ...).
 */
#ifndef TVM_TL_ASCEND_LAYOUT_ASCEND_LAYOUTS_H_
#define TVM_TL_ASCEND_LAYOUT_ASCEND_LAYOUTS_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {

// Ascend fractal layouts for Cube operands. A logical [.., rows, cols]
// tile maps to physical [.., cols / C0, rows / 16, 16, C0] (a grid of 16xC0
// fractals, row-major within a fractal) where C0 = 32 bytes / element_size
// (16 for bf16/fp16, 32 for fp8, 8 for fp32). In MajorK, cols is the reduce-K
// axis; in MajorMN, cols is the output M/N axis. The legacy NZ name is kept as
// an alias of MajorK during the transition.
Layout MakeAscendMajorKLayout(const tirx::Buffer &buffer, int k_align = 1);
Layout MakeAscendMajorMNLayout(const tirx::Buffer &buffer, int k_align = 1);
Layout MakeAscendL0CLayout(const tirx::Buffer &buffer);
Layout MakeAscendSFLayout(const tirx::Buffer &buffer);
Layout MakeAscendNZLayout(const tirx::Buffer &buffer);

// Ascend NZ column fractal C0 = 32 bytes / element_size = 256 / element_bits
// (16 for bf16/fp16, 32 for fp8, 8 for fp32).
int AscendC0(int element_bits);

// Ascend ND (row-major) identity layout for GM/UB operands; OutputShape ==
// InputShape. Used for copy-parameter derivation only (not anchored).
Layout makeAscendNDLayout(const tirx::Buffer &buffer);

/// Compute a 0-origin strided buffer view for a logical sub-region on a
/// layout-annotated physical buffer.
///
/// Given:
///   - phys_buf: the remapped buffer with physical (output-space) shape,
///     e.g. [M/16, K/C0, 16, C0] for NZ.
///   - layout: the layout whose forward map describes the physical layout.
///   - logical_region: one Range per layout input dim, [min, min+extent).
///
/// Produces a Buffer sharing phys_buf->data, with:
///   - shape = physical extents (from MapRegion)
///   - strides = row-major strides of phys_buf shape
///   - elem_offset = sum(phys_region[i].min * strides[i])
///
/// The result is a plain strided buffer — no layout knowledge needed by
/// the consumer.
///
/// Example (NZ): logical [32:64, 32:96] on [128, 128] bf16
///   → phys_region = [2:2, 2:4, 0:16, 0:16]
///   → shape = [2, 4, 16, 16], strides = [2048, 256, 16, 1], offset = 4608
tirx::Buffer MakeStridedSlice(const tirx::Buffer &phys_buf,
                              const Layout &layout,
                              const ffi::Array<Range> &logical_region);

// ── Ascend fractal layout analysis ──────────────────────────────────
//
// These utilities inspect a Layout's forward map to determine whether it
// matches the canonical Ascend fractal structure and, if so, which logical
// axis is grouped by C0 and which by 16.  The parser works by checking the
// trailing four output slots:
//
//   [axis_c0 / C0, axis_16 / 16, axis_16 % 16, axis_c0 % C0]
//
// This avoids structural-equality against constructor outputs and correctly
// distinguishes C0 from row-fractal even when C0 == 16 (bf16/fp16).

enum class AscendFractalKind {
  kMajorLike, // [col/C0, row/16, row%16, col%C0] or swapped
  kL0C,       // [col/16, row/16, row%16, col%16] (fixed 16×16 accumulator)
  kSF,        // [row/16, col/pack, row%16, col%pack] for MX scale factors
};

enum class LogicalAxis {
  kRow, // logical dim -2
  kCol, // logical dim -1
};

struct AscendFractalLayoutInfo {
  AscendFractalKind kind;
  PrimExpr c0;       // C0 value (or 16 for L0C, pack for SF)
  PrimExpr row_frac; // always 16
  PrimExpr rows;     // logical rows (dim -2)
  PrimExpr cols;     // logical cols (dim -1)
  PrimExpr outer0;   // full physical extent of C0-axis outer for matrix data or
                     // row16-axis outer for SF
  PrimExpr outer1;   // full physical extent of row16-axis outer for matrix data
                     // or C0-axis outer for SF
  LogicalAxis c0_axis;    // which logical axis is grouped by C0/pack
  LogicalAxis row16_axis; // which logical axis is grouped by 16
};

struct AscendFractalRegionInfo {
  Range outer0;    // C0-axis outer region for matrix data or row16-axis outer
                   // region for SF
  Range outer1;    // row16-axis outer region for matrix data or C0-axis outer
                   // region for SF
  Range row_inner; // row%16 region
  Range col_inner; // C0% (or 16%) region
};

/// Try to parse \p layout as an Ascend fractal layout for \p buf.
/// Returns true and fills \p out on success.
bool TryExtractAscendFractalLayout(const Layout &layout,
                                   const tirx::Buffer &buf,
                                   AscendFractalLayoutInfo *out);

/// Map a logical region through \p layout and extract the trailing four
/// physical output ranges into \p out.
bool TryExtractAscendFractalRegion(const Layout &layout,
                                   const ffi::Array<Range> &logical_region,
                                   AscendFractalRegionInfo *out);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_LAYOUT_ASCEND_LAYOUTS_H_
