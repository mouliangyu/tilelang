/*!
 * \file ascend/layout/ascend_layouts.cc
 * \brief Ascend fractal/NZ layout constructors and analysis (moved out of the
 *  generic src/layout/ tree).
 */

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>

#include "ascend/layout/ascend_layouts.h"
#include "layout/layout.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// Expand a 2-D base layout to the rank of \p buffer by prepending the leading
// (batch) dimensions of the buffer shape. Local copy of the swizzle helper so
// this file does not depend on gemm_layouts.cc internals.
static Layout ExpandLayout2D(const Layout &base, const Buffer &buffer) {
  Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer->shape.size() - 2);
  for (size_t i = 0; i + 2 < buffer->shape.size(); ++i) {
    leading_shape.push_back(buffer->shape[i]);
  }
  return base->Expand(leading_shape);
}

// Ascend fractal layout for Cube operands. A logical [.., rows, cols] tile is
// physically stored as [.., cols / C0, rows / 16, 16, C0], matching the MTE2
// ND2NZ convention where the contiguous axis (cols) is grouped by C0 and is the
// outer physical major. Within a fractal the row lane remains stride C0 and the
// column lane remains stride 1. MajorK means cols is the reduce-K axis; MajorMN
// means cols is the output M/N axis. The legacy NZ API remains an alias of
// MajorK while the frontend migrates to explicit major names.
static constexpr int kAscendRowFractal = 16;

// Column fractal C0 = 32 bytes / element_size, expressed from element bits.
int AscendC0(int element_bits) {
  ICHECK(element_bits > 0 && 256 % element_bits == 0)
      << "Ascend NZ layout: unsupported element bits=" << element_bits;
  return 256 / element_bits;
}

static Layout MakeAscendMajorKLayout2D(PrimExpr rows, PrimExpr cols,
                                       int element_bits) {
  PrimExpr c0 = Integer(AscendC0(element_bits));
  PrimExpr row_frac = Integer(kAscendRowFractal);
  Var i = InputPlaceholder(0); // row
  Var j = InputPlaceholder(1); // col
  PrimExpr ro = FloorDiv(i, row_frac);
  PrimExpr ri = FloorMod(i, row_frac);
  PrimExpr co = FloorDiv(j, c0);
  PrimExpr ci = FloorMod(j, c0);
  return Layout(Array<PrimExpr>{rows, cols}, {co, ro, ri, ci});
}

static void CheckStaticDivisible(const PrimExpr &value, int divisor,
                                 const std::string &what) {
  if (divisor <= 1) {
    return;
  }
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(value);
  const int64_t *int_value = as_const_int(simplified);
  ICHECK(int_value != nullptr)
      << what << " requires a static extent to verify divisibility by "
      << divisor << ", got " << value;
  ICHECK(*int_value % divisor == 0)
      << what << " requires K-axis extent to be divisible by " << divisor
      << ", got " << *int_value
      << ". For blockscaled GEMM, GM->L1 copy supports auto padding for "
         "tail K values, so allocate the L1/L0 K axis to a multiple of "
      << divisor << ".";
}

Layout MakeAscendMajorKLayout(const Buffer &buffer, int k_align) {
  ICHECK(buffer.defined() && buffer->shape.size() >= 2)
      << "MakeAscendMajorKLayout expects rank >= 2 buffer";
  size_t n = buffer->shape.size();
  PrimExpr rows = buffer->shape[n - 2];
  PrimExpr cols = buffer->shape[n - 1];
  CheckStaticDivisible(cols, k_align,
                       "Ascend MajorK layout for " + buffer->name);
  auto base = MakeAscendMajorKLayout2D(rows, cols, buffer->dtype.bits());
  return ExpandLayout2D(base, buffer);
}

Layout MakeAscendMajorMNLayout(const Buffer &buffer) {
  // Same physical convention as MajorK. The distinction is semantic: for a
  // [K, MN] logical tile, the contiguous cols axis is output M/N instead of the
  // reduce-K axis.
  return MakeAscendMajorKLayout(buffer);
}

Layout MakeAscendNZLayout(const Buffer &buffer) {
  return MakeAscendMajorKLayout(buffer);
}

static Layout MakeAscendL0CLayout2D(PrimExpr rows, PrimExpr cols) {
  PrimExpr frac = Integer(kAscendRowFractal);
  Var i = InputPlaceholder(0); // M row
  Var j = InputPlaceholder(1); // N col
  PrimExpr mo = FloorDiv(i, frac);
  PrimExpr mi = FloorMod(i, frac);
  PrimExpr no = FloorDiv(j, frac);
  PrimExpr ni = FloorMod(j, frac);
  return Layout(Array<PrimExpr>{rows, cols}, {no, mo, mi, ni});
}

Layout MakeAscendL0CLayout(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "L0C layout expects a defined buffer";
  ICHECK(buffer->shape.size() >= 2)
      << "L0C layout expects rank >= 2 buffer, got rank="
      << buffer->shape.size();
  size_t ndim = buffer->shape.size();
  PrimExpr rows = buffer->shape[ndim - 2];
  PrimExpr cols = buffer->shape[ndim - 1];
  auto base = MakeAscendL0CLayout2D(rows, cols);
  return ExpandLayout2D(base, buffer);
}

static Layout MakeAscendSFLayout2D(PrimExpr mn, PrimExpr k, int element_bits) {
  PrimExpr mn_frac = Integer(kAscendRowFractal);
  ICHECK(element_bits > 0 && 16 % element_bits == 0)
      << "SF layout expects element bits to divide 16, got " << element_bits;
  PrimExpr pack = Integer(16 / element_bits);
  Var i = InputPlaceholder(0); // M/N axis
  Var j = InputPlaceholder(1); // K axis in scale elements
  PrimExpr mo = FloorDiv(i, mn_frac);
  PrimExpr mi = FloorMod(i, mn_frac);
  PrimExpr ko = FloorDiv(j, pack);
  PrimExpr pi = FloorMod(j, pack);
  return Layout(Array<PrimExpr>{mn, k}, {mo, ko, mi, pi});
}

Layout MakeAscendSFLayout(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "SF layout expects a defined buffer";
  ICHECK(buffer->shape.size() >= 2)
      << "SF layout expects rank >= 2 buffer with trailing [MN, K], got rank="
      << buffer->shape.size();
  size_t ndim = buffer->shape.size();
  PrimExpr mn = buffer->shape[ndim - 2];
  PrimExpr k = buffer->shape[ndim - 1];
  ICHECK(buffer->dtype.bits() > 0 && 16 % buffer->dtype.bits() == 0)
      << "SF layout expects storage dtype whose bit width divides 16, got "
      << buffer->dtype;
  if (buffer->dtype.bits() != 16) {
    arith::Analyzer analyzer;
    PrimExpr k_extent = analyzer.Simplify(k);
    const int64_t *k_value = as_const_int(k_extent);
    ICHECK(k_value != nullptr)
        << "SF_K layout for non-uint16 storage requires a static K-axis "
           "extent to verify 2-byte alignment, got "
        << k;
    ICHECK((*k_value * buffer->dtype.bits()) % 16 == 0)
        << "SF_K layout for " << buffer->name << " with dtype " << buffer->dtype
        << " requires the K-axis extent to be 2-byte aligned. Got K="
        << *k_value << ". Pad the K-axis with identity e8m0 scale value 127.";
  }
  auto base = MakeAscendSFLayout2D(mn, k, buffer->dtype.bits());
  return ExpandLayout2D(base, buffer);
}

// ── Ascend fractal layout analysis ──────────────────────────────────

bool TryExtractAscendFractalLayout(const Layout &layout, const Buffer &buf,
                                   AscendFractalLayoutInfo *out) {
  if (!layout.defined() || !buf.defined() || buf->shape.size() < 2)
    return false;

  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  size_t n = buf->shape.size();
  PrimExpr rows = buf->shape[n - 2];
  PrimExpr cols = buf->shape[n - 1];
  PrimExpr row_frac = I(16);
  PrimExpr c0 = I(AscendC0(buf->dtype.bits()));
  PrimExpr sf_pack = I(0);
  bool can_be_sf = buf->dtype.bits() > 0 && 16 % buf->dtype.bits() == 0;
  if (can_be_sf) {
    sf_pack = I(16 / buf->dtype.bits());
  }

  Array<PrimExpr> fwd = layout->GetForwardIndex();
  if (fwd.size() < 4)
    return false;
  Array<PrimExpr> vars = layout->GetForwardVars();
  if (vars.size() < 2)
    return false;
  PrimExpr row = vars[vars.size() - 2];
  PrimExpr col = vars[vars.size() - 1];

  arith::Analyzer analyzer;
  Array<PrimExpr> input_shape = layout->InputShape();
  for (size_t i = 0; i < vars.size() && i < input_shape.size(); ++i) {
    if (const auto *var = vars[i].as<VarNode>()) {
      analyzer.Bind(GetRef<Var>(var), Range(0, input_shape[i]));
    }
  }
  auto same = [&](const PrimExpr &a, const PrimExpr &b) {
    return StructuralEqual()(analyzer.Simplify(a), analyzer.Simplify(b));
  };
  auto div_expr = [](const PrimExpr &v, const PrimExpr &f) {
    return FloorDiv(v, f);
  };
  auto ceildiv_expr = [](const PrimExpr &v, const PrimExpr &f) {
    return FloorDiv(v + f - Integer(1), f);
  };
  auto mod_expr = [](const PrimExpr &v, const PrimExpr &f) {
    return FloorMod(v, f);
  };

  size_t m = fwd.size();
  PrimExpr o0 = fwd[m - 4];
  PrimExpr o1 = fwd[m - 3];
  PrimExpr i0 = fwd[m - 2];
  PrimExpr i1 = fwd[m - 1];
  auto matches = [&](const PrimExpr &outer0_axis, const PrimExpr &outer1_axis,
                     const PrimExpr &inner0_axis, const PrimExpr &inner1_axis,
                     PrimExpr outer0_factor, PrimExpr outer1_factor,
                     PrimExpr inner0_factor, PrimExpr inner1_factor) {
    return same(o0, div_expr(outer0_axis, outer0_factor)) &&
           same(o1, div_expr(outer1_axis, outer1_factor)) &&
           same(i0, mod_expr(inner0_axis, inner0_factor)) &&
           same(i1, mod_expr(inner1_axis, inner1_factor));
  };
  auto matches_standard = [&](const PrimExpr &c0_axis,
                              const PrimExpr &row16_axis, PrimExpr c0_factor,
                              PrimExpr row16_factor) {
    return matches(c0_axis, row16_axis, row16_axis, c0_axis, c0_factor,
                   row16_factor, row16_factor, c0_factor);
  };
  auto matches_sf = [&](const PrimExpr &mn_axis, const PrimExpr &k_axis,
                        PrimExpr mn_factor, PrimExpr k_factor) {
    return matches(mn_axis, k_axis, mn_axis, k_axis, mn_factor, k_factor,
                   mn_factor, k_factor);
  };

  // Standard fractal: col/C0, row/16 (L1 NZ layout)
  if (matches_standard(col, row, c0, row_frac)) {
    bool is_l0c =
        buf.scope() == "shared.l0c" || buf.scope() == "shared.l0c.dyn";
    *out = {is_l0c ? AscendFractalKind::kL0C : AscendFractalKind::kMajorLike,
            c0,
            row_frac,
            rows,
            cols,
            ceildiv_expr(cols, c0),
            ceildiv_expr(rows, row_frac),
            LogicalAxis::kCol,
            LogicalAxis::kRow};
    return true;
  }

  // SF_K fractal: row/16, col/pack where pack = 16 bits / element bits.
  if (can_be_sf && matches_sf(row, col, row_frac, sf_pack)) {
    *out = {AscendFractalKind::kSF,
            sf_pack,
            row_frac,
            rows,
            cols,
            ceildiv_expr(rows, row_frac),
            ceildiv_expr(cols, sf_pack),
            LogicalAxis::kCol,
            LogicalAxis::kRow};
    return true;
  }

  // Swapped: row/C0, col/16
  if (matches_standard(row, col, c0, row_frac)) {
    *out = {AscendFractalKind::kMajorLike,
            c0,
            row_frac,
            rows,
            cols,
            ceildiv_expr(rows, c0),
            ceildiv_expr(cols, row_frac),
            LogicalAxis::kRow,
            LogicalAxis::kCol};
    return true;
  }

  // L0C fixed 16×16: col/16, row/16
  if (matches_standard(col, row, row_frac, row_frac)) {
    *out = {AscendFractalKind::kL0C,
            row_frac,
            row_frac,
            rows,
            cols,
            ceildiv_expr(cols, row_frac),
            ceildiv_expr(rows, row_frac),
            LogicalAxis::kCol,
            LogicalAxis::kRow};
    return true;
  }

  return false;
}

bool TryExtractAscendFractalRegion(const Layout &layout,
                                   const Array<Range> &logical_region,
                                   AscendFractalRegionInfo *out) {
  if (!layout.defined() || logical_region.size() != layout->InputDim())
    return false;
  Array<Range> phys = layout->MapRegion(logical_region);
  if (phys.size() < 4)
    return false;
  size_t n = phys.size();
  *out = {phys[n - 4], phys[n - 3], phys[n - 2], phys[n - 1]};
  return true;
}

// Ascend ND (row-major) identity layout for GM/UB operands. Forward is the
// identity [i0, .., i_{n-1}] -> [i0, .., i_{n-1}], so OutputShape ==
// InputShape. Used only for copy-parameter derivation in lowering;
// intentionally NOT anchored into layout_map/buffer_remap (an identity remap is
// a no-op, and a flattening layout would alias GM/UB buffers to 1D).
Layout makeAscendNDLayout(const Buffer &buffer) {
  int ndim = static_cast<int>(buffer->shape.size());
  Array<IterVar> iter_vars;
  Array<PrimExpr> forward_index;
  for (int i = 0; i < ndim; ++i) {
    auto iv = MakeIterVar("i" + std::to_string(i), buffer->shape[i]);
    iter_vars.push_back(iv);
    forward_index.push_back(iv->var);
  }
  return Layout(iter_vars, forward_index);
}

// ── MakeStridedSlice ──────────────────────────────────────────────────
// Given a physical (remapped) buffer and a layout, compute a 0-origin
// strided buffer view for the given logical sub-region.
//
// Steps:
// 1. MapRegion(logical_region) → physical region (one Range per output dim)
// 2. Compute row-major strides from phys_buf shape (or use existing strides)
// 3. offset = sum(phys_region[i].min * strides[i])
// 4. shape  = [phys_region[i].extent for each dim]
// 5. Return Buffer(same data, shape, strides, offset)
tirx::Buffer MakeStridedSlice(const tirx::Buffer &phys_buf,
                              const Layout &layout,
                              const Array<Range> &logical_region) {
  Array<Range> phys_region = layout->MapRegion(logical_region);
  int n = static_cast<int>(phys_region.size());

  ICHECK_EQ(n, static_cast<int>(phys_buf->shape.size()))
      << "MakeStridedSlice: physical region rank (" << n
      << ") != physical buffer rank (" << phys_buf->shape.size() << ")";

  // Use phys_buf's explicit strides if present, otherwise compute row-major.
  Array<PrimExpr> strides;
  if (!phys_buf->strides.empty()) {
    strides = phys_buf->strides;
  } else {
    strides.resize(n);
    PrimExpr cur = Integer(1);
    for (int i = n - 1; i >= 0; --i) {
      strides.Set(i, cur);
      cur = cur * phys_buf->shape[i];
    }
  }

  // Flat offset and shape from phys_region.
  PrimExpr offset = make_const(DataType::Int(64), 0);
  Array<PrimExpr> shape;
  shape.reserve(n);
  for (int i = 0; i < n; ++i) {
    offset = offset + cast(DataType::Int(64), phys_region[i]->min) *
                          cast(DataType::Int(64), strides[i]);
    shape.push_back(phys_region[i]->extent);
  }

  return Buffer(phys_buf->data, phys_buf->dtype, shape, strides, offset,
                phys_buf->name + "_slice", phys_buf->data_alignment,
                phys_buf->offset_factor, phys_buf->buffer_type);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.make_ascend_major_k_layout",
           [](const Buffer &buffer, int k_align) {
             return MakeAscendMajorKLayout(buffer, k_align);
           })
      .def("tl.make_ascend_major_mn_layout",
           [](const Buffer &buffer) { return MakeAscendMajorMNLayout(buffer); })
      .def("tl.make_ascend_l0c_layout",
           [](const Buffer &buffer) { return MakeAscendL0CLayout(buffer); })
      .def("tl.make_ascend_sf_layout",
           [](const Buffer &buffer) { return MakeAscendSFLayout(buffer); })
      .def("tl.make_ascend_nz_layout",
           [](const Buffer &buffer) { return MakeAscendNZLayout(buffer); })
      .def("tl.try_extract_fractal_layout",
           [](Layout layout, const Buffer &buf) -> Optional<Map<String, Any>> {
             AscendFractalLayoutInfo info;
             if (!TryExtractAscendFractalLayout(layout, buf, &info))
               return std::nullopt;
             int c0_axis_int = static_cast<int>(info.c0_axis);
             int row16_axis_int = static_cast<int>(info.row16_axis);
             int kind_int = static_cast<int>(info.kind);
             return Map<String, Any>{
                 {"kind", Integer(kind_int)},
                 {"c0", info.c0},
                 {"row_frac", info.row_frac},
                 {"rows", info.rows},
                 {"cols", info.cols},
                 {"outer0", info.outer0},
                 {"outer1", info.outer1},
                 {"c0_axis", Integer(c0_axis_int)},
                 {"row16_axis", Integer(row16_axis_int)},
             };
           })
      .def("tl.make_strided_slice", [](const Buffer &phys_buf, Layout layout,
                                       Array<Range> logical_region) {
        return MakeStridedSlice(phys_buf, layout, logical_region);
      });
}

} // namespace tl
} // namespace tvm
