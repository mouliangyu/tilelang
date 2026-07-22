/*!
 * \file tl/ascend/op/copy.cc
 * \brief Ascend implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "ascend/layout/ascend_layouts.h"
#include "ascend_mte_plan.h"
#include "backend/common/target_utils.h"
#include "layout/layout.h"
#include "oob_padding.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

namespace {

bool IsDMACopy(const CopyNode &op) { return GetDMAPath(op.src, op.dst) >= 0; }

bool IsInsideSimtVF(const LowerArgs &T) {
  if (!T.thread_var.defined()) {
    return false;
  }
  std::string name_hint = T.thread_var->name_hint;
  return name_hint.rfind("simtvf", 0) == 0;
}

void CollectFragmentLayouts(const PrimExpr &expr,
                            const Map<Var, PrimExpr> &bind_var_to_expr,
                            const LayoutMap &existing_layouts,
                            PrimExpr thread_extent, Range thread_bounds,
                            Map<Buffer, Layout> &result_map) {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      if (bind_var_to_expr.count(var)) {
        CollectFragmentLayouts(bind_var_to_expr[var], bind_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

PrimExpr ComputeFlatOffset(const Buffer &buf, const Array<Range> &range) {
  PrimExpr offset = make_const(DataType::Int(64), 0);
  for (size_t i = 0; i < range.size(); ++i) {
    PrimExpr stride = make_const(DataType::Int(64), 1);
    for (size_t j = i + 1; j < buf->shape.size(); ++j) {
      stride = stride * buf->shape[j];
    }
    offset = offset + range[i]->min * stride;
  }
  return offset;
}

// Row-major element strides of an ND (GM/UB) buffer, derived from its ND
// (identity) affine layout's output shape: strides[k] = product(shape[k+1..]).
// Keeps copy parameters layout-driven for the ND side as well.
Array<PrimExpr> AscendNDStrides(const Buffer &buf) {
  Array<PrimExpr> shape = makeAscendNDLayout(buf)->OutputShape();
  int n = static_cast<int>(shape.size());
  Array<PrimExpr> strides(static_cast<size_t>(n),
                          make_const(DataType::Int(32), 1));
  PrimExpr cur = make_const(DataType::Int(32), 1);
  for (int i = n - 1; i >= 0; --i) {
    strides.Set(i, cur);
    cur = cur * shape[i];
  }
  return strides;
}

// Map float32 source (L0C) + destination dtype → quant_pre hardware mode.
static int GetCCQuantPre(DataType src_dtype, DataType dst_dtype) {
  if (src_dtype.is_float() && src_dtype.bits() == 32) {
    if (dst_dtype.is_bfloat16())
      return 16; // F322BF16
    if (dst_dtype.is_float16())
      return 1; // F322F16
    if (dst_dtype.is_float() && dst_dtype.bits() == 8)
      return 13; // QF322FP8_PRE
    if (dst_dtype.is_int() && dst_dtype.bits() == 8)
      return 9; // REQ8
    if (dst_dtype.is_int() && dst_dtype.bits() == 4)
      return 22; // REQ4
    if (dst_dtype.is_int() && dst_dtype.bits() == 16)
      return 28; // DEQS16
  }
  return 0; // NoQuant
}

Stmt LowerDMACopy(const CopyNode &op, const LowerArgs &T,
                  arith::Analyzer *analyzer, int dma_path) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  auto I64 = [](int64_t v) { return make_const(DataType::Int(64), v); };
  int elem_bits = op.src->dtype.bits();

  // OOB tail clamping (and the empty-tile has_data guard) is performed before
  // AutoSchedule by the AscendInsertOOBPadding pass, which rewrites this copy's
  // ranges in place. By the time lowering runs, op.src_range / op.dst_range are
  // already the in-bounds ranges, so no clamp is needed here.
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  size_t ndim = src_range.size();
  PrimExpr inner_elems = src_range[ndim - 1]->extent;
  PrimExpr n_rows = I(1);
  for (size_t i = 0; i + 1 < ndim; ++i) {
    n_rows = n_rows * src_range[i]->extent;
  }

  PrimExpr row_bytes = AscendMTEBytesFromElements(inner_elems, elem_bits);

  // GM/UB row stride from the ND (identity) layout's row-major strides: the
  // outermost stride is the bytes between consecutive rows.
  PrimExpr src_row_stride_bytes =
      AscendMTEBytesFromElements(AscendNDStrides(op.src)[0], elem_bits);

  auto make_access_ptr = [&](const Buffer &buf, PrimExpr offset,
                             PrimExpr extent, int rw_mask) -> PrimExpr {
    return Call(
        DataType::Handle(), builtin::tvm_access_ptr(),
        {TypeAnnotation(buf->dtype), buf->data, offset, extent, I(rw_mask)});
  };
  PrimExpr total_elems = inner_elems * n_rows;
  PrimExpr src_offset = ComputeFlatOffset(op.src, src_range);
  PrimExpr dst_offset = ComputeFlatOffset(op.dst, dst_range);
  PrimExpr src_ptr = make_access_ptr(op.src, src_offset, total_elems, 1);
  PrimExpr dst_ptr = make_access_ptr(op.dst, dst_offset, total_elems, 2);

  PrimExpr sid = I(0);
  PrimExpr zero = I(0);
  PrimExpr l2_cache_ctrl = I(op.GetL2CacheCtrl());

  PrimExpr total_bytes = row_bytes * n_rows;
  PrimExpr call;
  if (dma_path == 0 || dma_path == 1) {
    auto src_layout =
        StridedLayout::FromBufferRange(op.src, src_range, elem_bits);
    auto dst_layout =
        StridedLayout::FromBufferRange(op.dst, dst_range, elem_bits);
    auto plan = PlanMTECopy(src_layout, dst_layout, analyzer);

    PrimExpr plan_row_bytes =
        AscendMTEBytesFromElements(plan.row_elems, elem_bits);
    PrimExpr plan_src_stride_bytes =
        AscendMTEBytesFromElements(plan.src_stride_elems, elem_bits);
    PrimExpr plan_dst_stride_bytes =
        AscendMTEBytesFromElements(plan.dst_stride_elems, elem_bits);

    PrimExpr plan_total_elems = plan.row_elems * plan.n_rows;
    PrimExpr plan_src_ptr =
        make_access_ptr(op.src, plan.src_offset, plan_total_elems, 1);
    PrimExpr plan_dst_ptr =
        make_access_ptr(op.dst, plan.dst_offset, plan_total_elems, 2);
    PrimExpr plan_total_bytes = plan_row_bytes * plan.n_rows;

    // Ascend GM->UB padding (data_select): an unaligned row is right-padded up
    // to the next 32B boundary and the pad lanes are filled from the pad-value
    // register. The register must have been set by a leading
    // ascend_set_copy_pad_value op.
    auto as_int_plan = [](PrimExpr e) -> int64_t {
      if (auto *imm = e.as<IntImmNode>()) {
        return imm->value;
      }
      return -1;
    };
    int64_t rb_plan = as_int_plan(plan_row_bytes);
    bool data_select = dma_path == 0 && op.GetDataSelect() != 0;
    // Padding right-pads each row up to the next 32B boundary, so it needs a
    // statically-known row byte length to compute the pad count. A symbolic
    // row extent would silently skip padding and read garbage tail lanes.
    ICHECK(!data_select || rb_plan > 0)
        << "Ascend padded GM->UB copy (pad_value / data_select) requires a "
           "statically-known row byte length, got a symbolic row extent.";
    int64_t right_pad_elems = 0;
    if (data_select && rb_plan > 0 && (rb_plan % 32) != 0) {
      int64_t aligned_bytes = (rb_plan + 31) / 32 * 32;
      right_pad_elems = ((aligned_bytes - rb_plan) * 8) / elem_bits;
    }
    bool do_pad = right_pad_elems > 0;

    // Collapse into a single contiguous burst only when the region is truly
    // contiguous on both sides: one row, or each side's inter-row stride equals
    // the row byte length (consecutive rows back-to-back). A small row
    // (row_bytes < 32) is NOT sufficient: strided rows must still be issued as
    // separate bursts, otherwise rows 2+ are silently dropped. The align_v2 MTE
    // intrinsics support byte-granular bursts, so sub-32B rows are valid in the
    // multi-row path.
    bool plan_single_row =
        analyzer->CanProveEqual(plan.n_rows, I(1)) ||
        (analyzer->CanProveEqual(plan_src_stride_bytes, plan_row_bytes) &&
         analyzer->CanProveEqual(plan_dst_stride_bytes, plan_row_bytes));

    if (dma_path == 0) {
      if (do_pad) {
        // Padded copy: rightPadding fills the row tail from the pad-value
        // register (set_mov_pad_val). The destination stride is the physical
        // UB row stride (plan_dst_stride_bytes), NOT the padded row width: the
        // dst buffer may be over-allocated wider than align32(row), and rows
        // must land at their real stride to avoid clobbering each other.
        call =
            Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                 {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows, plan_row_bytes,
                  zero, I(right_pad_elems), I(1), l2_cache_ctrl,
                  plan_src_stride_bytes, plan_dst_stride_bytes});
      } else if (plan_single_row) {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, I(1), plan_total_bytes,
                     zero, zero, zero, l2_cache_ctrl, plan_total_bytes,
                     plan_total_bytes});
      } else {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows,
                     plan_row_bytes, zero, zero, zero, l2_cache_ctrl,
                     plan_src_stride_bytes, plan_dst_stride_bytes});
      }
    } else {
      constexpr int kStoreDefaultL2CacheCtrl = 4;
      PrimExpr store_l2_cache_ctrl =
          I(op.GetL2CacheCtrl(kStoreDefaultL2CacheCtrl));
      if (plan_single_row) {
        call = Call(DataType::Void(), ascend_copy_ubuf_to_gm(),
                    {plan_dst_ptr, plan_src_ptr, sid, I(1), plan_total_bytes,
                     store_l2_cache_ctrl, plan_total_bytes, plan_total_bytes});
      } else {
        call = Call(DataType::Void(), ascend_copy_ubuf_to_gm(),
                    {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows,
                     plan_row_bytes, store_l2_cache_ctrl, plan_dst_stride_bytes,
                     plan_src_stride_bytes});
      }
    }
  } else if (dma_path == 2) {
    AscendFractalLayoutInfo dst_layout;
    bool has_dst_layout = false;
    if (auto layout = FindLayoutForBuffer(T.layout_map, op.dst)) {
      has_dst_layout =
          TryExtractAscendFractalLayout(layout.value(), op.dst, &dst_layout);
      ICHECK(has_dst_layout)
          << "GM->L1 copy expects an Ascend fractal layout on the L1 dst "
          << op.dst->name;
    }
    bool needs_transpose = op.GetTranspose();
    PrimExpr dst_n_value;
    if (has_dst_layout) {
      if (dst_layout.kind == AscendFractalKind::kSF) {
        needs_transpose = !needs_transpose;
        dst_n_value = dst_layout.outer1;
      } else {
        dst_n_value = dst_layout.outer1 * dst_layout.row_frac;
      }
    } else {
      size_t dst_ndim = op.dst_range.size();
      dst_n_value = I(1);
      for (size_t i = 0; i + 1 < dst_ndim; ++i) {
        dst_n_value = dst_n_value * op.dst_range[i]->extent;
      }
    }

    size_t src_ndim = src_range.size();
    ICHECK_GE(src_ndim, 2U)
        << "Ascend GM->L1 copy requires at least a 2D source region for "
           "buffer "
        << op.src->name;
    for (size_t i = 0; i + 2 < src_ndim; ++i) {
      ICHECK(analyzer->CanProve(
          op.src_range[i]->min + op.src_range[i]->extent <= op.src->shape[i],
          arith::ProofStrength::kSymbolicBound))
          << "Ascend GM->L1 auto padding only supports OOB on the trailing "
             "two source dimensions, got source ranges "
          << op.src_range << " for buffer " << op.src->name << " shape "
          << op.src->shape;
    }

    PrimExpr valid_inner_elems = src_range[src_ndim - 1]->extent;
    PrimExpr valid_n_rows = I(1);
    for (size_t i = 0; i + 1 < src_ndim; ++i) {
      valid_n_rows = valid_n_rows * src_range[i]->extent;
    }

    PrimExpr transpose_flag = I(needs_transpose ? 1 : 0);
    // For dn2nz (transpose), SDK expects nValue=width(cols) and
    // dValue=height(rows), which is the reverse of nd2nz.
    PrimExpr n_value = needs_transpose ? valid_inner_elems : valid_n_rows;
    PrimExpr d_value = needs_transpose ? valid_n_rows : valid_inner_elems;
    std::string physical_dtype;
    if (has_dst_layout && dst_layout.kind == AscendFractalKind::kSF &&
        op.src->dtype.bits() == 8 && op.dst->dtype.bits() == 8) {
      physical_dtype = "uint16_t";
      n_value = n_value / I(2);
    }
    call = Call(DataType::Void(), ascend_copy_gm_to_cbuf(),
                {dst_ptr, src_ptr, sid, src_row_stride_bytes, l2_cache_ctrl,
                 n_value, d_value, zero, zero, transpose_flag, dst_n_value,
                 StringImm(physical_dtype)});
    // OOB clamping, the empty-tile has_data guard and the padding fill for a
    // padded GM->L1 copy are all handled before lowering by the
    // AscendInsertOOBPadding pass, so lowering just emits the DMA call.

  } else if (dma_path == 3 || dma_path == 4) {
    // M/K positions go through intrinsic args; leading dims (e.g. version)
    // through the pointer offset.  MakeAccessPtrFromRegion splits offset/extent
    // the same way: leading dims → offset, last two (M×K) → extent.
    PrimExpr src_base_ptr =
        MakeAccessPtrFromRegion(BufferRegion(op.src, op.src_range), 1, false);

    AscendFractalLayoutInfo src_info;
    AscendFractalLayoutInfo dst_info;
    Optional<Layout> src_layout = FindLayoutForBuffer(T.layout_map, op.src);
    Optional<Layout> dst_layout = FindLayoutForBuffer(T.layout_map, op.dst);
    bool have_src_layout =
        src_layout.defined() &&
        TryExtractAscendFractalLayout(src_layout.value(), op.src, &src_info);
    bool have_dst_layout =
        dst_layout.defined() &&
        TryExtractAscendFractalLayout(dst_layout.value(), op.dst, &dst_info);

    PrimExpr m_start, k_start, m_step, k_step, src_stride, dst_stride;

    if (have_src_layout && have_dst_layout) {
      AscendFractalRegionInfo src_region;
      ICHECK(TryExtractAscendFractalRegion(src_layout.value(), op.src_range,
                                           &src_region))
          << "Failed to map L1->L0 source region through layout for buffer "
          << op.src->name;
      m_start = src_region.outer1->min;
      k_start = src_region.outer0->min;
      m_step = src_region.outer1->extent;
      k_step = src_region.outer0->extent;
      src_stride = src_info.outer1;
      dst_stride = dst_info.outer1;
    } else {
      // Fallback: derive fractal params from logical shape (legacy path)
      PrimExpr c0 = I(16);
      PrimExpr c0_k = I(AscendC0(elem_bits));
      size_t src_range_ndim = op.src_range.size();
      PrimExpr m_dim =
          src_range_ndim >= 2 ? op.src_range[src_range_ndim - 2]->extent : I(1);
      PrimExpr k_dim =
          src_range_ndim >= 2 ? op.src_range[src_range_ndim - 1]->extent : I(1);
      m_start = src_range_ndim >= 2 ? op.src_range[src_range_ndim - 2]->min / c0
                                    : I(0);
      k_start = src_range_ndim >= 2
                    ? op.src_range[src_range_ndim - 1]->min / c0_k
                    : I(0);
      m_step = m_dim / c0;
      k_step = k_dim / c0_k;
      // src/dst stride from logical shape
      size_t sn = op.src->shape.size();
      size_t dn = op.dst->shape.size();
      src_stride = sn >= 2 ? op.src->shape[sn - 2] / c0 : I(1);
      dst_stride = dn >= 2 ? op.dst->shape[dn - 2] / c0 : I(1);
    }

    // The user may request an explicit transpose via T.copy(..,
    // transpose=True).  Additionally, when the L1 source and the L0
    // destination carry different fractal majors (e.g. a K-major L1 tile
    // feeding an MN-major L0 fractal, or vice versa), an implicit transpose is
    // required to reconcile the two layouts.  The effective transpose is the
    // XOR of the user request and the major mismatch: two transposes cancel.
    //
    // L1 tiles are canonically K-major (reduce-K on the C0 axis); when the L1
    // source carries no explicit layout we assume K-major so the mismatch is
    // driven purely by the inferred L0 major.
    bool user_transpose = op.GetTranspose();
    LogicalAxis src_c0_axis =
        have_src_layout ? src_info.c0_axis : LogicalAxis::kCol;
    LogicalAxis dst_c0_axis =
        have_dst_layout ? dst_info.c0_axis : LogicalAxis::kCol;
    bool major_mismatch = src_c0_axis != dst_c0_axis;
    bool needs_transpose = user_transpose ^ major_mismatch;

    auto intrinsic =
        dma_path == 3 ? ascend_load_cbuf_to_ca() : ascend_load_cbuf_to_cb();
    Array<PrimExpr> ld_args = {
        dst_ptr,    src_base_ptr, m_start,
        k_start,    m_step,       k_step,
        src_stride, dst_stride,   I(needs_transpose ? 1 : 0)};

    if (op.sf.defined()) {
      const Buffer &sf_buf = op.sf.value();
      PrimExpr sf_ptr =
          MakeAccessPtrFromRegion(BufferRegion(sf_buf, op.sf_range), 1, false);

      AscendFractalLayoutInfo sf_info;
      Optional<Layout> sf_layout = FindLayoutForBuffer(T.layout_map, sf_buf);
      bool have_sf_layout =
          sf_layout.defined() &&
          TryExtractAscendFractalLayout(sf_layout.value(), sf_buf, &sf_info) &&
          sf_info.kind == AscendFractalKind::kSF;

      PrimExpr sf_m_start, sf_m_step, sf_k_start, sf_k_step, sf_src_stride,
          sf_dst_stride;

      if (have_sf_layout) {
        AscendFractalRegionInfo sf_region;
        ICHECK(TryExtractAscendFractalRegion(sf_layout.value(), op.sf_range,
                                             &sf_region))
            << "Failed to map L1->L0 SF region through layout for buffer "
            << sf_buf->name;
        // SF_K layout maps trailing [M, K-pairs] to physical
        // [M/16, K-pairs/pack, M%16, K-pairs%pack].  The MX load intrinsic
        // expects x=M/16 and y=K-pair/pack positions.
        sf_m_start = sf_region.outer0->min;
        sf_k_start = sf_region.outer1->min;
        sf_m_step = sf_region.outer0->extent;
        sf_k_step = sf_region.outer1->extent;
        sf_src_stride = sf_info.outer1;
        sf_dst_stride = sf_region.outer1->extent;
      } else {
        const int sf_range_ndim = static_cast<int>(op.sf_range.size());
        const int sf_shape_ndim = static_cast<int>(sf_buf->shape.size());
        // SF buffer layout is (..., K-pairs, M): K-pairs is the second-to-last
        // dim, M is the last. All positions are derived from the SF region
        // itself (independent of the data copy), and the M/K start positions
        // flow through the load_cbuf_to_ca_mx intrinsic parameters rather than
        // the pointer.
        const int sf_k_dim = sf_range_ndim >= 2 ? sf_range_ndim - 2 : -1;
        const int sf_m_dim = sf_range_ndim - 1;

        // SF pointer: M/K positions go through intrinsic args; leading dims
        // (e.g. pipeline version) go through the pointer offset.
        sf_ptr = MakeAccessPtrFromRegion(BufferRegion(sf_buf, op.sf_range), 1,
                                         false);
        sf_m_start = op.sf_range[sf_m_dim]->min / I(16);
        sf_m_step = op.sf_range[sf_m_dim]->extent / I(16);
        sf_k_start = sf_k_dim >= 0 ? op.sf_range[sf_k_dim]->min : I(0);
        sf_k_step = sf_k_dim >= 0 ? op.sf_range[sf_k_dim]->extent : I(1);
        sf_src_stride =
            sf_shape_ndim >= 2 ? sf_buf->shape[sf_shape_ndim - 2] : I(1);
        sf_dst_stride = sf_k_dim >= 0 ? op.sf_range[sf_k_dim]->extent : I(1);
      }
      ld_args.push_back(sf_ptr);        // [9]  sf_ptr
      ld_args.push_back(sf_m_start);    // [10] sf_x_start
      ld_args.push_back(sf_k_start);    // [11] sf_y_start
      ld_args.push_back(sf_m_step);     // [12] sf_x_step
      ld_args.push_back(sf_k_step);     // [13] sf_y_step
      ld_args.push_back(sf_src_stride); // [14] sf_src_stride
      ld_args.push_back(sf_dst_stride); // [15] sf_dst_stride
    }
    call = Call(DataType::Void(), intrinsic, ld_args);
  } else if (dma_path == 5) {
    AscendFractalLayoutInfo info;
    PrimExpr loop_src_stride = n_rows;

    if (T.layout_map.count(op.src)) {
      ICHECK(TryExtractAscendFractalLayout(T.layout_map[op.src], op.src, &info))
          << "L0C->UB copy expects an Ascend fractal layout on the L0C src "
          << op.src->name;
      loop_src_stride = info.outer1 * info.row_frac;
    }

    PrimExpr dual_dst_ctl_val = I(op.GetDualDstCtl());
    PrimExpr unit_flag_ctl_val = I(op.GetUnitFlagCtl());
    call = Call(DataType::Void(), ascend_copy_matrix_cc_to_ub(),
                {dst_ptr,
                 src_ptr,
                 sid,
                 inner_elems,
                 n_rows,
                 inner_elems,
                 loop_src_stride,
                 dual_dst_ctl_val,
                 I(op.GetSubBlockId()), // sub_blockid
                 zero,                  // clip_relu_pre
                 unit_flag_ctl_val,     // unit_flag_ctl
                 I(GetCCQuantPre(op.src->dtype, op.dst->dtype)), // quant_pre
                 zero,                                           // relu_pre
                 zero,                                           // split_en
                 I(1),                                           // NZ2ND_en
                 zero,                                           // quant_post
                 zero,                                           // relu_post
                 zero,   // clip_relu_post
                 zero,   // loop_enhance_en
                 zero,   // eltwise_op
                 zero,   // eltwise_antq_en
                 zero,   // loop_enhance_merge_en
                 zero,   // C0_pad_en
                 zero,   // wino_post_en
                 zero,   // broadcast_en
                 zero}); // NZ2DN_en
  } else if (dma_path == 6) {
    ICHECK(!op.GetNd2Nz())
        << "Ascend nd2nz UB->L1 copy must be emitted directly from the Python "
           "frontend (tilelang/language/copy_op.py); it should never reach "
           "tl.tileop.copy lowering. This indicates the Python bypass was not "
           "taken; check the call site.";
    PrimExpr n_burst_val = total_bytes / I(32);
    call = Call(DataType::Void(), ascend_copy_ubuf_to_cbuf(),
                {dst_ptr, src_ptr, zero, n_burst_val, I(1), zero, zero});
  } else if (dma_path == 7) {
    AscendFractalLayoutInfo info;
    PrimExpr loop_src_stride = n_rows;

    if (T.layout_map.count(op.src)) {
      ICHECK(TryExtractAscendFractalLayout(T.layout_map[op.src], op.src, &info))
          << "L0C->GM copy expects an Ascend fractal layout on the L0C src "
          << op.src->name;
      loop_src_stride = info.outer1 * info.row_frac;
    }
    // GM dst last-dim from its ND (identity) layout output shape.
    Array<PrimExpr> dst_nd_shape = makeAscendNDLayout(op.dst)->OutputShape();
    PrimExpr dst_last_dim = dst_nd_shape[dst_nd_shape.size() - 1];
    call =
        Call(DataType::Void(), ascend_copy_matrix_cc_to_gm(),
             {dst_ptr,
              src_ptr,
              sid,
              inner_elems,
              n_rows,
              dst_last_dim,
              loop_src_stride,
              I(op.GetL2CacheCtrl()), // [7] l2_cache_ctl
              zero,                   // [8] clip_relu_pre
              I(op.GetUnitFlagCtl()), // [9] unit_flag_ctl
              I(GetCCQuantPre(op.src->dtype, op.dst->dtype)), // [10] quant_pre
              zero,                                           // [11] relu_pre
              zero,                                           // [12] split_en
              I(1),                                           // [13] NZ2ND_en
              zero,                                           // [14] quant_post
              zero,                                           // [15] relu_post
              zero,   // [16] clip_relu_post
              zero,   // [17] loop_enhance_en
              zero,   // [18] eltwise_op
              zero,   // [19] eltwise_antq_en
              zero,   // [20] loop_enhance_merge_en
              zero,   // [21] C0_pad_en
              zero,   // [22] wino_post_en
              zero,   // [23] broadcast_en
              zero}); // [24] NZ2DN_en
  } else {
    LOG(FATAL) << "Unknown Ascend DMA path: " << dma_path;
  }

  // The empty-tile has_data guard is applied by AscendInsertOOBPadding (it
  // wraps this copy in IfThenElse before lowering), so emit the DMA call
  // directly.
  return Evaluate(call);
}

Stmt LowerAscendNormalCopy(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer) {
  if (!IsInsideSimtVF(T)) {
    return LowerNormalCopy(op, T, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map, analyzer,
                         false, T.buffer_remap, T.bind_var_to_expr},
                        level);
  }

  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                           analyzer, T.layout_map,
                           par_op->GetPredicate(T.thread_var),
                           /*parallel_loop=*/true, /*should_vectorize=*/true,
                           par_op->LoopLayoutRequiresPaddingGuard());
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    if (IsDMACopy(op)) {
      Map<Buffer, Layout> result_map;
      PrimExpr thread_extent = T.thread_bounds->extent;
      for (const auto &range : op.src_range) {
        CollectFragmentLayouts(range->min, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
        CollectFragmentLayouts(range->extent, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
      }
      for (const auto &range : op.dst_range) {
        CollectFragmentLayouts(range->min, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
        CollectFragmentLayouts(range->extent, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
      }

      if (op.sf.defined()) {
        const Buffer &sf_buf = op.sf.value();
        result_map.Set(sf_buf, MakeAscendSFLayout(sf_buf));
      }

      int dma_path = GetDMAPath(op.src, op.dst);
      if (dma_path == 2 && level == InferLevel::kFree &&
          !FindLayoutForBuffer(T.layout_map, op.dst).defined()) {
        // A standalone GM->L1 copy has no downstream L0/GEMM consumer to
        // anchor its layout. Preserve the canonical legacy K-col NZ layout as
        // a free-mode fallback.
        result_map.Set(op.dst, MakeAscendMajorKLayout(op.dst));
      } else if (dma_path == 3 || dma_path == 4) {
        int k_align = op.sf.defined() ? 64 : 1;
        Optional<Layout> existing_src_layout =
            FindLayoutForBuffer(T.layout_map, op.src);
        Layout src_layout = existing_src_layout.defined()
                                ? existing_src_layout.value()
                                : MakeAscendMajorKLayout(op.src);
        if (!existing_src_layout.defined()) {
          result_map.Set(op.src, src_layout);
        }
        AscendFractalLayoutInfo src_info;
        ICHECK(TryExtractAscendFractalLayout(src_layout, op.src, &src_info))
            << "Ascend L1->L0 copy expects a fractal layout on source L1 "
            << op.src->name;

        // The effective transpose is the XOR of the user's explicit request
        // and the implicit transpose caused by a K-major L1 source feeding an
        // MN-major L0 destination (or vice versa): two transposes cancel.
        // Mirror the logic in LowerDMACopy so the alignment guard covers both
        // explicit and implicit transposes.
        bool effective_transpose = op.GetTranspose();
        {
          Optional<Layout> dst_layout =
              FindLayoutForBuffer(T.layout_map, op.dst);
          AscendFractalLayoutInfo dst_info;
          if (dst_layout.defined() &&
              TryExtractAscendFractalLayout(dst_layout.value(), op.dst,
                                            &dst_info)) {
            bool major_mismatch = src_info.c0_axis != dst_info.c0_axis;
            effective_transpose = op.GetTranspose() ^ major_mismatch;
          }
        }

        // Transpose swaps the semantic roles of the physical C0 and row16
        // axes. Keep the stored layout physical, and use this operation-local
        // view to identify K and MN.
        LogicalAxis effective_c0_axis = src_info.c0_axis;
        LogicalAxis effective_row16_axis = src_info.row16_axis;
        if (effective_transpose) {
          std::swap(effective_c0_axis, effective_row16_axis);
        }
        auto axis_extent = [](const Buffer &buffer, LogicalAxis axis) {
          size_t ndim = buffer->shape.size();
          return axis == LogicalAxis::kRow ? buffer->shape[ndim - 2]
                                           : buffer->shape[ndim - 1];
        };
        auto check_alignment = [&](const PrimExpr &extent, int alignment,
                                   const std::string &what) {
          PrimExpr divisor = make_const(extent.dtype(), alignment);
          PrimExpr zero = make_const(extent.dtype(), 0);
          ICHECK(T.analyzer->CanProveEqual(FloorMod(extent, divisor), zero))
              << what << " must be divisible by " << alignment << ", got "
              << extent;
        };

        if (k_align > 1) {
          check_alignment(axis_extent(op.src, effective_c0_axis), k_align,
                          "Ascend blockscaled source L1 K allocation extent");
        }
        if (effective_transpose) {
          // After the L1->L0 transpose the source MN axis becomes the L0 C0
          // axis (must cover complete C0 groups) and the source K axis becomes
          // the L0 row16 axis (must cover complete 16-row groups).  With the
          // axes already swapped above, effective_c0_axis points at the source
          // MN axis and effective_row16_axis at the source K axis.
          int c0 = AscendC0(op.src->dtype.bits());
          int row_frac = 16;
          check_alignment(axis_extent(op.src, effective_c0_axis), c0,
                          "Ascend transposed source L1 MN allocation extent");
          check_alignment(axis_extent(op.src, effective_row16_axis), row_frac,
                          "Ascend transposed source L1 K allocation extent");
        }
      }

      // Layout constraints for Ascend Cube-side data buffers are now attached
      // at allocation / explicit annotation sites (e.g. alloc_l1(...,
      // major=...), alloc_l0a/b default MajorK, and GemmMAD for L0C).  Copy
      // inference should not guess or overwrite those data layouts based solely
      // on DMA path. It only contributes fragment layouts discovered in index
      // expressions above, plus SF_K layout for explicit MX scale operands.
      return result_map;
    }

    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (!IsInsideSimtVF(T) && !IsLocalBuffer(op.src) &&
        !IsLocalBuffer(op.dst)) {
      int dma_path = GetDMAPath(op.src, op.dst);
      if (dma_path >= 0) {
        return LowerDMACopy(op, T, analyzer, dma_path);
      }
    }

    return LowerAscendNormalCopy(op, T, analyzer);
  }
};

} // namespace ascend

namespace {

bool MatchAscendCopyTarget(Target target) { return TargetIsAscend(target); }

bool RegisterAscendCopy() {
  RegisterCopyImpl(CopyImpl{
      "ascend.Copy",
      MatchAscendCopyTarget,
      200,
      ascend::Copy::InferLayout,
      ascend::Copy::Lower,
  });
  return true;
}

const bool ascend_copy_registered = RegisterAscendCopy();

} // namespace

} // namespace tl
} // namespace tvm
