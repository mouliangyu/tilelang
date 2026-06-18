/*!
 * \file tl/ascend/op/copy.cc
 * \brief Ascend implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "ascend_mte_plan.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

namespace {

int GetDMAPath(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return 0;
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return 1;
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return 2;
  if (IsL1Buffer(src) && IsL0ABuffer(dst))
    return 3;
  if (IsL1Buffer(src) && IsL0BBuffer(dst))
    return 4;
  if (IsL0CBuffer(src) && IsSharedBuffer(dst))
    return 5;
  if (IsSharedBuffer(src) && IsL1Buffer(dst))
    return 6;
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return 7;
  return -1;
}

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

Stmt LowerDMACopy(const CopyNode &op, const LowerArgs &T,
                  arith::Analyzer *analyzer, int dma_path) {
  (void)T;
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  auto I64 = [](int64_t v) { return make_const(DataType::Int(64), v); };
  int elem_bytes = op.src->dtype.bytes();

  size_t ndim = op.src_range.size();
  PrimExpr inner_elems = op.src_range[ndim - 1]->extent;
  PrimExpr n_rows = I(1);
  for (size_t i = 0; i + 1 < ndim; ++i) {
    n_rows = n_rows * op.src_range[i]->extent;
  }

  PrimExpr row_bytes = inner_elems * I(elem_bytes);

  PrimExpr src_row_stride_bytes = I(elem_bytes);
  for (size_t j = 1; j < op.src->shape.size(); ++j) {
    src_row_stride_bytes = src_row_stride_bytes * op.src->shape[j];
  }

  auto make_access_ptr = [&](const Buffer &buf, PrimExpr offset,
                             PrimExpr extent, int rw_mask) -> PrimExpr {
    return Call(
        DataType::Handle(), builtin::tvm_access_ptr(),
        {TypeAnnotation(buf->dtype), buf->data, offset, extent, I(rw_mask)});
  };

  PrimExpr total_elems = inner_elems * n_rows;
  PrimExpr src_offset = ComputeFlatOffset(op.src, op.src_range);
  PrimExpr dst_offset = ComputeFlatOffset(op.dst, op.dst_range);
  PrimExpr src_ptr = make_access_ptr(op.src, src_offset, total_elems, 1);
  PrimExpr dst_ptr = make_access_ptr(op.dst, dst_offset, total_elems, 2);

  PrimExpr sid = I(0);
  PrimExpr zero = I(0);
  PrimExpr l2_cache_ctrl = I(op.GetL2CacheCtrl());

  PrimExpr total_bytes = row_bytes * n_rows;
  PrimExpr call;
  if (dma_path == 0 || dma_path == 1) {
    auto src_layout =
        StridedLayout::FromBufferRange(op.src, op.src_range, elem_bytes);
    auto dst_layout =
        StridedLayout::FromBufferRange(op.dst, op.dst_range, elem_bytes);
    auto plan = PlanMTECopy(src_layout, dst_layout, analyzer);

    PrimExpr plan_total_elems = plan.row_bytes / I(elem_bytes) * plan.n_rows;
    PrimExpr plan_src_ptr =
        make_access_ptr(op.src, plan.src_offset, plan_total_elems, 1);
    PrimExpr plan_dst_ptr =
        make_access_ptr(op.dst, plan.dst_offset, plan_total_elems, 2);
    PrimExpr plan_total_bytes = plan.row_bytes * plan.n_rows;

    auto as_int_plan = [](PrimExpr e) -> int64_t {
      if (auto *imm = e.as<IntImmNode>()) {
        return imm->value;
      }
      return -1;
    };
    int64_t nr_plan = as_int_plan(plan.n_rows);
    int64_t rb_plan = as_int_plan(plan.row_bytes);
    bool plan_single_row = nr_plan == 1 || (rb_plan >= 0 && rb_plan < 32);
    if (!plan_single_row) {
      ICHECK(rb_plan > 0 && rb_plan % 32 == 0)
          << "Ascend MTE multi-row copy requires row_bytes to be a multiple "
             "of 32B, but got row_bytes="
          << rb_plan;
    }

    if (dma_path == 0) {
      if (plan_single_row) {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, I(1), plan_total_bytes,
                     zero, zero, zero, l2_cache_ctrl, plan_total_bytes,
                     plan_total_bytes});
      } else {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows,
                     plan.row_bytes, zero, zero, zero, l2_cache_ctrl,
                     plan.src_stride_bytes, plan.dst_stride_bytes});
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
                     plan.row_bytes, store_l2_cache_ctrl, plan.dst_stride_bytes,
                     plan.src_stride_bytes});
      }
    }
  } else if (dma_path == 2) {
    PrimExpr transpose_flag = I(op.GetTranspose());
    size_t dst_ndim = op.dst_range.size();
    PrimExpr dst_n_rows = I(1);
    for (size_t i = 0; i + 1 < dst_ndim; ++i) {
      dst_n_rows = dst_n_rows * op.dst_range[i]->extent;
    }
    // For dn2nz (transpose), SDK expects nValue=width(cols) and
    // dValue=height(rows), which is the reverse of nd2nz.
    PrimExpr n_value = op.GetTranspose() ? inner_elems : n_rows;
    PrimExpr d_value = op.GetTranspose() ? n_rows : inner_elems;
    PrimExpr dst_n_value = dst_n_rows;
    call = Call(DataType::Void(), ascend_copy_gm_to_cbuf(),
                {dst_ptr, src_ptr, sid, src_row_stride_bytes, l2_cache_ctrl,
                 n_value, d_value, zero, zero, transpose_flag, dst_n_value});
  } else if (dma_path == 3 || dma_path == 4) {
    PrimExpr src_base_offset = make_const(DataType::Int(64), 0);
    for (size_t i = 0; i + 1 < op.src_range.size(); ++i) {
      PrimExpr stride = make_const(DataType::Int(64), 1);
      for (size_t j = i + 1; j < op.src->shape.size(); ++j) {
        stride = stride * op.src->shape[j];
      }
      src_base_offset = src_base_offset + op.src_range[i]->min * stride;
    }
    PrimExpr src_base_ptr =
        make_access_ptr(op.src, src_base_offset, total_elems, 1);

    PrimExpr c0 = I(16);
    size_t dst_ndim = op.dst_range.size();
    PrimExpr m_dim = dst_ndim >= 2 ? op.dst_range[dst_ndim - 2]->extent : I(1);
    PrimExpr k_dim = op.dst_range[dst_ndim - 1]->extent;
    PrimExpr k_start = op.src_range[op.src_range.size() - 1]->min;
    PrimExpr m_step = m_dim / c0;
    PrimExpr k_step = k_dim / c0;
    auto intrinsic =
        dma_path == 3 ? ascend_load_cbuf_to_ca() : ascend_load_cbuf_to_cb();
    call = Call(DataType::Void(), intrinsic,
                {dst_ptr, src_base_ptr, zero, k_start / c0, m_step, k_step,
                 m_step, m_step, zero});
  } else if (dma_path == 5) {
    PrimExpr dual_dst_ctl_val = I(op.GetDualDstCtl());
    PrimExpr unit_flag_ctl_val = I(op.GetUnitFlagCtl());
    call = Call(DataType::Void(), ascend_copy_matrix_cc_to_ub(),
                {dst_ptr,
                 src_ptr,
                 sid,
                 inner_elems,
                 n_rows,
                 inner_elems,
                 n_rows,
                 dual_dst_ctl_val,
                 zero,              // sub_blockid
                 zero,              // clip_relu_pre
                 unit_flag_ctl_val, // unit_flag_ctl
                 zero,              // quant_pre
                 zero,              // relu_pre
                 zero,              // split_en
                 I(1),              // NZ2ND_en
                 zero,              // quant_post
                 zero,              // relu_post
                 zero,              // clip_relu_post
                 zero,              // loop_enhance_en
                 zero,              // eltwise_op
                 zero,              // eltwise_antq_en
                 zero,              // loop_enhance_merge_en
                 zero,              // C0_pad_en
                 zero,              // wino_post_en
                 zero,              // broadcast_en
                 zero});            // NZ2DN_en
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
    PrimExpr dst_last_dim = op.dst->shape[op.dst->shape.size() - 1];
    call = Call(DataType::Void(), ascend_copy_matrix_cc_to_gm(),
                {dst_ptr,
                 src_ptr,
                 sid,
                 inner_elems,
                 n_rows,
                 dst_last_dim,
                 n_rows,
                 I(op.GetL2CacheCtrl()), // [7] l2_cache_ctl
                 zero,                   // [8] clip_relu_pre
                 I(op.GetUnitFlagCtl()), // [9] unit_flag_ctl
                 zero,                   // [10] quant_pre
                 zero,                   // [11] relu_pre
                 zero,                   // [12] split_en
                 I(1),                   // [13] NZ2ND_en
                 zero,                   // [14] quant_post
                 zero,                   // [15] relu_post
                 zero,                   // [16] clip_relu_post
                 zero,                   // [17] loop_enhance_en
                 zero,                   // [18] eltwise_op
                 zero,                   // [19] eltwise_antq_en
                 zero,                   // [20] loop_enhance_merge_en
                 zero,                   // [21] C0_pad_en
                 zero,                   // [22] wino_post_en
                 zero,                   // [23] broadcast_en
                 zero});                 // [24] NZ2DN_en
  } else {
    LOG(FATAL) << "Unknown Ascend DMA path: " << dma_path;
  }

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
