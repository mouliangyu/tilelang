/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file rewrite_dual_copy.cc
 * \brief Lower Ascend dual-copy operations using the enclosing cthread sid.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "op/copy.h"
#include "op/region.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::Map;
using ffi::ObjectRef;
using ffi::Optional;
using ffi::String;

namespace {

Call RewriteRegion(const Call &region_call, int split_axis,
                   const PrimExpr &offset, const PrimExpr &split_extent) {
  RegionOp region(region_call->args);
  Array<Range> ranges = region->GetRanges();
  ICHECK_GE(split_axis, 0);
  ICHECK_LT(split_axis, static_cast<int>(ranges.size()));

  Array<PrimExpr> indices;
  indices.reserve(ranges.size());
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    PrimExpr index = ranges[dimension]->min;
    if (static_cast<int>(dimension) == split_axis)
      index = index + offset;
    indices.push_back(index);
  }

  Array<PrimExpr> arguments = {BufferLoad(region->GetBuffer(), indices),
                               Integer(region->GetAccessMask())};
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    arguments.push_back(static_cast<int>(dimension) == split_axis
                            ? split_extent
                            : ranges[dimension]->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), arguments);
}

Map<String, ObjectRef>
StripDualCopyAnnotations(const Map<String, ObjectRef> &annotations,
                         bool strip_double) {
  Map<String, ObjectRef> result;
  for (const auto &[key, value] : annotations) {
    if (key == "dual_dst_ctl" || key == "unit_flag_ctl" ||
        (strip_double && key == "double"))
      continue;
    result.Set(key, value);
  }
  return result;
}

class DualCopyRewriter : public StmtMutator {
private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent)
      return StmtMutator::VisitStmt_(op);
    const auto *iter_var = op->node.as<IterVarNode>();
    if (iter_var == nullptr || iter_var->thread_tag != "cthread")
      return StmtMutator::VisitStmt_(op);

    Optional<Var> outer_sid = sid_;
    sid_ = iter_var->var;
    Stmt body = VisitStmt(op->body);
    sid_ = std::move(outer_sid);
    if (body.same_as(op->body))
      return ffi::GetRef<Stmt>(op);
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt rewritten = StmtMutator::VisitStmt_(op);
    const auto *evaluate = rewritten.as<EvaluateNode>();
    ICHECK(evaluate != nullptr);
    const auto *call_node = evaluate->value.as<CallNode>();
    if (call_node == nullptr)
      return rewritten;
    Call call = ffi::GetRef<Call>(call_node);

    static const Op &copy_op = Op::Get("tl.tileop.copy");
    static const Op &post_copy_op = Op::Get("tl.ascend_nd2nz_post_copy");
    static const Op &access_ptr_op = Op::Get("tl.access_ptr");
    if (call->op.same_as(post_copy_op) && call->args.size() == 8) {
      Var sid = RequireSid(call);
      const auto *split = call->args[6].as<IntImmNode>();
      ICHECK(split) << "ascend_nd2nz_post_copy dual split_dim must be an "
                       "IntImm (0=M, 1=N along trailing-2 axes)";
      ICHECK(split->value == 0 || split->value == 1)
          << "ascend_nd2nz_post_copy dual split_dim must be 0 or 1, got "
          << split->value;
      PrimExpr half_extent = call->args[7];
      const auto *destination_ptr = call->args[0].as<CallNode>();
      ICHECK(destination_ptr && destination_ptr->op.same_as(access_ptr_op))
          << "ascend_nd2nz_post_copy dst arg must be a tl.access_ptr Call; got "
          << call->args[0];
      ICHECK_EQ(destination_ptr->args.size(), 3);
      const auto *destination_load =
          destination_ptr->args[0].as<BufferLoadNode>();
      ICHECK(destination_load)
          << "tl.access_ptr first arg must be a BufferLoad";
      int ndim = static_cast<int>(destination_load->indices.size());
      ICHECK_GE(ndim, 2)
          << "nd2nz dst BufferLoad must have at least 2 indices, got " << ndim;
      int split_axis = ndim - 2 + split->value;
      Array<PrimExpr> indices = destination_load->indices;
      indices.Set(split_axis,
                  indices[split_axis] + sid * std::move(half_extent));
      Call new_destination_ptr(destination_ptr->dtype, access_ptr_op,
                               {BufferLoad(destination_load->buffer, indices),
                                destination_ptr->args[1],
                                destination_ptr->args[2]});
      Array<PrimExpr> arguments = {new_destination_ptr};
      for (size_t index = 1; index < 6; ++index)
        arguments.push_back(call->args[index]);
      return Evaluate(Call(call->dtype, post_copy_op, arguments));
    }

    if (!call->op.same_as(copy_op))
      return rewritten;
    Copy copy(call->args, call->annotations);
    if (copy->annotations.Get("double")) {
      // Fixpipe runs once on AIC and implements L0C-to-UB dual destination in
      // hardware. Preserve dual_dst_ctl; only AIV-side copies need sid-based
      // software expansion.
      if (IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst))
        return rewritten;
      Var sid = RequireSid(call);
      int dual_control = copy->GetDualDstCtl();
      int source_axis = static_cast<int>(copy->src_range.size()) - 2 +
                        (dual_control == 1 ? 0 : 1);
      int destination_axis = static_cast<int>(copy->dst_range.size()) - 2 +
                             (dual_control == 1 ? 0 : 1);
      PrimExpr split_extent = copy->src_range[source_axis]->extent;
      Call destination =
          RewriteRegion(Downcast<Call>(call->args[1]), destination_axis,
                        sid * split_extent, split_extent);
      return Evaluate(Call(call->dtype, copy_op,
                           {Downcast<Call>(call->args[0]), destination},
                           StripDualCopyAnnotations(call->annotations, true)));
    }

    if (!IsGlobalBuffer(copy->src) || !copy->annotations.Get("dual_dst_ctl"))
      return rewritten;
    Var sid = RequireSid(call);
    int dual_control = copy->GetDualDstCtl();
    int source_axis = static_cast<int>(copy->src_range.size()) - 2 +
                      (dual_control == 1 ? 0 : 1);
    PrimExpr split_extent =
        floordiv(copy->src_range[source_axis]->extent, Integer(2));
    Call source = RewriteRegion(Downcast<Call>(call->args[0]), source_axis,
                                sid * split_extent, split_extent);
    return Evaluate(Call(call->dtype, copy_op,
                         {source, Downcast<Call>(call->args[1])},
                         StripDualCopyAnnotations(call->annotations, false)));
  }

  Var RequireSid(const Call &call) const {
    TVM_FFI_CHECK(sid_.has_value(), ValueError)
        << "RewriteDualCopy found a dual-copy operation outside a cthread "
           "scope. Run CorePartition and NormalizeMixedKernelSid before "
           "RewriteDualCopy. Operation: "
        << call;
    return sid_.value();
  }

  Optional<Var> sid_;
};

} // namespace

tvm::transform::Pass RewriteDualCopy() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    func.CopyOnWrite()->body = DualCopyRewriter()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.RewriteDualCopy",
                                             {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteDualCopy", RewriteDualCopy);
}

} // namespace tl
} // namespace tvm
