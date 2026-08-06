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
 * \file normalize_task.cc
 * \brief Mark canonical AutoSchedule task boundaries and hardware pipes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

#include "ascend/transform/auto_schedule/task_analysis.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using ffi::GetRef;

namespace {

constexpr uint16_t kSpecializedPipes =
    static_cast<uint16_t>(ResourcePipe::kMTE1) |
    static_cast<uint16_t>(ResourcePipe::kMTE2) |
    static_cast<uint16_t>(ResourcePipe::kMTE3) |
    static_cast<uint16_t>(ResourcePipe::kCube) |
    static_cast<uint16_t>(ResourcePipe::kVector) |
    static_cast<uint16_t>(ResourcePipe::kFixpipe);

uint16_t ResolvePipe(uint16_t mask) {
  if ((mask & kSpecializedPipes) == 0)
    return mask | static_cast<uint16_t>(ResourcePipe::kScalar);
  return mask;
}

uint16_t GetAscendCopyPipeMask(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  if (IsL1Buffer(src) && IsL0ABuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE1);
  if (IsL1Buffer(src) && IsL0BBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE1);
  if (IsL0CBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kFixpipe);
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kFixpipe);
  if (IsSharedBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  return static_cast<uint16_t>(ResourcePipe::kMTE2);
}

uint16_t GetAscendCallPipeMask(const Call &call) {
  static const auto copy_op = Op::Get("tl.tileop.copy");
  static const auto gemm_op = Op::Get("tl.tileop.gemm");
  if (call->op.same_as(copy_op)) {
    Copy copy(call->args, call->annotations);
    return GetAscendCopyPipeMask(copy->src, copy->dst);
  }
  if (call->op.same_as(tl::ascend_nd2nz_scatter()))
    return static_cast<uint16_t>(ResourcePipe::kVector);
  if (call->op.same_as(tl::ascend_nd2nz_post_copy()))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  if (call->op.same_as(tl::ascend_fill_l1()))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (call->op.same_as(gemm_op))
    return static_cast<uint16_t>(ResourcePipe::kCube);
  if (call->op.same_as(builtin::call_extern()) && !call->args.empty()) {
    if (const auto *sym = call->args[0].as<StringImmNode>()) {
      const std::string &name = sym->value;
      if (name == "debug_print_msg" || name == "debug_print_var" ||
          name == "debug_print_buffer_value")
        return static_cast<uint16_t>(ResourcePipe::kScalar);
      if (name == "debug_print_buffer_dump" && call->args.size() > 3) {
        if (const auto *scope_call = call->args[3].as<CallNode>()) {
          if (!scope_call->args.empty()) {
            if (const auto *load = scope_call->args[0].as<BufferLoadNode>()) {
              String scope = load->buffer.scope();
              return static_cast<uint16_t>(scope == "shared" ||
                                                   scope == "shared.dyn"
                                               ? ResourcePipe::kVector
                                               : ResourcePipe::kCube);
            }
          }
        }
      }
      if (name == "debug_print_global_buffer_dump")
        return static_cast<uint16_t>(ResourcePipe::kMTE3);
    }
  }
  return 0;
}

class PipeAnalyzer : public StmtExprVisitor {
public:
  static uint16_t Analyze(const Stmt &stmt) {
    PipeAnalyzer analyzer;
    analyzer(stmt);
    return ResolvePipe(analyzer.pipe_mask_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    pipe_mask_ |= GetAscendCallPipeMask(GetRef<Call>(op));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF")
      pipe_mask_ |= static_cast<uint16_t>(ResourcePipe::kVector);
  }

  uint16_t pipe_mask_{0};
};

} // namespace

uint16_t AnalyzeTaskPipe(const Stmt &stmt) {
  return PipeAnalyzer::Analyze(stmt);
}

namespace {

bool IsTaskBoundary(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>())
    return loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled;
  return stmt.as<EvaluateNode>() || stmt.as<BufferStoreNode>() ||
         stmt.as<BindNode>() || stmt.as<WhileNode>() || stmt.as<SBlockNode>();
}

class TaskNormalizer : public StmtMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    TaskNormalizer normalizer;
    func.CopyOnWrite()->body = normalizer(func->body);
    return func;
  }

private:
  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tl::attr::kAscendTask) {
        ++task_marker_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --task_marker_depth_;
        const auto *normalized = result.as<AttrStmtNode>();
        ICHECK(normalized && normalized->attr_key == tl::attr::kAscendTask);
        return AttrStmt(
            MakeTaskPipeAnnotation(AnalyzeTaskPipe(normalized->body),
                                   normalized->node),
            normalized->attr_key, normalized->value, normalized->body,
            normalized->span);
      }
      if (attr->attr_key == tl::attr::kAscendPerCoreTask) {
        TVM_FFI_CHECK_EQ(task_marker_depth_, 0, ValueError)
            << "T.PerCoreTask cannot be nested inside T.Task. Place T.Task "
               "candidate regions inside T.PerCoreTask instead.";
        ++per_core_task_depth_;
        Stmt body = VisitStmt(attr->body);
        --per_core_task_depth_;
        return AttrStmt(
            MakeTaskPipeAnnotation(AnalyzeTaskPipe(body), attr->node),
            attr->attr_key, attr->value, std::move(body), attr->span);
      }
    }

    if (schedule_body_depth_ == 0 || task_marker_depth_ > 0)
      return StmtMutator::VisitStmt(stmt);

    if (!IsTaskBoundary(stmt))
      return StmtMutator::VisitStmt(stmt);

    if (per_core_task_depth_ > 0 && stmt.as<BindNode>()) {
      TVM_FFI_CHECK(!HasTaskBufferAccess(stmt), ValueError)
          << "An unmarked scalar Bind inside T.PerCoreTask may not access "
             "buffers: synchronization with statements outside the "
             "PerCoreTask is emitted only at T.Task candidate sites, so an "
             "external buffer dependency could not be synchronized at this "
             "Bind. Dependency-free scalar Bind control is allowed. Move the "
             "buffer access into a supported same-pipe T.Task candidate. Bind="
          << stmt;
    }

    return AttrStmt(MakeTaskPipeAnnotation(AnalyzeTaskPipe(stmt),
                                           IntImm(DataType::Int(32), 0)),
                    tl::attr::kAscendTask, IntImm(DataType::Int(32), 1), stmt,
                    stmt->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root")
      return StmtMutator::VisitStmt_(op);
    ++schedule_body_depth_;
    Stmt result = StmtMutator::VisitStmt_(op);
    --schedule_body_depth_;
    return result;
  }

  int schedule_body_depth_{0};
  int task_marker_depth_{0};
  int per_core_task_depth_{0};
};

} // namespace

tvm::transform::Pass NormalizeTask() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return TaskNormalizer::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeTask", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeTask", NormalizeTask);
}

} // namespace tl
} // namespace tvm
