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
 * \file auto_schedule.cc
 * \brief AutoSchedule pass for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/ir_structure.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "auto_schedule.h"
#include "backend/common/target_utils.h"
#include "buffer_version.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

// AttrStmt wrappers that AutoSchedule supports.
static const AutoScheduleAttrSpec kAutoScheduleAttrs[] = {
    {tirx::attr::tilelang_assume, true},
    {tl::attr::kAssumeRequiresRuntimeCheck, false},
};

class PerCoreTaskControlValidator : public StmtExprVisitor {
public:
  explicit PerCoreTaskControlValidator(uint16_t pipe_mask)
      : expected_pipe_("PIPE_" + GetResourcePipeName(pipe_mask)) {}

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *mode = op->args[0].as<IntImmNode>();
      ICHECK(mode) << "Cross-core synchronization mode must be a constant "
                      "integer, got "
                   << op->args[0];
      ICHECK_EQ(mode->value, 0)
          << "Cross-core synchronization inside T.PerCoreTask only supports "
             "mode 0, got "
          << mode->value;
      const auto *pipe = op->args[1].as<StringImmNode>();
      ICHECK(pipe) << "Inter-core synchronization pipe must be a constant "
                      "string, got "
                   << op->args[1];
      ICHECK_EQ(pipe->value, expected_pipe_)
          << "Inter-core synchronization inside T.PerCoreTask must use its "
             "inferred pipe "
          << expected_pipe_ << ", got " << pipe->value;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::string expected_pipe_;
};

class TaskMarkerControlValidator : public StmtExprVisitor {
public:
  TaskMarkerControlValidator(bool reject_candidate_selection,
                             bool allow_cross_core_sync)
      : reject_candidate_selection_(reject_candidate_selection),
        allow_cross_core_sync_(allow_cross_core_sync) {}

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK_NE(op->attr_key, tl::attr::kAscendTask)
        << "Nested T.Task regions are not supported";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ICHECK(!reject_candidate_selection_)
        << "Control flow selecting a T.Task candidate must be outside the "
           "T.Task scope";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    TVM_FFI_CHECK(allow_cross_core_sync_ ||
                      (!op->op.same_as(tl::ascend_cross_core_set_flag()) &&
                       !op->op.same_as(tl::ascend_cross_core_wait_flag())),
                  ValueError)
        << "Inter-core synchronization inside T.Task must be the task's only "
           "statement";
    StmtExprVisitor::VisitExpr_(op);
  }

  bool reject_candidate_selection_;
  bool allow_cross_core_sync_;
};

static bool IsStandaloneCrossCoreSync(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (!evaluate)
    return false;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

static bool IsPerCoreTaskScalarControl(const TaskNode *task) {
  if (task->stmt.as<BindNode>())
    return true;
  const auto *evaluate = task->stmt.as<EvaluateNode>();
  if (!evaluate)
    return false;
  if (is_zero(evaluate->value))
    return true;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

static bool IsLetDeclTask(const TaskNode *task) {
  return task != nullptr && task->stmt.as<BindNode>() != nullptr;
}

const AutoScheduleAttrSpec *FindAutoScheduleAttr(const String &key) {
  for (const AutoScheduleAttrSpec &spec : kAutoScheduleAttrs) {
    if (key == spec.key)
      return &spec;
  }
  return nullptr;
}

// Mutator to extract the body of tilelang_root block and optional
// buffer version overrides annotation.
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;
  BufferVersionOverrideMap buffer_versions_overrides;
  ManualMultiBufferTable manual_multi_buffer_versions;
  std::set<std::string> unlimit_memory_scopes;

  void VisitStmt_(const SBlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      body = op->body;
      auto ann_val = op->annotations.Get("tl.buffer_versions_map");
      if (ann_val.has_value()) {
        auto *map_obj = ann_val.value().as<ffi::MapObj>();
        if (map_obj) {
          for (const auto &it : *map_obj) {
            if (auto *var = it.first.as<VarNode>()) {
              if (auto imm = it.second.as<int>()) {
                buffer_versions_overrides[GetRef<Var>(var)] = imm.value();
              }
            }
          }
        }
      }
      auto manual_ann = op->annotations.Get("tl.manual_multi_buffer");
      if (manual_ann.has_value()) {
        if (auto *map_obj = manual_ann.value().as<ffi::MapObj>()) {
          for (const auto &it : *map_obj) {
            if (auto *var = it.first.as<VarNode>()) {
              if (auto imm = it.second.as<int>()) {
                manual_multi_buffer_versions[var] = imm.value();
              }
            }
          }
        }
      }
      auto mem_ann = op->annotations.Get("tl.unlimit_memory_scopes");
      if (mem_ann.has_value()) {
        if (auto *arr = mem_ann.value().as<ffi::ArrayObj>()) {
          for (const auto &it : *arr) {
            if (auto str_opt = it.as<ffi::String>()) {
              unlimit_memory_scopes.insert(str_opt.value());
            }
          }
        }
      }
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
};

static bool IsAutoScheduleAttrWrapper(const AttrStmtNode *op) {
  return FindAutoScheduleAttr(op->attr_key) != nullptr;
}

struct CrossCorePipeInfo {
  SpecialRegister special_register;
  CoreMask core_mask;
};

static CrossCorePipeInfo GetCrossCorePipeInfo(const PrimExpr &pipe_arg) {
  const auto *pipe_imm = pipe_arg.as<StringImmNode>();
  ICHECK(pipe_imm)
      << "Auto-scheduled cross-core synchronization requires a constant pipe "
         "string";
  const String &pipe = pipe_imm->value;
  if (pipe == "PIPE_S")
    return {SpecialRegister::kPipeScalar, kCoreUnassigned};
  if (pipe == "PIPE_V")
    return {SpecialRegister::kPipeVector, kCoreVector};
  if (pipe == "PIPE_M")
    return {SpecialRegister::kPipeCube, kCoreCube};
  if (pipe == "PIPE_MTE1")
    return {SpecialRegister::kPipeMTE1, kCoreCube};
  if (pipe == "PIPE_MTE2")
    return {SpecialRegister::kPipeMTE2, kCoreUnassigned};
  if (pipe == "PIPE_MTE3")
    return {SpecialRegister::kPipeMTE3, kCoreVector};
  if (pipe == "PIPE_FIX")
    return {SpecialRegister::kPipeFixpipe, kCoreCube};
  LOG(FATAL) << "Ascend AutoSchedule does not support cross-core pipe \""
             << pipe
             << "\". Expected one of PIPE_S, PIPE_V, PIPE_M, PIPE_MTE1, "
                "PIPE_MTE2, PIPE_MTE3, or PIPE_FIX";
  return {SpecialRegister::kNone, kCoreUnassigned};
}

static Optional<Call> GetCrossCoreCall(const Call &call) {
  if (!call->op.same_as(tl::ascend_cross_core_set_flag()) &&
      !call->op.same_as(tl::ascend_cross_core_wait_flag())) {
    return std::nullopt;
  }
  ICHECK_EQ(call->args.size(), 3)
      << "ascend_cross_core_set_flag/wait_flag expects exactly 3 arguments";
  const auto *mode = call->args[0].as<IntImmNode>();
  ICHECK(mode)
      << "Ascend AutoSchedule requires cross-core mode_id to be constant 0";
  ICHECK_EQ(mode->value, 0)
      << "Ascend AutoSchedule only supports cross-core synchronization with "
         "mode_id=0; disable AutoSchedule for manually scheduled mode_id="
      << mode->value;
  return call;
}

static std::vector<Call> CollectCrossCoreCalls(const TaskNode *task) {
  class Collector : public StmtExprVisitor {
  public:
    std::vector<Call> calls;

  private:
    void VisitExpr_(const CallNode *op) final {
      Call call = GetRef<Call>(op);
      if (GetCrossCoreCall(call).defined())
        calls.push_back(call);
      StmtExprVisitor::VisitExpr_(op);
    }
  } collector;
  collector(task->stmt);
  return collector.calls;
}

static uint16_t CollectSpecialRegisterWrites(const Stmt &stmt) {
  class Collector : public StmtExprVisitor {
  public:
    uint16_t mask{0};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(tl::loop_break())) {
        mask |= static_cast<uint16_t>(SpecialRegister::kLoopControl);
      } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
        mask |= static_cast<uint16_t>(SpecialRegister::kHf32Mode);
      } else if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
        mask |= static_cast<uint16_t>(SpecialRegister::kPadValue);
      } else if (op->op.same_as(tl::ascend_set_atomic()) ||
                 op->op.same_as(tl::ascend_set_atomic_none())) {
        mask |= static_cast<uint16_t>(SpecialRegister::kAtomicMode);
      } else if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
                 op->op.same_as(tl::ascend_cross_core_wait_flag())) {
        Call call = GetRef<Call>(op);
        ICHECK(GetCrossCoreCall(call).defined());
        mask |= static_cast<uint16_t>(
            GetCrossCorePipeInfo(op->args[1]).special_register);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  } collector;
  collector(stmt);
  return collector.mask;
}

static bool ContainsDualCopy(const Stmt &stmt) {
  class Detector : public StmtExprVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (found)
        return;
      static const auto copy_op = Op::Get("tl.tileop.copy");
      if (op->op.same_as(copy_op)) {
        Copy copy(op->args, op->annotations);
        found = copy->GetDualDstCtl() != 0;
      } else if (op->op.same_as(tl::ascend_nd2nz_post_copy()) &&
                 op->args.size() == 8) {
        found = true;
      }
      if (!found)
        StmtExprVisitor::VisitExpr_(op);
    }
  } detector;
  detector(stmt);
  return detector.found;
}

// Detect multiple kernel launches in the PrimFunc body.
// In tilelang, when multiple T.Kernel() blocks are used, the IR structure is:
//   root block body:
//     AttrStmt(tl.assume_requires_runtime_check, ...)
//       AttrStmt(tl.assume, ...)
//         SeqStmt [
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel1_subtree),
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel2_subtree),
//         ]
// Each kernel subtree contains its own launch_threads and tilelang_root block.
// This class finds that SeqStmt and returns each child as a separate kernel.
class MultiKernelDetector {
public:
  static bool Detect(const Stmt &func_body, std::vector<Stmt> &kernel_stmts,
                     Stmt &prefix_wrapper) {
    std::vector<Stmt> stmts;
    const Stmt *inner = &func_body;

    // Peel through root block.
    if (const auto *block = inner->as<SBlockNode>()) {
      inner = &block->body;
    } else if (const auto *br = inner->as<SBlockRealizeNode>()) {
      inner = &br->block->body;
    }

    // Peel through Attribute chains.
    while (const auto *attr = inner->as<AttrStmtNode>()) {
      if (!IsAutoScheduleAttrWrapper(attr))
        break;
      inner = &attr->body;
    }

    // Check if we have a SeqStmt with multiple children that each contain
    // a launch_thread (thread_extent)
    const auto *seq = inner->as<SeqStmtNode>();
    if (!seq || seq->seq.size() < 2)
      return false;

    int kernel_count = 0;
    for (const auto &child : seq->seq) {
      if (ContainsLaunchThread(child)) {
        kernel_count++;
      }
    }

    if (kernel_count < 2)
      return false;

    for (const auto &child : seq->seq) {
      kernel_stmts.push_back(child);
    }
    return true;
  }

private:
  static bool ContainsLaunchThread(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tirx::attr::thread_extent) {
        return true;
      }
    }
    return false;
  }
};

// Mutator that replaces the inner SeqStmt (inside root block -> assume-wrapper
// chain) with a new body. Used to reassemble multi-kernel results.
class InnerSeqStmtReplacer : public StmtMutator {
public:
  explicit InnerSeqStmtReplacer(Stmt new_inner) : new_inner_(new_inner) {}

  Stmt VisitStmt_(const SBlockRealizeNode *op) override {
    auto new_block_body = this->VisitStmt(op->block->body);
    if (new_block_body.same_as(op->block->body))
      return GetRef<Stmt>(op);
    auto new_block =
        SBlock(op->block->iter_vars, op->block->reads, op->block->writes,
               op->block->name_hint, new_block_body, op->block->init,
               op->block->alloc_buffers, op->block->match_buffers,
               op->block->annotations);
    return SBlockRealize(op->iter_values, op->predicate, new_block);
  }

  Stmt VisitStmt_(const SBlockNode *op) override {
    auto new_body = this->VisitStmt(op->body);
    if (new_body.same_as(op->body))
      return GetRef<Stmt>(op);
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (IsAutoScheduleAttrWrapper(op)) {
      auto new_body = this->VisitStmt(op->body);
      if (new_body.same_as(op->body))
        return GetRef<Stmt>(op);
      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override { return new_inner_; }

private:
  Stmt new_inner_;
};

// Mutator to replace the body of tilelang_root block
class TilelangRootBodyReplacer : public StmtMutator {
public:
  TilelangRootBodyReplacer(Stmt new_body,
                           ascend::BufferVersionMap buffer_versions)
      : new_body_(std::move(new_body)),
        buffer_versions_(std::move(buffer_versions)) {}

  Stmt VisitStmt_(const SBlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      Map<String, Any> annotations = op->annotations;
      annotations.erase(ascend::kManualMultiBuffer);
      if (!buffer_versions_.empty()) {
        annotations.Set(ascend::kManualMultiBuffer, buffer_versions_);
      }
      return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                    new_body_, op->init, op->alloc_buffers, op->match_buffers,
                    annotations, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
  ascend::BufferVersionMap buffer_versions_;
};

// Visitor to build IRStructure from TIR statements
//
// Inherits `ConstrVisitor` so the constraint stack (For ranges, LetStmt
// bindings, AttrStmt thread_extent/virtual_thread/tilelang_assume,
// IfThenElse branch guards) is maintained automatically. Whenever a fresh
// TaskNode is constructed, `StampCtx` snapshots `GetConstrSet()` onto it
// for later use by RegionsMayConflict.
//
// Caveat: each VisitStmt_ override below replaces ConstrVisitor's default —
// for nodes that introduce a scope (For, LetStmt, AttrStmt, IfThenElse),
// the override must manually call `MakeGuard(...)` around the body
// recursion to keep the stack accurate.
class IRStructureBuilder : public TaskAwareConstrVisitor {
public:
  // Build an ordered list of IRStructure nodes from a statement. The old
  // SequenceNode wrapper is gone: a SeqStmt simply contributes its children to
  // the current list inline.
  std::vector<std::shared_ptr<IRStructure>> Build(const Stmt &stmt) {
    auto list = BuildList(stmt);
    if (list.empty()) {
      LOG(WARNING) << "IRStructureBuilder: produced empty list after visiting "
                      "statement. "
                   << "This may indicate an unhandled statement type.";
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = stmt;
      list.push_back(std::move(task_node));
    }
    return list;
  }

protected:
  // Visit `stmt` into a fresh accumulator list and return it. Nested visits
  // (loop bodies, if-branches) use this to collect their own ordered list
  // without disturbing the caller's.
  std::vector<std::shared_ptr<IRStructure>> BuildList(const Stmt &stmt) {
    std::vector<std::shared_ptr<IRStructure>> saved;
    saved.swap(result_);
    VisitStmt(stmt);
    std::vector<std::shared_ptr<IRStructure>> out;
    out.swap(result_);
    result_.swap(saved);
    return out;
  }

  // Stamp the current ConstrVisitor stack snapshot onto a freshly-built
  // TaskNode, so dependency proving and latency estimation see the symbol
  // environment (ancestor For ranges, Let bindings, thread_extents,
  // tilelang_assume, branch guards) the region was written under.
  void StampCtx(TaskNode *task) { task->outer_ctx = GetConstrSet(); }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled) {
      // Sequential For -> ControlNode
      auto control_node = std::make_shared<ControlNode>();
      control_node->control = GetRef<For>(op);
      control_node->task = std::make_shared<TaskNode>();
      control_node->task->stmt =
          For(op->loop_var, op->min, op->extent, op->kind, Evaluate(0),
              op->thread_binding, op->annotations, op->step, op->span);
      // The loop task itself sees the context outside the loop.
      StampCtx(control_node->task.get());
      ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(op->min)),
                        control_node->task.get());
      ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(op->extent)),
                        control_node->task.get());
      if (op->step.defined())
        ApplyTaskAccesses(
            AnalyzeTaskAccesses(Evaluate(GetRef<PrimExpr>(op->step.get()))),
            control_node->task.get());

      // Process the loop body inside the loop scope (push loop_var range).
      {
        auto guard_1 =
            MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        auto guard_2 = MakeGuard(op->extent > 0);
        control_node->children = BuildList(op->body);
      }

      result_.push_back(std::move(control_node));
    } else {
      ThrowUnmarkedTask("AutoSchedule", "parallel loop");
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    ThrowUnmarkedTask("AutoSchedule", "Evaluate");
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    ThrowUnmarkedTask("AutoSchedule", "BufferStore");
  }

  // Stamp a guard condition onto every schedule candidate produced for an
  // if-branch
  void StampConditionOnBranch(IRStructure *node, const PrimExpr &cond) {
    if (!node)
      return;
    auto stamp_task = [&](IRStructure *item, TaskNode *sink) {
      item->PrependCondition(cond);
      if (sink) {
        ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(cond)), sink);
      }
    };
    if (node->IsTask()) {
      stamp_task(node, static_cast<TaskNode *>(node));
    } else if (node->IsControl()) {
      stamp_task(node, static_cast<ControlNode *>(node)->task.get());
    } else {
      // Post-scheduling-only metadata does not appear pre-scheduling; stamp
      // defensively.
      node->PrependCondition(cond);
    }
  }

  // Stamp a supported AttrStmt onto every schedule candidate produced for its
  // body. Attr metadata remains an ordered guard delta and is not materialized
  // as a schedulable task.
  void StampAttributeOnBranch(IRStructure *node, const Any &attr_node,
                              const String &key, const PrimExpr &value,
                              const Span &span) {
    if (!node)
      return;
    auto stamp_task = [&](IRStructure *item, TaskNode *sink) {
      item->PrependAttribute(attr_node, key, value, span);
      if (sink) {
        if (const auto *expr_node = attr_node.as<PrimExprNode>()) {
          ApplyTaskAccesses(
              AnalyzeTaskAccesses(Evaluate(GetRef<PrimExpr>(expr_node))), sink);
        }
      }
    };
    if (node->IsTask()) {
      stamp_task(node, static_cast<TaskNode *>(node));
    } else if (node->IsControl()) {
      stamp_task(node, static_cast<ControlNode *>(node)->task.get());
    } else {
      node->PrependAttribute(attr_node, key, value, span);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // No dedicated IfNode: the guard is stamped onto the branch's schedule
    // candidates as a `conditions` delta (see StampConditionOnBranch).
    // then/else become ordinary schedule candidates distinguished only by their
    // conditions; both branches' candidates are spliced into the current list.
    std::vector<std::shared_ptr<IRStructure>> then_list;
    {
      auto guard = MakeGuard(op->condition);
      then_list = BuildList(op->then_case);
    }
    for (auto &node : then_list) {
      StampConditionOnBranch(node.get(), op->condition);
      result_.push_back(std::move(node));
    }

    if (op->else_case) {
      auto guard = MakeGuard(Not(op->condition));
      auto else_list = BuildList(op->else_case.value());
      for (auto &node : else_list) {
        StampConditionOnBranch(node.get(), Not(op->condition));
        result_.push_back(std::move(node));
      }
    }
  }

  void VisitStmt_(const BindNode *op) override {
    ThrowUnmarkedTask("AutoSchedule", "Bind");
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tl::attr::kAscendTask) {
      ICHECK_EQ(task_marker_depth_, 0)
          << "Nested T.Task regions are not supported";

      TaskAnnotations annotations = ParseTaskAnnotations(op->node);
      Stmt task_body = op->body;

      bool inside_per_core_task = per_core_task_depth_ > 0;
      ++task_marker_depth_;
      TaskMarkerControlValidator task_control_validator(
          inside_per_core_task, IsStandaloneCrossCoreSync(task_body));
      task_control_validator(task_body);
      --task_marker_depth_;

      TVM_FFI_CHECK(task_body.as<EvaluateNode>() ||
                        !ContainsDualCopy(task_body),
                    ValueError)
          << "T.Task does not support dual_copy after it is combined with "
             "other statements or expanded into a multi-statement body; "
             "keep dual_copy as a standalone single-pipe task. Body="
          << task_body;

      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = task_body;
      task_node->SetPipeMask(RequireTaskPipe(annotations, "T.Task"));
      StampCtx(task_node.get());
      task_node->SetSpecialWriteMask(
          CollectSpecialRegisterWrites(task_node->stmt));
      ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(task_node->stmt),
                             task_node.get());
      ApplyTaskAccesses(AnalyzeTaskAccesses(task_node->stmt), task_node.get());
      uint16_t issue_pipe_mask = task_node->GetPipeMask();
      if (task_node->GetSpecialWriteMask() != 0) {
        issue_pipe_mask |= static_cast<uint16_t>(ResourcePipe::kScalar);
      }
      TVM_FFI_CHECK((issue_pipe_mask & (issue_pipe_mask - 1)) == 0, ValueError)
          << "T.Task supports exactly one Ascend hardware pipe, but its body "
             "uses multiple pipes (mask="
          << issue_pipe_mask
          << "). Split the statements into separate T.Task regions. Body="
          << task_node->stmt;
      uint16_t hbm_mask = task_node->GetHbmMask();
      TVM_FFI_CHECK((hbm_mask & (hbm_mask - 1)) == 0, ValueError)
          << "T.Task requires one Ascend core affinity, but its body "
             "uses both AIC and AIV HBM paths. Split GM-to-L1 and GM-to-UB "
             "operations into separate T.Task regions. Body="
          << task_node->stmt;
      TaskCost cost = RequireTaskCost(annotations, "T.Task");
      task_node->SetLatency(cost.latency);
      task_node->SetII(cost.ii);
      if (inside_per_core_task && HasTaskBufferAccess(task_node.get()))
        per_core_task_candidates_.push_back(task_node);
      result_.push_back(std::move(task_node));
      return;
    }

    if (op->attr_key == tl::attr::kAscendPerCoreTask) {
      ICHECK_EQ(per_core_task_depth_, 0)
          << "Nested T.PerCoreTask regions are not supported";

      TaskAnnotations annotations = ParseTaskAnnotations(op->node);
      uint16_t pipe_mask = RequireTaskPipe(annotations, "T.PerCoreTask");
      TVM_FFI_CHECK((pipe_mask & (pipe_mask - 1)) == 0, ValueError)
          << "T.PerCoreTask requires exactly one annotated hardware pipe, but "
             "its mask is "
          << pipe_mask;
      Stmt per_core_body = op->body;
      std::vector<std::shared_ptr<TaskNode>> saved_task_candidates;
      saved_task_candidates.swap(per_core_task_candidates_);
      ++per_core_task_depth_;
      std::vector<std::shared_ptr<IRStructure>> body_list =
          BuildList(per_core_body);
      --per_core_task_depth_;
      std::vector<std::shared_ptr<TaskNode>> task_candidates;
      task_candidates.swap(per_core_task_candidates_);
      per_core_task_candidates_.swap(saved_task_candidates);
      ICHECK(!task_candidates.empty())
          << "T.PerCoreTask must contain at least one dependency-bearing task "
             "after NormalizeTask; run NormalizeTask immediately before "
             "AutoSchedule";
      std::unordered_set<const TaskNode *> candidate_tasks;
      for (const std::shared_ptr<TaskNode> &candidate : task_candidates) {
        ICHECK(candidate);
        ICHECK(candidate_tasks.insert(candidate.get()).second)
            << "Duplicate T.Task candidate TaskNode";
      }
      bool has_dependency_stmt = false;

      std::function<void(IRStructure *)> collect_sites;
      collect_sites = [&](IRStructure *node) {
        if (!node)
          return;
        if (node->IsControl()) {
          auto *control = static_cast<ControlNode *>(node);
          for (const auto &child : control->children)
            collect_sites(child.get());
          return;
        }

        auto *task = static_cast<TaskNode *>(node);
        ICHECK(!task->IsPerCoreTask())
            << "Nested T.PerCoreTask regions are not supported";
        const bool has_access = HasTaskBufferAccess(task);
        if (!has_access) {
          ICHECK(IsPerCoreTaskScalarControl(task))
              << "T.PerCoreTask only permits scalar Bind/condition/loop "
                 "control and inter-core synchronization in addition to "
                 "operations on its inferred pipe; got "
              << task->stmt;
          return;
        }
        ICHECK(candidate_tasks.count(task))
            << "NormalizeTask did not assign a T.Task candidate marker "
               "to a dependency-bearing statement in T.PerCoreTask; got "
            << task->stmt;
        uint16_t task_pipe_mask = task->GetPipeMask();
        ICHECK_NE(task_pipe_mask, 0)
            << "Cannot infer an Ascend pipe for T.PerCoreTask candidate "
            << task->stmt;
        ICHECK_EQ(task_pipe_mask & (task_pipe_mask - 1), 0)
            << "Each dependency-bearing statement in T.PerCoreTask must use "
               "exactly one hardware pipe, but statement mask="
            << task_pipe_mask << ", statement=" << task->stmt;
        ICHECK_EQ(task_pipe_mask, pipe_mask)
            << "T.PerCoreTask candidate pipe does not match its annotated "
               "pipe. Annotated mask="
            << pipe_mask << ", statement mask=" << task_pipe_mask
            << ", statement=" << task->stmt;
        has_dependency_stmt = true;
      };
      for (const auto &node : body_list)
        collect_sites(node.get());
      ICHECK(has_dependency_stmt)
          << "T.PerCoreTask must contain at least one dependency-bearing "
             "statement on its annotated pipe";

      PerCoreTaskControlValidator control_validator(pipe_mask);
      control_validator(per_core_body);
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = per_core_body;
      ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(per_core_body),
                             task_node.get());
      ApplyTaskAccesses(AnalyzeTaskAccesses(per_core_body), task_node.get());
      task_node->SetPipeMask(pipe_mask);
      task_node->SetPerCoreTaskPipeMask(pipe_mask);
      // Cross-core calls inside PerCoreTask are internal sequencing of this
      // compound task. They migrate with its concrete candidates at emission
      // time and intentionally do not expose special-register writes to the
      // outer scheduler; placement comes from the task's inferred pipe/memory.
      StampCtx(task_node.get());
      TaskCost cost = RequireTaskCost(annotations, "T.PerCoreTask");
      task_node->SetLatency(cost.latency);
      task_node->SetII(cost.ii);
      result_.push_back(std::move(task_node));
      return;
    }

    const AutoScheduleAttrSpec *attr_spec = FindAutoScheduleAttr(op->attr_key);
    ICHECK(attr_spec != nullptr)
        << "AutoSchedule IRStructureBuilder only supports AttrStmt "
           "keys listed in kAutoScheduleAttrs inside the schedulable "
           "tilelang_root body. "
           "Unsupported attr_key: "
        << op->attr_key;

    // Supported AttrStmts become ordered guard deltas on every candidate.
    // tl.assume additionally enters the prover environment while its body is
    // built; transparent attrs only preserve their source nesting.
    std::vector<std::shared_ptr<IRStructure>> body_list;
    if (attr_spec->is_assumption) {
      auto guard = MakeGuard(Downcast<PrimExpr>(op->node), /*is_assume=*/true);
      body_list = BuildList(op->body);
    } else {
      body_list = BuildList(op->body);
    }
    if (body_list.empty()) {
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = Evaluate(0);
      StampCtx(task_node.get());
      body_list.push_back(std::move(task_node));
    }
    for (auto &node : body_list) {
      StampAttributeOnBranch(node.get(), op->node, op->attr_key, op->value,
                             op->span);
      result_.push_back(std::move(node));
    }
  }

  void VisitStmt_(const WhileNode *op) override {
    ThrowUnmarkedTask("AutoSchedule", "While");
  }

  void VisitStmt_(const SBlockNode *op) override {
    ThrowUnmarkedTask("AutoSchedule", "block");
  }

private:
  std::vector<std::shared_ptr<IRStructure>> result_;
  int per_core_task_depth_{0};
  int task_marker_depth_{0};
  std::vector<std::shared_ptr<TaskNode>> per_core_task_candidates_;
};

// Result of scheduling one kernel segment before serializing scheduled markers.
struct ScheduledKernelResult {
  std::vector<std::shared_ptr<IRStructure>> root;
  FlagInfoList flag_infos;
  std::vector<MultiVersionBufferInfo> buffer_infos;
};

using ScheduledMetadata = Map<String, Any>;

static Stmt WrapScheduledGuards(Stmt body, const GuardList &guards) {
  for (auto it = guards.rbegin(); it != guards.rend(); ++it)
    body = (*it)->Wrap(std::move(body));
  return body;
}

class PerCoreTaskSyncMigrator : public StmtMutator {
public:
  PerCoreTaskSyncMigrator(std::vector<Stmt> before, std::vector<Stmt> after)
      : before_(std::move(before)), after_(std::move(after)) {}

  Stmt Rewrite(Stmt body) {
    Stmt result = VisitStmt(std::move(body));
    ICHECK_GT(num_candidates_, 0)
        << "Cannot find a T.Task marker in normalized PerCoreTask";
    return result;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kAscendTask)
      return StmtMutator::VisitStmt_(op);
    Stmt candidate = VisitStmt(op->body);
    if (!HasTaskBufferAccess(candidate))
      return candidate;
    ++num_candidates_;
    Array<Stmt> sequence;
    sequence.reserve(before_.size() + 1 + after_.size());
    for (const Stmt &stmt : before_)
      sequence.push_back(stmt);
    sequence.push_back(std::move(candidate));
    for (const Stmt &stmt : after_)
      sequence.push_back(stmt);
    return SeqStmt::Flatten(sequence);
  }

  std::vector<Stmt> before_;
  std::vector<Stmt> after_;
  int num_candidates_{0};
};

class ScheduledTIREmitter {
public:
  Stmt Emit(const std::vector<std::shared_ptr<IRStructure>> &root,
            const FlagInfoList &flag_infos) {
    Array<Stmt> statements;
    for (const FlagInfo &info : flag_infos) {
      if (info.mode == 0) {
        statements.push_back(EmitSyntheticTask(
            MakeAscendSetFlag(info.hardevent, info.flag_var), info.core_mask));
      } else if (info.mode == 1) {
        statements.push_back(EmitSyntheticTask(
            MakeAscendCrossCoreSetFlag(4, info.hardevent, info.flag_var),
            info.core_mask));
      }
    }
    for (const auto &node : root)
      statements.push_back(EmitNode(node.get()));
    for (const FlagInfo &info : flag_infos) {
      if (info.mode == 0) {
        statements.push_back(EmitSyntheticTask(
            MakeAscendWaitFlag(info.hardevent, info.flag_var), info.core_mask));
      } else if (info.mode == 2) {
        statements.push_back(EmitSyntheticTask(
            MakeAscendCrossCoreWaitFlag(4, info.hardevent, info.flag_var),
            info.core_mask));
      }
    }
    return SeqStmt::Flatten(statements);
  }

private:
  Stmt EmitList(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    Array<Stmt> statements;
    statements.reserve(nodes.size());
    for (const auto &node : nodes)
      statements.push_back(EmitNode(node.get()));
    return SeqStmt::Flatten(statements);
  }

  Stmt EmitSyntheticTask(Stmt stmt, CoreMask core_mask) {
    return EmitSyntheticTask(std::move(stmt), core_mask, 0);
  }

  Stmt EmitSyntheticTask(Stmt stmt, CoreMask core_mask, int stage) {
    ScheduledMetadata metadata;
    metadata.Set(kAscendScheduledCoreMask, Integer(core_mask));
    metadata.Set(kAscendScheduledStage, Integer(stage));
    return AttrStmt(metadata, attr::kAscendTask, Integer(1), std::move(stmt));
  }

  void
  EmitAdjacentTasks(Array<Stmt> *result,
                    const std::map<CoreMask, std::vector<Stmt>> &statements,
                    int stage) {
    for (const auto &[core_mask, stmts] : statements) {
      ICHECK(IsConcreteCore(core_mask))
          << "Scheduled statement must belong to one concrete core, got mask "
          << static_cast<int>(core_mask);
      for (const Stmt &stmt : stmts) {
        result->push_back(EmitSyntheticTask(stmt, core_mask, stage));
      }
    }
  }

  static std::vector<Stmt>
  GetPerCoreSync(const std::map<CoreMask, std::vector<Stmt>> &statements,
                 CoreMask core_mask) {
    auto it = statements.find(core_mask);
    return it == statements.end() ? std::vector<Stmt>() : it->second;
  }

  Stmt EmitNode(const IRStructure *node) {
    ICHECK(node != nullptr);
    Array<Stmt> result;
    bool migrate_per_core_sync =
        node->IsTask() && static_cast<const TaskNode *>(node)->IsPerCoreTask();
    if (!migrate_per_core_sync)
      EmitAdjacentTasks(&result, node->before, node->GetStage());

    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      ScheduledMetadata metadata;
      metadata.Set(kAscendScheduledStage, Integer(node->GetStage()));
      metadata.Set(kAscendScheduledCoreMask, Integer(task->GetCoreMask()));
      Stmt task_body = task->stmt;
      if (task->IsPerCoreTask()) {
        task_body = PerCoreTaskSyncMigrator(
                        GetPerCoreSync(node->before, task->GetCoreMask()),
                        GetPerCoreSync(node->after, task->GetCoreMask()))
                        .Rewrite(std::move(task_body));
      }
      result.push_back(
          AttrStmt(metadata, attr::kAscendTask, Integer(1),
                   WrapScheduledGuards(std::move(task_body), node->GetGuards()),
                   task->stmt->span));
    } else {
      const auto *control = static_cast<const ControlNode *>(node);
      For loop = control->control;
      loop.CopyOnWrite()->body = EmitList(control->children);
      ScheduledMetadata metadata;
      metadata.Set(kAscendScheduledStage, Integer(node->GetStage()));
      result.push_back(
          AttrStmt(metadata, attr::kAscendTask, Integer(1),
                   WrapScheduledGuards(std::move(loop), node->GetGuards()),
                   control->control->span));
    }
    if (!migrate_per_core_sync)
      EmitAdjacentTasks(&result, node->after, node->GetStage());
    return SeqStmt::Flatten(result);
  }
};

static void WarnOnUnappliedBufferVersionOverrides(
    const BufferVersionOverrideMap &buffer_versions_overrides,
    const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      multi_buffered_storages;
  for (const MultiVersionBufferInfo &info : buffer_infos) {
    if (info.num_versions > 1) {
      multi_buffered_storages.insert(info.buffer->data);
    }
  }

  for (const auto &[buffer_var, num_versions] : buffer_versions_overrides) {
    if (num_versions > 1 && !multi_buffered_storages.count(buffer_var)) {
      LOG(WARNING) << "Ascend AutoSchedule did not enable multi-buffering for "
                      "buffer \""
                   << buffer_var->name_hint
                   << "\" although T.annotate_buffer_versions " << "requested "
                   << num_versions
                   << " versions. The buffer must be eligible in a scheduled "
                      "serial loop and written before it is read.";
    }
  }
}

struct CrossCoreTaskInfo {
  TaskNode *task;
  ControlNode *control_node;
  PrimExpr flag_id;
  String pipe;
  CoreMask core_hint;
};

static void
AssignCrossCoreTasks(const std::vector<TaskNodeWithContext> &all_tasks) {
  std::vector<CrossCoreTaskInfo> sync_tasks;
  sync_tasks.reserve(all_tasks.size());
  CoreMask kernel_core_mask = kCoreUnassigned;
  for (const TaskNodeWithContext &task_ctx : all_tasks) {
    // PerCoreTask consumes its embedded cross-core synchronization internally.
    // Keep the core already inferred from its ordinary operation instead
    // of merging those calls into an outer synchronization group.
    if (task_ctx.task->IsPerCoreTask()) {
      CoreMask core_mask = task_ctx.task->GetCoreMask();
      if (IsConcreteCore(core_mask))
        kernel_core_mask |= core_mask;
      continue;
    }
    std::vector<Call> calls = CollectCrossCoreCalls(task_ctx.task);
    if (calls.empty()) {
      CoreMask core_mask = task_ctx.task->GetCoreMask();
      if (IsConcreteCore(core_mask))
        kernel_core_mask |= core_mask;
      continue;
    }
    for (const Call &call : calls) {
      const auto *pipe_imm = call->args[1].as<StringImmNode>();
      ICHECK(pipe_imm);
      CrossCorePipeInfo pipe_info = GetCrossCorePipeInfo(call->args[1]);
      sync_tasks.push_back({task_ctx.task, task_ctx.control_node, call->args[2],
                            pipe_imm->value, pipe_info.core_mask});
    }
  }
  if (sync_tasks.empty())
    return;

  struct SyncGroup {
    ControlNode *control_node;
    PrimExpr flag_id;
    std::vector<CrossCoreTaskInfo *> tasks;
    CoreMask core_hints{kCoreUnassigned};
  };
  std::vector<SyncGroup> groups;
  ffi::StructuralEqual structural_equal;
  for (CrossCoreTaskInfo &sync_task : sync_tasks) {
    SyncGroup *group = nullptr;
    for (SyncGroup &candidate : groups) {
      if (candidate.control_node == sync_task.control_node &&
          structural_equal(candidate.flag_id, sync_task.flag_id)) {
        group = &candidate;
        break;
      }
    }
    if (!group) {
      groups.push_back(
          {sync_task.control_node, sync_task.flag_id, {}, /*hints=*/0});
      group = &groups.back();
    }
    group->tasks.push_back(&sync_task);
    group->core_hints |= sync_task.core_hint;
  }

  for (const SyncGroup &group : groups) {
    if (group.core_hints == kCoreBroadcast) {
      std::ostringstream pipes;
      for (size_t i = 0; i < group.tasks.size(); ++i) {
        if (i)
          pipes << ", ";
        pipes << group.tasks[i]->pipe;
      }
      LOG(FATAL) << "Cross-core set/wait group for flag " << group.flag_id
                 << " resolves to both Vector and Cube cores from pipes "
                 << pipes.str()
                 << "; all calls in one synchronization group must run on "
                    "the same core";
    }

    CoreMask core_mask = group.core_hints;
    if (core_mask == kCoreUnassigned && IsConcreteCore(kernel_core_mask)) {
      // PIPE_S and PIPE_MTE2 exist on both core types. A pure kernel supplies
      // the only possible placement when neither side of the pair is pinned.
      core_mask = kernel_core_mask;
    } else if (!IsConcreteCore(core_mask)) {
      LOG(FATAL)
          << "Cannot infer the core for cross-core set/wait group with flag "
          << group.flag_id
          << ": PIPE_S and PIPE_MTE2 are ambiguous in a mixed or empty "
             "kernel. Use PIPE_V/PIPE_MTE3 for Vector or "
             "PIPE_M/PIPE_MTE1/PIPE_FIX for Cube on at least one call in "
             "the group";
    }
    for (const CrossCoreTaskInfo *sync_task : group.tasks)
      sync_task->task->SetCoreMask(core_mask);
  }
}

// Collect every mode-0 CrossCore flag range used explicitly by the source
// kernel. Track enclosing loops, flat Bind declarations, branch guards, and
// assumptions so ConstIntBound can resolve expressions such as
// indexmod(iter, num_versions) + base. PerCoreTask bodies are visited as
// ordinary statements because their explicit flags still share the
// kernel-global hardware namespace.
class ExplicitCrossCoreFlagCollector : public TaskAwareConstrVisitor {
public:
  CrossCoreFlagReservation Collect(const Stmt &stmt) {
    operator()(stmt);
    return std::move(reservation_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    if (GetCrossCoreCall(call).defined()) {
      const PrimExpr &flag_id = call->args[2];
      arith::Analyzer analyzer;
      GetConstrSet().Populate(analyzer);
      arith::ConstIntBound bound = analyzer.const_int_bound(flag_id);
      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_flag_id("unconstrained_cross_core_flag_id",
                                flag_id.dtype());
      arith::ConstIntBound unconstrained_bound =
          unconstrained_analyzer.const_int_bound(unconstrained_flag_id);
      if (bound->min_value == arith::ConstIntBound::kNegInf ||
          bound->max_value == arith::ConstIntBound::kPosInf ||
          (bound->min_value == unconstrained_bound->min_value &&
           bound->max_value == unconstrained_bound->max_value)) {
        reservation_.has_unbounded_id = true;
      } else {
        ICHECK_LE(bound->min_value, bound->max_value)
            << "Ascend AutoSchedule inferred an invalid explicit cross-core "
               "flag range ["
            << bound->min_value << ", " << bound->max_value << "]";
        reservation_.explicit_id_ranges.emplace_back(bound->min_value,
                                                     bound->max_value);
      }
    }
    ConstrVisitor::VisitExpr_(op);
  }

  CrossCoreFlagReservation reservation_;
};

static CrossCoreFlagReservation
CollectExplicitCrossCoreFlags(const Stmt &kernel_body) {
  ExplicitCrossCoreFlagCollector collector;
  return collector.Collect(kernel_body);
}

// Assign scalar, register-buffer, and special-register producers to the core
// mask union of their consumers. CrossCore synchronization remains a dedicated
// side-effecting path and contributes no special-register demand.
class CoreMaskPropagator {
public:
  static void Assign(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNodeWithContext> &all_tasks) {
    CoreMaskPropagator analyzer(root, all_tasks);
    analyzer.Run_();
  }

private:
  static constexpr int kSpecialRegisterBits =
      std::numeric_limits<uint16_t>::digits;
  using VarDemandMap =
      std::unordered_map<Var, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  using BufferDemandMap =
      std::unordered_map<Buffer, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  using SpecialRegisterDemandMap =
      std::unordered_map<const TaskNode *, CoreMask>;
  using SpecialRegisterReadDemand = std::array<CoreMask, kSpecialRegisterBits>;
  using SpecialRegisterWriters =
      std::array<std::vector<TaskNode *>, kSpecialRegisterBits>;
  struct SpecialRegisterScopeDemand {
    SpecialRegisterReadDemand readers{};
    SpecialRegisterWriters writers;
  };

  CoreMaskPropagator(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNodeWithContext> &all_tasks)
      : root_(root), all_tasks_(all_tasks) {}

  void Run_() {
    CollectSpecialRegisterDemand_(root_);
    AssignSpecialRegisterWriters_();
    AssignCrossCoreTasks(all_tasks_);
    for (const auto &node : root_)
      SeedReads_(node.get(), kCoreBroadcast);
    PropagateReadsToFixedPoint_();
    AssignScalarWriters_();
  }

  static CoreMask GetConcreteCoreMask_(const TaskNode *task) {
    CoreMask core_mask = task->GetCoreMask();
    return IsConcreteCore(core_mask) ? core_mask : kCoreUnassigned;
  }

  // Special registers have one independent instance per core. Return both
  // reader demand and concrete descendant writers so readers in an outer
  // sibling list can place writers nested under a control node. loop_break is
  // body-local and is therefore the only register not propagated outward.
  SpecialRegisterScopeDemand CollectSpecialRegisterDemand_(
      const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    SpecialRegisterScopeDemand scope;

    for (const auto &node : nodes) {
      if (node->IsControl()) {
        const auto *control = static_cast<const ControlNode *>(node.get());
        SpecialRegisterScopeDemand child_scope =
            CollectSpecialRegisterDemand_(control->children);
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if (bit == static_cast<uint16_t>(SpecialRegister::kLoopControl))
            continue;
          scope.readers[bit_index] |= child_scope.readers[bit_index];
          scope.writers[bit_index].insert(
              scope.writers[bit_index].end(),
              child_scope.writers[bit_index].begin(),
              child_scope.writers[bit_index].end());
        }
      } else {
        const auto *task = static_cast<const TaskNode *>(node.get());
        CoreMask core_mask = GetConcreteCoreMask_(task);
        uint16_t read_mask = node->GetSpecialReadMask();
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if ((read_mask & bit) != 0)
            scope.readers[bit_index] |= core_mask;
        }
      }

      if (!node->IsTask())
        continue;
      auto *task = static_cast<TaskNode *>(node.get());
      // CrossCore bits only order their target hardware pipe. The calls are
      // placed by matching pipe hints in their flag group, not by demand.
      if (!CollectCrossCoreCalls(task).empty())
        continue;
      uint16_t write_mask = task->GetSpecialWriteMask();
      for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
        uint16_t bit = static_cast<uint16_t>(1U << bit_index);
        if ((write_mask & bit) != 0)
          scope.writers[bit_index].push_back(task);
      }
    }

    for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
      for (TaskNode *writer : scope.writers[bit_index])
        special_writer_demand_[writer] |= scope.readers[bit_index];
    }
    return scope;
  }

  void AssignSpecialRegisterWriters_() {
    for (const TaskNodeWithContext &task_ctx : all_tasks_) {
      TaskNode *task = task_ctx.task;
      // loop_break controls the enclosing loop itself, so it must survive in
      // every core clone of that loop even when all concrete work is
      // nested below another ControlNode and contributes no same-level demand.
      if (task->ContainsLoopBreak()) {
        task->SetCoreMask(kCoreBroadcast);
        continue;
      }
      auto it = special_writer_demand_.find(task);
      if (it == special_writer_demand_.end())
        continue;
      AssignTaskFromDemand_(task, it->second);
    }
  }

  bool AddReadsDemand_(const TaskNode *task, CoreMask mask) {
    if (!mask)
      return false;
    bool grew = false;
    for (const Var &var : task->GetReadVars()) {
      CoreMask &demand = var_demand_[var];
      if ((demand | mask) != demand) {
        demand |= mask;
        grew = true;
      }
    }
    for (const BufferRegion &region : task->GetReadRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      CoreMask &demand = buffer_demand_[region->buffer];
      if ((demand | mask) != demand) {
        demand |= mask;
        grew = true;
      }
    }
    return grew;
  }

  void AddGuardDemand_(const IRStructure *node, CoreMask mask) {
    if (!mask)
      return;
    for (const auto &guard : node->GetGuards()) {
      if (!guard->IsCondition())
        continue;
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      for (const Var &var : UndefinedVars(condition_guard->condition))
        var_demand_[var] |= mask;
    }
  }

  static CoreMask GetLoopBodyCoverage_(const IRStructure *node) {
    if (!node)
      return 0;
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      // A loop break controls whether the loop continues, so every core
      // that receives a clone of the loop must evaluate the break condition.
      if (task->ContainsLoopBreak())
        return kCoreBroadcast;
      return GetConcreteCoreMask_(task);
    }
    CoreMask mask = kCoreUnassigned;
    const auto *control = static_cast<const ControlNode *>(node);
    for (const auto &child : control->children)
      mask |= GetLoopBodyCoverage_(child.get());
    return mask;
  }

  void SeedReads_(const IRStructure *node, CoreMask enclosing) {
    if (!node)
      return;
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      CoreMask task_mask = task->GetCoreMask();
      CoreMask mask = IsConcreteCore(task_mask) ? task_mask : enclosing;
      AddGuardDemand_(task, mask);
      AddReadsDemand_(task, mask);
      return;
    }

    const auto *control = static_cast<const ControlNode *>(node);
    CoreMask coverage = GetLoopBodyCoverage_(node);
    AddGuardDemand_(control, coverage);
    if (control->task)
      AddReadsDemand_(control->task.get(), coverage);
    for (const auto &child : control->children)
      SeedReads_(child.get(), coverage);
  }

  CoreMask GetWriteDemand_(const TaskNode *task) const {
    CoreMask mask = kCoreUnassigned;
    auto special_it = special_writer_demand_.find(task);
    if (special_it != special_writer_demand_.end())
      mask |= special_it->second;
    for (const Var &var : task->GetWriteVars()) {
      auto it = var_demand_.find(var);
      if (it != var_demand_.end())
        mask |= it->second;
    }
    for (const BufferRegion &region : task->GetWriteRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      auto it = buffer_demand_.find(region->buffer);
      if (it != buffer_demand_.end())
        mask |= it->second;
    }
    return mask;
  }

  void PropagateReadsToFixedPoint_() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (const TaskNodeWithContext &task_ctx : all_tasks_) {
        TaskNode *task = task_ctx.task;
        if (!task->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(task))
          continue;
        if (AddReadsDemand_(task, GetWriteDemand_(task)))
          changed = true;
      }
    }
  }

  static void AssignTaskFromDemand_(TaskNode *task, CoreMask demand) {
    if (demand != kCoreUnassigned)
      task->SetCoreMask(demand);
  }

  void AssignScalarWriters_() {
    for (const TaskNodeWithContext &task_ctx : all_tasks_) {
      TaskNode *task = task_ctx.task;
      if (task->ContainsLoopBreak())
        continue; // Assigned broadcast above; never demote from local demand.
      if (task->GetCoreMask() != kCoreBroadcast)
        continue;
      if (!task->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(task))
        continue;
      CoreMask demand = GetWriteDemand_(task);
      if (demand == kCoreUnassigned)
        task->SetCoreMask(kCoreUnassigned);
      else
        AssignTaskFromDemand_(task, demand);
    }
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const std::vector<TaskNodeWithContext> &all_tasks_;
  VarDemandMap var_demand_;
  BufferDemandMap buffer_demand_;
  SpecialRegisterDemandMap special_writer_demand_;
};

// Build and schedule one kernel segment, leaving core partitioning and final
// Stmt emission to the downstream CorePartition pass.
static ScheduledKernelResult ScheduleSingleKernel(
    const Stmt &kernel_body, Target target,
    const BufferVersionOverrideMap &buffer_versions_overrides = {},
    const ManualMultiBufferTable &manual_multi_buffer_versions = {},
    const std::set<std::string> &unlimit_memory_scopes = {}) {
  // Build IRStructure from the body to schedule
  ScheduledKernelResult result;
  IRStructureBuilder builder;
  auto ir_structure = builder.Build(kernel_body);

  ICHECK(!ir_structure.empty()) << "IRStructure is empty (empty body?)";

  std::vector<TaskNodeWithContext> costed_tasks;
  CollectAllTaskNodesWithContext(ir_structure, costed_tasks);
  for (const TaskNodeWithContext &task_ctx : costed_tasks) {
    TVM_FFI_CHECK_GT(task_ctx.task->GetII(), 0, ValueError)
        << "AutoSchedule found a task without latency/II annotations: "
        << task_ctx.task->stmt
        << ". Run tl.transform.LatencyEstimator immediately before "
           "AutoSchedule.";
  }

  // Schedule IRStructure in place with the Z3 scheduler.
  ScheduleBuilder unit_builder;
  int64_t shared_mem_limit = GetSharedMemoryLimit(target);
  if (TargetIsAscend(target) && !HasSimtVF(kernel_body)) {
    shared_mem_limit = 248 * 1024;
  }
  unit_builder.SetMemoryLimit("shared", shared_mem_limit);
  unit_builder.SetMemoryLimit("shared.l1", GetL1MemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0c", GetL0CMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0a", GetL0AMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0b", GetL0BMemoryLimit(target));
  constexpr int64_t kUnlimitedMemory = 1LL << 40;
  for (const auto &scope : unlimit_memory_scopes) {
    unit_builder.SetMemoryLimit(scope, kUnlimitedMemory);
  }
  unit_builder.SetBufferVersionsOverride(buffer_versions_overrides);
  unit_builder.SetManualMultiBuffer(manual_multi_buffer_versions);
  unit_builder.ScheduleList(ir_structure);

  // Assign core masks based on pipe and buffer scopes.
  {
    std::vector<TaskNodeWithContext> all_tasks;
    CollectAllTaskNodesWithContext(ir_structure, all_tasks);
    for (auto &task_ctx : all_tasks) {
      auto *t = task_ctx.task;
      // Special-register writers are placed from their concrete use sites
      // after ordinary compute and data-movement tasks have been pinned.
      if (t->GetSpecialWriteMask() != 0) {
        t->SetCoreMask(kCoreUnassigned);
        continue;
      }
      uint16_t m = t->GetPipeMask();
      if (m & static_cast<uint16_t>(ResourcePipe::kCube)) {
        t->SetCoreMask(kCoreCube);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kVector)) {
        t->SetCoreMask(kCoreVector);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE1) ||
                 m & static_cast<uint16_t>(ResourcePipe::kFixpipe)) {
        t->SetCoreMask(kCoreCube);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE3)) {
        t->SetCoreMask(kCoreVector);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE2)) {
        bool to_l1 = false;
        for (auto &r : t->GetWriteRegions()) {
          if (IsL1Buffer(r->buffer)) {
            to_l1 = true;
            break;
          }
        }
        t->SetCoreMask(to_l1 ? kCoreCube : kCoreVector);
      } else if (t->UsesPipe(ResourcePipe::kScalar) || m == 0) {
        // A scalar / mask-0 task issues on the S pipe, but its core is still
        // fixed by the on-chip memory it touches: UB is Vector-side, while
        // L1/L0A/L0B/L0C are Cube-side.
        bool touches_ub = false, touches_cube = false;
        auto scan = [&](const Array<BufferRegion> &regions) {
          for (auto &r : regions) {
            if (IsSharedBuffer(r->buffer))
              touches_ub = true;
            else if (IsL1Buffer(r->buffer) || IsL0ABuffer(r->buffer) ||
                     IsL0BBuffer(r->buffer) || IsL0CBuffer(r->buffer))
              touches_cube = true;
          }
        };
        scan(t->GetReadRegions());
        scan(t->GetWriteRegions());
        if (touches_ub && !touches_cube) {
          t->SetCoreMask(kCoreVector);
        } else if (touches_cube && !touches_ub) {
          t->SetCoreMask(kCoreCube);
        } else {
          t->SetCoreMask(kCoreBroadcast);
        }
      }
    }

    CoreMaskPropagator::Assign(ir_structure, all_tasks);
  }

  // Fallback: pin every unresolved task -- plus any still-broadcast
  // PerCoreTask, which has no core-specific memory placement to demote it -- to
  // a single core: Cube if the kernel has any concrete Cube-side work, else
  // Vector.
  {
    std::vector<TaskNodeWithContext> all_tasks;
    CollectAllTaskNodesWithContext(ir_structure, all_tasks);
    std::vector<TaskNode *> unresolved;
    bool has_cube_core = false;
    for (auto &task_ctx : all_tasks) {
      CoreMask core_mask = task_ctx.task->GetCoreMask();
      if (core_mask == kCoreCube)
        has_cube_core = true;
      else if (core_mask == kCoreUnassigned ||
               (core_mask == kCoreBroadcast && task_ctx.task->IsPerCoreTask()))
        unresolved.push_back(task_ctx.task);
    }
    CoreMask fallback_core = has_cube_core ? kCoreCube : kCoreVector;
    for (auto *t : unresolved)
      t->SetCoreMask(fallback_core);
  }

  CrossCoreFlagReservation cross_core_flag_reservation =
      CollectExplicitCrossCoreFlags(kernel_body);

  BarrierBuilder barrier_builder(manual_multi_buffer_versions,
                                 std::move(cross_core_flag_reservation));
  barrier_builder.Build(ir_structure);
  result.flag_infos = barrier_builder.GetFlagInfos();
  result.buffer_infos = barrier_builder.GetBufferInfos();
  WarnOnUnappliedBufferVersionOverrides(buffer_versions_overrides,
                                        result.buffer_infos);
  result.root = std::move(ir_structure);
  return result;
}

static void AddBufferVersion(ascend::BufferVersionMap *buffer_versions,
                             const Var &data, int64_t num_versions) {
  ICHECK_GT(num_versions, 0);
  if (num_versions == 1) {
    return;
  }
  if (buffer_versions->count(data)) {
    const auto *existing = (*buffer_versions)[data].as<IntImmNode>();
    ICHECK(existing);
    ICHECK_EQ(existing->value, num_versions)
        << "Conflicting manual and automatic buffer version counts for storage "
        << data << ": " << existing->value << " vs " << num_versions;
    return;
  }
  buffer_versions->Set(
      data, IntImm(DataType::Int(32), static_cast<int64_t>(num_versions)));
}

static ascend::BufferVersionMap BuildBufferVersionStagingMap(
    const ManualMultiBufferTable &manual_buffer_versions,
    const std::vector<MultiVersionBufferInfo> &automatic_buffer_versions) {
  ascend::BufferVersionMap result;
  for (const auto &[data_node, num_versions] : manual_buffer_versions) {
    AddBufferVersion(&result, GetRef<Var>(data_node), num_versions);
  }
  for (const MultiVersionBufferInfo &info : automatic_buffer_versions) {
    AddBufferVersion(&result, info.new_buffer->data, info.num_versions);
  }
  return result;
}

static Stmt SerializeScheduledKernel(const Stmt &kernel_subtree,
                                     const TilelangRootBodyExtractor &extractor,
                                     const ScheduledKernelResult &result) {
  ScheduledTIREmitter emitter;
  Stmt scheduled_body = emitter.Emit(result.root, result.flag_infos);
  ascend::BufferVersionMap buffer_versions = BuildBufferVersionStagingMap(
      extractor.manual_multi_buffer_versions, result.buffer_infos);
  TilelangRootBodyReplacer replacer(scheduled_body, std::move(buffer_versions));
  Stmt serialized = replacer(kernel_subtree);
  if (!result.buffer_infos.empty()) {
    serialized = RewriteAllocBuffers(serialized, result.buffer_infos);
    BroadcastMultiBufferRewriter broadcast(result.buffer_infos);
    serialized = broadcast(serialized);
  }
  return serialized;
}

// Build and schedule each kernel, serializing the scheduled tree into ordinary
// TIR markers for CorePartition to consume.
tvm::transform::Pass AutoSchedule() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    // Detect multiple kernel launches in the PrimFunc body.
    // When multiple T.Kernel() blocks are used, the IR has a SeqStmt
    // containing separate kernel subtrees, each with its own tilelang_root.
    std::vector<Stmt> kernel_stmts;
    Stmt prefix_wrapper;
    bool is_multi_kernel =
        MultiKernelDetector::Detect(func->body, kernel_stmts, prefix_wrapper);

    if (!is_multi_kernel) {
      TilelangRootBodyExtractor extractor;
      extractor(func->body);
      TVM_FFI_CHECK(extractor.body.defined(), ValueError)
          << "AutoSchedule could not find a tilelang_root body";

      auto overrides = extractor.buffer_versions_overrides;
      ScheduledKernelResult result =
          ScheduleSingleKernel(extractor.body, target, overrides,
                               extractor.manual_multi_buffer_versions,
                               extractor.unlimit_memory_scopes);
      Stmt body = SerializeScheduledKernel(func->body, extractor, result);
      return PrimFunc(func->params, body, func->ret_type, func->buffer_map,
                      func->attrs);
    }

    Array<Stmt> combined_stmts;
    for (const Stmt &kernel_subtree : kernel_stmts) {
      TilelangRootBodyExtractor extractor;
      extractor(kernel_subtree);
      Stmt serialized_subtree = kernel_subtree;
      if (extractor.body.defined()) {
        auto overrides = extractor.buffer_versions_overrides;
        ScheduledKernelResult result =
            ScheduleSingleKernel(extractor.body, target, overrides,
                                 extractor.manual_multi_buffer_versions,
                                 extractor.unlimit_memory_scopes);
        serialized_subtree =
            SerializeScheduledKernel(kernel_subtree, extractor, result);
      }
      combined_stmts.push_back(serialized_subtree);
    }

    Stmt new_inner = combined_stmts.size() == 1 ? combined_stmts[0]
                                                : Stmt(SeqStmt(combined_stmts));
    InnerSeqStmtReplacer seq_replacer(new_inner);
    Stmt body = seq_replacer(func->body);
    return PrimFunc(func->params, body, func->ret_type, func->buffer_map,
                    func->attrs);
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

Layout ExpandLayoutToBuffer(const Buffer &layout_buffer,
                            const Buffer &target_buffer, const Layout &layout) {
  if (!layout.defined() ||
      target_buffer->shape.size() < layout_buffer->shape.size()) {
    return Layout();
  }
  size_t leading_ndim =
      target_buffer->shape.size() - layout_buffer->shape.size();
  if (leading_ndim == 0) {
    return layout;
  }
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(target_buffer->shape[i]);
  }
  return layout->Expand(leading_shape);
}

Layout FindLayoutForBuffer(const Map<Buffer, Layout> &layout_map,
                           const Buffer &buffer) {
  if (layout_map.count(buffer)) {
    return layout_map[buffer];
  }
  for (const auto &[layout_buffer, layout] : layout_map) {
    if (layout_buffer->data.same_as(buffer->data) ||
        layout_buffer->name == buffer->name) {
      Layout expanded = ExpandLayoutToBuffer(layout_buffer, buffer, layout);
      if (expanded.defined()) {
        return expanded;
      }
    }
  }
  return Layout();
}

bool AddLayoutForBuffer(Map<Buffer, Layout> *layout_map, const Buffer &buffer) {
  if (!layout_map || layout_map->count(buffer)) {
    return false;
  }
  Layout layout = FindLayoutForBuffer(*layout_map, buffer);
  if (!layout.defined()) {
    return false;
  }
  layout_map->Set(buffer, layout);
  return true;
}

// StmtMutator to rewrite alloc_buffers in SBlock nodes for double buffering
class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    bool needs_update = false;
    bool layout_update = false;
    Array<Buffer> new_alloc_buffers;
    Map<String, Any> new_annotations = op->annotations;
    bool has_layout_map = false;
    Map<Buffer, Layout> layout_map;
    if (auto layout_map_ref = op->annotations.Get(attr::kLayoutMap)) {
      if (auto map = layout_map_ref.value().as<Map<Buffer, Layout>>()) {
        layout_map = map.value();
        has_layout_map = true;
      }
    }
    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }
    if (has_layout_map) {
      for (const auto &buffer : new_alloc_buffers) {
        layout_update |= AddLayoutForBuffer(&layout_map, buffer);
      }
    }
    if (layout_update) {
      new_annotations.Set(attr::kLayoutMap, layout_map);
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
    }
    if (layout_update) {
      new_block->annotations = new_annotations;
    }
    return Stmt(new_block);
  }

  const std::vector<MultiVersionBufferInfo> &buffer_infos_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (buffer_infos.empty()) {
    return stmt;
  }
  AllocBufferRewriter rewriter(buffer_infos);
  return rewriter(stmt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
