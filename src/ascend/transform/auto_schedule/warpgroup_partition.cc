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
 * \file warpgroup_partition.cc
 * \brief Warpgroup partition and IRStructure-to-Stmt conversion for TileLang
 * AutoSchedule
 */

#include "./warpgroup_partition.h"

#include "../auto_schedule.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
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
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/task_analysis.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/region.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

class LetStmtVarRenamer : public StmtExprMutator {
public:
  explicit LetStmtVarRenamer(const std::string &suffix) : suffix_(suffix) {}

  Stmt Rename(Stmt stmt) {
    CollectLetVars(stmt);
    if (var_remap_.empty())
      return stmt;
    return VisitStmt(stmt);
  }

private:
  void CollectLetVars(const Stmt &stmt) {
    class Collector : public StmtExprVisitor {
    public:
      explicit Collector(Map<Var, Var> &remap, const std::string &suffix)
          : remap_(remap), suffix_(suffix) {}
      void VisitStmt_(const BindNode *op) final {
        remap_.Set(op->var, op->var.copy_with_suffix(suffix_));
        StmtExprVisitor::VisitExpr(op->value);
      }
      Map<Var, Var> &remap_;
      const std::string &suffix_;
    };
    Collector c(var_remap_, suffix_);
    c(stmt);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    auto it = var_remap_.find(op->var);
    Var new_var = it != var_remap_.end() ? (*it).second : op->var;
    PrimExpr new_value = VisitExpr(op->value);
    return Bind(new_var, new_value, op->span);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) {
      return (*it).second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::string suffix_;
  Map<Var, Var> var_remap_;
};

static Stmt RenameLetStmtVars(Stmt stmt, const std::string &suffix) {
  return LetStmtVarRenamer(suffix).Rename(std::move(stmt));
}

// Like tirx::Substitute, but also remaps a buffer's backing `data` var at
// *use* sites. The generic IRSubstitute only rewrites buffer->data at
// definition sites (VisitBufferDef); a buffer produced by T.make_tensor is
// backed by a LetStmt-bound var and only appears as a use inside the scheduled
// region (e.g. the regions of a T.copy), so renaming that let var via the
// generic Substitute leaves the buffer pointing at the old, now-undefined var.
class LetVarSubstituter : public StmtExprMutator {
public:
  explicit LetVarSubstituter(const Map<Var, PrimExpr> &vmap) : vmap_(vmap) {}

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = vmap_.find(GetRef<Var>(op));
    return it != vmap_.end() ? (*it).second : StmtExprMutator::VisitExpr_(op);
  }

  Buffer VisitBufferUse(const Buffer &buffer) final {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end())
      return (*it).second;
    auto vit = vmap_.find(buffer->data);
    if (vit == vmap_.end())
      return buffer;
    Buffer new_buf = buffer;
    new_buf.CopyOnWrite()->data = Downcast<Var>((*vit).second);
    buffer_remap_[buffer] = new_buf;
    return new_buf;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (FindAutoScheduleAttr(op->attr_key) != nullptr) {
      Any new_node = op->node;
      bool node_changed = false;
      if (const auto *expr_node = op->node.as<PrimExprNode>()) {
        PrimExpr old_node = GetRef<PrimExpr>(expr_node);
        PrimExpr rewritten_node = VisitExpr(old_node);
        node_changed = !rewritten_node.same_as(old_node);
        new_node = std::move(rewritten_node);
      }
      PrimExpr new_value = VisitExpr(op->value);
      Stmt new_body = VisitStmt(op->body);
      if (!node_changed && new_value.same_as(op->value) &&
          new_body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return AttrStmt(new_node, op->attr_key, new_value, new_body, op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  const Map<Var, PrimExpr> &vmap_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

static Stmt SubstituteLetVars(Stmt stmt, const Map<Var, PrimExpr> &vmap) {
  if (vmap.empty())
    return stmt;
  return LetVarSubstituter(vmap)(std::move(stmt));
}

static PrimExpr SubstituteLetVarsExpr(PrimExpr expr,
                                      const Map<Var, PrimExpr> &vmap) {
  if (vmap.empty())
    return expr;
  return LetVarSubstituter(vmap)(std::move(expr));
}

// Copy a node's guards (If conditions + supported AttrStmt wrappers) onto its
// clone, applying the let-var remap so renamed warpgroup-local vars referenced
// inside a guard stay defined.
static void CloneGuards(const IRStructure *src, IRStructure *dst,
                        const Map<Var, PrimExpr> &var_remap) {
  GuardList remapped;
  remapped.reserve(src->GetGuards().size());
  for (const auto &guard : src->GetGuards()) {
    if (guard->IsCondition()) {
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      PrimExpr condition =
          SubstituteLetVarsExpr(condition_guard->condition, var_remap);
      remapped.push_back(
          std::make_unique<ConditionGuard>(std::move(condition)));
    } else {
      const auto *attribute_guard =
          static_cast<const AttributeGuard *>(guard.get());
      Any node = attribute_guard->node;
      if (const auto *expr_node = node.as<PrimExprNode>()) {
        node = SubstituteLetVarsExpr(GetRef<PrimExpr>(expr_node), var_remap);
      }
      PrimExpr value = attribute_guard->value;
      if (value.defined()) {
        value = SubstituteLetVarsExpr(std::move(value), var_remap);
      }
      remapped.push_back(std::make_unique<AttributeGuard>(
          std::move(node), attribute_guard->key, std::move(value),
          attribute_guard->span));
    }
  }
  dst->SetGuards(std::move(remapped));
}

static void CloneSchedulingMetadata(const IRStructure *src, IRStructure *dst,
                                    int warpgroup_id,
                                    const Map<Var, PrimExpr> &var_remap) {
  CloneGuards(src, dst, var_remap);
  dst->SetStage(src->GetStage());
  dst->SetStartTime(src->GetStartTime());
  auto copy_stmt_list = [&](const std::vector<Stmt> &src_stmts,
                            std::vector<Stmt> *dst_stmts) {
    dst_stmts->clear();
    dst_stmts->reserve(src_stmts.size());
    for (const auto &stmt : src_stmts) {
      dst_stmts->push_back(SubstituteLetVars(stmt, var_remap));
    }
  };
  if (auto it = src->before.find(warpgroup_id); it != src->before.end()) {
    copy_stmt_list(it->second, &dst->before[warpgroup_id]);
  }
  if (auto it = src->after.find(warpgroup_id); it != src->after.end()) {
    copy_stmt_list(it->second, &dst->after[warpgroup_id]);
  }
}

bool IsLetDeclTask(const TaskNode *task) {
  return task->stmt.as<BindNode>() != nullptr;
}

// A LetDecl whose bound value is a handle/pointer.
bool IsHandleLetDeclTask(const TaskNode *task) {
  const auto *bind = task->stmt.as<BindNode>();
  return bind != nullptr && bind->value.dtype().is_handle();
}

// Helper: check if an IRStructure node is a LetDecl task.
bool IsLetDeclNode(const IRStructure *node) {
  if (!node)
    return false;
  if (node->IsTask()) {
    return IsLetDeclTask(static_cast<const TaskNode *>(node));
  }
  return false;
}

// Helper: check if an IRStructure subtree contains any LetDecl tasks
bool ContainsLetDecl(const IRStructure *node) {
  if (!node)
    return false;
  if (IsLetDeclNode(node))
    return true;
  if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    for (const auto &child : ctrl->children) {
      if (ContainsLetDecl(child.get()))
        return true;
    }
  }
  return false;
}

// Filter an ordered list of children for a warpgroup. LetDecl tasks are
// always kept (needed by every warp group); other nodes are kept only if they
// participate in `warpgroup_id`. Drops nodes that filter to null.
std::vector<std::shared_ptr<IRStructure>>
CloneIRStructureListWithWarpgroupFilter(
    const std::vector<std::shared_ptr<IRStructure>> &children, int warpgroup_id,
    Map<Var, PrimExpr> &var_remap) {
  std::vector<std::shared_ptr<IRStructure>> out;
  for (const auto &child : children) {
    auto new_child = CloneIRStructureWithWarpgroupFilter(
        child.get(), warpgroup_id, var_remap);
    if (new_child) {
      out.push_back(std::move(new_child));
    }
  }
  return out;
}

// Helper function to clone IRStructure with warpgroup filter.
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap) {
  if (!node)
    return nullptr;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    if (IsLetDeclTask(task)) {
      const auto *bind = task->stmt.as<BindNode>();
      // A LetDecl marked broadcast is replicated onto every warpgroup; give
      // each clone a fresh variable so the two cores use independent names.
      if (IsWarpgroupBroadcast(task->GetWarpgroupId())) {
        auto new_var = bind->var.copy_with_suffix("");
        // Substitute previously renamed variables in the value expression.
        PrimExpr new_value = var_remap.empty()
                                 ? bind->value
                                 : Substitute(bind->value, var_remap);
        var_remap.Set(bind->var, new_var);
        auto new_task = std::make_shared<TaskNode>();
        new_task->stmt = Bind(new_var, new_value);
        CloneSchedulingMetadata(task, new_task.get(), warpgroup_id, var_remap);
        return new_task;
      }
      // A LetDecl assigned a concrete warpgroup only lives on that core (the
      // demand-lattice pass proved all its consumers are there), so filter it
      // like a normal task and keep its original variable name — downstream
      // consumers on this core reference that name and get no remap for it.
      if (!node->containWarpgroupId(warpgroup_id))
        return nullptr;
      PrimExpr new_value =
          var_remap.empty() ? bind->value : Substitute(bind->value, var_remap);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmt = Bind(bind->var, new_value);
      CloneSchedulingMetadata(task, new_task.get(), warpgroup_id, var_remap);
      return new_task;
    }

    // Non-LetDecl tasks: only include if warp group matches
    if (!node->containWarpgroupId(warpgroup_id))
      return nullptr;
    auto cloned = task->Clone();
    // Substitute renamed LetDecl variables in task statements
    if (!var_remap.empty()) {
      auto ct = static_cast<TaskNode *>(cloned.get());
      ct->stmt = SubstituteLetVars(ct->stmt, var_remap);
    }
    CloneSchedulingMetadata(task, cloned.get(), warpgroup_id, var_remap);
    return cloned;
  } else if (node->IsControl()) {
    // A ControlNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks.
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto ctrl = static_cast<ControlNode *>(node);
    auto new_ctrl = std::make_shared<ControlNode>();
    For new_for = ctrl->control;
    auto new_loop_var = ctrl->control->loop_var.copy_with_suffix("");
    new_for.CopyOnWrite()->loop_var = new_loop_var;
    var_remap.Set(ctrl->control->loop_var, new_loop_var);
    new_for.CopyOnWrite()->min = Substitute(ctrl->control->min, var_remap);
    new_for.CopyOnWrite()->extent =
        Substitute(ctrl->control->extent, var_remap);
    if (ctrl->control->step.has_value()) {
      new_for.CopyOnWrite()->step =
          Substitute(ctrl->control->step.value(), var_remap);
    }
    new_ctrl->control = new_for;
    // Clone the task and apply var_remap so each warpgroup gets its own copy
    // with correctly renamed LetDecl variables.
    if (ctrl->task) {
      auto cloned_task =
          std::static_pointer_cast<TaskNode>(ctrl->task->Clone());
      if (!var_remap.empty()) {
        cloned_task->stmt = SubstituteLetVars(cloned_task->stmt, var_remap);
      }
      new_ctrl->task = std::move(cloned_task);
    }
    new_ctrl->children = CloneIRStructureListWithWarpgroupFilter(
        ctrl->children, warpgroup_id, var_remap);
    CloneSchedulingMetadata(node, new_ctrl.get(), warpgroup_id, var_remap);
    return new_ctrl;
  }
  LOG(FATAL);
  return nullptr;
}

// Entry point overload — creates a fresh var_remap and filters the root list.
std::vector<std::shared_ptr<IRStructure>> CloneIRStructureWithWarpgroupFilter(
    const std::vector<std::shared_ptr<IRStructure>> &root, int warpgroup_id) {
  Map<Var, PrimExpr> var_remap;
  auto cloned =
      CloneIRStructureListWithWarpgroupFilter(root, warpgroup_id, var_remap);
  return cloned;
}

// Wrap a single stmt in an if-guard. For SimtVF / SimdVF blocks this leaves
// the block (and its alloc_buffers / launch_thread attrs) intact under the
// IfThenElse — codegen's IfThenElseNode handler then naturally emits the
// `if (cond) { asc_vf_call<...>(...) }` pattern around the call site while
// the helper function definition stays at file scope. This avoids pushing
// the guard into the helper body (which made every thread evaluate the
// per-iteration condition redundantly).
static Stmt GuardWithIf(const Stmt &body, const PrimExpr &condition) {
  return IfThenElse(condition, body);
}

// Re-wrap a body in its guard deltas, innermost-first. A conditional guard
// becomes `if (expr) body`; an attribute guard restores its original AttrStmt.
// Reverse iteration preserves source nesting order: the outermost guard (front
// of the list) ends up as the outermost wrapper.
static Stmt GuardWithGuards(Stmt body, const GuardList &guards) {
  for (auto it = guards.rbegin(); it != guards.rend(); ++it) {
    body = (*it)->Wrap(std::move(body));
  }
  return body;
}

// PerCoreTask is one scheduler node, but its source control flow may contain
// several mutually-exclusive dependency-bearing candidates plus scalar control
// tasks. NormalizeTask marks both kinds; only candidates receive migrated
// synchronization.
class PerCoreTaskSyncMigrator : public StmtMutator {
public:
  PerCoreTaskSyncMigrator(std::vector<Stmt> before, std::vector<Stmt> after,
                          std::function<Stmt(Stmt)> emit_candidate)
      : before_(std::move(before)), after_(std::move(after)),
        emit_candidate_(std::move(emit_candidate)) {}

  Stmt Rewrite(Stmt body) {
    Stmt result = VisitStmt(std::move(body));
    ICHECK_GT(num_candidates_, 0)
        << "Cannot find a T.Task marker in normalized PerCoreTask";
    return result;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kAscendTask) {
      Stmt candidate = VisitStmt(op->body);
      if (!HasTaskBufferAccess(candidate)) {
        return candidate;
      }
      ++num_candidates_;
      return WrapCandidate(emit_candidate_(std::move(candidate)));
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt WrapCandidate(Stmt stmt) const {
    std::vector<Stmt> sequence;
    sequence.reserve(before_.size() + 1 + after_.size());
    sequence.insert(sequence.end(), before_.begin(), before_.end());
    sequence.push_back(std::move(stmt));
    sequence.insert(sequence.end(), after_.begin(), after_.end());
    return SeqStmt::Flatten(sequence);
  }

  std::vector<Stmt> before_;
  std::vector<Stmt> after_;
  std::function<Stmt(Stmt)> emit_candidate_;
  int num_candidates_{0};
};

static bool IsPerCoreTaskNode(const IRStructure *node) {
  return node && node->IsTask() &&
         static_cast<const TaskNode *>(node)->IsPerCoreTask();
}

static std::vector<Stmt>
CollectSchedulingStmts(const std::map<int, std::vector<Stmt>> &by_warpgroup) {
  std::vector<Stmt> result;
  for (const auto &[_, stmts] : by_warpgroup)
    result.insert(result.end(), stmts.begin(), stmts.end());
  return result;
}

// Per-warpgroup state for lowering conditional LetDecl tasks.
struct ConditionalLetDeclCtx {
  std::unordered_map<const TaskNode *, Buffer> buf_of;
  Map<Var, PrimExpr> vmap;
};

static Stmt ConvertIRStructureToStmtInner(IRStructure *structure, Var sid,
                                          const ConditionalLetDeclCtx &ctx);
static Stmt ConvertIRStructureToStmt(IRStructure *structure, Var sid,
                                     const ConditionalLetDeclCtx &ctx);
static Stmt
ConvertIRStructureToStmt(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                         Var sid, const ConditionalLetDeclCtx &ctx);

// Register every guarded scalar LetDecl as a local.var buffer.
static void PrescanConditionalLetDecls(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    ConditionalLetDeclCtx *ctx) {
  for (const auto &child : nodes) {
    IRStructure *node = child.get();
    if (node->IsTask() && IsLetDeclTask(static_cast<const TaskNode *>(node)) &&
        node->HasGuards()) {
      const auto *task = static_cast<const TaskNode *>(node);
      const auto *bind = task->stmt.as<BindNode>();
      // Handle/pointer LetDecls (e.g. a T.make_tensor pointer backing a
      // Buffer's `data`) can't become a size-1 local.var scalar — `data` must
      // stay a Var. Leave them out of the local.var machinery; they are emitted
      // as a plain unconditional Bind below.
      if (IsHandleLetDeclTask(task)) {
        continue;
      }
      Buffer buf =
          decl_buffer({IntImm(DataType::Int(32), 1)}, bind->value.dtype(),
                      bind->var->name_hint, "local.var");
      ctx->buf_of[task] = buf;
      ctx->vmap.Set(bind->var, BufferLoad(buf, {IntImm(DataType::Int(32), 0)}));
    }
    if (node->IsControl()) {
      PrescanConditionalLetDecls(
          static_cast<const ControlNode *>(node)->children, ctx);
    }
  }
}

static Stmt ConvertIRStructureToStmt(IRStructure *structure, Var sid,
                                     const ConditionalLetDeclCtx &ctx) {
  Stmt body = ConvertIRStructureToStmtInner(structure, sid, ctx);
  if (structure && structure->HasGuards()) {
    // A guarded LetDecl is lowered to a mutable local.var scalar:
    //   T v = 0;
    //   if (cond) v = value;          // conditional guard
    //   assume(fact) { v = value; }   // assume guard
    if (structure->IsTask() &&
        IsLetDeclTask(static_cast<const TaskNode *>(structure))) {
      const auto *task = static_cast<const TaskNode *>(structure);
      const auto *bind = task->stmt.as<BindNode>();
      if (IsHandleLetDeclTask(task)) {
        return body;
      }
      auto it = ctx.buf_of.find(task);
      ICHECK(it != ctx.buf_of.end())
          << "Guarded LetDecl not registered by prescan: " << bind->var;
      Buffer buf = it->second;
      Map<String, Any> annotations;
      annotations.Set(tl::attr::kLocalVarInit, make_zero(buf->dtype));
      Stmt alloc = AllocBuffer(buf, annotations);
      // Emit one nested guard per delta (outer..inner) rather than ANDing
      // conditions into a single predicate, matching the guarding of
      // non-LetDecl bodies below.
      Stmt store = GuardWithGuards(
          BufferStore(buf, bind->value, {IntImm(DataType::Int(32), 0)}),
          structure->GetGuards());
      return SeqStmt({alloc, store});
    }
    body = GuardWithGuards(body, structure->GetGuards());
  }
  return body;
}

// Top-level entry for emitting one warpgroup body
Stmt EmitWarpgroupBody(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                       Var sid) {
  ConditionalLetDeclCtx ctx;
  PrescanConditionalLetDecls(nodes, &ctx);
  Stmt body = ConvertIRStructureToStmt(nodes, sid, ctx);
  return SubstituteLetVars(body, ctx.vmap);
}

// Emit an ordered list of scheduled nodes into a single flattened SeqStmt,
// interleaving each node's `before` / body / `after` sync statements. This is
// the root-level emission (and any non-pipelined loop body), replacing the old
// SequenceNode branch.
static Stmt
ConvertIRStructureToStmt(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                         Var sid, const ConditionalLetDeclCtx &ctx) {
  std::vector<Stmt> stmts;
  for (const auto &child : nodes) {
    auto node = child.get();
    if (!IsPerCoreTaskNode(node)) {
      for (auto &[_, before] : node->before) {
        for (auto &stmt : before) {
          stmts.push_back(stmt);
        }
      }
    }
    stmts.push_back(ConvertIRStructureToStmt(node, sid, ctx));
    if (!IsPerCoreTaskNode(node)) {
      for (auto &[_, after] : node->after) {
        for (auto &stmt : after) {
          stmts.push_back(stmt);
        }
      }
    }
  }
  return SeqStmt::Flatten(stmts);
}

static Stmt ConvertIRStructureToStmtInner(IRStructure *structure, Var sid,
                                          const ConditionalLetDeclCtx &ctx) {
  if (!structure) {
    return Evaluate(0);
  }

  if (structure->IsTask()) {
    auto task = static_cast<TaskNode *>(structure);
    if (!task->stmt.defined()) {
      return Evaluate(0);
    }
    if (task->IsPerCoreTask()) {
      auto emit_candidate = [&](Stmt candidate) {
        auto candidate_task = std::make_shared<TaskNode>();
        candidate_task->stmt = std::move(candidate);
        return ConvertIRStructureToStmtInner(candidate_task.get(), sid, ctx);
      };
      return PerCoreTaskSyncMigrator(CollectSchedulingStmts(task->before),
                                     CollectSchedulingStmts(task->after),
                                     std::move(emit_candidate))
          .Rewrite(task->stmt);
    }
    // Check for dual_copy: rewrite to copies with sid-based offset
    {
      if (auto *eval = task->stmt.as<EvaluateNode>()) {
        if (auto *call = eval->value.as<CallNode>()) {
          static const auto copy_op = Op::Get("tl.tileop.copy");
          static const auto post_copy_op = Op::Get("tl.ascend_nd2nz_post_copy");
          static const auto access_ptr_op = Op::Get("tl.access_ptr");
          if (call->op.same_as(post_copy_op) && call->args.size() == 8) {
            const auto *split_imm = call->args[6].as<IntImmNode>();
            ICHECK(split_imm)
                << "ascend_nd2nz_post_copy dual split_dim must be "
                   "an IntImm (0=M, 1=N along trailing-2 axes)";
            int split_dim_neg2 = split_imm->value;
            ICHECK(split_dim_neg2 == 0 || split_dim_neg2 == 1)
                << "ascend_nd2nz_post_copy dual split_dim must be 0 or 1, got "
                << split_dim_neg2;
            PrimExpr half_extent = call->args[7];
            // args[0] = tl.access_ptr(BufferLoad(buf, mins), extent, rw_mask)
            const auto *dst_ap = call->args[0].as<CallNode>();
            ICHECK(dst_ap && dst_ap->op.same_as(access_ptr_op))
                << "ascend_nd2nz_post_copy dst arg must be a tl.access_ptr "
                   "Call; got "
                << call->args[0];
            ICHECK_EQ(dst_ap->args.size(), 3);
            const auto *dst_load = dst_ap->args[0].as<BufferLoadNode>();
            ICHECK(dst_load) << "tl.access_ptr first arg must be a BufferLoad";
            int ndim = static_cast<int>(dst_load->indices.size());
            ICHECK_GE(ndim, 2)
                << "nd2nz dst BufferLoad must have at least 2 indices, got "
                << ndim;
            int split_axis = ndim - 2 + split_dim_neg2;
            Array<PrimExpr> new_indices;
            new_indices.reserve(ndim);
            for (int d = 0; d < ndim; ++d) {
              if (d == split_axis) {
                new_indices.push_back(dst_load->indices[d] + sid * half_extent);
              } else {
                new_indices.push_back(dst_load->indices[d]);
              }
            }
            BufferLoad new_dst_load(dst_load->buffer, new_indices);
            Call new_dst_ap(dst_ap->dtype, access_ptr_op,
                            {new_dst_load, dst_ap->args[1], dst_ap->args[2]});
            Array<PrimExpr> new_args = {new_dst_ap};
            for (size_t i = 1; i < 6; ++i) {
              new_args.push_back(call->args[i]);
            }
            Call new_call(call->dtype, post_copy_op, new_args);
            return Evaluate(new_call);
          }
          if (call->op.same_as(copy_op)) {
            Copy cp(call->args, call->annotations);
            if (cp->annotations.Get("double")) {
              int dual_ctl = cp->GetDualDstCtl();
              auto &src = cp->src;
              auto &dst = cp->dst;
              auto &src_r = cp->src_range;
              auto &dst_r = cp->dst_range;
              // Use last-2-dims relative indexing so that leading
              // dimensions (e.g. buffer version) don't shift the split
              // axis (matching the nd2nz path above).
              int src_split_dim =
                  static_cast<int>(src_r.size()) - 2 + (dual_ctl == 1 ? 0 : 1);
              int dst_split_dim =
                  static_cast<int>(dst_r.size()) - 2 + (dual_ctl == 1 ? 0 : 1);
              // Remove dual_dst_ctl, unit_flag_ctl, and double from annotations
              Map<String, ObjectRef> new_ann;
              for (auto &kv : call->annotations) {
                if (kv.first != "dual_dst_ctl" && kv.first != "unit_flag_ctl" &&
                    kv.first != "double")
                  new_ann.Set(kv.first, kv.second);
              }
              // Use RegionOp to get buffer/range info from the Call args
              auto src_call = Downcast<Call>(call->args[0]);
              auto dst_call = Downcast<Call>(call->args[1]);
              RegionOp dst_rop(dst_call->args);
              int dst_mask = dst_rop->GetAccessMask();
              auto dst_ranges = dst_rop->GetRanges();
              auto dst_buf = dst_rop->GetBuffer();
              // Single copy with sid-based offset: offset = sid * half
              PrimExpr off = sid * src_r[src_split_dim]->extent;
              // Build new dst region args
              Array<PrimExpr> dst_indices;
              for (size_t d = 0; d < dst_ranges.size(); ++d) {
                if (static_cast<int>(d) == dst_split_dim)
                  dst_indices.push_back(dst_ranges[d]->min + off);
                else
                  dst_indices.push_back(dst_ranges[d]->min);
              }
              auto dst_load = BufferLoad(dst_buf, dst_indices);
              Array<PrimExpr> dst_args = {dst_load, Integer(dst_mask)};
              for (size_t d = 0; d < dst_ranges.size(); ++d) {
                if (static_cast<int>(d) == dst_split_dim)
                  dst_args.push_back(src_r[src_split_dim]->extent);
                else
                  dst_args.push_back(dst_ranges[d]->extent);
              }
              auto new_dst =
                  Call(DataType::Handle(), RegionOp::Get(), dst_args);
              Array<PrimExpr> new_args = {src_call, new_dst};
              auto new_call = Call(call->dtype, copy_op, new_args, new_ann);
              return Evaluate(new_call);
            } else if (IsGlobalBuffer(cp->src) &&
                       cp->annotations.Get("dual_dst_ctl")) {
              int dual_ctl = cp->GetDualDstCtl();
              auto &src_r = cp->src_range;
              int src_split_dim =
                  static_cast<int>(src_r.size()) - 2 + (dual_ctl == 1 ? 0 : 1);
              PrimExpr half_ext =
                  floordiv(src_r[src_split_dim]->extent, Integer(2));
              // Strip dual_dst_ctl and unit_flag_ctl from annotations
              Map<String, ObjectRef> new_ann;
              for (auto &kv : call->annotations) {
                if (kv.first != "dual_dst_ctl" && kv.first != "unit_flag_ctl")
                  new_ann.Set(kv.first, kv.second);
              }
              PrimExpr off = sid * half_ext;
              auto src_call = Downcast<Call>(call->args[0]);
              auto dst_call = Downcast<Call>(call->args[1]);
              // Rebuild src (GM) region: half extent, sid-based offset
              RegionOp src_rop(src_call->args);
              int src_mask = src_rop->GetAccessMask();
              auto src_ranges = src_rop->GetRanges();
              auto src_buf = src_rop->GetBuffer();
              Array<PrimExpr> src_indices;
              for (size_t d = 0; d < src_ranges.size(); ++d) {
                if (static_cast<int>(d) == src_split_dim)
                  src_indices.push_back(src_ranges[d]->min + off);
                else
                  src_indices.push_back(src_ranges[d]->min);
              }
              auto src_load = BufferLoad(src_buf, src_indices);
              Array<PrimExpr> src_args = {src_load, Integer(src_mask)};
              for (size_t d = 0; d < src_ranges.size(); ++d) {
                if (static_cast<int>(d) == src_split_dim)
                  src_args.push_back(half_ext);
                else
                  src_args.push_back(src_ranges[d]->extent);
              }
              auto new_src =
                  Call(DataType::Handle(), RegionOp::Get(), src_args);
              Array<PrimExpr> new_args = {new_src, dst_call};
              auto new_call = Call(call->dtype, copy_op, new_args, new_ann);
              return Evaluate(new_call);
            }
          }
        }
      }
      return task->stmt;
    }
  } else if (structure->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(structure);
    Var loop_var = ctrl->control->loop_var;
    PrimExpr loop_start = ctrl->control->min;
    PrimExpr loop_extent = ctrl->control->extent;
    PrimExpr loop_step = ctrl->control->step.has_value()
                             ? ctrl->control->step.value()
                             : IntImm(DataType::Int(32), 1);
    int min_stages = 100, max_stages = -1;
    for (auto &child : ctrl->children) {
      auto node = child.get();
      min_stages = std::min(min_stages, node->GetStage());
      max_stages = std::max(max_stages, node->GetStage());
    }
    // Stage-aware prologue/epilogue emission: substitute loop_var and wrap
    // each task in an `if (cond)` guard so that within a single extended
    // loop the right stages run for each iteration. Separate unrolled
    // prologue/epilogue blocks are disabled for now (enable_pro/enable_epi
    // forced to false).
    if (min_stages == max_stages) {
      Stmt body = ConvertIRStructureToStmt(ctrl->children, sid, ctx);
      // Filter out scheduling-only annotations now that AutoSchedule is done.
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      filtered_annotations.erase("multi_buffer_eligible");
      filtered_annotations.erase("no_conflict");
      return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                 ctrl->control->thread_binding, filtered_annotations);
    }
    // Stage-aware single-loop body: substitute loop_var per task and wrap
    // each task in a stage-appropriate `if (cond)` guard. Separate
    // prologue/epilogue unrolled blocks are intentionally NOT emitted.
    std::vector<Stmt> steady;
    for (auto &child : ctrl->children) {
      auto node = child.get();
      Map<Var, PrimExpr> substitution, substitution_cond;
      substitution.Set(loop_var,
                       loop_var - loop_step * (max_stages - node->GetStage()));
      substitution_cond.Set(
          loop_var,
          Max(loop_start,
              Min(loop_start + loop_extent - loop_step,
                  loop_var - loop_step * (max_stages - node->GetStage()))));
      PrimExpr condition =
          And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
      if (node->GetStage() == min_stages) {
        condition = loop_var >= loop_start;
      }
      if (node->GetStage() == max_stages) {
        condition = loop_var < loop_start + loop_extent;
      }
      if (IsLetDeclNode(node)) {
        std::vector<Stmt> guarded;
        for (const auto &[_, before] : node->before) {
          for (const auto &stmt : before) {
            guarded.push_back(
                Substitute(IfThenElse(condition, stmt), substitution));
          }
        }
        guarded.push_back(Substitute(ConvertIRStructureToStmt(node, sid, ctx),
                                     substitution_cond));
        for (const auto &[_, after] : node->after) {
          for (const auto &stmt : after) {
            guarded.push_back(
                Substitute(IfThenElse(condition, stmt), substitution));
          }
        }
        steady.push_back(SeqStmt::Flatten(guarded));
      } else {
        std::vector<Stmt> stmts;
        if (!IsPerCoreTaskNode(node)) {
          for (const auto &[_, before] : node->before) {
            for (const auto &stmt : before) {
              stmts.push_back(GuardWithIf(stmt, condition));
            }
          }
        }
        stmts.push_back(
            GuardWithIf(ConvertIRStructureToStmt(node, sid, ctx), condition));
        if (!IsPerCoreTaskNode(node)) {
          for (const auto &[_, after] : node->after) {
            for (const auto &stmt : after) {
              stmts.push_back(GuardWithIf(stmt, condition));
            }
          }
        }
        Stmt stmt = SeqStmt::Flatten(stmts);
        steady.push_back(Substitute(stmt, substitution));
      }
    }
    Stmt new_body = SeqStmt::Flatten(steady);
    auto new_var = loop_var.copy_with_suffix("");
    // Filter out scheduling-only annotations now that AutoSchedule is done.
    Map<String, Any> filtered_annotations = ctrl->control->annotations;
    filtered_annotations.erase("num_stages");
    filtered_annotations.erase("multi_buffer_eligible");
    filtered_annotations.erase("no_conflict");
    Map<Var, PrimExpr> sub_var;
    sub_var.Set(loop_var, new_var);
    For for_op =
        For(new_var, loop_start,
            ctrl->control->extent + loop_step * (max_stages - min_stages),
            ctrl->control->kind, Substitute(new_body, sub_var),
            ctrl->control->thread_binding, filtered_annotations);
    return for_op;
  }

  LOG(FATAL)
      << "Failed to convert IRStructure to Stmt, returning empty statement";
  return Evaluate(0);
}
} // namespace tl
} // namespace tvm
