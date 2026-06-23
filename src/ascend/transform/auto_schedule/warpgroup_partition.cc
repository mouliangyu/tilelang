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
#include <unordered_set>
#include <utility>
#include <vector>

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

bool IsLetDeclTask(const TaskNode *task) {
  return task->stmts.size() == 1 && task->stmts[0].as<BindNode>() != nullptr;
}

// Helper: check if an IRStructure node is a LetDecl task (or a ScheduleUnit
// wrapping one)
bool IsLetDeclNode(const IRStructure *node) {
  if (!node)
    return false;
  if (node->IsTask()) {
    return IsLetDeclTask(static_cast<const TaskNode *>(node));
  }
  if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return unit->child && unit->child->IsTask() &&
           IsLetDeclTask(static_cast<const TaskNode *>(unit->child.get()));
  }
  return false;
}

// Helper: check if an IRStructure subtree contains any LetDecl tasks
bool ContainsLetDecl(const IRStructure *node) {
  if (!node)
    return false;
  if (IsLetDeclNode(node))
    return true;
  if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      if (ContainsLetDecl(child.get()))
        return true;
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    return ContainsLetDecl(ctrl->child.get());
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    return ContainsLetDecl(wrapper->child.get());
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return ContainsLetDecl(unit->child.get());
  } else if (node->IsIf()) {
    auto if_node = static_cast<const IfNode *>(node);
    if (ContainsLetDecl(if_node->then_child.get()))
      return true;
    if (if_node->else_child && ContainsLetDecl(if_node->else_child.get()))
      return true;
  }
  return false;
}

// Helper function to clone IRStructure with warpgroup filter.
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap) {
  if (!node)
    return nullptr;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // LetDecl tasks are always included in every warp group clone.
    // Create a fresh variable copy so the two warp groups use different names.
    if (IsLetDeclTask(task)) {
      const auto *bind = task->stmts[0].as<BindNode>();
      auto new_var = bind->var.copy_with_suffix("");
      // Substitute previously renamed variables in the value expression.
      PrimExpr new_value =
          var_remap.empty() ? bind->value : Substitute(bind->value, var_remap);
      var_remap.Set(bind->var, new_var);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmts.push_back(Bind(new_var, new_value));
      return new_task;
    }

    // Non-LetDecl tasks: only include if warp group matches
    if (!node->containWarpgroupId(warpgroup_id))
      return nullptr;
    auto cloned = task->Clone();
    // Substitute renamed LetDecl variables in task statements
    if (!var_remap.empty()) {
      auto ct = static_cast<TaskNode *>(cloned.get());
      for (size_t i = 0; i < ct->stmts.size(); ++i) {
        ct->stmts[i] = SubstituteLetVars(ct->stmts[i], var_remap);
      }
    }
    return cloned;
  } else if (node->IsSequence()) {
    // A SequenceNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks (which are always needed).
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto seq = static_cast<SequenceNode *>(node);
    auto new_seq = std::make_shared<SequenceNode>();
    for (const auto &child : seq->children) {
      auto new_child = CloneIRStructureWithWarpgroupFilter(
          child.get(), warpgroup_id, var_remap);
      if (new_child) {
        new_seq->children.push_back(std::move(new_child));
      }
    }
    if (new_seq->children.empty())
      return nullptr;
    return new_seq;
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
        for (size_t i = 0; i < cloned_task->stmts.size(); ++i) {
          cloned_task->stmts[i] =
              SubstituteLetVars(cloned_task->stmts[i], var_remap);
        }
      }
      new_ctrl->task = std::move(cloned_task);
    }
    new_ctrl->child = CloneIRStructureWithWarpgroupFilter(
        ctrl->child.get(), warpgroup_id, var_remap);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_shared<WrapperNode>();
    // Apply var_remap to the wrapper statement so that renamed LetDecl
    // variables are correctly substituted in Bind values / AttrStmt values.
    new_wrapper->wrapper = var_remap.empty()
                               ? wrapper->wrapper
                               : Substitute(wrapper->wrapper, var_remap);
    new_wrapper->child = CloneIRStructureWithWarpgroupFilter(
        wrapper->child.get(), warpgroup_id, var_remap);
    return new_wrapper;
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    bool child_is_let_decl = IsLetDeclNode(unit->child.get());

    // Include the ScheduleUnit if the child is a LetDecl or the warp group
    // matches.
    if (!child_is_let_decl && !node->containWarpgroupId(warpgroup_id))
      return nullptr;

    auto new_unit = std::make_shared<ScheduleUnit>();
    new_unit->child = CloneIRStructureWithWarpgroupFilter(
        unit->child.get(), warpgroup_id, var_remap);
    new_unit->stage = unit->stage;

    if (!child_is_let_decl) {
      // Copy before/after for the target warp group
      new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
      new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
      // Substitute renamed LetDecl variables in before/after stmts
      if (!var_remap.empty()) {
        for (auto &s : new_unit->before[warpgroup_id]) {
          s = SubstituteLetVars(s, var_remap);
        }
        for (auto &s : new_unit->after[warpgroup_id]) {
          s = SubstituteLetVars(s, var_remap);
        }
      }
    }
    return new_unit;
  } else if (node->IsIf()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto if_node = static_cast<IfNode *>(node);
    auto new_if = std::make_shared<IfNode>();
    new_if->condition = var_remap.empty()
                            ? if_node->condition
                            : Substitute(if_node->condition, var_remap);
    if (if_node->task) {
      auto cloned_task =
          std::static_pointer_cast<TaskNode>(if_node->task->Clone());
      if (!var_remap.empty()) {
        for (size_t i = 0; i < cloned_task->stmts.size(); ++i) {
          cloned_task->stmts[i] =
              SubstituteLetVars(cloned_task->stmts[i], var_remap);
        }
      }
      new_if->task = std::move(cloned_task);
    }
    new_if->then_child = CloneIRStructureWithWarpgroupFilter(
        if_node->then_child.get(), warpgroup_id, var_remap);
    if (if_node->else_child) {
      new_if->else_child = CloneIRStructureWithWarpgroupFilter(
          if_node->else_child.get(), warpgroup_id, var_remap);
    }
    // Return nullptr if both branches are empty
    if (!new_if->then_child && !new_if->else_child)
      return nullptr;
    return new_if;
  }
  LOG(FATAL);
  return nullptr;
}

// Entry point overload — creates a fresh var_remap per call
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id) {
  Map<Var, PrimExpr> var_remap;
  return CloneIRStructureWithWarpgroupFilter(node, warpgroup_id, var_remap);
}

std::shared_ptr<IRStructure>
RemoveUnusedLetDecls(std::shared_ptr<IRStructure> root) {
  if (!root)
    return nullptr;

  // Phase 1: Collect LetDecl definitions and variable references from
  // non-LetDecl nodes (task stmts and ScheduleUnit before/after).
  struct LetDeclEntry {
    const VarNode *var;
    PrimExpr value;
  };
  std::vector<LetDeclEntry> let_decls;
  std::unordered_set<const VarNode *> referenced_vars;

  std::function<void(const IRStructure *)> collect =
      [&](const IRStructure *node) {
        if (!node)
          return;
        if (node->IsTask()) {
          auto task = static_cast<const TaskNode *>(node);
          if (IsLetDeclTask(task)) {
            const auto *bind = task->stmts[0].as<BindNode>();
            let_decls.push_back({bind->var.get(), bind->value});
          } else {
            VarRefCollector collector;
            for (const auto &stmt : task->stmts) {
              collector(stmt);
            }
            referenced_vars.insert(collector.vars.begin(),
                                   collector.vars.end());
          }
        } else if (node->IsSequence()) {
          for (const auto &child :
               static_cast<const SequenceNode *>(node)->children) {
            collect(child.get());
          }
        } else if (node->IsControl()) {
          auto ctrl = static_cast<const ControlNode *>(node);
          collect(ctrl->task.get());
          collect(ctrl->child.get());
          // Also collect variable references from the For loop bounds
          // (min, extent, step) so their LetDecls are not removed.
          VarRefCollector for_collector;
          for_collector(ctrl->control->min);
          for_collector(ctrl->control->extent);
          if (ctrl->control->step.has_value()) {
            for_collector(ctrl->control->step.value());
          }
          referenced_vars.insert(for_collector.vars.begin(),
                                 for_collector.vars.end());
        } else if (node->IsWrapper()) {
          auto wrapper = static_cast<const WrapperNode *>(node);
          collect(wrapper->task.get());
          collect(wrapper->child.get());
          // Also collect variable references from the wrapper statement
          // (Bind value / AttrStmt value) so their LetDecls are not removed.
          VarRefCollector wrapper_collector;
          wrapper_collector(wrapper->wrapper);
          referenced_vars.insert(wrapper_collector.vars.begin(),
                                 wrapper_collector.vars.end());
        } else if (node->IsScheduleUnit()) {
          auto unit = static_cast<const ScheduleUnit *>(node);
          collect(unit->child.get());
          VarRefCollector collector;
          for (const auto &[_, stmts] : unit->before) {
            for (const auto &s : stmts)
              collector(s);
          }
          for (const auto &[_, stmts] : unit->after) {
            for (const auto &s : stmts)
              collector(s);
          }
          referenced_vars.insert(collector.vars.begin(), collector.vars.end());
        } else if (node->IsIf()) {
          auto if_node = static_cast<const IfNode *>(node);
          collect(if_node->task.get());
          collect(if_node->then_child.get());
          if (if_node->else_child) {
            collect(if_node->else_child.get());
          }
          // Collect variable references from the condition
          VarRefCollector cond_collector;
          cond_collector(if_node->condition);
          referenced_vars.insert(cond_collector.vars.begin(),
                                 cond_collector.vars.end());
        }
      };
  collect(root.get());

  // Phase 2: Transitive closure — if a LetDecl var is referenced,
  // all vars in its value expression are transitively referenced too.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &entry : let_decls) {
      if (referenced_vars.count(entry.var)) {
        VarRefCollector collector;
        collector(entry.value);
        for (const auto *v : collector.vars) {
          if (!referenced_vars.count(v)) {
            referenced_vars.insert(v);
            changed = true;
          }
        }
      }
    }
  }

  // Phase 3: Filter the tree — remove LetDecl tasks for unused vars.
  std::function<std::shared_ptr<IRStructure>(
      const std::shared_ptr<IRStructure> &)>
      filter_tree = [&](const std::shared_ptr<IRStructure> &node)
      -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;
    if (node->IsTask()) {
      if (IsLetDeclTask(static_cast<const TaskNode *>(node.get()))) {
        const auto *bind =
            static_cast<const TaskNode *>(node.get())->stmts[0].as<BindNode>();
        if (!referenced_vars.count(bind->var.get())) {
          return nullptr; // Remove unused LetDecl
        }
      }
      return node;
    } else if (node->IsSequence()) {
      auto seq = static_cast<const SequenceNode *>(node.get());
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        auto filtered = filter_tree(child);
        if (filtered)
          new_seq->children.push_back(std::move(filtered));
      }
      if (new_seq->children.empty())
        return nullptr;
      return new_seq;
    } else if (node->IsControl()) {
      auto ctrl = static_cast<const ControlNode *>(node.get());
      auto new_ctrl = std::make_shared<ControlNode>();
      new_ctrl->control = ctrl->control;
      new_ctrl->task = ctrl->task;
      new_ctrl->child = filter_tree(ctrl->child);
      if (!new_ctrl->child)
        return nullptr;
      return new_ctrl;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<const WrapperNode *>(node.get());
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->wrapper = wrapper->wrapper;
      new_wrapper->task = wrapper->task;
      new_wrapper->child = filter_tree(wrapper->child);
      return new_wrapper;
    } else if (node->IsScheduleUnit()) {
      auto unit = static_cast<const ScheduleUnit *>(node.get());
      auto new_unit = std::make_shared<ScheduleUnit>();
      new_unit->before = unit->before;
      new_unit->after = unit->after;
      new_unit->stage = unit->stage;
      new_unit->child = filter_tree(unit->child);
      if (!new_unit->child)
        return nullptr;
      return new_unit;
    } else if (node->IsIf()) {
      auto if_node = static_cast<const IfNode *>(node.get());
      auto new_if = std::make_shared<IfNode>();
      new_if->condition = if_node->condition;
      new_if->task = if_node->task;
      new_if->then_child = filter_tree(if_node->then_child);
      if (if_node->else_child) {
        new_if->else_child = filter_tree(if_node->else_child);
      }
      return new_if;
    }
    return node;
  };

  return filter_tree(root);
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

Stmt ConvertIRStructureToStmt(IRStructure *structure, Var sid) {
  if (!structure) {
    return Evaluate(0);
  }

  if (structure->IsTask()) {
    auto task = static_cast<TaskNode *>(structure);
    if (task->stmts.empty()) {
      return Evaluate(0);
    }
    // Check for dual_copy: rewrite to copies with sid-based offset
    if (task->stmts.size() == 1) {
      if (auto *eval = task->stmts[0].as<EvaluateNode>()) {
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
      return task->stmts[0];
    } else {
      return SeqStmt(task->stmts);
    }
  } else if (structure->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(structure);
    std::vector<Stmt> stmts;
    for (const auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      for (auto &[_, before] : unit->before) {
        for (auto &stmt : before) {
          stmts.push_back(stmt);
        }
      }
      Stmt child_stmt = ConvertIRStructureToStmt(unit->child.get(), sid);
      stmts.push_back(child_stmt);
      for (auto &[_, after] : unit->after) {
        for (auto &stmt : after) {
          stmts.push_back(stmt);
        }
      }
    }
    auto flattened = SeqStmt::Flatten(stmts);
    return flattened;
  } else if (structure->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(structure);
    Var loop_var = ctrl->control->loop_var;
    PrimExpr loop_start = ctrl->control->min;
    PrimExpr loop_extent = ctrl->control->extent;
    PrimExpr loop_step = ctrl->control->step.has_value()
                             ? ctrl->control->step.value()
                             : IntImm(DataType::Int(32), 1);
    int min_stages = 100, max_stages = -1;
    if (ctrl->child->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        min_stages = std::min(min_stages, unit->stage);
        max_stages = std::max(max_stages, unit->stage);
      }
    }
    // Stage-aware prologue/epilogue emission: substitute loop_var and wrap
    // each task in an `if (cond)` guard so that within a single extended
    // loop the right stages run for each iteration. Separate unrolled
    // prologue/epilogue blocks are disabled for now (enable_pro/enable_epi
    // forced to false).
    if (!ctrl->child->IsSequence() || min_stages == max_stages) {
      std::vector<Stmt> stmts;
      if (ctrl->child->IsScheduleUnit()) {
        auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
        for (auto &[_, before] : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(ConvertIRStructureToStmt(unit->child.get(), sid));
        for (auto &[_, after] : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
      } else if (ctrl->child->IsSequence()) {
        auto seq = static_cast<SequenceNode *>(ctrl->child.get());
        for (auto &child : seq->children) {
          ICHECK(child->IsScheduleUnit());
          auto unit = static_cast<ScheduleUnit *>(child.get());
          for (auto &[_, before] : unit->before) {
            for (auto &stmt : before) {
              stmts.push_back(stmt);
            }
          }
          stmts.push_back(ConvertIRStructureToStmt(unit->child.get(), sid));
          for (auto &[_, after] : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        }
      } else if (ctrl->child->IsTask()) {
        auto task = static_cast<TaskNode *>(ctrl->child.get());
        stmts.push_back(ConvertIRStructureToStmt(task, sid));
      } else {
        LOG(FATAL);
      }
      Stmt body = SeqStmt::Flatten(stmts);
      // Filter out scheduling-only annotations now that AutoSchedule is done.
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      filtered_annotations.erase("multi_buffer_eligible");
      return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                 ctrl->control->thread_binding, filtered_annotations);
    }
    // Stage-aware single-loop body: substitute loop_var per task and wrap
    // each task in a stage-appropriate `if (cond)` guard. Separate
    // prologue/epilogue unrolled blocks are intentionally NOT emitted.
    auto seq = static_cast<SequenceNode *>(ctrl->child.get());
    std::vector<Stmt> steady;
    for (auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      Map<Var, PrimExpr> substitution, substitution_cond;
      substitution.Set(loop_var,
                       loop_var - loop_step * (max_stages - unit->stage));
      substitution_cond.Set(
          loop_var, Max(loop_start, Min(loop_start + loop_extent - loop_step,
                                        loop_var - loop_step * (max_stages -
                                                                unit->stage))));
      PrimExpr condition =
          And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
      if (unit->stage == min_stages) {
        condition = loop_var >= loop_start;
      }
      if (unit->stage == max_stages) {
        condition = loop_var < loop_start + loop_extent;
      }
      if (IsLetDeclNode(unit->child.get())) {
        std::vector<Stmt> guarded;
        for (const auto &[_, before] : unit->before) {
          for (const auto &stmt : before) {
            guarded.push_back(
                Substitute(IfThenElse(condition, stmt), substitution));
          }
        }
        guarded.push_back(
            Substitute(ConvertIRStructureToStmt(unit->child.get(), sid),
                       substitution_cond));
        for (const auto &[_, after] : unit->after) {
          for (const auto &stmt : after) {
            guarded.push_back(
                Substitute(IfThenElse(condition, stmt), substitution));
          }
        }
        steady.push_back(SeqStmt::Flatten(guarded));
      } else {
        std::vector<Stmt> stmts;
        for (const auto &[_, before] : unit->before) {
          for (const auto &stmt : before) {
            stmts.push_back(GuardWithIf(stmt, condition));
          }
        }
        stmts.push_back(GuardWithIf(
            ConvertIRStructureToStmt(unit->child.get(), sid), condition));
        for (const auto &[_, after] : unit->after) {
          for (const auto &stmt : after) {
            stmts.push_back(GuardWithIf(stmt, condition));
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
    Map<Var, PrimExpr> sub_var;
    sub_var.Set(loop_var, new_var);
    For for_op =
        For(new_var, loop_start,
            ctrl->control->extent + loop_step * (max_stages - min_stages),
            ctrl->control->kind, Substitute(new_body, sub_var),
            ctrl->control->thread_binding, filtered_annotations);
    return for_op;
  } else if (structure->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(structure);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ConvertIRStructureToStmt(wrapper->child.get(), sid);
    }
    if (const auto *bind = wrapper->wrapper.as<BindNode>()) {
      return SeqStmt::Flatten(Bind(bind->var, bind->value), body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
    }
  } else if (structure->IsIf()) {
    auto if_node = static_cast<const IfNode *>(structure);
    Stmt then_stmt = ConvertIRStructureToStmt(if_node->then_child.get(), sid);
    Optional<Stmt> else_stmt;
    if (if_node->else_child) {
      else_stmt = ConvertIRStructureToStmt(if_node->else_child.get(), sid);
    }
    return IfThenElse(if_node->condition, then_stmt, else_stmt);
  }

  LOG(FATAL)
      << "Failed to convert IRStructure to Stmt, returning empty statement";
  return Evaluate(0);
}
} // namespace tl
} // namespace tvm
