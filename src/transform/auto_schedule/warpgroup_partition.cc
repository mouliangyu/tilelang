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
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

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

#include "../../op/builtin.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
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
      void VisitStmt_(const LetStmtNode *op) final {
        remap_.Set(op->var, op->var.copy_with_suffix(suffix_));
        StmtExprVisitor::VisitStmt_(op);
      }
      Map<Var, Var> &remap_;
      const std::string &suffix_;
    };
    Collector c(var_remap_, suffix_);
    c(stmt);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    auto it = var_remap_.find(op->var);
    Var new_var = it != var_remap_.end() ? (*it).second : op->var;
    PrimExpr new_value = VisitExpr(op->value);
    Stmt new_body = VisitStmt(op->body);
    return LetStmt(new_var, new_value, new_body, op->span);
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

bool IsLetDeclTask(const TaskNode *task) {
  return task->stmts.size() == 1 && task->stmts[0].as<LetStmtNode>() != nullptr;
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
      const auto *let = task->stmts[0].as<LetStmtNode>();
      auto new_var = let->var.copy_with_suffix("");
      // Substitute previously renamed variables in the value expression.
      PrimExpr new_value =
          var_remap.empty() ? let->value : Substitute(let->value, var_remap);
      var_remap.Set(let->var, new_var);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmts.push_back(LetStmt(new_var, new_value, Evaluate(0)));
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
        ct->stmts[i] = Substitute(ct->stmts[i], var_remap);
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
          cloned_task->stmts[i] = Substitute(cloned_task->stmts[i], var_remap);
        }
      }
      new_ctrl->task = std::move(cloned_task);
    }
    new_ctrl->SetPromote(ctrl->hasPromote());
    new_ctrl->child = CloneIRStructureWithWarpgroupFilter(
        ctrl->child.get(), warpgroup_id, var_remap);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_shared<WrapperNode>();
    // Apply var_remap to the wrapper statement so that renamed LetDecl
    // variables are correctly substituted in LetStmt values / AttrStmt values.
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
    new_unit->stage = unit->stage;
    new_unit->child = CloneIRStructureWithWarpgroupFilter(
        unit->child.get(), warpgroup_id, var_remap);

    if (!child_is_let_decl) {
      // Copy before/after for the target warp group
      new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
      new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
      // Substitute renamed LetDecl variables in before/after stmts
      if (!var_remap.empty()) {
        for (auto &s : new_unit->before[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
        for (auto &s : new_unit->after[warpgroup_id]) {
          s = Substitute(s, var_remap);
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
          cloned_task->stmts[i] = Substitute(cloned_task->stmts[i], var_remap);
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

// For each child of a root SequenceNode, apply
// CloneIRStructureWithWarpgroupFilter individually.
std::vector<std::shared_ptr<IRStructure>>
CloneIRStructureChildrenWithWarpgroupFilter(SequenceNode *root_seq,
                                            int warpgroup_id,
                                            Map<Var, PrimExpr> &var_remap) {
  std::vector<std::shared_ptr<IRStructure>> result;
  result.reserve(root_seq->children.size());
  for (const auto &child : root_seq->children) {
    result.push_back(CloneIRStructureWithWarpgroupFilter(
        child.get(), warpgroup_id, var_remap));
  }
  return result;
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
            const auto *let = task->stmts[0].as<LetStmtNode>();
            let_decls.push_back({let->var.get(), let->value});
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
          // (LetStmt value / AttrStmt value) so their LetDecls are not removed.
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
        const auto *let = static_cast<const TaskNode *>(node.get())
                              ->stmts[0]
                              .as<LetStmtNode>();
        if (!referenced_vars.count(let->var.get())) {
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
      new_ctrl->SetPromote(ctrl->hasPromote());
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
      new_unit->stage = unit->stage;
      new_unit->before = unit->before;
      new_unit->after = unit->after;
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

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

Stmt ConvertIRStructureToStmt(IRStructure *structure) {
  if (!structure) {
    return Evaluate(0);
  }

  if (structure->IsTask()) {
    auto task = static_cast<TaskNode *>(structure);
    if (task->stmts.empty()) {
      return Evaluate(0);
    } else if (task->stmts.size() == 1) {
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
      Stmt child_stmt = ConvertIRStructureToStmt(unit->child.get());
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
    if (!ctrl->hasPromote() || !ctrl->child->IsSequence() ||
        min_stages == max_stages || true) {
      std::vector<Stmt> stmts;
      if (ctrl->child->IsScheduleUnit()) {
        auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
        for (auto &[_, before] : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(ConvertIRStructureToStmt(unit->child.get()));
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
          stmts.push_back(ConvertIRStructureToStmt(unit->child.get()));
          for (auto &[_, after] : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        }
      } else if (ctrl->child->IsTask()) {
        auto task = static_cast<TaskNode *>(ctrl->child.get());
        stmts.push_back(ConvertIRStructureToStmt(task));
      } else {
        LOG(FATAL);
      }
      Stmt body = SeqStmt::Flatten(stmts);
      // Filter out "num_stages" annotation
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                 ctrl->control->thread_binding, filtered_annotations);
    }
    LOG(FATAL) << "Has Promote in control";
  } else if (structure->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(structure);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ConvertIRStructureToStmt(wrapper->child.get());
    }
    if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
    }
  } else if (structure->IsIf()) {
    auto if_node = static_cast<const IfNode *>(structure);
    Stmt then_stmt = ConvertIRStructureToStmt(if_node->then_child.get());
    Optional<Stmt> else_stmt;
    if (if_node->else_child) {
      else_stmt = ConvertIRStructureToStmt(if_node->else_child.get());
    }
    return IfThenElse(if_node->condition, then_stmt, else_stmt);
  }

  LOG(FATAL)
      << "Failed to convert IRStructure to Stmt, returning empty statement";
  return Evaluate(0);
}
} // namespace tl
} // namespace tvm
