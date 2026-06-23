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
 * \file ir_structure.cc
 * \brief IR structure analysis for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./ir_structure.h"
#include "op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

// BufferAccessPoint::task — defined here so the inline `static_cast` sees the
// full TaskNode definition via ir_structure.h.
const TaskNode *BufferAccessPoint::task() const {
  ICHECK(!path.empty() && path.back() && path.back()->IsTask());
  return static_cast<const TaskNode *>(path.back());
}

// SequenceNode member function implementations
uint16_t SequenceNode::GetPipeMask() const {
  uint16_t mask = 0;
  for (const auto &child : children) {
    if (child)
      mask |= child->GetPipeMask();
  }
  return mask;
}

uint16_t SequenceNode::GetHbmMask() const {
  uint16_t mask = 0;
  for (const auto &child : children) {
    if (child)
      mask |= child->GetHbmMask();
  }
  return mask;
}

std::vector<BufferRegion> SequenceNode::GetReadRegions() const {
  std::vector<BufferRegion> all_read_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_read_regions = child->GetReadRegions();
      all_read_regions.insert(all_read_regions.end(),
                              child_read_regions.begin(),
                              child_read_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  // This is a simplified deduplication - in practice might need more
  // sophisticated logic
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_read_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<Var> SequenceNode::GetReadVars() const {
  std::vector<Var> all_vars;
  for (const auto &child : children) {
    if (child) {
      auto child_read_vars = child->GetReadVars();
      all_vars.insert(all_vars.end(), child_read_vars.begin(),
                      child_read_vars.end());
    }
  }
  return all_vars;
}

std::vector<Var> SequenceNode::GetIndirectReadVars() const {
  std::vector<Var> all_vars;
  for (const auto &child : children) {
    if (child) {
      auto child_vars = child->GetIndirectReadVars();
      all_vars.insert(all_vars.end(), child_vars.begin(), child_vars.end());
    }
  }
  return all_vars;
}

std::vector<BufferRegion> SequenceNode::GetIndirectReadRegions() const {
  std::vector<BufferRegion> all_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_regions = child->GetIndirectReadRegions();
      all_regions.insert(all_regions.end(), child_regions.begin(),
                         child_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<BufferRegion> SequenceNode::GetWriteRegions() const {
  std::vector<BufferRegion> all_write_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_write_regions = child->GetWriteRegions();
      all_write_regions.insert(all_write_regions.end(),
                               child_write_regions.begin(),
                               child_write_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_write_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<Var> SequenceNode::GetWriteVars() const {
  std::vector<Var> all_vars;
  for (const auto &child : children) {
    if (child) {
      auto child_write_vars = child->GetWriteVars();
      all_vars.insert(all_vars.end(), child_write_vars.begin(),
                      child_write_vars.end());
    }
  }
  return all_vars;
}

int64_t SequenceNode::GetLatency() const { return latency_; }

int64_t SequenceNode::GetII() const { return ii_; }

void SequenceNode::SetPipeMask(uint16_t mask) {
  for (auto &child : children) {
    if (child)
      child->SetPipeMask(mask);
  }
}

void SequenceNode::SetReadRegions(const std::vector<BufferRegion> &regions) {
  // Not clear what this means for SequenceNode - maybe set on first child?
  if (!children.empty() && children[0]) {
    children[0]->SetReadRegions(regions);
  }
}

void SequenceNode::SetWriteRegions(const std::vector<BufferRegion> &regions) {
  if (!children.empty() && children[0]) {
    children[0]->SetWriteRegions(regions);
  }
}

void SequenceNode::SetLatency(int64_t latency) { latency_ = latency; }

void SequenceNode::SetII(int64_t ii) { ii_ = ii; }

std::shared_ptr<IRStructure> SequenceNode::Clone() const {
  auto new_seq = std::make_shared<SequenceNode>();
  new_seq->children.reserve(children.size());
  for (const auto &child : children) {
    if (child) {
      new_seq->children.push_back(child->Clone());
    } else {
      new_seq->children.push_back(nullptr);
    }
  }
  // Copy latency and II
  new_seq->SetLatency(GetLatency());
  new_seq->SetII(GetII());
  return new_seq;
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>();
  // Copy statements
  new_task->stmts = stmts;
  // Copy resource pipe mask
  new_task->SetPipeMask(GetPipeMask());
  new_task->SetHbmMask(GetHbmMask());
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  // Copy indirect read sets (propagated through scalar Let bindings)
  new_task->SetIndirectReads(indirect_read_regions_, indirect_read_vars_,
                             indirect_computed_);
  // Copy latency and II
  new_task->SetLatency(GetLatency());
  new_task->SetII(GetII());
  for (const auto &shape : tensor_core_shapes_) {
    new_task->AddTensorCoreShape(shape.m, shape.n, shape.k, shape.input_dtype,
                                 shape.accum_dtype);
  }
  for (const auto &info : copy_infos_) {
    new_task->AddCopyInfo(info.src_scope, info.dst_scope, info.dtype,
                          info.bytes, info.pipe_mask);
  }
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  // Copy loop_break cache
  new_task->contains_loop_break_cache_ = contains_loop_break_cache_;
  return new_task;
}

bool TaskNode::ContainsLoopBreak() const {
  // Return cached result if available
  if (contains_loop_break_cache_.has_value()) {
    return contains_loop_break_cache_.value();
  }

  // Check if any statement in this task contains a loop_break call
  bool found_loop_break = false;
  for (const auto &stmt : stmts) {
    // Helper function to check if a statement contains loop_break
    auto contains_loop_break = [](const Stmt &stmt) -> bool {
      bool found = false;
      PostOrderVisit(stmt, [&found](const ObjectRef &node) {
        if (found)
          return;
        if (const auto *call = node.as<CallNode>()) {
          if (call->op.same_as(tl::loop_break())) {
            found = true;
          }
        }
      });
      return found;
    };

    if (contains_loop_break(stmt)) {
      found_loop_break = true;
      break;
    }
  }

  contains_loop_break_cache_ = found_loop_break;
  return found_loop_break;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>();
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone child if exists
  if (child) {
    new_ctrl->child = child->Clone();
  }
  // Copy latency and II
  new_ctrl->SetLatency(GetLatency());
  new_ctrl->SetII(GetII());
  return new_ctrl;
}

std::shared_ptr<IRStructure> WrapperNode::Clone() const {
  auto new_wrapper = std::make_shared<WrapperNode>();
  // Copy var and value (TVM objects with reference counting)
  new_wrapper->wrapper = wrapper;
  // Clone child if exists
  if (child) {
    new_wrapper->child = child->Clone();
  }
  // Copy latency and II
  new_wrapper->SetLatency(GetLatency());
  new_wrapper->SetII(GetII());
  return new_wrapper;
}

std::shared_ptr<IRStructure> ScheduleUnit::Clone() const {
  auto new_unit = std::make_shared<ScheduleUnit>();
  // Copy var and value (TVM objects with reference counting)
  // Clone child if exists
  if (child) {
    new_unit->child = child->Clone();
  }
  // Copy latency and II
  new_unit->SetLatency(GetLatency());
  new_unit->SetII(GetII());
  new_unit->SetStartTime(GetStartTime());
  new_unit->stage = stage;
  return new_unit;
}

std::shared_ptr<IRStructure> IfNode::Clone() const {
  auto new_if = std::make_shared<IfNode>();
  new_if->condition = condition;
  if (then_child) {
    new_if->then_child = then_child->Clone();
  }
  if (else_child) {
    new_if->else_child = else_child->Clone();
  }
  if (task) {
    new_if->task = std::static_pointer_cast<TaskNode>(task->Clone());
  }
  new_if->SetLatency(GetLatency());
  new_if->SetII(GetII());
  return new_if;
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(IRStructure *node,
                                    std::vector<TaskNodeWithContext> &all_tasks,
                                    ControlNode *current_control_node) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    TaskNodeWithContext task_ctx;
    task_ctx.task = task;
    task_ctx.control_node = current_control_node;

    // Calculate tripcount if inside a loop
    if (current_control_node) {
      task_ctx.tripcount = current_control_node->GetTripCount();
    } else {
      task_ctx.tripcount = 1; // Not inside a loop
    }

    all_tasks.push_back(task_ctx);
  } else if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      CollectAllTaskNodesWithContext(child.get(), all_tasks,
                                     current_control_node);
    }
  } else if (node->IsControl()) {
    auto control = static_cast<const ControlNode *>(node);
    // When entering a control node, update the current control context
    CollectAllTaskNodesWithContext(control->child.get(), all_tasks,
                                   const_cast<ControlNode *>(control));
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    // Wrapper nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(wrapper->child.get(), all_tasks,
                                   current_control_node);
  } else if (node->IsScheduleUnit()) {
    auto promote = static_cast<const ScheduleUnit *>(node);
    // Promote nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(promote->child.get(), all_tasks,
                                   current_control_node);
  } else if (node->IsIf()) {
    auto if_node = static_cast<const IfNode *>(node);
    // Recurse into both branches
    CollectAllTaskNodesWithContext(if_node->then_child.get(), all_tasks,
                                   current_control_node);
    if (if_node->else_child) {
      CollectAllTaskNodesWithContext(if_node->else_child.get(), all_tasks,
                                     current_control_node);
    }
  } else {
    LOG(FATAL);
  }
}

// ============================================================================
// CollectFirstAccessPoints / CollectLastAccessPoints implementations
// ============================================================================

static bool LoopMustExecute(const ControlNode *ctrl) {
  if (!ctrl->control.defined())
    return false;
  const ForNode *for_node = ctrl->control.get();
  const int64_t *extent_ptr = as_const_int(for_node->extent);
  if (extent_ptr && *extent_ptr >= 1) {
    if (for_node->step.has_value()) {
      const int64_t *step_ptr = as_const_int(for_node->step.value());
      if (step_ptr && *step_ptr >= 1)
        return true;
      if (!step_ptr)
        return false;
    } else {
      return true;
    }
  }
  return false;
}

// An entry in the unified sequence helper. `node` is the IR-tree node to
// recurse into; `path_prefix` lists the real IR-tree nodes to push onto the
// access-point path before recursing (so the recorded chain reflects ancestor
// wrappers — typically the enclosing `ScheduleUnit`, plus the `IfNode` itself
// when we descend into an if-branch); `start_time` is the entry's offset
// relative to the enclosing sequence's start.
//
// At a regular Sequence level, each entry corresponds to one `ScheduleUnit`
// child: `node = unit->child`, `path_prefix = {unit}`. When the helper merges
// an if-group and descends into branches, each branch becomes a new
// SequenceEntry whose `path_prefix` extends to include the enclosing IfNode,
// so inner if-groups (and arbitrary nesting) are handled by the same helper.
struct SequenceEntry {
  const IRStructure *node;
  std::vector<const IRStructure *> path_prefix;
  int64_t start_time;
};

static const IfNode *AsIfNode(const IRStructure *node) {
  if (node && node->IsIf())
    return static_cast<const IfNode *>(node);
  return nullptr;
}

static bool SequenceCollectFirstAccessPoints(
    const std::vector<SequenceEntry> &entries, const TaskAccessPredicate &pred,
    std::set<BufferAccessPoint> &result, std::vector<const IRStructure *> &path,
    int64_t base_offset) {
  int n = static_cast<int>(entries.size());

  auto push_prefix = [&](const SequenceEntry &e) {
    for (auto *p : e.path_prefix)
      path.push_back(p);
  };
  auto pop_prefix = [&](const SequenceEntry &e) {
    for (size_t k = 0; k < e.path_prefix.size(); ++k)
      path.pop_back();
  };
  auto dispatch = [&](const SequenceEntry &e) -> bool {
    push_prefix(e);
    bool m = e.node->CollectFirstAccessPoints(pred, result, path,
                                              base_offset + e.start_time);
    pop_prefix(e);
    return m;
  };

  int break_idx = -1;
  for (int j = 0; j < n; ++j) {
    if (entries[j].node->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx > 0) {
      std::vector<SequenceEntry> before(entries.begin(),
                                        entries.begin() + break_idx);
      if (SequenceCollectFirstAccessPoints(before, pred, result, path,
                                           base_offset))
        return true;
    }
    if (dispatch(entries[break_idx]))
      return true;
    if (break_idx + 1 < n) {
      std::vector<SequenceEntry> after(entries.begin() + break_idx + 1,
                                       entries.end());
      SequenceCollectFirstAccessPoints(after, pred, result, path, base_offset);
    }
    return false;
  }
  int i = 0;
  while (i < n) {
    const IfNode *if_node = AsIfNode(entries[i].node);
    if (if_node) {
      int group_start = i;
      while (i + 1 < n) {
        const IfNode *next_if = AsIfNode(entries[i + 1].node);
        if (!next_if)
          break;
        if (!StructuralEqual()(if_node->condition, next_if->condition))
          break;
        ++i;
      }
      int group_end = i;
      ++i;
      // Condition task belongs to the IfNode itself. Use the first entry's
      // ancestor chain + the IfNode for the task's path; the start_time also
      // comes from the first entry of the group.
      const SequenceEntry &first_entry = entries[group_start];
      push_prefix(first_entry);
      path.push_back(if_node);
      bool task_matched = false;
      if (if_node->task) {
        task_matched = if_node->task->CollectFirstAccessPoints(
            pred, result, path, base_offset + first_entry.start_time);
      }
      path.pop_back();
      pop_prefix(first_entry);
      if (task_matched)
        return true;

      // Build branch virtual sub-sequences. Each branch entry inherits the
      // outer ancestor chain (entries[j].path_prefix) and extends it with the
      // enclosing IfNode, so the branch's children record the full chain.
      std::vector<SequenceEntry> then_entries, else_entries;
      for (int j = group_start; j <= group_end; ++j) {
        const SequenceEntry &je = entries[j];
        const IfNode *cur_if = AsIfNode(je.node);
        ICHECK(cur_if);
        auto make_branch_entry = [&](const std::shared_ptr<IRStructure> &branch,
                                     std::vector<SequenceEntry> &out) {
          if (!branch)
            return;
          SequenceEntry sub;
          sub.node = branch.get();
          sub.path_prefix = je.path_prefix;
          sub.path_prefix.push_back(cur_if);
          sub.start_time = je.start_time;
          out.push_back(std::move(sub));
        };
        make_branch_entry(cur_if->then_child, then_entries);
        make_branch_entry(cur_if->else_child, else_entries);
      }
      bool then_must = false, else_must = false;
      if (!then_entries.empty())
        then_must = SequenceCollectFirstAccessPoints(then_entries, pred, result,
                                                     path, base_offset);
      if (!else_entries.empty())
        else_must = SequenceCollectFirstAccessPoints(else_entries, pred, result,
                                                     path, base_offset);
      if (then_must && else_must)
        return true;
    } else {
      if (dispatch(entries[i]))
        return true;
      ++i;
    }
  }
  return false;
}

static bool SequenceCollectLastAccessPoints(
    const std::vector<SequenceEntry> &entries, const TaskAccessPredicate &pred,
    std::set<BufferAccessPoint> &result, std::vector<const IRStructure *> &path,
    int64_t base_offset) {
  int n = static_cast<int>(entries.size());

  auto push_prefix = [&](const SequenceEntry &e) {
    for (auto *p : e.path_prefix)
      path.push_back(p);
  };
  auto pop_prefix = [&](const SequenceEntry &e) {
    for (size_t k = 0; k < e.path_prefix.size(); ++k)
      path.pop_back();
  };
  auto dispatch = [&](const SequenceEntry &e) -> bool {
    push_prefix(e);
    bool m = e.node->CollectLastAccessPoints(pred, result, path,
                                             base_offset + e.start_time);
    pop_prefix(e);
    return m;
  };

  int break_idx = -1;
  for (int j = n - 1; j >= 0; --j) {
    if (entries[j].node->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx + 1 < n) {
      std::vector<SequenceEntry> after(entries.begin() + break_idx + 1,
                                       entries.end());
      SequenceCollectLastAccessPoints(after, pred, result, path, base_offset);
    }
    dispatch(entries[break_idx]);
    if (break_idx > 0) {
      std::vector<SequenceEntry> before(entries.begin(),
                                        entries.begin() + break_idx);
      SequenceCollectLastAccessPoints(before, pred, result, path, base_offset);
    }
    return false;
  }
  int i = n - 1;
  while (i >= 0) {
    const IfNode *if_node = AsIfNode(entries[i].node);
    if (if_node) {
      int group_end = i;
      while (i - 1 >= 0) {
        const IfNode *prev_if = AsIfNode(entries[i - 1].node);
        if (!prev_if)
          break;
        if (!StructuralEqual()(if_node->condition, prev_if->condition))
          break;
        --i;
      }
      int group_start = i;
      --i;

      std::vector<SequenceEntry> then_entries, else_entries;
      for (int j = group_start; j <= group_end; ++j) {
        const SequenceEntry &je = entries[j];
        const IfNode *cur_if = AsIfNode(je.node);
        ICHECK(cur_if);
        auto make_branch_entry = [&](const std::shared_ptr<IRStructure> &branch,
                                     std::vector<SequenceEntry> &out) {
          if (!branch)
            return;
          SequenceEntry sub;
          sub.node = branch.get();
          sub.path_prefix = je.path_prefix;
          sub.path_prefix.push_back(cur_if);
          sub.start_time = je.start_time;
          out.push_back(std::move(sub));
        };
        make_branch_entry(cur_if->then_child, then_entries);
        make_branch_entry(cur_if->else_child, else_entries);
      }
      bool then_must = false, else_must = false;
      if (!then_entries.empty())
        then_must = SequenceCollectLastAccessPoints(then_entries, pred, result,
                                                    path, base_offset);
      if (!else_entries.empty())
        else_must = SequenceCollectLastAccessPoints(else_entries, pred, result,
                                                    path, base_offset);
      if (then_must && else_must)
        return true;

      // After branches, attribute the condition task of the last if-unit to
      // the group. Uses entries[group_end]'s ancestor chain.
      const SequenceEntry &last_entry = entries[group_end];
      const IfNode *last_if = AsIfNode(last_entry.node);
      ICHECK(last_if);
      push_prefix(last_entry);
      path.push_back(last_if);
      bool task_matched = false;
      if (last_if->task) {
        task_matched = last_if->task->CollectLastAccessPoints(
            pred, result, path, base_offset + last_entry.start_time);
      }
      path.pop_back();
      pop_prefix(last_entry);
      if (task_matched)
        return true;
    } else {
      if (dispatch(entries[i]))
        return true;
      --i;
    }
  }
  return false;
}

// Build a SequenceEntry list from a real SequenceNode's children. Each
// ScheduleUnit child becomes one entry — the ScheduleUnit goes into the
// path_prefix and dispatch lands directly on its inner child.
static std::vector<SequenceEntry> MakeSequenceEntriesFromUnits(
    const std::vector<std::shared_ptr<IRStructure>> &children) {
  std::vector<SequenceEntry> entries;
  entries.reserve(children.size());
  for (const auto &c : children) {
    ICHECK(c->IsScheduleUnit())
        << "Sequence child must be a ScheduleUnit post-scheduling";
    const ScheduleUnit *unit = static_cast<const ScheduleUnit *>(c.get());
    ICHECK(unit->child)
        << "ScheduleUnit must wrap a child IRStructure post-scheduling";
    SequenceEntry e;
    e.node = unit->child.get();
    e.path_prefix = {unit};
    e.start_time = unit->GetStartTime();
    entries.push_back(std::move(e));
  }
  return entries;
}

bool TaskNode::CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                        std::set<BufferAccessPoint> &result,
                                        std::vector<const IRStructure *> &path,
                                        int64_t time_offset) const {
  if (pred(this)) {
    std::vector<const IRStructure *> point_path = path;
    point_path.push_back(this);
    result.emplace(time_offset, std::move(point_path));
    return true;
  }
  return false;
}

bool TaskNode::CollectLastAccessPoints(const TaskAccessPredicate &pred,
                                       std::set<BufferAccessPoint> &result,
                                       std::vector<const IRStructure *> &path,
                                       int64_t time_offset) const {
  if (pred(this)) {
    std::vector<const IRStructure *> point_path = path;
    point_path.push_back(this);
    result.emplace(time_offset, std::move(point_path));
    return true;
  }
  return false;
}

bool ScheduleUnit::CollectFirstAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  if (!child)
    return false;
  path.push_back(this);
  bool r = child->CollectFirstAccessPoints(pred, result, path, time_offset);
  path.pop_back();
  return r;
}

bool ScheduleUnit::CollectLastAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  if (!child)
    return false;
  path.push_back(this);
  bool r = child->CollectLastAccessPoints(pred, result, path, time_offset);
  path.pop_back();
  return r;
}

bool WrapperNode::CollectFirstAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  if (task) {
    if (task->CollectFirstAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  if (child) {
    if (child->CollectFirstAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  path.pop_back();
  return false;
}

bool WrapperNode::CollectLastAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  if (child) {
    if (child->CollectLastAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  if (task) {
    if (task->CollectLastAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  path.pop_back();
  return false;
}

bool IfNode::CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                      std::set<BufferAccessPoint> &result,
                                      std::vector<const IRStructure *> &path,
                                      int64_t time_offset) const {
  path.push_back(this);
  if (task) {
    if (task->CollectFirstAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  bool then_must = false, else_must = false;
  if (then_child)
    then_must =
        then_child->CollectFirstAccessPoints(pred, result, path, time_offset);
  if (else_child)
    else_must =
        else_child->CollectFirstAccessPoints(pred, result, path, time_offset);
  path.pop_back();
  return then_must && else_must;
}

bool IfNode::CollectLastAccessPoints(const TaskAccessPredicate &pred,
                                     std::set<BufferAccessPoint> &result,
                                     std::vector<const IRStructure *> &path,
                                     int64_t time_offset) const {
  path.push_back(this);
  bool then_must = false, else_must = false;
  if (then_child)
    then_must =
        then_child->CollectLastAccessPoints(pred, result, path, time_offset);
  if (else_child)
    else_must =
        else_child->CollectLastAccessPoints(pred, result, path, time_offset);
  if (then_must && else_must) {
    path.pop_back();
    return true;
  }
  if (task) {
    if (task->CollectLastAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  path.pop_back();
  return false;
}

bool ControlNode::CollectFirstAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  if (task) {
    if (task->CollectFirstAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    path.push_back(seq);
    body_must = SequenceCollectFirstAccessPoints(
        MakeSequenceEntriesFromUnits(seq->children), pred, result, path,
        time_offset);
    path.pop_back();
  } else if (child) {
    body_must =
        child->CollectFirstAccessPoints(pred, result, path, time_offset);
  }
  path.pop_back();
  if (body_must && LoopMustExecute(this))
    return true;
  return false;
}

bool ControlNode::CollectLastAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  int64_t trip_count = GetTripCount();
  int64_t ii_per_iter = GetIIperIter();
  int64_t last_iter_offset =
      (trip_count > 0 ? (trip_count - 1) : 0) * ii_per_iter;
  int64_t body_offset = time_offset + last_iter_offset;
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    path.push_back(seq);
    body_must = SequenceCollectLastAccessPoints(
        MakeSequenceEntriesFromUnits(seq->children), pred, result, path,
        body_offset);
    path.pop_back();
  } else if (child) {
    body_must = child->CollectLastAccessPoints(pred, result, path, body_offset);
  }
  if (body_must && LoopMustExecute(this)) {
    path.pop_back();
    return true;
  }
  if (task) {
    if (task->CollectLastAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  path.pop_back();
  return false;
}

bool SequenceNode::CollectFirstAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  bool r = SequenceCollectFirstAccessPoints(
      MakeSequenceEntriesFromUnits(children), pred, result, path, time_offset);
  path.pop_back();
  return r;
}

bool SequenceNode::CollectLastAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  bool r = SequenceCollectLastAccessPoints(
      MakeSequenceEntriesFromUnits(children), pred, result, path, time_offset);
  path.pop_back();
  return r;
}

inline bool WarpgroupMatches(const TaskNode *task, int wg_id) {
  int task_wg = task->GetWarpgroupId();
  return IsWarpgroupBroadcast(wg_id) || IsWarpgroupBroadcast(task_wg) ||
         wg_id == task_wg;
}

TaskAccessPredicate MakeBufferAccessPredicate(const Buffer &buffer,
                                              bool is_write, int wg_id) {
  return [buffer, is_write, wg_id](const TaskNode *task) -> bool {
    if (!WarpgroupMatches(task, wg_id))
      return false;
    const auto &regions =
        is_write ? task->GetWriteRegions() : task->GetReadRegions();
    for (const auto &region : regions) {
      if (region->buffer == buffer)
        return true;
    }
    return false;
  };
}

// ============================================================================
// Dependency analysis implementation
// ============================================================================

bool RegionsMayConflict(const TaskNode *a_task, const BufferRegion &a_region,
                        const TaskNode *b_task, const BufferRegion &b_region,
                        std::optional<For> current_loop, bool cross) {
  if (!a_region->buffer.same_as(b_region->buffer))
    return false;
  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  Map<Var, PrimExpr> a_sub, b_sub;
  if (current_loop.has_value()) {
    const ForNode *fn = current_loop->get();
    Var lv = fn->loop_var;
    if (!cross) {
      // Same iteration: producer and consumer share one loop-var value.
      Var lv_s(lv->name_hint + "_s", lv.dtype());
      ana.Bind(lv_s, Range::FromMinExtent(fn->min, fn->extent));
      a_sub.Set(lv, lv_s);
      b_sub.Set(lv, lv_s);
    } else {
      // Loop back-edge: consumer runs d>=1 iterations after the producer.
      // lv_p ranges the loop; lv_c = lv_p + d with d in [1, extent-1].
      Var lv_p(lv->name_hint + "_p", lv.dtype());
      Var d(lv->name_hint + "_d", lv.dtype());
      ana.Bind(lv_p, Range::FromMinExtent(fn->min, fn->extent));
      // d in [1, extent-1]: at least one later iteration exists.
      ana.Bind(
          d, Range::FromMinExtent(IntImm(lv.dtype(), 1),
                                  max(fn->extent - 1, IntImm(lv.dtype(), 1))));
      a_sub.Set(lv, lv_p);
      b_sub.Set(lv, lv_p + d);
    }
  }
  // Feed each side's surrounding-scope ConstrSet to the prover, after
  // substituting the current_loop var with that side's loop-var map.
  if (a_task) {
    a_task->outer_ctx.Substitute(a_sub).Populate(ana);
  }
  if (b_task) {
    b_task->outer_ctx.Substitute(b_sub).Populate(ana);
  }

  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr a_min = Substitute(a_rng[i]->min, a_sub);
    PrimExpr a_max = Substitute(a_rng[i]->min + a_rng[i]->extent - 1, a_sub);
    PrimExpr b_min = Substitute(b_rng[i]->min, b_sub);
    PrimExpr b_max = Substitute(b_rng[i]->min + b_rng[i]->extent - 1, b_sub);
    try {
      a_min = ana.Simplify(a_min);
      a_max = ana.Simplify(a_max);
      b_min = ana.Simplify(b_min);
      b_max = ana.Simplify(b_max);
      if (ana.CanProve(a_max < b_min || b_max < a_min)) {
        return false; // disjoint on this dim ⇒ regions disjoint
      }
    } catch (...) {
      // conservative — treat this dim as non-disjoint
    }
  }
  return true; // no dim proved disjoint ⇒ conservatively overlap
}

BufferRegion TaskRegionFor(const TaskNode *task, const Buffer &buffer,
                           bool is_write) {
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  std::optional<BufferRegion> found;
  for (const auto &r : regions) {
    if (r->buffer.same_as(buffer)) {
      ICHECK(!found) << "Buffer " << buffer->name
                     << " appears in more than one region of the task";
      found = r;
    }
  }
  ICHECK(found) << "Buffer " << buffer->name << " not found in task regions";
  return *found;
}

std::vector<DepInfo>
AnalyzeDependencies(const std::vector<ScheduleUnit *> &units,
                    std::optional<For> current_loop) {
  const size_t n = units.size();
  std::vector<DepInfo> deps;

  std::set<Buffer> buffers;
  for (size_t i = 0; i < n; ++i) {
    for (const auto &region : units[i]->GetReadRegions()) {
      buffers.insert(region->buffer);
    }
    for (const auto &region : units[i]->GetWriteRegions()) {
      buffers.insert(region->buffer);
    }
  }

  for (const auto &buffer : buffers) {
    auto touches = [&](size_t u) {
      for (const auto &region : units[u]->GetReadRegions()) {
        if (region->buffer.same_as(buffer))
          return true;
      }
      for (const auto &region : units[u]->GetWriteRegions()) {
        if (region->buffer.same_as(buffer))
          return true;
      }
      return false;
    };

    for (size_t i = 0; i < n; ++i) {
      if (!touches(i))
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (i == j && !current_loop.has_value())
          continue;
        if (!touches(j))
          continue;
        bool same_iter = (i < j);
        bool cross = current_loop.has_value() && (i >= j);
        if (!same_iter && !cross)
          continue;

        auto a_w = units[i]->GetLastAccessPoints(buffer, /*is_write=*/true);
        auto a_r = units[i]->GetLastAccessPoints(buffer, /*is_write=*/false);
        auto b_r = units[j]->GetFirstAccessPoints(buffer, /*is_write=*/false);
        auto b_w = units[j]->GetFirstAccessPoints(buffer, /*is_write=*/true);

        std::set<BufferAccessPoint> prod_pts, cons_pts;
        int64_t latency = -1;

        auto consider = [&](const std::set<BufferAccessPoint> &a_pts,
                            bool a_write,
                            const std::set<BufferAccessPoint> &b_pts,
                            bool b_write) {
          for (const auto &a : a_pts) {
            BufferRegion rA = TaskRegionFor(a.task(), buffer, a_write);
            for (const auto &b : b_pts) {
              BufferRegion rB = TaskRegionFor(b.task(), buffer, b_write);
              if (!RegionsMayConflict(a.task(), rA, b.task(), rB, current_loop,
                                      cross))
                continue;
              prod_pts.insert(a);
              cons_pts.insert(b);
              int64_t l = a.timestamp + a.task()->GetLatency() - b.timestamp;
              if (l < 0)
                l = 0;
              if (l > latency)
                latency = l;
            }
          }
        };
        consider(a_w, /*a_write=*/true, b_r, /*b_write=*/false);
        consider(a_w, /*a_write=*/true, b_w, /*b_write=*/true);
        consider(a_r, /*a_write=*/false, b_w, /*b_write=*/true);

        if (latency >= 0) {
          deps.push_back(
              {units[i], units[j], buffer,
               std::vector<BufferAccessPoint>(prod_pts.begin(), prod_pts.end()),
               std::vector<BufferAccessPoint>(cons_pts.begin(), cons_pts.end()),
               cross, latency});
        }
      }
    }
  }
  return deps;
}

} // namespace tl
} // namespace tvm
