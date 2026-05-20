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
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

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

#include "../../op/builtin.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// SequenceNode member function implementations
uint16_t SequenceNode::GetPipeMask() const {
  uint16_t mask = 0;
  for (const auto &child : children) {
    if (child)
      mask |= child->GetPipeMask();
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
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  // Copy latency and II
  new_task->SetLatency(GetLatency());
  new_task->SetII(GetII());
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  // Copy loop_break cache
  new_task->contains_loop_break_cache_ = contains_loop_break_cache_;
  return new_task;
}

void TaskNode::CollectBufferAccessInfo(
    int num_wgs, std::set<BufferAccessInfo> &result) const {
  int wg_id = GetWarpgroupId();
  // Collect write buffers
  for (const auto &region : GetWriteRegions()) {
    if (wg_id != -1) {
      result.emplace(region->buffer, true, wg_id);
    } else {
      for (int i = 0; i < num_wgs; ++i) {
        result.emplace(region->buffer, true, i);
      }
    }
  }
  // Collect read buffers
  for (const auto &region : GetReadRegions()) {
    if (wg_id != -1) {
      result.emplace(region->buffer, false, wg_id);
    } else {
      for (int i = 0; i < num_wgs; ++i) {
        result.emplace(region->buffer, false, i);
      }
    }
  }
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

void ControlNode::CollectBufferAccessInfo(
    int num_wgs, std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, result);
  }
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, result);
  }
}

void WrapperNode::CollectBufferAccessInfo(
    int num_wgs, std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, result);
  }
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, result);
  }
}

void ScheduleUnit::CollectBufferAccessInfo(
    int num_wgs, std::set<BufferAccessInfo> &result) const {
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, result);
  }
}

void SequenceNode::CollectBufferAccessInfo(
    int num_wgs, std::set<BufferAccessInfo> &result) const {
  for (const auto &child : children) {
    if (child) {
      child->CollectBufferAccessInfo(num_wgs, result);
    }
  }
}

void IfNode::CollectBufferAccessInfo(int num_wgs,
                                     std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, result);
  }
  if (then_child) {
    then_child->CollectBufferAccessInfo(num_wgs, result);
  }
  if (else_child) {
    else_child->CollectBufferAccessInfo(num_wgs, result);
  }
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

static const IfNode *TryGetIfNode(const ScheduleUnit *unit) {
  if (!unit || !unit->child)
    return nullptr;
  if (unit->child->IsIf())
    return static_cast<const IfNode *>(unit->child.get());
  return nullptr;
}

static bool TaskMatchesAccess(const TaskNode *task, const Buffer &buffer,
                              bool is_write, int wg_id) {
  int task_wg = task->GetWarpgroupId();
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  for (const auto &region : regions) {
    if (region->buffer == buffer &&
        (task_wg == wg_id || IsWarpgroupBroadcast(task_wg)))
      return true;
  }
  return false;
}

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

// Build synthetic ScheduleUnits wrapping the then/else branches of an
// if-group so the recursive helper can treat them as a virtual sequence.
// Each synthetic unit inherits the start_time of its enclosing if-statement
// unit, so `base_offset + unit->GetStartTime()` lands on the correct absolute
// timestamp during recursion. The `owners` vector keeps the synthetic units
// alive for the duration of the caller; the parallel out-vector contains raw
// pointers into it.
static void
WrapBranchesForIfGroup(const std::vector<const ScheduleUnit *> &nodes,
                       int group_start, int group_end,
                       std::vector<std::shared_ptr<ScheduleUnit>> &then_owners,
                       std::vector<std::shared_ptr<ScheduleUnit>> &else_owners,
                       std::vector<const ScheduleUnit *> &then_units,
                       std::vector<const ScheduleUnit *> &else_units) {
  for (int j = group_start; j <= group_end; ++j) {
    const ScheduleUnit *unit_j = nodes[j];
    const IfNode *cur_if = TryGetIfNode(unit_j);
    ICHECK(cur_if);
    auto wrap_branch = [&](const std::shared_ptr<IRStructure> &branch,
                           std::vector<std::shared_ptr<ScheduleUnit>> &owners,
                           std::vector<const ScheduleUnit *> &out) {
      if (!branch)
        return;
      auto fresh = std::make_shared<ScheduleUnit>();
      fresh->child = branch;
      fresh->SetStartTime(unit_j->GetStartTime());
      owners.push_back(fresh);
      out.push_back(fresh.get());
    };
    wrap_branch(cur_if->then_child, then_owners, then_units);
    wrap_branch(cur_if->else_child, else_owners, else_units);
  }
}

static bool
SequenceCollectFirstAccessPoints(const std::vector<const ScheduleUnit *> &nodes,
                                 const Buffer &buffer, bool is_write, int wg_id,
                                 std::set<BufferAccessPoint> &result,
                                 int64_t base_offset) {
  int n = static_cast<int>(nodes.size());
  int break_idx = -1;
  for (int j = 0; j < n; ++j) {
    if (nodes[j]->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx > 0) {
      std::vector<const ScheduleUnit *> before(nodes.begin(),
                                               nodes.begin() + break_idx);
      if (SequenceCollectFirstAccessPoints(before, buffer, is_write, wg_id,
                                           result, base_offset))
        return true;
    }
    if (nodes[break_idx]->CollectFirstAccessPoints(
            buffer, is_write, wg_id, result,
            base_offset + nodes[break_idx]->GetStartTime()))
      return true;
    if (break_idx + 1 < n) {
      std::vector<const ScheduleUnit *> after(nodes.begin() + break_idx + 1,
                                              nodes.end());
      SequenceCollectFirstAccessPoints(after, buffer, is_write, wg_id, result,
                                       base_offset);
    }
    return false;
  }
  int i = 0;
  while (i < n) {
    const IfNode *if_node = TryGetIfNode(nodes[i]);
    if (if_node) {
      int group_start = i;
      while (i + 1 < n) {
        const IfNode *next_if = TryGetIfNode(nodes[i + 1]);
        if (!next_if)
          break;
        if (!StructuralEqual()(if_node->condition, next_if->condition))
          break;
        ++i;
      }
      int group_end = i;
      ++i;
      if (if_node->task) {
        if (if_node->task->CollectFirstAccessPoints(
                buffer, is_write, wg_id, result,
                base_offset + nodes[group_start]->GetStartTime()))
          return true;
      }
      std::vector<std::shared_ptr<ScheduleUnit>> then_owners, else_owners;
      std::vector<const ScheduleUnit *> then_units, else_units;
      WrapBranchesForIfGroup(nodes, group_start, group_end, then_owners,
                             else_owners, then_units, else_units);
      bool then_must = false, else_must = false;
      if (!then_units.empty())
        then_must = SequenceCollectFirstAccessPoints(
            then_units, buffer, is_write, wg_id, result, base_offset);
      if (!else_units.empty())
        else_must = SequenceCollectFirstAccessPoints(
            else_units, buffer, is_write, wg_id, result, base_offset);
      if (then_must && else_must)
        return true;
    } else {
      if (nodes[i]->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                             base_offset +
                                                 nodes[i]->GetStartTime()))
        return true;
      ++i;
    }
  }
  return false;
}

static bool
SequenceCollectLastAccessPoints(const std::vector<const ScheduleUnit *> &nodes,
                                const Buffer &buffer, bool is_write, int wg_id,
                                std::set<BufferAccessPoint> &result,
                                int64_t base_offset) {
  int n = static_cast<int>(nodes.size());
  int break_idx = -1;
  for (int j = n - 1; j >= 0; --j) {
    if (nodes[j]->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx + 1 < n) {
      std::vector<const ScheduleUnit *> after(nodes.begin() + break_idx + 1,
                                              nodes.end());
      SequenceCollectLastAccessPoints(after, buffer, is_write, wg_id, result,
                                      base_offset);
    }
    nodes[break_idx]->CollectLastAccessPoints(
        buffer, is_write, wg_id, result,
        base_offset + nodes[break_idx]->GetStartTime());
    if (break_idx > 0) {
      std::vector<const ScheduleUnit *> before(nodes.begin(),
                                               nodes.begin() + break_idx);
      SequenceCollectLastAccessPoints(before, buffer, is_write, wg_id, result,
                                      base_offset);
    }
    return false;
  }
  int i = n - 1;
  while (i >= 0) {
    const IfNode *if_node = TryGetIfNode(nodes[i]);
    if (if_node) {
      int group_end = i;
      while (i - 1 >= 0) {
        const IfNode *prev_if = TryGetIfNode(nodes[i - 1]);
        if (!prev_if)
          break;
        if (!StructuralEqual()(if_node->condition, prev_if->condition))
          break;
        --i;
      }
      int group_start = i;
      --i;
      std::vector<std::shared_ptr<ScheduleUnit>> then_owners, else_owners;
      std::vector<const ScheduleUnit *> then_units, else_units;
      WrapBranchesForIfGroup(nodes, group_start, group_end, then_owners,
                             else_owners, then_units, else_units);
      bool then_must = false, else_must = false;
      if (!then_units.empty())
        then_must = SequenceCollectLastAccessPoints(
            then_units, buffer, is_write, wg_id, result, base_offset);
      if (!else_units.empty())
        else_must = SequenceCollectLastAccessPoints(
            else_units, buffer, is_write, wg_id, result, base_offset);
      if (then_must && else_must)
        return true;
      const IfNode *last_if = TryGetIfNode(nodes[group_end]);
      ICHECK(last_if);
      if (last_if->task) {
        if (last_if->task->CollectLastAccessPoints(
                buffer, is_write, wg_id, result,
                base_offset + nodes[group_end]->GetStartTime()))
          return true;
      }
    } else {
      if (nodes[i]->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                            base_offset +
                                                nodes[i]->GetStartTime()))
        return true;
      --i;
    }
  }
  return false;
}

bool TaskNode::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                        int wg_id,
                                        std::set<BufferAccessPoint> &result,
                                        int64_t time_offset) const {
  if (TaskMatchesAccess(this, buffer, is_write, wg_id)) {
    result.emplace(this, time_offset);
    return true;
  }
  return false;
}

bool TaskNode::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                       int wg_id,
                                       std::set<BufferAccessPoint> &result,
                                       int64_t time_offset) const {
  if (TaskMatchesAccess(this, buffer, is_write, wg_id)) {
    result.emplace(this, time_offset);
    return true;
  }
  return false;
}

bool ScheduleUnit::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                            int wg_id,
                                            std::set<BufferAccessPoint> &result,
                                            int64_t time_offset) const {
  if (child)
    return child->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                           time_offset);
  return false;
}

bool ScheduleUnit::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                           int wg_id,
                                           std::set<BufferAccessPoint> &result,
                                           int64_t time_offset) const {
  if (child)
    return child->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                          time_offset);
  return false;
}

bool WrapperNode::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                           int wg_id,
                                           std::set<BufferAccessPoint> &result,
                                           int64_t time_offset) const {
  if (task) {
    if (task->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                       time_offset))
      return true;
  }
  if (child) {
    if (child->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                        time_offset))
      return true;
  }
  return false;
}

bool WrapperNode::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                          int wg_id,
                                          std::set<BufferAccessPoint> &result,
                                          int64_t time_offset) const {
  if (child) {
    if (child->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                       time_offset))
      return true;
  }
  if (task) {
    if (task->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                      time_offset))
      return true;
  }
  return false;
}

bool IfNode::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                      int wg_id,
                                      std::set<BufferAccessPoint> &result,
                                      int64_t time_offset) const {
  if (task) {
    if (task->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                       time_offset))
      return true;
  }
  bool then_must = false, else_must = false;
  if (then_child)
    then_must = then_child->CollectFirstAccessPoints(buffer, is_write, wg_id,
                                                     result, time_offset);
  if (else_child)
    else_must = else_child->CollectFirstAccessPoints(buffer, is_write, wg_id,
                                                     result, time_offset);
  return then_must && else_must;
}

bool IfNode::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                     int wg_id,
                                     std::set<BufferAccessPoint> &result,
                                     int64_t time_offset) const {
  bool then_must = false, else_must = false;
  if (then_child)
    then_must = then_child->CollectLastAccessPoints(buffer, is_write, wg_id,
                                                    result, time_offset);
  if (else_child)
    else_must = else_child->CollectLastAccessPoints(buffer, is_write, wg_id,
                                                    result, time_offset);
  if (then_must && else_must)
    return true;
  if (task) {
    if (task->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                      time_offset))
      return true;
  }
  return false;
}

bool ControlNode::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                           int wg_id,
                                           std::set<BufferAccessPoint> &result,
                                           int64_t time_offset) const {
  if (task) {
    if (task->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                       time_offset))
      return true;
  }
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    std::vector<const ScheduleUnit *> ordered;
    ordered.reserve(seq->children.size());
    for (const auto &c : seq->children) {
      ICHECK(c->IsScheduleUnit())
          << "Sequence child must be a ScheduleUnit post-scheduling";
      ordered.push_back(static_cast<const ScheduleUnit *>(c.get()));
    }
    body_must = SequenceCollectFirstAccessPoints(ordered, buffer, is_write,
                                                 wg_id, result, time_offset);
  } else if (child) {
    body_must = child->CollectFirstAccessPoints(buffer, is_write, wg_id, result,
                                                time_offset);
  }
  if (body_must && LoopMustExecute(this))
    return true;
  return false;
}

bool ControlNode::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                          int wg_id,
                                          std::set<BufferAccessPoint> &result,
                                          int64_t time_offset) const {
  int64_t trip_count = GetTripCount();
  int64_t ii_per_iter = GetIIperIter();
  int64_t last_iter_offset =
      (trip_count > 0 ? (trip_count - 1) : 0) * ii_per_iter;
  int64_t body_offset = time_offset + last_iter_offset;
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    std::vector<const ScheduleUnit *> ordered;
    ordered.reserve(seq->children.size());
    for (const auto &c : seq->children) {
      ICHECK(c->IsScheduleUnit())
          << "Sequence child must be a ScheduleUnit post-scheduling";
      ordered.push_back(static_cast<const ScheduleUnit *>(c.get()));
    }
    body_must = SequenceCollectLastAccessPoints(ordered, buffer, is_write,
                                                wg_id, result, body_offset);
  } else if (child) {
    body_must = child->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                               body_offset);
  }
  if (body_must && LoopMustExecute(this))
    return true;
  if (task) {
    if (task->CollectLastAccessPoints(buffer, is_write, wg_id, result,
                                      time_offset))
      return true;
  }
  return false;
}

bool SequenceNode::CollectFirstAccessPoints(const Buffer &buffer, bool is_write,
                                            int wg_id,
                                            std::set<BufferAccessPoint> &result,
                                            int64_t time_offset) const {
  std::vector<const ScheduleUnit *> ordered;
  ordered.reserve(children.size());
  for (const auto &c : children) {
    ICHECK(c->IsScheduleUnit())
        << "Sequence child must be a ScheduleUnit post-scheduling";
    ordered.push_back(static_cast<const ScheduleUnit *>(c.get()));
  }
  return SequenceCollectFirstAccessPoints(ordered, buffer, is_write, wg_id,
                                          result, time_offset);
}

bool SequenceNode::CollectLastAccessPoints(const Buffer &buffer, bool is_write,
                                           int wg_id,
                                           std::set<BufferAccessPoint> &result,
                                           int64_t time_offset) const {
  std::vector<const ScheduleUnit *> ordered;
  ordered.reserve(children.size());
  for (const auto &c : children) {
    ICHECK(c->IsScheduleUnit())
        << "Sequence child must be a ScheduleUnit post-scheduling";
    ordered.push_back(static_cast<const ScheduleUnit *>(c.get()));
  }
  return SequenceCollectLastAccessPoints(ordered, buffer, is_write, wg_id,
                                         result, time_offset);
}

} // namespace tl
} // namespace tvm
