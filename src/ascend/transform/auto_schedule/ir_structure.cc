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
 * \brief IRStructure node cloning and debug helpers for TileLang.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_hash.h>
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
#include <limits>
#include <map>
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

namespace {

void RebuildIRStructureParentsImpl(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    IRStructure *parent) {
  for (const std::shared_ptr<IRStructure> &node : nodes) {
    if (!node)
      continue;
    node->SetParent(parent);
    if (!node->IsControl())
      continue;
    auto *control = static_cast<ControlNode *>(node.get());
    if (control->task)
      control->task->SetParent(control);
    RebuildIRStructureParentsImpl(control->children, control);
  }
}

} // namespace

std::vector<IRStructure *> IRStructure::Path() {
  std::vector<IRStructure *> path;
  for (IRStructure *node = this; node != nullptr; node = node->GetParent())
    path.push_back(node);
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<const IRStructure *> IRStructure::Path() const {
  std::vector<const IRStructure *> path;
  for (const IRStructure *node = this; node != nullptr;
       node = node->GetParent())
    path.push_back(node);
  std::reverse(path.begin(), path.end());
  return path;
}

void RebuildIRStructureParents(
    const std::vector<std::shared_ptr<IRStructure>> &roots) {
  RebuildIRStructureParentsImpl(roots, nullptr);
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>(CloneExtraInfo());
  // Copy statement
  new_task->stmt = stmt;
  // Copy resource pipe mask
  new_task->SetPipeMask(GetPipeMask());
  new_task->SetHbmMask(GetHbmMask());
  new_task->SetSpecialWriteMask(GetSpecialWriteMask());
  new_task->SetReadsPadValue(ReadsPadValueRegister());
  new_task->SetReadsAtomic(ReadsAtomicRegister());
  if (IsPerCoreTask())
    new_task->MarkPerCoreTask();
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  new_task->SetReadVars(GetReadVars());
  new_task->SetWriteVars(GetWriteVars());
  new_task->outer_ctx = outer_ctx;
  new_task->SetCoreMask(GetCoreMask());
  new_task->CopyBaseMetadataFrom(*this);
  return new_task;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>(CloneExtraInfo());
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone children
  new_ctrl->children.reserve(children.size());
  for (const auto &child : children) {
    new_ctrl->children.push_back(child ? child->Clone() : nullptr);
  }
  if (task) {
    new_ctrl->task = std::static_pointer_cast<TaskNode>(task->Clone());
    new_ctrl->task->SetParent(new_ctrl.get());
  }
  RebuildIRStructureParentsImpl(new_ctrl->children, new_ctrl.get());
  new_ctrl->CopyBaseMetadataFrom(*this);
  return new_ctrl;
}

void CollectAllTaskNodes(IRStructure *node,
                         std::vector<TaskNode *> &all_tasks) {
  if (!node)
    return;
  if (node->IsTask()) {
    all_tasks.push_back(static_cast<TaskNode *>(node));
    return;
  }
  ICHECK(node->IsControl());
  auto *control = static_cast<ControlNode *>(node);
  for (const auto &child : control->children)
    CollectAllTaskNodes(child.get(), all_tasks);
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
void PrintIRStructure(const IRStructure *node, int indent) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  for (const auto &guard : node->GetGuards()) {
    if (guard->IsAttribute()) {
      const auto *attribute_guard =
          static_cast<const AttributeGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [attr " << attribute_guard->key << "] "
                << attribute_guard->node;
    } else {
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [guard] " << condition_guard->condition;
    }
  }

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmt: " << task->stmt;
    for (auto &region : task->GetReadRegions()) {
      LOG(INFO) << indent_str << "  Read Region: " << region;
    }
    for (auto &region : task->GetWriteRegions()) {
      LOG(INFO) << indent_str << "  Write Region: " << region;
    }
    for (auto &var : task->GetReadVars()) {
      LOG(INFO) << indent_str << "  Read Var: " << var;
    }
    for (auto &var : task->GetWriteVars()) {
      LOG(INFO) << indent_str << "  Write Var: " << var;
    }
    LOG(INFO) << indent_str << "  pipe_mask: 0x" << std::hex
              << task->GetPipeMask() << std::dec;
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str
              << "  core_mask: " << static_cast<int>(task->GetCoreMask());
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    LOG(INFO) << indent_str << "  Loop body: " << control->children.size()
              << " children";
    for (size_t i = 0; i < control->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(control->children[i].get(), indent + 2);
    }
  }
}

} // namespace tl
} // namespace tvm
