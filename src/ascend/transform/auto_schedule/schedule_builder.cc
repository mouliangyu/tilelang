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
 * \file schedule_builder.cc
 * \brief ScheduleUnitBuilder and task gathering for TileLang AutoSchedule
 */

#include "./schedule_builder.h"

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
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

void GatherTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                     std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  for (const auto &node : nodes) {
    if (node->IsTask()) {
      task_nodes.emplace_back(node);
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node.get());
      GatherTaskNodes(seq->children, task_nodes);
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node.get());
      if (wrapper->task)
        task_nodes.emplace_back(wrapper->task);
      if (wrapper->child)
        GatherTaskNodesSingle(wrapper->child, task_nodes);
    } else if (node->IsControl()) {
      task_nodes.emplace_back(node);
    } else if (node->IsIf()) {
      task_nodes.emplace_back(node);
    } else {
      LOG(FATAL) << "Unknown node type in GatherTaskNodes";
    }
  }
}

void GatherTaskNodesSingle(
    const std::shared_ptr<IRStructure> &node,
    std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  return GatherTaskNodes({node}, task_nodes);
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasVarDependency(const IRStructure *a, const IRStructure *b) {
  for (const auto &write_var_a : a->GetWriteVars()) {
    for (const auto &read_var_b : b->GetReadVars()) {
      if (SameVar(write_var_a, read_var_b))
        return true;
    }
  }
  return false;
}

bool HasRegisterRegion(const IRStructure *node) {
  return CountRegisterRegions(node) > 0;
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b) {
  if ((a->GetPipeMask() & b->GetPipeMask()) != 0)
    return true;
  return (a->GetHbmMask() & b->GetHbmMask()) != 0;
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) For each SequenceNode, its children are reordered by Z3
scheduler and wrapped in ScheduleUnits. 2) For each ControlNode, its child is a
SequenceNode with Z3-scheduled children wrapped in ScheduleUnits. 3) For each
IfNode, its then_child and else_child are recursively scheduled (if exist).
*/
void ScheduleUnitBuilder::ScheduleRecursive(
    std::shared_ptr<IRStructure> &node) {
  if (!node)
    return;

  if (node->IsTask()) {
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());

    std::vector<std::shared_ptr<IRStructure>> seq_children, origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      ScheduleRecursive(child);
    }

    seq->children = Z3SchedulePython(origin_children);
    int64_t overall_latency = 0;
    for (const auto &child : seq->children) {
      overall_latency += child->GetLatency();
    }
    seq->SetLatency(overall_latency);
    seq->SetII(overall_latency);
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());

    if (ctrl->child) {
      if (ctrl->child->IsSequence()) {
        auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes(seq_body->children, origin_children);
        for (auto &child : origin_children) {
          ScheduleRecursive(child);
        }
        Z3SchedulePythonLoop(ctrl);
      } else if (ctrl->child->IsWrapper()) {
        auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
        for (auto &child : origin_children) {
          ScheduleRecursive(child);
        }
        Z3SchedulePythonLoop(ctrl);
      } else {
        ScheduleRecursive(ctrl->child);
        auto old_child = ctrl->child;
        auto seq_node = std::make_shared<SequenceNode>();
        seq_node->children = {old_child};
        ctrl->child = seq_node;
        Z3SchedulePythonLoop(ctrl);
      }
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      ScheduleRecursive(child);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = Z3SchedulePython(origin_children);
    int64_t overall_latency = 0;
    for (const auto &child : seq_node->children) {
      overall_latency += child->GetLatency();
    }
    seq_node->SetLatency(overall_latency);
    seq_node->SetII(overall_latency);
    node = seq_node;
    return;
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node.get());
    if (if_node->then_child) {
      ScheduleRecursive(if_node->then_child);
    }
    if (if_node->else_child) {
      ScheduleRecursive(if_node->else_child);
    }
    if_node->SetLatency(
        std::max(if_node->then_child ? if_node->then_child->GetLatency() : 0,
                 if_node->else_child ? if_node->else_child->GetLatency() : 0));
    if_node->SetII(
        std::max(if_node->then_child ? if_node->then_child->GetII() : 0,
                 if_node->else_child ? if_node->else_child->GetII() : 0));
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

} // namespace tl
} // namespace tvm
