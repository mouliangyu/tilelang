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

#include "../../op/builtin.h"
#include "../../op/gemm_py.h"
#include "../../op/utils.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
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

bool SameBuffer(const BufferRegion &a, const BufferRegion &b) {
  return a->buffer.same_as(b->buffer);
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasDependency(const IRStructure *a, const IRStructure *b) {
  if (a->ContainsLoopBreak())
    return true;
  if (b->ContainsLoopBreak())
    return true;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        return true;
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &write_var_a : a->GetWriteVars()) {
    for (const auto &read_var_b : b->GetReadVars()) {
      if (SameVar(write_var_a, read_var_b))
        return true;
    }
  }
  return false;
}

bool HasRegisterDependency(const IRStructure *a, const IRStructure *b) {
  if (a->ContainsLoopBreak())
    return true;
  if (b->ContainsLoopBreak())
    return true;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    if (IsSharedBuffer(write_region_a.get()->buffer) ||
        IsL1Buffer(write_region_a.get()->buffer))
      continue;
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        return true;
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    if (IsSharedBuffer(read_region_a.get()->buffer) ||
        IsL1Buffer(read_region_a.get()->buffer))
      continue;
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        return true;
    }
  }
  return false;
}

std::set<Buffer> GetSharedDependencies(const IRStructure *a,
                                       const IRStructure *b) {
  std::set<Buffer> deps;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    if (!IsSharedBuffer(write_region_a->buffer) &&
        !IsL1Buffer(write_region_a->buffer))
      continue;
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        deps.insert(write_region_a->buffer);
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        deps.insert(write_region_a->buffer);
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    if (!IsSharedBuffer(read_region_a->buffer) &&
        !IsL1Buffer(read_region_a->buffer))
      continue;
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        deps.insert(read_region_a->buffer);
    }
  }
  return deps;
}

bool HasRegisterRegion(const IRStructure *node) {
  return CountRegisterRegions(node) > 0;
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b) {
  return (a->GetPipeMask() & b->GetPipeMask()) != 0;
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) For each SequenceNode, its children are reordered by Z3
scheduler and wrapped in ScheduleUnits. 2) For each ControlNode, its child is a
SequenceNode with Z3-scheduled children wrapped in ScheduleUnits. 3) For each
IfNode, its then_child and else_child are recursively scheduled (if exist).
*/
void ScheduleUnitBuilder::ScheduleRecursive(
    std::shared_ptr<IRStructure> &node, const std::set<Buffer> &used_buffers) {
  if (!node)
    return;

  auto ChildrenScheduleHelper =
      [&](std::vector<std::shared_ptr<IRStructure>> origin_children)
      -> std::vector<std::shared_ptr<IRStructure>> {
    std::vector<IRStructure *> child_nodes;
    child_nodes.reserve(origin_children.size());
    for (const auto &child : origin_children) {
      child_nodes.push_back(child.get());
    }

    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes = Z3SchedulePython(child_nodes);

    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != child_nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < child_nodes.size(); ++i) {
        node_to_index[child_nodes[i]] = i;
      }

      std::vector<std::shared_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());

      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        size_t old_idx = it->second;
        reordered_children.emplace_back(origin_children[old_idx]);
      }

      origin_children = reordered_children;
    }
    for (auto &node : origin_children) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->child = std::shared_ptr<IRStructure>(node);
      node = unit;
    }
    return origin_children;
  };

  if (node->IsTask()) {
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());

    std::vector<std::shared_ptr<IRStructure>> seq_children, origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child, child_used_buffers);
    }

    seq->children = ChildrenScheduleHelper(origin_children);
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
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else if (ctrl->child->IsWrapper()) {
        auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
        for (auto &child : origin_children) {
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else {
        ScheduleRecursive(ctrl->child, used_buffers);
        auto old_child = ctrl->child;
        auto seq_node = std::make_shared<SequenceNode>();
        seq_node->children = {old_child};
        ctrl->child = seq_node;
        Z3SchedulePythonLoop(ctrl, used_buffers);
      }
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child, child_used_buffers);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = ChildrenScheduleHelper(origin_children);
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
      ScheduleRecursive(if_node->then_child, used_buffers);
    }
    if (if_node->else_child) {
      ScheduleRecursive(if_node->else_child, used_buffers);
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
