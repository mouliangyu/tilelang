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
 * \brief ScheduleBuilder and task gathering for TileLang AutoSchedule
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

bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask) {
  if (((a->GetPipeMask() & b->GetPipeMask()) & mask) != 0)
    return true;
  return (a->GetHbmMask() & b->GetHbmMask()) != 0;
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) Each ordered child list (the root list or a
ControlNode body) is reordered by the Z3 scheduler and the children carry
scheduling metadata directly. 2) For each ControlNode, its `children` are the
Z3-scheduled loop body. Conditional execution is carried by each node's
`conditions` delta (no dedicated IfNode), so guarded tasks are ordinary
schedule candidates.
*/
void ScheduleBuilder::ScheduleList(
    std::vector<std::shared_ptr<IRStructure>> &children) {
  std::vector<For> loop_stack;
  std::vector<std::shared_ptr<IRStructure>> origin_children = children;
  for (size_t i = 0; i < origin_children.size(); ++i) {
    origin_children[i]->SetIndex(i);
  }
  for (auto &child : origin_children) {
    ScheduleRecursive(child, loop_stack);
  }
  children = Z3SchedulePython(origin_children);
  for (size_t i = 0; i < children.size(); ++i) {
    children[i]->SetIndex(i);
  }
}

void ScheduleBuilder::ScheduleRecursive(std::shared_ptr<IRStructure> &node,
                                        std::vector<For> &loop_stack) {
  if (!node)
    return;

  if (node->IsTask()) {
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children = ctrl->children;
    for (size_t i = 0; i < origin_children.size(); ++i) {
      origin_children[i]->SetIndex(i);
    }
    // Children see this loop in their loop stack; this node's own Z3 schedule
    // sees only its enclosing ancestors (not itself), so pop before scheduling.
    loop_stack.push_back(ctrl->control);
    for (auto &child : origin_children) {
      ScheduleRecursive(child, loop_stack);
    }
    loop_stack.pop_back();
    Z3SchedulePythonLoop(ctrl, loop_stack);
    for (size_t i = 0; i < ctrl->children.size(); ++i) {
      ctrl->children[i]->SetIndex(i);
    }
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

} // namespace tl
} // namespace tvm
