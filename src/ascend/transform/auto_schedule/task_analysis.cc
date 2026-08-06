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

#include "./task_analysis.h"

#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>

#include "./memory_detector.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

class ResourceAnalyzer : public StmtExprVisitor {
public:
  static TaskResourceUsage Analyze(const Stmt &stmt) {
    ResourceAnalyzer analyzer;
    analyzer(stmt);
    return analyzer.usage_;
  }

private:
  static uint16_t GetHbmPortBit(const Buffer &src, const Buffer &dst) {
    bool aiv = (IsGlobalBuffer(src) && IsSharedBuffer(dst)) ||
               (IsSharedBuffer(src) && IsGlobalBuffer(dst));
    bool aic = (IsGlobalBuffer(src) && IsL1Buffer(dst)) ||
               (IsL0CBuffer(src) && IsGlobalBuffer(dst));
    if (aiv)
      return static_cast<uint16_t>(HbmPort::kAiv);
    if (aic)
      return static_cast<uint16_t>(HbmPort::kAic);
    return 0;
  }

  void VisitExpr_(const CallNode *op) final {
    static const auto copy_op = Op::Get("tl.tileop.copy");

    if (op->op.same_as(copy_op)) {
      Copy copy_obj(op->args, op->annotations);
      const CopyNode *copy = copy_obj.get();
      usage_.hbm_mask |= GetHbmPortBit(copy->src, copy->dst);
      if (copy_obj->GetDataSelect() != 0)
        usage_.reads_pad_value = true;
      if (IsGlobalBuffer(copy->dst))
        usage_.reads_atomic = true;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  // Task-level blocks are atomic to all task analyses.
  void VisitStmt_(const SBlockNode *) final {}

  TaskResourceUsage usage_;
};

} // namespace

bool HasTaskBufferAccess(const Stmt &stmt) {
  TaskAccessInfo accesses = AnalyzeTaskAccesses(stmt);
  return !accesses.read_regions.empty() || !accesses.write_regions.empty();
}

bool HasTaskBufferAccess(const TaskNode *task) {
  ICHECK(task != nullptr);
  return !task->GetReadRegions().empty() || !task->GetWriteRegions().empty();
}

[[noreturn]] void ThrowUnmarkedTask(const char *consumer,
                                    const char *node_kind) {
  TVM_FFI_THROW(ValueError)
      << consumer << " found an unmarked " << node_kind
      << ". Run tl.transform.NormalizeTask immediately before " << consumer
      << ".";
}

TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &stmt) {
  return ResourceAnalyzer::Analyze(stmt);
}

TaskAccessInfo AnalyzeTaskAccesses(const Stmt &stmt) {
  MemoryAccessDetector memory_detector;
  memory_detector.Analyze(stmt);
  TaskAccessInfo result;
  result.read_regions = memory_detector.GetReadRegions();
  result.write_regions = memory_detector.GetWriteRegions();
  result.read_vars = memory_detector.GetReadVars();
  result.write_vars = memory_detector.GetWriteVars();
  return result;
}

void ApplyTaskResourceUsage(const TaskResourceUsage &usage, TaskNode *task) {
  ICHECK(task != nullptr);
  task->SetHbmMask(usage.hbm_mask);
  if (usage.reads_pad_value)
    task->SetReadsPadValue(true);
  if (usage.reads_atomic)
    task->SetReadsAtomic(true);
}

void ApplyTaskAccesses(const TaskAccessInfo &accesses, TaskNode *task) {
  ICHECK(task != nullptr);
  for (const BufferRegion &region : accesses.read_regions)
    task->AddReadRegion(region);
  for (const BufferRegion &region : accesses.write_regions)
    task->AddWriteRegion(region);
  for (const Var &var : accesses.read_vars)
    task->AddReadVar(var);
  for (const Var &var : accesses.write_vars)
    task->AddWriteVar(var);
}

} // namespace tl
} // namespace tvm
