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
 * \file latency_estimator.cc
 * \brief Annotate every Ascend AutoSchedule task with latency and II.
 */

#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "./task_cost_estimator.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using ffi::GetRef;

namespace {

using TaskCostMap =
    std::unordered_map<Stmt, TaskCost, ObjectPtrHash, ObjectPtrEqual>;

class TaskCostCollector : public TaskAwareConstrVisitor {
public:
  static TaskCostMap Collect(const PrimFunc &func) {
    TaskCostCollector collector;
    collector(func->body);
    return std::move(collector.costs_);
  }

private:
  bool InScheduleBody() const { return schedule_body_depth_ > 0; }

  std::shared_ptr<TaskNode> NewTask(const Stmt &stmt) const {
    auto task = std::make_shared<TaskNode>();
    task->stmt = stmt;
    task->outer_ctx = GetConstrSet();
    return task;
  }

  void RecordCost(const Stmt &key, const std::shared_ptr<TaskNode> &task,
                  const TaskAnnotations &annotations) {
    TaskCost estimated{task->GetLatency(), task->GetII()};
    TaskCost cost = ResolveTaskCost(estimated, annotations);
    task->SetLatency(cost.latency);
    task->SetII(cost.ii);
    costs_[key] = cost;
  }

  void EstimateMarkedTask(const AttrStmtNode *op) {
    TaskAnnotations annotations = ParseTaskAnnotations(op->node);
    auto task = NewTask(op->body);
    task->SetPipeMask(RequireTaskPipe(annotations, "T.Task"));
    ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(op->body), task.get());
    ApplyTaskAccesses(AnalyzeTaskAccesses(op->body), task.get());
    estimator_.EstimateComposedTask(task.get());
    Stmt marker = GetRef<Stmt>(op);
    RecordCost(marker, task, annotations);
    if (per_core_task_depth_ > 0 && HasTaskBufferAccess(task.get()))
      per_core_task_candidates_.push_back({marker, task});
  }

  void EstimatePerCoreTask(const AttrStmtNode *op) {
    TVM_FFI_CHECK_EQ(per_core_task_depth_, 0, ValueError)
        << "Nested T.PerCoreTask regions are not supported";
    TaskAnnotations annotations = ParseTaskAnnotations(op->node);

    std::vector<PerCoreTaskCandidate> saved_candidates;
    saved_candidates.swap(per_core_task_candidates_);
    ++per_core_task_depth_;
    VisitStmt(op->body);
    --per_core_task_depth_;
    std::vector<PerCoreTaskCandidate> candidates;
    candidates.swap(per_core_task_candidates_);
    per_core_task_candidates_.swap(saved_candidates);

    TVM_FFI_CHECK(!candidates.empty(), ValueError)
        << "T.PerCoreTask must contain at least one normalized T.Task "
           "candidate before LatencyEstimator";
    auto task = NewTask(op->body);
    uint16_t pipe = RequireTaskPipe(annotations, "T.PerCoreTask");
    for (const PerCoreTaskCandidate &candidate : candidates) {
      uint16_t candidate_pipe = candidate.task->GetPipeMask();
      TVM_FFI_CHECK(candidate_pipe != 0 &&
                        (candidate_pipe & (candidate_pipe - 1)) == 0,
                    ValueError)
          << "Each dependency-bearing statement in T.PerCoreTask must use "
             "exactly one hardware pipe, but statement mask="
          << candidate_pipe << ", statement=" << candidate.task->stmt;
      TVM_FFI_CHECK_EQ(candidate_pipe, pipe, ValueError)
          << "T.PerCoreTask candidate pipe does not match its annotated pipe. "
             "Annotated mask="
          << pipe << ", statement mask=" << candidate_pipe
          << ", statement=" << candidate.task->stmt;
    }
    task->SetPipeMask(pipe);
    estimator_.EstimatePerCoreTask(task.get(), op->body, candidates);
    RecordCost(GetRef<Stmt>(op), task, annotations);
  }

  void VisitStmt_(const ForNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    if (op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    ThrowUnmarkedTask("LatencyEstimator", "parallel loop");
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    ThrowUnmarkedTask("LatencyEstimator", "Evaluate");
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    ThrowUnmarkedTask("LatencyEstimator", "BufferStore");
  }

  void VisitStmt_(const BindNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    ThrowUnmarkedTask("LatencyEstimator", "Bind");
  }

  void VisitStmt_(const WhileNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    ThrowUnmarkedTask("LatencyEstimator", "While");
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "tilelang_root") {
      ++schedule_body_depth_;
      VisitStmt(op->body);
      --schedule_body_depth_;
      return;
    }
    if (InScheduleBody()) {
      ThrowUnmarkedTask("LatencyEstimator", "block");
    }
    ConstrVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (!InScheduleBody()) {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    if (op->attr_key == tl::attr::kAscendTask) {
      EstimateMarkedTask(op);
      return;
    }
    if (op->attr_key == tl::attr::kAscendPerCoreTask) {
      EstimatePerCoreTask(op);
      return;
    }
    ConstrVisitor::VisitStmt_(op);
  }

  TaskCostEstimator estimator_;
  TaskCostMap costs_;
  int schedule_body_depth_{0};
  int per_core_task_depth_{0};
  std::vector<PerCoreTaskCandidate> per_core_task_candidates_;
};

class TaskAnnotationRewriter : public StmtMutator {
public:
  explicit TaskAnnotationRewriter(const TaskCostMap &costs) : costs_(costs) {}

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  Stmt VisitStmt(const Stmt &stmt) final {
    auto it = costs_.find(stmt);
    if (it == costs_.end())
      return StmtMutator::VisitStmt(stmt);

    const auto *attr = stmt.as<AttrStmtNode>();
    ICHECK(attr && (attr->attr_key == tl::attr::kAscendTask ||
                    attr->attr_key == tl::attr::kAscendPerCoreTask));
    Stmt body = attr->body;
    if (attr->attr_key == tl::attr::kAscendPerCoreTask)
      body = VisitStmt(body);
    return AttrStmt(MakeTaskAnnotations(it->second, attr->node), attr->attr_key,
                    attr->value, std::move(body), attr->span);
  }

  const TaskCostMap &costs_;
};

PrimFunc RunLatencyEstimator(PrimFunc func) {
  TaskCostMap costs = TaskCostCollector::Collect(func);
  TVM_FFI_CHECK(!costs.empty(), ValueError)
      << "LatencyEstimator found no schedulable tasks inside tilelang_root";
  TaskAnnotationRewriter rewriter(costs);
  func.CopyOnWrite()->body = rewriter.Rewrite(func->body);
  return func;
}

} // namespace

tvm::transform::Pass LatencyEstimatorPass() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return RunLatencyEstimator(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LatencyEstimator", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LatencyEstimator", LatencyEstimatorPass);
}

} // namespace tl
} // namespace tvm
