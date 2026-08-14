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
 * \file assign_core.cc
 * \brief Assign scheduled tasks to Ascend vector and cube cores.
 */

#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

namespace {

static bool IsLetDeclTask(const TaskNode *task) {
  return task != nullptr && GetFlatTaskBind(task->stmt).defined();
}
struct CrossCorePipeInfo {
  SpecialRegister special_register;
  CoreMask core_mask;
};

static CrossCorePipeInfo GetCrossCorePipeInfo(const PrimExpr &pipe_arg) {
  const auto *pipe_imm = pipe_arg.as<StringImmNode>();
  ICHECK(pipe_imm)
      << "Auto-scheduled cross-core synchronization requires a constant pipe "
         "string";
  const String &pipe = pipe_imm->value;
  if (pipe == "PIPE_S")
    return {SpecialRegister::kPipeScalar, kCoreUnassigned};
  if (pipe == "PIPE_V")
    return {SpecialRegister::kPipeVector, kCoreVector};
  if (pipe == "PIPE_M")
    return {SpecialRegister::kPipeCube, kCoreCube};
  if (pipe == "PIPE_MTE1")
    return {SpecialRegister::kPipeMTE1, kCoreCube};
  if (pipe == "PIPE_MTE2")
    return {SpecialRegister::kPipeMTE2, kCoreUnassigned};
  if (pipe == "PIPE_MTE3")
    return {SpecialRegister::kPipeMTE3, kCoreVector};
  if (pipe == "PIPE_FIX")
    return {SpecialRegister::kPipeFixpipe, kCoreCube};
  LOG(FATAL) << "Ascend AutoSchedule does not support cross-core pipe \""
             << pipe
             << "\". Expected one of PIPE_S, PIPE_V, PIPE_M, PIPE_MTE1, "
                "PIPE_MTE2, PIPE_MTE3, or PIPE_FIX";
  return {SpecialRegister::kNone, kCoreUnassigned};
}

static Optional<Call> GetCrossCoreCall(const Call &call) {
  if (!call->op.same_as(tl::ascend_cross_core_set_flag()) &&
      !call->op.same_as(tl::ascend_cross_core_wait_flag())) {
    return std::nullopt;
  }
  ICHECK_EQ(call->args.size(), 3)
      << "ascend_cross_core_set_flag/wait_flag expects exactly 3 arguments";
  const auto *mode = call->args[0].as<IntImmNode>();
  ICHECK(mode)
      << "Ascend AutoSchedule requires cross-core mode_id to be constant 0";
  ICHECK_EQ(mode->value, 0)
      << "Ascend AutoSchedule only supports cross-core synchronization with "
         "mode_id=0; disable AutoSchedule for manually scheduled mode_id="
      << mode->value;
  return call;
}

static std::vector<Call> CollectCrossCoreCalls(const TaskNode *task) {
  class Collector : public StmtExprVisitor {
  public:
    std::vector<Call> calls;

  private:
    void VisitExpr_(const CallNode *op) final {
      Call call = GetRef<Call>(op);
      if (GetCrossCoreCall(call).defined())
        calls.push_back(call);
      StmtExprVisitor::VisitExpr_(op);
    }
  } collector;
  collector(task->stmt);
  return collector.calls;
}

using CrossCoreCallMap = std::unordered_map<TaskNode *, std::vector<Call>>;
using CrossCoreHintMap =
    std::unordered_map<PrimExpr, CoreMask, ffi::StructuralHash,
                       ffi::StructuralEqual>;

static void AssignCrossCoreTasksInList(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    const CrossCoreCallMap &cross_core_calls, CoreMask kernel_core_mask) {
  std::vector<TaskNode *> scoped_tasks;
  for (const auto &node : nodes) {
    if (!node->IsTask())
      continue;
    auto *task = static_cast<TaskNode *>(node.get());
    // PerCoreTask consumes its embedded cross-core synchronization internally.
    // Keep the core already inferred from its ordinary operation instead
    // of merging those calls into outer placement inference.
    if (task->IsPerCoreTask())
      continue;
    auto calls_it = cross_core_calls.find(task);
    if (calls_it == cross_core_calls.end())
      continue;
    scoped_tasks.push_back(task);
  }

  CrossCoreHintMap flag_core_hints;
  for (TaskNode *task : scoped_tasks) {
    for (const Call &call : cross_core_calls.at(task))
      flag_core_hints[call->args[2]] |=
          GetCrossCorePipeInfo(call->args[1]).core_mask;
  }

  std::unordered_map<TaskNode *, CoreMask> task_core_masks;
  for (TaskNode *task : scoped_tasks) {
    for (const Call &call : cross_core_calls.at(task)) {
      const PrimExpr &flag_id = call->args[2];
      CoreMask core_mask = flag_core_hints.at(flag_id);
      if (core_mask == kCoreBroadcast) {
        LOG(FATAL) << "Cross-core set/wait calls with flag " << flag_id
                   << " in one control scope resolve to both Vector and Cube "
                      "cores; all matching calls must run on the same core";
      }
      if (core_mask == kCoreUnassigned && IsConcreteCore(kernel_core_mask)) {
        // PIPE_S and PIPE_MTE2 exist on both core types. A pure kernel supplies
        // the only possible placement when neither side of the pair is pinned.
        core_mask = kernel_core_mask;
      } else if (!IsConcreteCore(core_mask)) {
        LOG(FATAL)
            << "Cannot infer the core for cross-core set/wait calls with flag "
            << flag_id << " in one control scope"
            << ": PIPE_S and PIPE_MTE2 are ambiguous in a mixed or empty "
               "kernel. Use PIPE_V/PIPE_MTE3 for Vector or "
               "PIPE_M/PIPE_MTE1/PIPE_FIX for Cube on at least one call in "
               "that scope";
      }
      task_core_masks[task] |= core_mask;
    }
  }
  for (const auto &[task, core_mask] : task_core_masks) {
    if (core_mask == kCoreBroadcast) {
      LOG(FATAL) << "Cross-core synchronization calls in one task resolve to "
                    "both Vector and Cube cores: "
                 << task->stmt;
    }
    task->SetCoreMask(core_mask);
  }

  // A flag ID may be reused sequentially in another control scope. Recurse
  // into each ordered child list so only calls with the same immediate parent
  // influence the same core inference.
  for (const auto &node : nodes) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node.get());
      AssignCrossCoreTasksInList(control->children, cross_core_calls,
                                 kernel_core_mask);
    }
  }
}

static void
AssignCrossCoreTasks(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNode *> &all_tasks) {
  CoreMask kernel_core_mask = kCoreUnassigned;
  CrossCoreCallMap cross_core_calls;
  for (TaskNode *task : all_tasks) {
    if (task->IsPerCoreTask()) {
      CoreMask core_mask = task->GetCoreMask();
      if (IsConcreteCore(core_mask))
        kernel_core_mask |= core_mask;
      continue;
    }
    std::vector<Call> calls = CollectCrossCoreCalls(task);
    if (!calls.empty()) {
      cross_core_calls.emplace(task, std::move(calls));
      continue;
    }
    CoreMask core_mask = task->GetCoreMask();
    if (IsConcreteCore(core_mask))
      kernel_core_mask |= core_mask;
  }
  AssignCrossCoreTasksInList(root, cross_core_calls, kernel_core_mask);
}

// Assign scalar, register-buffer, and special-register producers to the core
// mask union of their consumers. CrossCore synchronization remains a dedicated
// side-effecting path and contributes no special-register demand.
class CoreMaskPropagator {
public:
  static void Assign(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNode *> &all_tasks) {
    CoreMaskPropagator analyzer(root, all_tasks);
    analyzer.Run_();
  }

private:
  static constexpr int kSpecialRegisterBits =
      std::numeric_limits<uint16_t>::digits;
  using VarDemandMap =
      std::unordered_map<Var, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  using BufferDemandMap =
      std::unordered_map<Buffer, CoreMask, ObjectPtrHash, ObjectPtrEqual>;
  using SpecialRegisterDemandMap =
      std::unordered_map<const TaskNode *, CoreMask>;
  using SpecialRegisterReadDemand = std::array<CoreMask, kSpecialRegisterBits>;
  using SpecialRegisterWriters =
      std::array<std::vector<TaskNode *>, kSpecialRegisterBits>;
  struct SpecialRegisterScopeDemand {
    SpecialRegisterReadDemand readers{};
    SpecialRegisterWriters writers;
  };

  CoreMaskPropagator(const std::vector<std::shared_ptr<IRStructure>> &root,
                     const std::vector<TaskNode *> &all_tasks)
      : root_(root), all_tasks_(all_tasks) {}

  void Run_() {
    AssignInitialCoreMasks_();
    CollectSpecialRegisterDemand_(root_);
    AssignSpecialRegisterWriters_();
    AssignCrossCoreTasks(root_, all_tasks_);
    for (const auto &node : root_)
      SeedReads_(node.get(), kCoreBroadcast);
    PropagateReadsToFixedPoint_();
    AssignScalarWriters_();
    AssignFallbackCore_();
  }

  void AssignInitialCoreMasks_() {
    for (TaskNode *task : all_tasks_) {
      // Special-register writers are placed from their concrete use sites
      // after ordinary compute and data-movement tasks have been pinned.
      if (task->GetSpecialWriteMask() != 0) {
        task->SetCoreMask(kCoreUnassigned);
        continue;
      }
      uint16_t pipe_mask = task->GetPipeMask();
      if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kCube)) {
        task->SetCoreMask(kCoreCube);
      } else if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kVector)) {
        task->SetCoreMask(kCoreVector);
      } else if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1) ||
                 pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe)) {
        task->SetCoreMask(kCoreCube);
      } else if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3)) {
        task->SetCoreMask(kCoreVector);
      } else if (pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2)) {
        bool to_l1 = false;
        for (const BufferRegion &region : task->GetWriteRegions()) {
          if (IsL1Buffer(region->buffer)) {
            to_l1 = true;
            break;
          }
        }
        task->SetCoreMask(to_l1 ? kCoreCube : kCoreVector);
      } else if (task->UsesPipe(ResourcePipe::kScalar) || pipe_mask == 0) {
        // A scalar / mask-0 task issues on the S pipe, but its core is still
        // fixed by the on-chip memory it touches: UB is Vector-side, while
        // L1/L0A/L0B/L0C are Cube-side.
        bool touches_ub = false;
        bool touches_cube = false;
        auto scan = [&](const Array<BufferRegion> &regions) {
          for (const BufferRegion &region : regions) {
            if (IsSharedBuffer(region->buffer))
              touches_ub = true;
            else if (IsL1Buffer(region->buffer) ||
                     IsL0ABuffer(region->buffer) ||
                     IsL0BBuffer(region->buffer) || IsL0CBuffer(region->buffer))
              touches_cube = true;
          }
        };
        scan(task->GetReadRegions());
        scan(task->GetWriteRegions());
        if (touches_ub && !touches_cube) {
          task->SetCoreMask(kCoreVector);
        } else if (touches_cube && !touches_ub) {
          task->SetCoreMask(kCoreCube);
        } else {
          task->SetCoreMask(kCoreBroadcast);
        }
      }
    }
  }

  static CoreMask GetConcreteCoreMask_(const TaskNode *task) {
    CoreMask core_mask = task->GetCoreMask();
    return IsConcreteCore(core_mask) ? core_mask : kCoreUnassigned;
  }

  // Special registers have one independent instance per core. Return both
  // reader demand and concrete descendant writers so readers in an outer
  // sibling list can place writers nested under a control node. loop_break is
  // body-local and is therefore the only register not propagated outward.
  SpecialRegisterScopeDemand CollectSpecialRegisterDemand_(
      const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    SpecialRegisterScopeDemand scope;

    for (const auto &node : nodes) {
      if (node->IsControl()) {
        const auto *control = static_cast<const ControlNode *>(node.get());
        SpecialRegisterScopeDemand child_scope =
            CollectSpecialRegisterDemand_(control->children);
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if (bit == static_cast<uint16_t>(SpecialRegister::kLoopControl))
            continue;
          scope.readers[bit_index] |= child_scope.readers[bit_index];
          scope.writers[bit_index].insert(
              scope.writers[bit_index].end(),
              child_scope.writers[bit_index].begin(),
              child_scope.writers[bit_index].end());
        }
      } else {
        const auto *task = static_cast<const TaskNode *>(node.get());
        CoreMask core_mask = GetConcreteCoreMask_(task);
        uint16_t read_mask = node->GetSpecialReadMask();
        for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
          uint16_t bit = static_cast<uint16_t>(1U << bit_index);
          if ((read_mask & bit) != 0)
            scope.readers[bit_index] |= core_mask;
        }
      }

      if (!node->IsTask())
        continue;
      auto *task = static_cast<TaskNode *>(node.get());
      // CrossCore bits only order their target hardware pipe. The calls are
      // placed by matching flag uses in their immediate control scope, not by
      // demand.
      if (!CollectCrossCoreCalls(task).empty())
        continue;
      uint16_t write_mask = task->GetSpecialWriteMask();
      for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
        uint16_t bit = static_cast<uint16_t>(1U << bit_index);
        if ((write_mask & bit) != 0)
          scope.writers[bit_index].push_back(task);
      }
    }

    for (int bit_index = 0; bit_index < kSpecialRegisterBits; ++bit_index) {
      for (TaskNode *writer : scope.writers[bit_index])
        special_writer_demand_[writer] |= scope.readers[bit_index];
    }
    return scope;
  }

  void AssignSpecialRegisterWriters_() {
    for (TaskNode *task : all_tasks_) {
      // loop_break controls the enclosing loop itself, so it must survive in
      // every core clone of that loop even when all concrete work is
      // nested below another ControlNode and contributes no same-level demand.
      if (task->ContainsLoopBreak()) {
        task->SetCoreMask(kCoreBroadcast);
        continue;
      }
      auto it = special_writer_demand_.find(task);
      if (it == special_writer_demand_.end())
        continue;
      AssignTaskFromDemand_(task, it->second);
    }
  }

  bool AddReadsDemand_(const TaskNode *task, CoreMask mask) {
    if (!mask)
      return false;
    bool grew = false;
    for (const Var &var : task->GetReadVars()) {
      CoreMask &demand = var_demand_[var];
      if ((demand | mask) != demand) {
        demand |= mask;
        grew = true;
      }
    }
    for (const BufferRegion &region : task->GetReadRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      CoreMask &demand = buffer_demand_[region->buffer];
      if ((demand | mask) != demand) {
        demand |= mask;
        grew = true;
      }
    }
    return grew;
  }

  void AddGuardDemand_(const IRStructure *node, CoreMask mask) {
    if (!mask)
      return;
    for (const auto &guard : node->GetGuards()) {
      if (!guard->IsCondition())
        continue;
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      for (const Var &var : UndefinedVars(condition_guard->condition))
        var_demand_[var] |= mask;
    }
  }

  static CoreMask GetLoopBodyCoverage_(const IRStructure *node) {
    if (!node)
      return 0;
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      // A loop break controls whether the loop continues, so every core
      // that receives a clone of the loop must evaluate the break condition.
      if (task->ContainsLoopBreak())
        return kCoreBroadcast;
      return GetConcreteCoreMask_(task);
    }
    CoreMask mask = kCoreUnassigned;
    const auto *control = static_cast<const ControlNode *>(node);
    for (const auto &child : control->children)
      mask |= GetLoopBodyCoverage_(child.get());
    return mask;
  }

  void SeedReads_(const IRStructure *node, CoreMask enclosing) {
    if (!node)
      return;
    if (node->IsTask()) {
      const auto *task = static_cast<const TaskNode *>(node);
      CoreMask task_mask = task->GetCoreMask();
      CoreMask mask = IsConcreteCore(task_mask) ? task_mask : enclosing;
      AddGuardDemand_(task, mask);
      AddReadsDemand_(task, mask);
      return;
    }

    const auto *control = static_cast<const ControlNode *>(node);
    CoreMask coverage = GetLoopBodyCoverage_(node);
    AddGuardDemand_(control, coverage);
    if (control->task)
      AddReadsDemand_(control->task.get(), coverage);
    for (const auto &child : control->children)
      SeedReads_(child.get(), coverage);
  }

  CoreMask GetWriteDemand_(const TaskNode *task) const {
    CoreMask mask = kCoreUnassigned;
    auto special_it = special_writer_demand_.find(task);
    if (special_it != special_writer_demand_.end())
      mask |= special_it->second;
    for (const Var &var : task->GetWriteVars()) {
      auto it = var_demand_.find(var);
      if (it != var_demand_.end())
        mask |= it->second;
    }
    for (const BufferRegion &region : task->GetWriteRegions()) {
      if (!IsRegisterRegion(region))
        continue;
      auto it = buffer_demand_.find(region->buffer);
      if (it != buffer_demand_.end())
        mask |= it->second;
    }
    return mask;
  }

  void PropagateReadsToFixedPoint_() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (TaskNode *task : all_tasks_) {
        if (!task->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(task))
          continue;
        if (AddReadsDemand_(task, GetWriteDemand_(task)))
          changed = true;
      }
    }
  }

  static void AssignTaskFromDemand_(TaskNode *task, CoreMask demand) {
    if (demand != kCoreUnassigned)
      task->SetCoreMask(demand);
  }

  void AssignScalarWriters_() {
    for (TaskNode *task : all_tasks_) {
      if (task->ContainsLoopBreak())
        continue; // Assigned broadcast above; never demote from local demand.
      if (task->GetCoreMask() != kCoreBroadcast)
        continue;
      if (!task->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(task))
        continue;
      CoreMask demand = GetWriteDemand_(task);
      if (demand == kCoreUnassigned)
        task->SetCoreMask(kCoreUnassigned);
      else
        AssignTaskFromDemand_(task, demand);
    }
  }

  void AssignFallbackCore_() {
    bool has_cube_core = false;
    for (TaskNode *task : all_tasks_) {
      if (task->GetCoreMask() == kCoreCube)
        has_cube_core = true;
    }
    CoreMask fallback_core = has_cube_core ? kCoreCube : kCoreVector;
    for (TaskNode *task : all_tasks_) {
      CoreMask core_mask = task->GetCoreMask();
      if (core_mask == kCoreUnassigned ||
          (core_mask == kCoreBroadcast && task->IsPerCoreTask()))
        task->SetCoreMask(fallback_core);
    }
  }

  const std::vector<std::shared_ptr<IRStructure>> &root_;
  const std::vector<TaskNode *> &all_tasks_;
  VarDemandMap var_demand_;
  BufferDemandMap buffer_demand_;
  SpecialRegisterDemandMap special_writer_demand_;
};

void AssignCoreMasks(std::vector<std::shared_ptr<IRStructure>> &root) {
  std::vector<TaskNode *> all_tasks;
  CollectAllTaskNodes(root, all_tasks);
  CoreMaskPropagator::Assign(root, all_tasks);
}

} // namespace

tvm::transform::Pass AssignCore() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "AssignCore",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          AssignCoreMasks(scheduled_tir.tree);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AssignCore", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AssignCore", AssignCore);
}

} // namespace tl
} // namespace tvm
