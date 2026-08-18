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

#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "./ir_structure.h"
#include "./memory_detector.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;

inline ffi::Optional<Bind> GetFlatTaskBind(const Stmt &stmt) {
  if (const auto *bind = stmt.as<BindNode>())
    return ffi::GetRef<Bind>(bind);
  const auto *attribute = stmt.as<AttrStmtNode>();
  if (attribute && (attribute->attr_key == tl::attr::kAscendTask ||
                    attribute->attr_key == tl::attr::kScheduleUnit))
    return GetFlatTaskBind(attribute->body);
  return std::nullopt;
}

inline Stmt ReplaceFlatTaskBind(const Stmt &stmt, const Bind &replacement) {
  if (stmt.as<BindNode>())
    return replacement;
  const auto *attribute = stmt.as<AttrStmtNode>();
  ICHECK(attribute && (attribute->attr_key == tl::attr::kAscendTask ||
                       attribute->attr_key == tl::attr::kScheduleUnit))
      << "Expected a flat Task/schedule-unit wrapper around Bind, got " << stmt;
  return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                  ReplaceFlatTaskBind(attribute->body, replacement),
                  attribute->span);
}

class TaskAwareConstrVisitor : public ConstrVisitor {
protected:
  static ffi::Optional<Bind> GetFlatBind(const Stmt &stmt) {
    return GetFlatTaskBind(stmt);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    std::vector<Guard> bind_guards;
    for (const Stmt &stmt : op->seq) {
      VisitStmt(stmt);
      if (ffi::Optional<Bind> bind = GetFlatBind(stmt); bind.defined()) {
        bind_guards.push_back(
            MakeGuard(bind.value()->var, bind.value()->value));
      }
    }
  }
};

struct RegionElementUpperBound {
  int64_t elements{1};
  // False only for the last-resort extent=1 heuristic. Constant, analyzer, and
  // static-buffer-shape bounds are all usable for tightening an equal copy
  // region from the other side.
  bool reliable{true};
};

// Estimate a region's maximum element count under the current task context.
// A symbolic extent first uses Analyzer bounds, then the corresponding static
// buffer dimension, and finally the legacy extent=1 heuristic. Static buffer
// dimensions also cap weak dtype-wide Analyzer bounds.
inline RegionElementUpperBound
EstimateRegionElementUpperBound(const Buffer &buffer, const Region &ranges,
                                arith::Analyzer *analyzer) {
  RegionElementUpperBound result;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const PrimExpr &extent_expr = ranges[i]->extent;
    int64_t extent = 1;
    bool extent_reliable = true;

    if (const int64_t *extent_int = as_const_int(extent_expr)) {
      extent = *extent_int;
    } else {
      const int64_t *shape_extent =
          i < buffer->shape.size() ? as_const_int(buffer->shape[i]) : nullptr;
      arith::ConstIntBound bound = analyzer->const_int_bound(extent_expr);

      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_extent("unconstrained_extent", extent_expr.dtype());
      int64_t unconstrained_max =
          unconstrained_analyzer.const_int_bound(unconstrained_extent)
              ->max_value;
      bool has_finite_bound = bound->max_value >= 0 &&
                              bound->max_value != arith::ConstIntBound::kPosInf;
      bool has_useful_bound =
          has_finite_bound && bound->max_value < unconstrained_max;

      if (shape_extent != nullptr) {
        extent = has_finite_bound ? std::min(bound->max_value, *shape_extent)
                                  : *shape_extent;
      } else if (has_useful_bound) {
        extent = bound->max_value;
      } else {
        extent_reliable = false;
      }
    }

    result.elements *= extent;
    result.reliable = result.reliable && extent_reliable;
  }
  return result;
}

inline PrimExpr RegionElementCountExpr(const Region &ranges) {
  PrimExpr elements = make_const(DataType::Int(64), 1);
  for (const Range &range : ranges)
    elements = elements * cast(DataType::Int(64), range->extent);
  return elements;
}

inline int64_t ElementsToBytes(int64_t elements, DataType dtype) {
  int64_t elem_bits = dtype.bits() * dtype.lanes();
  return (elements * elem_bits + 7) / 8;
}

// Hardware resource properties derived from one task body.
struct TaskResourceUsage {
  uint16_t pipe_mask{0};
  uint16_t hbm_mask{0};
  bool reads_pad_value{false};
  bool reads_atomic{false};
};

// Direct buffer and scalar-variable accesses made by one statement. Accesses
// remain separate from hardware resource classification because AutoSchedule
// also adds accesses from surrounding guards and loop expressions to a task.
struct TaskAccessInfo {
  std::vector<BufferRegion> read_regions;
  std::vector<BufferRegion> write_regions;
  std::vector<Var> read_vars;
  std::vector<Var> write_vars;
};

namespace task_analysis_detail {

inline constexpr uint16_t kSpecializedPipes =
    static_cast<uint16_t>(ResourcePipe::kMTE1) |
    static_cast<uint16_t>(ResourcePipe::kMTE2) |
    static_cast<uint16_t>(ResourcePipe::kMTE3) |
    static_cast<uint16_t>(ResourcePipe::kCube) |
    static_cast<uint16_t>(ResourcePipe::kVector) |
    static_cast<uint16_t>(ResourcePipe::kFixpipe);

inline uint16_t ResolvePipe(uint16_t mask) {
  if ((mask & kSpecializedPipes) == 0)
    return mask | static_cast<uint16_t>(ResourcePipe::kScalar);
  return mask;
}

inline uint16_t GetCopyPipeMask(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  if (IsL1Buffer(src) && IsL0ABuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE1);
  if (IsL1Buffer(src) && IsL0BBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE1);
  if (IsL0CBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kFixpipe);
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kFixpipe);
  if (IsSharedBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  return static_cast<uint16_t>(ResourcePipe::kMTE2);
}

inline uint16_t GetCallPipeMask(const Call &call) {
  static const auto copy_op = Op::Get("tl.tileop.copy");
  static const auto gemm_op = Op::Get("tl.tileop.gemm");
  if (call->op.same_as(copy_op)) {
    Copy copy(call->args, call->annotations);
    return GetCopyPipeMask(copy->src, copy->dst);
  }
  if (call->op.same_as(tl::ascend_nd2nz_scatter()))
    return static_cast<uint16_t>(ResourcePipe::kVector);
  if (call->op.same_as(tl::ascend_nd2nz_post_copy()))
    return static_cast<uint16_t>(ResourcePipe::kMTE3);
  if (call->op.same_as(tl::ascend_fill_l1()))
    return static_cast<uint16_t>(ResourcePipe::kMTE2);
  if (call->op.same_as(gemm_op))
    return static_cast<uint16_t>(ResourcePipe::kCube);
  if (call->op.same_as(builtin::call_extern()) && !call->args.empty()) {
    if (const auto *sym = call->args[0].as<StringImmNode>()) {
      const std::string &name = sym->value;
      if (name == "debug_print_msg" || name == "debug_print_var" ||
          name == "debug_print_buffer_value")
        return static_cast<uint16_t>(ResourcePipe::kScalar);
      if (name == "debug_print_buffer_dump" && call->args.size() > 3) {
        if (const auto *scope_call = call->args[3].as<CallNode>()) {
          if (!scope_call->args.empty()) {
            if (const auto *load = scope_call->args[0].as<BufferLoadNode>()) {
              String scope = load->buffer.scope();
              return static_cast<uint16_t>(scope == "shared" ||
                                                   scope == "shared.dyn"
                                               ? ResourcePipe::kVector
                                               : ResourcePipe::kCube);
            }
          }
        }
      }
      if (name == "debug_print_global_buffer_dump")
        return static_cast<uint16_t>(ResourcePipe::kMTE3);
    }
  }
  return 0;
}

inline uint16_t GetHbmResourceMask(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && IsSharedBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv);
  if (IsSharedBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAiv) |
           static_cast<uint16_t>(HbmPort::kStore);
  if (IsGlobalBuffer(src) && IsL1Buffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic);
  if (IsL0CBuffer(src) && IsGlobalBuffer(dst))
    return static_cast<uint16_t>(HbmPort::kAic) |
           static_cast<uint16_t>(HbmPort::kStore);
  return 0;
}

inline SpecialRegister
SpecialRegisterForCrossCorePipe(const PrimExpr &pipe_arg) {
  const auto *pipe = pipe_arg.as<StringImmNode>();
  ICHECK(pipe != nullptr)
      << "Cross-core synchronization pipe must be a constant string, got "
      << pipe_arg;
  if (pipe->value == "PIPE_S")
    return SpecialRegister::kPipeScalar;
  if (pipe->value == "PIPE_V")
    return SpecialRegister::kPipeVector;
  if (pipe->value == "PIPE_M")
    return SpecialRegister::kPipeCube;
  if (pipe->value == "PIPE_MTE1")
    return SpecialRegister::kPipeMTE1;
  if (pipe->value == "PIPE_MTE2")
    return SpecialRegister::kPipeMTE2;
  if (pipe->value == "PIPE_MTE3")
    return SpecialRegister::kPipeMTE3;
  if (pipe->value == "PIPE_FIX")
    return SpecialRegister::kPipeFixpipe;
  LOG(FATAL) << "Unsupported cross-core synchronization pipe " << pipe->value;
}

class SpecialRegisterWriteAnalyzer : public StmtExprVisitor {
public:
  static uint16_t Analyze(const Stmt &stmt) {
    SpecialRegisterWriteAnalyzer analyzer;
    analyzer(stmt);
    return analyzer.mask_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::loop_break())) {
      Add(SpecialRegister::kLoopControl);
    } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
      Add(SpecialRegister::kHf32Mode);
    } else if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      Add(SpecialRegister::kPadValue);
    } else if (op->op.same_as(tl::ascend_set_atomic()) ||
               op->op.same_as(tl::ascend_set_atomic_none())) {
      Add(SpecialRegister::kAtomicMode);
    } else if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
               op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects 3 arguments";
      const auto *mode = op->args[0].as<IntImmNode>();
      ICHECK(mode != nullptr && mode->value == 0)
          << "Auto-scheduled cross-core synchronization requires mode 0";
      Add(SpecialRegisterForCrossCorePipe(op->args[1]));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void Add(SpecialRegister reg) { mask_ |= static_cast<uint16_t>(reg); }

  uint16_t mask_{0};
};

class ResourceAnalyzer : public StmtExprVisitor {
public:
  static TaskResourceUsage Analyze(const Stmt &stmt) {
    ResourceAnalyzer analyzer;
    analyzer(stmt);
    analyzer.usage_.pipe_mask = ResolvePipe(analyzer.usage_.pipe_mask);
    return analyzer.usage_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    static const auto copy_op = Op::Get("tl.tileop.copy");

    usage_.pipe_mask |= GetCallPipeMask(ffi::GetRef<Call>(op));
    if (op->op.same_as(copy_op)) {
      Copy copy_obj(op->args, op->annotations);
      const CopyNode *copy = copy_obj.get();
      usage_.hbm_mask |= GetHbmResourceMask(copy->src, copy->dst);
      if (copy_obj->GetDataSelect() != 0)
        usage_.reads_pad_value = true;
      if (IsGlobalBuffer(copy->dst))
        usage_.reads_atomic = true;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  // Task-level blocks are atomic to all task analyses.
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF")
      usage_.pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
  }

  TaskResourceUsage usage_;
};

} // namespace task_analysis_detail

inline TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &stmt) {
  return task_analysis_detail::ResourceAnalyzer::Analyze(stmt);
}

inline uint16_t AnalyzeSpecialRegisterWrites(const Stmt &stmt) {
  return task_analysis_detail::SpecialRegisterWriteAnalyzer::Analyze(stmt);
}

inline TaskAccessInfo AnalyzeTaskAccesses(const Stmt &stmt) {
  MemoryAccessDetector memory_detector;
  memory_detector.Analyze(stmt);
  TaskAccessInfo result;
  result.read_regions = memory_detector.GetReadRegions();
  result.write_regions = memory_detector.GetWriteRegions();
  result.read_vars = memory_detector.GetReadVars();
  result.write_vars = memory_detector.GetWriteVars();
  return result;
}

inline bool HasTaskBufferAccess(const Stmt &stmt) {
  TaskAccessInfo accesses = AnalyzeTaskAccesses(stmt);
  return !accesses.read_regions.empty() || !accesses.write_regions.empty();
}

inline bool HasTaskBufferAccess(const TaskNode *task) {
  ICHECK(task != nullptr);
  return !task->GetReadRegions().empty() || !task->GetWriteRegions().empty();
}

[[noreturn]] inline void FatalUnmarkedTask(const char *consumer,
                                           const char *node_kind,
                                           const char *prerequisite = nullptr) {
  if (prerequisite != nullptr) {
    LOG(FATAL) << consumer << " found an unmarked " << node_kind
               << ". Run tl.transform." << prerequisite << " before "
               << consumer << ".";
  }
  LOG(FATAL) << consumer << " found an unmarked " << node_kind
             << " after internal task normalization";
}

inline void ApplyTaskResourceUsage(const TaskResourceUsage &usage,
                                   TaskNode *task) {
  ICHECK(task != nullptr);
  task->SetPipeMask(usage.pipe_mask);
  task->SetHbmMask(usage.hbm_mask);
  if (usage.reads_pad_value)
    task->SetReadsPadValue(true);
  if (usage.reads_atomic)
    task->SetReadsAtomic(true);
}

inline void ApplyTaskAccesses(const TaskAccessInfo &accesses, TaskNode *task) {
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
