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

#ifndef TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_TASK_ANALYSIS_H_
#define TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_TASK_ANALYSIS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "./ir_structure.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;

class TaskAwareConstrVisitor : public ConstrVisitor {
protected:
  static ffi::Optional<Bind> GetFlatBind(const Stmt &stmt) {
    if (const auto *bind = stmt.as<BindNode>())
      return ffi::GetRef<Bind>(bind);
    const auto *attr_stmt = stmt.as<AttrStmtNode>();
    if (attr_stmt && attr_stmt->attr_key == tl::attr::kAscendTask) {
      if (const auto *wrapped_bind = attr_stmt->body.as<BindNode>())
        return ffi::GetRef<Bind>(wrapped_bind);
    }
    return std::nullopt;
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

// Non-pipe statement properties shared by LatencyEstimator and AutoSchedule.
// Pipe classification has its own analysis because NormalizeTask persists that
// result on the canonical T.Task for downstream passes to consume.
struct TaskResourceUsage {
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

bool HasTaskBufferAccess(const Stmt &stmt);
bool HasTaskBufferAccess(const TaskNode *task);

[[noreturn]] void ThrowUnmarkedTask(const char *consumer,
                                    const char *node_kind);

uint16_t AnalyzeTaskPipe(const Stmt &stmt);
TaskResourceUsage AnalyzeTaskResourceUsage(const Stmt &stmt);
TaskAccessInfo AnalyzeTaskAccesses(const Stmt &stmt);

void ApplyTaskResourceUsage(const TaskResourceUsage &usage, TaskNode *task);
void ApplyTaskAccesses(const TaskAccessInfo &accesses, TaskNode *task);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_TASK_ANALYSIS_H_
