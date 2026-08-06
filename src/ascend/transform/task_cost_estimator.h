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
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "./auto_schedule/ir_structure.h"
#include "./latency_estimator_params.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./auto_schedule/task_analysis.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::String;

struct PerCoreTaskCandidate {
  Stmt marker;
  std::shared_ptr<TaskNode> task;
};

// Estimator-only operation details. They deliberately do not live on
// AutoSchedule's TaskNode: the scheduler consumes the resulting latency/II,
// not the copy descriptors or Cube shapes used to derive them.
struct TaskCostFeatures {
  struct CubeShape {
    int64_t m;
    int64_t n;
    int64_t k;
    DataType input_dtype;
  };

  struct CopyInfo {
    String src_scope;
    String dst_scope;
    int64_t bytes;
  };

  std::vector<CubeShape> cube_shapes;
  std::vector<CopyInfo> copy_infos;
};

// Internal cost model for Ascend AutoSchedule tasks. The public transform pass
// is implemented separately in latency_estimator.cc.
class TaskCostEstimator {
public:
  // Estimate latency and initiation interval for one Ascend task.
  void Estimate(TaskNode *task);

  // Estimate one canonical Task marker from its statement structure.
  // Sequential statements add, mutually-exclusive branches take their
  // maximum. Unsynchronized sequencing exposes each preceding statement's II
  // and keeps the latest completion time across all statements; serial loops
  // apply the same rule across iterations. Parallel loops and VF blocks remain
  // atomic and use Estimate(TaskNode*) directly.
  void EstimateComposedTask(TaskNode *task);

  // Estimate a PerCoreTask from canonical Task candidate markers.
  // Candidate bodies are serial, candidates within one inter-core phase run in
  // parallel across cores, and mode-0 cross-core waits delimit serial phases.
  void EstimatePerCoreTask(TaskNode *per_core_task, const Stmt &body,
                           const std::vector<PerCoreTaskCandidate> &candidates);

private:
  struct SerialTaskCost {
    int64_t latency{0};
    int64_t ii{0};
  };

  static int64_t CheckedSerialTaskAdd(int64_t lhs, int64_t rhs,
                                      const char *field) {
    ICHECK_GE(lhs, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_GE(rhs, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_LE(lhs, std::numeric_limits<int64_t>::max() - rhs)
        << "Serial task " << field << " overflow while aggregating operations";
    return lhs + rhs;
  }

  static int64_t CheckedSerialTaskMultiply(int64_t value, int64_t factor,
                                           const char *field) {
    ICHECK_GE(value, 0) << "Serial task " << field << " must be non-negative";
    ICHECK_GE(factor, 0) << "Serial task trip count must be non-negative";
    if (value == 0 || factor == 0)
      return 0;
    ICHECK_LE(value, std::numeric_limits<int64_t>::max() / factor)
        << "Serial task " << field << " overflow while multiplying by loop "
        << "trip count " << factor;
    return value * factor;
  }

  SerialTaskCost EstimateStmtCost(const Stmt &stmt, const ConstrSet &outer_ctx);
  SerialTaskCost EstimateLeafTaskCost(const Stmt &stmt,
                                      const ConstrSet &outer_ctx);

  AscendLatencyParams params_;

  // ====================================================================
  // Every divisor is guarded so an unknown bandwidth/throughput contributes
  // zero rather than causing a divide-by-zero.
  //
  // The estimator is structured by physical resource:
  //   1. Memory latency: walk read/write regions, classify (src,dst) scope
  //      pair to a physical path, charge `base + bytes/bandwidth`.
  //   2. Compute latency:
  //        Cube   -> (M*K*N) / cube_throughput_<dtype> + base
  //        Vector -> max(op-count via OperationCounter,
  //                      bytes / valu_bandwidth)
  //        Scalar -> one cycle per atomic task
  //   3. II by pipe: take the max of the bandwidth- and throughput-limited
  //      initiation intervals for each active physical resource.
  //   4. Total latency = memory_latency + compute_latency
  //      under sequential assumption; pipeline overlap is handled by the
  //      scheduler (Z3) via the II we emit.
  // ====================================================================

  static int64_t safe_div_ceil(int64_t numer, int64_t denom) {
    return denom > 0 ? (numer + denom - 1) / denom : 0;
  }
  // Read `tl.vf_latency` annotation from SIMT_VF / SIMD_VF blocks within
  // the task. Returns 0 when no annotation is present.
  static int64_t ReadVFLatencyFromBlocks(const TaskNode *task) {
    struct VFLatencyReader : public StmtVisitor {
      void VisitStmt_(const SBlockNode *op) override {
        if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
          auto it = op->annotations.find("tl.vf_latency");
          if (it != op->annotations.end()) {
            if (auto *imm = (*it).second.as<IntImmNode>()) {
              latency = std::max(latency, imm->value);
            }
          }
        }
        StmtVisitor::VisitStmt_(op);
      }
      int64_t latency{0};
    };

    VFLatencyReader reader;
    reader(task->stmt);
    return reader.latency;
  }

  // Classification of (scope -> scope) for an MTE-style copy. Matches the
  // PipeMask the IRStructureBuilder will have set on the task, but lets
  // EstimateAscend pick the *specific* physical port (e.g. L1->L0A vs
  // L1->L0B) from the region's buffer.
  enum class AscendPath {
    kUnknown,
    kGmToL1,    // MTE2 (AIC)
    kGmToUb,    // MTE2 (AIV)
    kUbToGm,    // MTE3
    kUbToL1,    // MTE3 (cross-core)
    kL1ToL0a,   // MTE1
    kL1ToL0b,   // MTE1
    kL1ToBt,    // MTE1
    kL1ToFpBuf, // MTE1
    kL0cToUb,   // Fixpipe
    kL0cToGm,   // Fixpipe
  };

  // Pick the bandwidth (bytes/cycle) for a physical path. Returns 0 when
  // the path or the parameter is unknown — `safe_div` then collapses that
  // term to zero in the formulas below.
  int64_t BandwidthForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
      return params_.aic_mte2_to_l1_bandwidth;
    case AscendPath::kGmToUb:
      return params_.aiv_mte2_to_ub_bandwidth;
    case AscendPath::kUbToGm:
      return params_.aiv_mte3_to_gm_bandwidth;
    case AscendPath::kUbToL1:
      return params_.aiv_mte3_to_aic_l1_bandwidth;
    case AscendPath::kL1ToL0a:
      return params_.l1_to_l0a_bandwidth;
    case AscendPath::kL1ToL0b:
      return params_.l1_to_l0b_bandwidth;
    case AscendPath::kL1ToBt:
      return params_.l1_to_bt_bandwidth;
    case AscendPath::kL1ToFpBuf:
      return params_.l1_to_fp_buf_bandwidth;
    case AscendPath::kL0cToUb:
      return params_.fixpipe_bandwidth;
    case AscendPath::kL0cToGm:
      return params_.fixpipe_bandwidth;
    default:
      return 0;
    }
  }

  // Two-segment (knee) parameters for a path: {knee, flat, seg_base, seg_bw}.
  // seg_bw == 0 means "no two-segment fit for this path" and the caller falls
  // back to the legacy single-segment base + bytes/bw model.
  struct SegParams {
    int64_t knee;
    int64_t flat;
    int64_t seg_base;
    int64_t seg_bw;
  };
  SegParams SegParamsForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToUb:
      return {params_.mte2_gm_to_ub_knee, params_.mte2_gm_to_ub_flat,
              params_.mte2_gm_to_ub_seg_base, params_.mte2_gm_to_ub_seg_bw};
    case AscendPath::kUbToGm:
      return {params_.mte3_ub_to_gm_knee, params_.mte3_ub_to_gm_flat,
              params_.mte3_ub_to_gm_seg_base, params_.mte3_ub_to_gm_seg_bw};
    case AscendPath::kGmToL1:
      return {params_.mte2_gm_to_l1_knee, params_.mte2_gm_to_l1_flat,
              params_.mte2_gm_to_l1_seg_base, params_.mte2_gm_to_l1_seg_bw};
    default:
      return {0, 0, 0, 0};
    }
  }

  // Latency (cycles) for moving `bytes` over `path`. Uses the two-segment knee
  // model when the path has a measured fit, else the legacy single segment.
  int64_t MemLatencyForPath(AscendPath path, int64_t bytes) const {
    SegParams sp = SegParamsForPath(path);
    if (sp.seg_bw > 0) {
      if (bytes <= sp.knee)
        return sp.flat;
      return sp.seg_base + safe_div_ceil(bytes, sp.seg_bw);
    }
    return BaseLatencyForPath(path) +
           safe_div_ceil(bytes, BandwidthForPath(path));
  }

  int64_t BaseLatencyForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
    case AscendPath::kGmToUb:
      return params_.mte2_base_latency;
    case AscendPath::kUbToGm:
    case AscendPath::kUbToL1:
      return params_.mte3_base_latency;
    case AscendPath::kL1ToL0a:
    case AscendPath::kL1ToL0b:
    case AscendPath::kL1ToBt:
    case AscendPath::kL1ToFpBuf:
      return params_.mte1_base_latency;
    case AscendPath::kL0cToUb:
    case AscendPath::kL0cToGm:
      return params_.fixpipe_base_latency;
    default:
      return 0;
    }
  }

  // Best-effort classification of a single region into an AscendPath from
  // its buffer scope. For reads, `path = ? -> dst_of_pipe`; for writes,
  // `path = src_of_pipe -> ?`. The task's pipe mask narrows it further.
  AscendPath ClassifyRegion(const String &scope, uint16_t pipe_mask,
                            bool is_read) const {
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);

    std::string s = scope.operator std::string();
    if (has_mte2) {
      if (s == "global" && !is_read)
        return AscendPath::kUnknown;
      if (s == "global" && is_read)
        return AscendPath::kGmToUb; // default
      if (s == "shared.l1")
        return AscendPath::kGmToL1;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kGmToUb;
    }
    if (has_mte3) {
      if (s == "global")
        return AscendPath::kUbToGm;
      if (s == "shared.l1")
        return AscendPath::kUbToL1;
    }
    if (has_mte1) {
      if (s == "shared.l0a")
        return AscendPath::kL1ToL0a;
      if (s == "shared.l0b")
        return AscendPath::kL1ToL0b;
      if (s == "shared.bt")
        return AscendPath::kL1ToBt;
    }
    if (has_fix) {
      if (s == "shared.l0c" && is_read) {
        // L0C is the source; dest scope comes from the *other* region.
        return AscendPath::kL0cToUb; // refined by paired region below
      }
      if (s == "global")
        return AscendPath::kL0cToGm;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kL0cToUb;
    }
    return AscendPath::kUnknown;
  }

  AscendPath ClassifyCopy(const String &src_scope,
                          const String &dst_scope) const {
    std::string src = src_scope.operator std::string();
    std::string dst = dst_scope.operator std::string();
    if (src == "global" && dst == "shared.l1")
      return AscendPath::kGmToL1;
    if (src == "global" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kGmToUb;
    if ((src == "shared" || src == "shared.dyn") && dst == "global")
      return AscendPath::kUbToGm;
    if ((src == "shared" || src == "shared.dyn") && dst == "shared.l1")
      return AscendPath::kUbToL1;
    if (src == "shared.l1" && dst == "shared.l0a")
      return AscendPath::kL1ToL0a;
    if (src == "shared.l1" && dst == "shared.l0b")
      return AscendPath::kL1ToL0b;
    if (src == "shared.l1" && dst == "shared.bt")
      return AscendPath::kL1ToBt;
    if (src == "shared.l1" && dst == "shared.fp")
      return AscendPath::kL1ToFpBuf;
    if (src == "shared.l0c" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kL0cToUb;
    if (src == "shared.l0c" && dst == "global")
      return AscendPath::kL0cToGm;
    return AscendPath::kUnknown;
  }

  // Per-dtype Cube throughput (MACs/cycle).
  int64_t CubeThroughputFor(DataType dtype) const {
    if (dtype.is_float16())
      return params_.cube_throughput_fp16 ? params_.cube_throughput_fp16
                                          : params_.cube_fallback_throughput;
    if (dtype.is_bfloat16())
      return params_.cube_throughput_bf16 ? params_.cube_throughput_bf16
                                          : params_.cube_fallback_throughput;
    if (dtype.is_float() && dtype.bits() == 32)
      return params_.cube_throughput_fp32 ? params_.cube_throughput_fp32
                                          : params_.cube_fallback_throughput;
    if (dtype.is_float8() || (dtype.is_float() && dtype.bits() == 8))
      return params_.cube_throughput_fp8 ? params_.cube_throughput_fp8
                                         : params_.cube_fallback_throughput;
    if (dtype.is_int() && dtype.bits() == 8)
      return params_.cube_throughput_int8 ? params_.cube_throughput_int8
                                          : params_.cube_fallback_throughput;
    return params_.cube_fallback_throughput;
  }

  // Ascend latency / II estimation entry point.
  void EstimateAscend(TaskNode *task, const TaskCostFeatures &cost_features,
                      arith::Analyzer *analyzer) {
    uint16_t pipe_mask = task->GetPipeMask();
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);
    bool has_cube = pipe_mask & static_cast<uint16_t>(ResourcePipe::kCube);
    bool has_vec = pipe_mask & static_cast<uint16_t>(ResourcePipe::kVector);
    bool has_scl = pipe_mask & static_cast<uint16_t>(ResourcePipe::kScalar);
    bool has_dma = has_mte1 || has_mte2 || has_mte3 || has_fix;
    int64_t measured_vf_latency = ReadVFLatencyFromBlocks(task);

    // -----------------------------------------------------------------
    // 1. Memory latency: per-physical-path, base + bytes / bandwidth.
    //    Bytes are aggregated per path for the II step.
    // -----------------------------------------------------------------
    std::unordered_map<int, int64_t> bytes_per_path;
    int64_t memory_latency = 0;

    auto add_memory_access = [&](AscendPath path, int64_t bytes) {
      int64_t lat =
          params_.mte_descriptor_cycles + MemLatencyForPath(path, bytes);
      memory_latency += lat;
      bytes_per_path[static_cast<int>(path)] += bytes;
    };

    if (!cost_features.copy_infos.empty()) {
      for (const TaskCostFeatures::CopyInfo &info : cost_features.copy_infos) {
        add_memory_access(ClassifyCopy(info.src_scope, info.dst_scope),
                          info.bytes);
      }
    } else {
      auto visit_region = [&](const BufferRegion &region, bool is_read) {
        const Buffer &buffer = region->buffer;
        String scope = buffer.scope();
        AscendPath path = ClassifyRegion(scope, pipe_mask, is_read);
        int64_t bytes = CalculateAccessBytes(region, analyzer);
        add_memory_access(path, bytes);
      };
      for (const auto &r : task->GetReadRegions())
        visit_region(r, /*is_read=*/true);
      for (const auto &r : task->GetWriteRegions())
        visit_region(r, /*is_read=*/false);
    }

    // -----------------------------------------------------------------
    // 2. Compute latency.
    // -----------------------------------------------------------------
    int64_t compute_latency = 0;
    int64_t vector_latency = 0;

    if (has_vec) {
      if (measured_vf_latency > 0) {
        // User-annotated latency — use it directly, skip model estimation.
        vector_latency = measured_vf_latency;
      } else {
        // Op-count path (TIR visitor).
        OperationCounter counter(&params_);
        counter(task->stmt);
        int64_t op_count_latency = counter.GetEstimatedLatency();

        // Bandwidth-bound path: total bytes the VALU consumes / valu_bandwidth.
        int64_t valu_bytes = 0;
        for (const auto &r : task->GetReadRegions()) {
          String scope = r->buffer.scope();
          if (scope == "local" || scope == "local.fragment" ||
              scope == "shared" || scope == "shared.dyn") {
            valu_bytes += CalculateAccessBytes(r, analyzer);
          }
        }
        int64_t bw_latency = safe_div_ceil(valu_bytes, params_.valu_bandwidth);

        vector_latency = std::max(op_count_latency, bw_latency);
      }
      compute_latency = std::max(compute_latency, vector_latency);
    }

    if (has_cube) {
      // matmul latency = base + total_MACs / per-dtype throughput.
      // Use recorded GEMM shapes, or fall back to one unit MMAD.
      int64_t matmul_latency = params_.cube_base_latency;
      if (!cost_features.cube_shapes.empty()) {
        for (const TaskCostFeatures::CubeShape &shape :
             cost_features.cube_shapes) {
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          matmul_latency += safe_div_ceil(ops, throughput);
        }
      } else if (params_.cube_unit_m > 0 && params_.cube_unit_k > 0 &&
                 params_.cube_unit_n > 0) {
        // Single unit mad fallback.
        int64_t unit_ops =
            params_.cube_unit_m * params_.cube_unit_k * params_.cube_unit_n;
        int64_t throughput = CubeThroughputFor(DataType::Float(16));
        matmul_latency =
            params_.cube_base_latency + safe_div_ceil(unit_ops, throughput);
      }
      compute_latency = std::max(compute_latency, matmul_latency);
    }

    if (has_scl && !has_vec) {
      // Pure scalar tasks (set/wait_flag bodies, address compute). Charge
      // a flat one-cycle cost. A task carries exactly one statement.
      compute_latency = std::max<int64_t>(compute_latency, 1);
    }

    int64_t total_latency = memory_latency + compute_latency;

    // -----------------------------------------------------------------
    // 4. II: per-pipe analysis. Take the max across all active pipes.
    // -----------------------------------------------------------------
    int64_t ii = 1;
    int64_t memory_ii = 1;
    for (const auto &kv : bytes_per_path) {
      AscendPath path = static_cast<AscendPath>(kv.first);
      // Steady-state throughput: the saturated large-packet rate (seg_bw)
      // when the path has a two-segment fit, else the headline bandwidth.
      SegParams sp = SegParamsForPath(path);
      int64_t bw = sp.seg_bw > 0 ? sp.seg_bw : BandwidthForPath(path);
      memory_ii = std::max(memory_ii, safe_div_ceil(kv.second, bw));
    }

    int64_t cube_ii = params_.cube_min_ii;
    if (has_cube) {
      if (!cost_features.cube_shapes.empty()) {
        for (const TaskCostFeatures::CubeShape &shape :
             cost_features.cube_shapes) {
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          cube_ii = std::max(cube_ii, safe_div_ceil(ops, throughput));
        }
      }
      cube_ii = std::max(cube_ii, params_.cube_pipeline_depth);
    }

    int64_t vector_ii = 1;
    if (has_vec) {
      if (measured_vf_latency > 0) {
        vector_ii = measured_vf_latency;
      } else {
        int64_t denom =
            params_.vector_queue_depth > 0 ? params_.vector_queue_depth : 1;
        vector_ii = std::max(vector_ii, safe_div_ceil(vector_latency, denom));
      }
    }

    if (has_dma && !has_cube && !has_vec) {
      // DMA-only: II = max bandwidth-limited II across active paths.
      ii = std::max(ii, memory_ii);
    } else if (has_cube && !has_dma && !has_vec) {
      // Cube-only: II is bounded by pipeline depth and MAC throughput.
      ii = std::max(ii, cube_ii);
    } else if (has_vec && !has_dma && !has_cube) {
      // Vector-only: II = compute_latency / vector_queue_depth (fallback to
      // compute_latency when queue depth is unknown).
      ii = std::max(ii, vector_ii);
    } else {
      // Mixed / fallback: keep task granularity, but bound II by the slowest
      // active pipe instead of forcing it to the full sequential latency.
      if (has_dma)
        ii = std::max(ii, memory_ii);
      if (has_cube)
        ii = std::max(ii, cube_ii);
      if (has_vec)
        ii = std::max(ii, vector_ii);
    }

    task->SetLatency(total_latency);
    task->SetII(ii);
  }

  // VALU operation counter.
  class OperationCounter : public StmtExprVisitor {
  public:
    // Loop dimension information
    struct LoopDimension {
      int64_t trip_count;
      int depth;
    };

    int64_t total_latency = 0;
    const AscendLatencyParams *params = nullptr;

    // Track loop dimensions for loop-invariant detection
    std::vector<LoopDimension> loop_stack;
    std::unordered_map<const VarNode *, int> var_to_depth;

    explicit OperationCounter(const AscendLatencyParams *params)
        : params(params) {}

    // Operator to visit a statement
    void operator()(const Stmt &stmt) { VisitStmt(stmt); }

    // Get estimated latency
    int64_t GetEstimatedLatency() const { return total_latency; }

    // Helper function to update latency for an operation based on throughput
    void update_operation(
        int64_t throughput,
        const std::unordered_set<const VarNode *> &contained_vars) {
      int64_t effective_factor =
          CalculateEffectiveParallelFactor(contained_vars);

      if (throughput > 0) {
        int64_t cycles_for_operations =
            (effective_factor + throughput - 1) / throughput;
        total_latency += std::max<int64_t>(cycles_for_operations, 1);
      } else {
        total_latency += 1;
      }
    }

    // Analyze which loop variables are contained in an expression
    std::unordered_set<const VarNode *>
    AnalyzeContainedLoopVars(const PrimExpr &expr) {
      class VarCollector : public ExprVisitor {
      public:
        const std::unordered_map<const VarNode *, int> &var_to_depth;
        std::unordered_set<const VarNode *> collected_vars;

        VarCollector(
            const std::unordered_map<const VarNode *, int> &var_to_depth)
            : var_to_depth(var_to_depth) {}

        void VisitExpr_(const VarNode *op) final {
          if (var_to_depth.count(op)) {
            collected_vars.insert(op);
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VarCollector collector(var_to_depth);
      collector(expr);
      return collector.collected_vars;
    }

    // Calculate effective parallel factor based on contained loop variables
    int64_t CalculateEffectiveParallelFactor(
        const std::unordered_set<const VarNode *> &contained_vars) {
      if (contained_vars.empty()) {
        return 1; // Completely loop-invariant
      }

      // Find the maximum depth of contained loop variables
      int max_depth = -1;
      for (const VarNode *var : contained_vars) {
        auto it = var_to_depth.find(var);
        if (it != var_to_depth.end()) {
          max_depth = std::max(max_depth, it->second);
        }
      }

      // Calculate effective parallel factor: product of trip counts for loops
      // with depth <= max_depth
      int64_t effective_factor = 1;
      for (const auto &loop : loop_stack) {
        if (loop.depth <= max_depth) {
          effective_factor *= loop.trip_count;
        }
      }

      return effective_factor;
    }

    void VisitExpr_(const AddNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->add_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->sub_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MulNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mul_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const DivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const ModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorDivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const EQNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const AndNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const OrNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NotNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MinNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MaxNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));

      // Check for special math functions by name
      if (op->op.as<OpNode>()) {
        auto op_node = op->op.as<OpNode>();
        std::string op_name = op_node->name;

        // Check for special math functions
        if (op_name == "exp2" || op_name == "log2" || op_name == "exp" ||
            op_name == "log" || op_name == "sin" || op_name == "cos" ||
            op_name == "tan" || op_name == "asin" || op_name == "acos" ||
            op_name == "atan" || op_name == "sinh" || op_name == "cosh" ||
            op_name == "tanh" || op_name == "sqrt" || op_name == "rsqrt" ||
            op_name == "pow" || op_name == "erf" || op_name == "sigmoid") {
          // Special math functions have lower throughput
          update_operation(params->special_func_throughput, contained_vars);
        } else if (op_name.find("copy") != std::string::npos ||
                   op_name.find("gemm") != std::string::npos) {
          // Data movement and Cube latency are modeled separately.
        } else {
          update_operation(params->default_operation_throughput,
                           contained_vars);
        }
      } else {
        update_operation(params->default_operation_throughput, contained_vars);
      }
      // Visit arguments
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      // Calculate trip count for the loop
      int64_t trip_count = 1;
      PrimExpr loop_extent = op->extent;
      PrimExpr loop_step = op->step.has_value() ? op->step.value()
                                                : IntImm(DataType::Int(32), 1);

      // Try to get constant values
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        if (const int64_t *step_ptr = as_const_int(loop_step)) {
          int64_t extent = *extent_ptr;
          int64_t step = *step_ptr;
          if (step > 0) {
            // ceil(extent / step) = (extent + step - 1) / step
            trip_count = (extent + step - 1) / step;
          } else {
            trip_count = extent; // Invalid step, use extent
          }
        } else {
          trip_count = 100; // Non-constant step, use default
        }
      } else {
        trip_count = 100; // Non-constant extent, use default
      }

      // Create loop dimension information
      LoopDimension loop_dim{.trip_count = trip_count,
                             .depth = static_cast<int>(loop_stack.size())};

      // Push loop onto stack and update mapping
      loop_stack.push_back(loop_dim);
      var_to_depth[op->loop_var.get()] = loop_dim.depth;

      // Visit loop body
      StmtExprVisitor::VisitStmt_(op);

      // Pop loop from stack
      var_to_depth.erase(op->loop_var.get());
      loop_stack.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (op->value.defined()) {
        StmtExprVisitor::VisitExpr(op->value);
      }
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      // Count operations in indices
      for (const auto &index : op->indices) {
        StmtExprVisitor::VisitExpr(index);
      }
      // Count operations in value
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const SeqStmtNode *op) final {
      for (const auto &child : op->seq) {
        StmtExprVisitor::VisitStmt(child);
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const BindNode *op) final {
      // Let binding: the value expression is evaluated once
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);

      // For if-then-else branches, we need to take the maximum latency of both
      // paths Save current latency
      int64_t old_latency = total_latency;

      // Count latency in then branch
      StmtExprVisitor::VisitStmt(op->then_case);
      int64_t then_latency = total_latency;

      // Restore latency and count else branch
      total_latency = old_latency;
      if (op->else_case) {
        StmtExprVisitor::VisitStmt(op->else_case.value());
      }
      int64_t else_latency = total_latency;

      // Take the maximum of both branches
      total_latency = old_latency + std::max(then_latency - old_latency,
                                             else_latency - old_latency);
    }

    void VisitStmt_(const SBlockNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }
  };

  // Helper function to calculate total bytes accessed in a region.
  int64_t CalculateAccessBytes(const BufferRegion &region,
                               arith::Analyzer *analyzer) {
    RegionElementUpperBound estimate = EstimateRegionElementUpperBound(
        region->buffer, region->region, analyzer);
    return ElementsToBytes(estimate.elements, DataType(region->buffer->dtype));
  }
};

} // namespace tl
} // namespace tvm
