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
 * \file auto_schedule.cc
 * \brief AutoSchedule pass for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
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
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm_py.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "auto_schedule.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Mutator to extract the body of tilelang_root block and optional
// buffer version overrides annotation.
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;
  std::map<std::string, int64_t> buffer_versions_overrides;

  void VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      body = op->body;
      auto ann_val = op->annotations.Get("tl.buffer_versions_map");
      if (ann_val.has_value()) {
        auto *map_obj = ann_val.value().as<ffi::MapObj>();
        if (map_obj) {
          for (auto it : *map_obj) {
            if (auto *var = it.first.as<VarNode>()) {
              if (auto imm = it.second.as<int>()) {
                buffer_versions_overrides[var->name_hint] = imm.value();
              }
            }
          }
        }
      }
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
};

// Detect multiple kernel launches in the PrimFunc body.
// In tilelang, when multiple T.Kernel() blocks are used, the IR structure is:
//   root block body:
//     AttrStmt(tl.assume, ...)
//       AttrStmt(tl.assume, ...)
//         SeqStmt [
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel1_subtree),
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel2_subtree),
//         ]
// Each kernel subtree contains its own launch_threads and tilelang_root block.
// This class finds that SeqStmt and returns each child as a separate kernel.
class MultiKernelDetector {
public:
  static bool Detect(const Stmt &func_body, std::vector<Stmt> &kernel_stmts,
                     Stmt &prefix_wrapper) {
    std::vector<Stmt> stmts;
    const Stmt *inner = &func_body;

    // Peel through root block -> BlockRealize
    if (const auto *br = inner->as<BlockRealizeNode>()) {
      inner = &br->block->body;
    }

    // Peel through AttrStmt(tl.assume, ...) chains
    while (const auto *attr = inner->as<AttrStmtNode>()) {
      if (attr->attr_key != "tl.assume")
        break;
      inner = &attr->body;
    }

    // Check if we have a SeqStmt with multiple children that each contain
    // a launch_thread (thread_extent)
    const auto *seq = inner->as<SeqStmtNode>();
    if (!seq || seq->seq.size() < 2)
      return false;

    int kernel_count = 0;
    for (const auto &child : seq->seq) {
      if (ContainsLaunchThread(child)) {
        kernel_count++;
      }
    }

    if (kernel_count < 2)
      return false;

    for (const auto &child : seq->seq) {
      kernel_stmts.push_back(child);
    }
    return true;
  }

private:
  static bool ContainsLaunchThread(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::thread_extent) {
        return true;
      }
    }
    return false;
  }
};

// Mutator that replaces the inner SeqStmt (inside root block -> tl.assume
// chain) with a new body. Used to reassemble multi-kernel results.
class InnerSeqStmtReplacer : public StmtMutator {
public:
  explicit InnerSeqStmtReplacer(Stmt new_inner) : new_inner_(new_inner) {}

  Stmt VisitStmt_(const BlockRealizeNode *op) override {
    auto new_block_body = this->VisitStmt(op->block->body);
    if (new_block_body.same_as(op->block->body))
      return GetRef<Stmt>(op);
    auto new_block =
        Block(op->block->iter_vars, op->block->reads, op->block->writes,
              op->block->name_hint, new_block_body, op->block->init,
              op->block->alloc_buffers, op->block->match_buffers,
              op->block->annotations);
    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "tl.assume") {
      auto new_body = this->VisitStmt(op->body);
      if (new_body.same_as(op->body))
        return GetRef<Stmt>(op);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override { return new_inner_; }

private:
  Stmt new_inner_;
};

// Mutator to replace the body of tilelang_root block
class TilelangRootBodyReplacer : public StmtMutator {
public:
  explicit TilelangRootBodyReplacer(Stmt new_body) : new_body_(new_body) {}

  Stmt VisitStmt_(const BlockNode *op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                   new_body_, op->init, op->alloc_buffers, op->match_buffers,
                   op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::shared_ptr<IRStructure> Build(const Stmt &stmt,
                                     Target target = Target()) {
    target_ = target;
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING)
          << "IRStructureBuilder: root_ is null after visiting statement. "
          << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:
  void VisitStmt_(const SeqStmtNode *op) override {
    auto seq_node = std::make_shared<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial) {
      // Sequential For -> ControlNode
      auto control_node = std::make_shared<ControlNode>();
      control_node->control = GetRef<For>(op);
      control_node->task = std::make_shared<TaskNode>();
      control_node->task->stmts.push_back(
          For(op->loop_var, op->min, op->extent, op->kind, Evaluate(0),
              op->thread_binding, op->annotations, op->step, op->span));
      AnalyzeResourceUsage(Evaluate(op->min), control_node->task.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), control_node->task.get(),
                           true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             control_node->task.get(), true);

      // Process the loop body
      VisitStmt(op->body);
      if (root_) {
        control_node->child = std::move(root_);
      } else {
      }

      root_ = std::move(control_node);
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage)
      AnalyzeResourceUsage(op->body, task_node.get());
      AnalyzeResourceUsage(Evaluate(op->min), task_node.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), task_node.get(), true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             task_node.get(), true);

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // If statement -> IfNode with independently schedulable branches
    auto if_node = std::make_shared<IfNode>();
    if_node->condition = op->condition;

    // Create task for condition expression resource analysis
    auto cond_task = std::make_shared<TaskNode>();
    cond_task->stmts.push_back(Evaluate(op->condition));
    AnalyzeMemoryExpr(op->condition, cond_task.get());
    AnalyzeResourceUsage(Evaluate(op->condition), cond_task.get(), true);
    if_node->task = std::move(cond_task);

    // Recursively build then branch
    VisitStmt(op->then_case);
    if (root_) {
      if_node->then_child = std::move(root_);
    }

    // Recursively build else branch (if present)
    if (op->else_case) {
      VisitStmt(op->else_case.value());
      if (root_) {
        if_node->else_child = std::move(root_);
      }
    }

    root_ = std::move(if_node);
  }

  void VisitStmt_(const LetStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetLetDecl(op));
    AnalyzeResourceUsage(GetLetDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetAttrDecl(op));
    AnalyzeResourceUsage(GetAttrDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode *op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by
    // TilelangRootBodyExtractor If we encounter it here, it means we're
    // processing the entire function body (not extracted), which should only
    // happen when there's no tilelang_root block
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }

private:
  std::shared_ptr<IRStructure> root_;
  Target target_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node,
                            bool only_variables = false) {
    // Recursively analyze statements to determine resource usage (Ascend pipe
    // model)
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      uint16_t pipe_mask{0};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
            : m(m), n(n), k(k) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;

      // Target and block_size for GemmInst determination
      Target target;
      int64_t block_size;

      ResourceAnalyzer(TaskNode *node, Target target = Target(),
                       int64_t block_size = 128)
          : task_node(node), target(target), block_size(block_size) {}

      // Determine Ascend pipe from copy src/dst buffer scopes
      static uint16_t GetCopyPipeMask(const Buffer &src, const Buffer &dst) {
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
        // Default: treat as MTE2 data movement
        return static_cast<uint16_t>(ResourcePipe::kMTE2);
      }

      void VisitExpr_(const CallNode *op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_py_op = Op::Get("tl.tileop.gemm_py");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");

        if (op->op.same_as(copy_op)) {
          Copy copy_obj(op->args, op->annotations);
          const CopyNode *copy = copy_obj.get();
          pipe_mask |= GetCopyPipeMask(copy->src, copy->dst);
        } else if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
          // SimdVF MicroAPI scatter writing into a UB tmp.
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
        } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
          // Per-D-group copy_ubuf_to_cbuf: UB → L1 = MTE3.
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kMTE3);
        } else if (op->op.same_as(gemm_py_op) || op->op.same_as(gemm_op)) {
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kCube);
          GemmPy gemm_py(op->args);
          tensor_core_shapes.emplace_back(gemm_py->m_, gemm_py->n_,
                                          gemm_py->k_);
        } else if (op->op.same_as(builtin::call_extern())) {
          if (auto sym = op->args[0].as<StringImmNode>()) {
            const std::string &name = sym->value;
            if (name == "debug_print_msg" || name == "debug_print_var" ||
                name == "debug_print_buffer_value") {
              // (AscendC::)printf -> S pipe
              pipe_mask |= static_cast<uint16_t>(ResourcePipe::kScalar);
            } else if (name == "debug_print_buffer_dump") {
              auto scope = op->args[3]
                               .as<CallNode>()
                               ->args[0]
                               .as<BufferLoadNode>()
                               ->buffer.scope();
              if (scope == "shared" || scope == "shared.dyn") {
                pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
              } else {
                pipe_mask |= static_cast<uint16_t>(ResourcePipe::kCube);
              }
            } else if (name == "debug_print_global_buffer_dump") {
              // AscendC::DumpTensor(GlobalTensor<T>) reads GM
              pipe_mask |= static_cast<uint16_t>(ResourcePipe::kMTE3);
            }
          }
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitStmt_(const BlockNode *op) override {
        if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
        }
      }
    };

    ResourceAnalyzer analyzer(task_node, target_);
    analyzer(stmt);

    if (!only_variables) {
      // Set task node pipe mask
      task_node->SetPipeMask(analyzer.pipe_mask);
      // Set Tensor Core shape information if available
      for (const auto &shape : analyzer.tensor_core_shapes) {
        if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
          task_node->AddTensorCoreShape(shape.m, shape.n, shape.k);
        }
      }
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();
    std::vector<Var> read_vars = memory_detector.GetReadVars();
    std::vector<Var> write_vars = memory_detector.GetWriteVars();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    for (const auto &var : read_vars) {
      task_node->AddReadVar(var);
    }

    for (const auto &var : write_vars) {
      task_node->AddWriteVar(var);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.Estimate(task_node);
  }

  void AnalyzeMemoryExpr(const PrimExpr &expr, TaskNode *task_node) {
    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(expr);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.Estimate(task_node);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt);

// Result of scheduling a single kernel segment
struct ScheduledKernelResult {
  Stmt scheduled_body;
  std::vector<Buffer> barrier_buffers;
  Map<ObjectRef, ObjectRef> barrier_map;
  std::vector<MultiVersionBufferInfo> buffer_infos;
};

// Recursively check if any TaskNode has the given warpgroup_id.
static bool HasWarpgroupId(IRStructure *node, int wg_id) {
  if (!node)
    return false;
  if (node->IsTask())
    return static_cast<TaskNode *>(node)->GetWarpgroupId() == wg_id;
  if (node->IsSequence()) {
    for (auto &c : static_cast<SequenceNode *>(node)->children)
      if (HasWarpgroupId(c.get(), wg_id))
        return true;
  } else if (node->IsControl()) {
    return HasWarpgroupId(static_cast<ControlNode *>(node)->child.get(), wg_id);
  } else if (node->IsWrapper()) {
    return HasWarpgroupId(static_cast<WrapperNode *>(node)->child.get(), wg_id);
  } else if (node->IsScheduleUnit()) {
    return HasWarpgroupId(static_cast<ScheduleUnit *>(node)->child.get(),
                          wg_id);
  } else if (node->IsIf()) {
    auto ifn = static_cast<IfNode *>(node);
    return HasWarpgroupId(ifn->then_child.get(), wg_id) ||
           (ifn->else_child && HasWarpgroupId(ifn->else_child.get(), wg_id));
  }
  return false;
}

// Handles IRStructure building, Z3 scheduling (with Ascend sync insertion),
// and Stmt conversion.
static ScheduledKernelResult ScheduleSingleKernel(
    const Stmt &kernel_body, Target target,
    const std::map<std::string, int64_t> &buffer_versions_overrides = {}) {
  // Build IRStructure from the body to schedule
  ScheduledKernelResult result;
  IRStructureBuilder builder;
  auto ir_structure = builder.Build(kernel_body, target);

  ICHECK(ir_structure) << "IRStructure is null (empty body?)";

  // Build ScheduleUnits from IRStructure (Z3 scheduling + Ascend sync)
  ScheduleUnitBuilder unit_builder;
  unit_builder.SetMemoryLimit("shared", GetSharedMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l1", GetL1MemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0c", GetL0CMemoryLimit(target));
  unit_builder.SetBufferVersionsOverride(buffer_versions_overrides);
  unit_builder.ScheduleRecursive(ir_structure);

  // Assign warpgroup IDs based on pipe and buffer scopes
  {
    std::vector<TaskNodeWithContext> all_tasks;
    CollectAllTaskNodesWithContext(ir_structure.get(), all_tasks);
    for (auto &task_ctx : all_tasks) {
      auto *t = task_ctx.task;
      uint16_t m = t->GetPipeMask();
      if (m & static_cast<uint16_t>(ResourcePipe::kCube)) {
        t->SetWarpgroupId(1);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kVector)) {
        t->SetWarpgroupId(0);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE1) ||
                 m & static_cast<uint16_t>(ResourcePipe::kFixpipe)) {
        t->SetWarpgroupId(1);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE3)) {
        t->SetWarpgroupId(0);
      } else if (m & static_cast<uint16_t>(ResourcePipe::kMTE2)) {
        bool to_l1 = false;
        for (auto &r : t->GetWriteRegions()) {
          if (IsL1Buffer(r->buffer)) {
            to_l1 = true;
            break;
          }
        }
        t->SetWarpgroupId(to_l1 ? 1 : 0);
      }
    }
  }

  EventIdMap event_id_map;
  FlagInfoList flag_infos;
  LoopNestingInfo loop_info;
  Var sid("sid", DataType::Int(32));
  AnalyzeAndInsertBarriers(ir_structure.get(), event_id_map, flag_infos, sid,
                           result.barrier_buffers, result.barrier_map,
                           loop_info, result.buffer_infos);

  Stmt wg0_body = Evaluate(0);
  Stmt wg1_body = Evaluate(0);

  bool has_wg0 = HasWarpgroupId(ir_structure.get(), 0);
  bool has_wg1 = HasWarpgroupId(ir_structure.get(), 1);

  if (has_wg0) {
    auto wg = CloneIRStructureWithWarpgroupFilter(ir_structure.get(), 0);
    wg0_body = ConvertIRStructureToStmt(wg.get(), sid);
  }
  if (has_wg1) {
    auto wg = CloneIRStructureWithWarpgroupFilter(ir_structure.get(), 1);
    wg1_body = ConvertIRStructureToStmt(wg.get(), sid);
  }

  if (!flag_infos.empty()) {
    std::vector<Stmt> stmts[2];
    for (const auto &info : flag_infos) {
      if (info.mode == 0) {
        stmts[info.warpgroup_id].push_back(
            makeAscendSetFlag(info.hardevent, info.flag_var));
      } else if (info.mode == 1) {
        stmts[info.warpgroup_id].push_back(
            makeAscendCrossCoreSetFlag(4, info.hardevent, info.flag_var));
      }
    }
    stmts[0].push_back(wg0_body);
    stmts[1].push_back(wg1_body);
    for (const auto &info : flag_infos) {
      if (info.mode == 0) {
        stmts[info.warpgroup_id].push_back(
            makeAscendWaitFlag(info.hardevent, info.flag_var));
      } else if (info.mode == 2) {
        stmts[info.warpgroup_id].push_back(
            makeAscendCrossCoreWaitFlag(4, info.hardevent, info.flag_var));
      }
    }
    if (has_wg0) {
      wg0_body = SeqStmt::Flatten(stmts[0]);
    }
    if (has_wg1) {
      wg1_body = SeqStmt::Flatten(stmts[1]);
    }
  }
  if (has_wg0 && has_wg1) {
    Array<Stmt> blocks;
    blocks.push_back(
        Block({}, {}, {}, "CUBE", wg1_body, std::nullopt, {}, {}, {}));
    Map<String, ObjectRef> vec_annotations;
    vec_annotations.Set("vector_count", Integer(2));
    IterVar sid_thread(Range(0, 2), sid, IterVarType::kThreadIndex, "cthread");
    Stmt vec_body =
        AttrStmt(sid_thread, tir::attr::thread_extent, Integer(2), wg0_body);
    blocks.push_back(Block({}, {}, {}, "VECTOR", vec_body, std::nullopt, {}, {},
                           vec_annotations));
    result.scheduled_body = SeqStmt(blocks);
  } else if (has_wg0) {
    result.scheduled_body = wg0_body;
  } else {
    result.scheduled_body = wg1_body;
  }
  return result;
}

// The main pass function
tvm::transform::Pass AutoSchedule() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    // Detect multiple kernel launches in the PrimFunc body.
    // When multiple T.Kernel() blocks are used, the IR has a SeqStmt
    // containing separate kernel subtrees, each with its own tilelang_root.
    std::vector<Stmt> kernel_stmts;
    Stmt prefix_wrapper;
    bool is_multi_kernel =
        MultiKernelDetector::Detect(func->body, kernel_stmts, prefix_wrapper);

    if (!is_multi_kernel) {
      // --- Single-kernel path ---
      TilelangRootBodyExtractor extractor;
      extractor(func->body);
      Stmt body_to_schedule;

      if (extractor.body.defined()) {
        body_to_schedule = extractor.body;
      } else {
        LOG(FATAL);
        body_to_schedule = func->body;
      }

      auto overrides = extractor.buffer_versions_overrides;
      auto kr = ScheduleSingleKernel(body_to_schedule, target, overrides);

      TilelangRootBodyReplacer replacer(kr.scheduled_body);
      Stmt final_body = replacer(func->body);
      if (!kr.buffer_infos.empty()) {
        final_body = RewriteAllocBuffers(final_body, kr.buffer_infos);
      }
      final_body = ReNestLetStmts(final_body);

      auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                               func->buffer_map, func->attrs);
      return new_func;
    }

    // --- Multi-kernel path ---
    Array<Stmt> combined_stmts;

    for (size_t i = 0; i < kernel_stmts.size(); ++i) {
      const Stmt &kernel_subtree = kernel_stmts[i];

      TilelangRootBodyExtractor extractor;
      extractor(kernel_subtree);

      if (!extractor.body.defined()) {
        if (!combined_stmts.empty()) {
          combined_stmts.push_back(
              AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                       Evaluate(0)));
        }
        combined_stmts.push_back(kernel_subtree);
        continue;
      }

      Stmt body_to_schedule = extractor.body;

      auto overrides = extractor.buffer_versions_overrides;
      auto result = ScheduleSingleKernel(body_to_schedule, target, overrides);
      Stmt scheduled_body = result.scheduled_body;

      TilelangRootBodyReplacer replacer(scheduled_body);
      Stmt scheduled_subtree = replacer(kernel_subtree);

      if (!combined_stmts.empty()) {
        combined_stmts.push_back(
            AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                     Evaluate(0)));
      }
      combined_stmts.push_back(scheduled_subtree);
    }

    // Reassemble: replace the inner SeqStmt in the PrimFunc body with the
    // new combined statements
    Stmt new_inner;
    if (combined_stmts.size() == 1) {
      new_inner = combined_stmts[0];
    } else {
      new_inner = SeqStmt(combined_stmts);
    }

    InnerSeqStmtReplacer seq_replacer(new_inner);
    Stmt final_body = seq_replacer(func->body);

    final_body = ReNestLetStmts(final_body);

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Re-write LetStmt to nest them properly
// Example transformation:
//   SeqStmt {
//     let x = 42 { Evaluate(0) }     // standalone, empty body
//     let y = x+1 { Evaluate(0) }    // standalone, empty body
//     compute(x, y)                   // actual work
//     store(result)
//   }
// becomes:
//   let x = 42 {
//     let y = x+1 {
//       SeqStmt {
//         compute(x, y)
//         store(result)
//       }
//     }
//   }
class LetStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> stmts;
    for (const auto &stmt : op->seq) {
      stmts.push_back(this->VisitStmt(stmt));
    }

    Array<Stmt> flat_stmts;
    for (const auto &stmt : stmts) {
      if (const auto *inner_seq = stmt.as<SeqStmtNode>()) {
        for (const auto &inner_stmt : inner_seq->seq) {
          flat_stmts.push_back(inner_stmt);
        }
      } else {
        flat_stmts.push_back(stmt);
      }
    }
    stmts = flat_stmts;

    for (int i = static_cast<int>(stmts.size()) - 2; i >= 0; --i) {
      if (const auto *let = stmts[i].as<LetStmtNode>()) {
        if (IsEmptyBody(let->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i, LetStmt(let->var, let->value, absorbed_body));
        }
      } else if (const auto *attr = stmts[i].as<AttrStmtNode>()) {
        if (IsEmptyBody(attr->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i,
              AttrStmt(attr->node, attr->attr_key, attr->value, absorbed_body));
        }
      }
    }

    if (stmts.empty())
      return Evaluate(0);
    if (stmts.size() == 1)
      return stmts[0];

    return SeqStmt(stmts);
  }

private:
  // Check if a statement body is Evaluate(0) — the empty placeholder
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *imm = eval->value.as<IntImmNode>()) {
        return imm->value == 0;
      }
    }
    return false;
  }

  // Collect stmts[start .. end) into a single Stmt
  static Stmt CollectRemaining(const Array<Stmt> &stmts, int start) {
    int n = static_cast<int>(stmts.size());
    if (start >= n) {
      return Evaluate(0);
    }
    if (start == n - 1) {
      return stmts[start];
    }
    Array<Stmt> remaining;
    for (int j = start; j < n; ++j) {
      remaining.push_back(stmts[j]);
    }
    return SeqStmt(remaining);
  }

  // Keep stmts[0..index), replace stmts[index] with new_stmt,
  // discard everything after (already absorbed into new_stmt body)
  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &stmts, int index,
                                        Stmt new_stmt) {
    Array<Stmt> result;
    for (int j = 0; j < index; ++j) {
      result.push_back(stmts[j]);
    }
    result.push_back(new_stmt);
    return result;
  }
};

// Recursively flatten all nested SeqStmt nodes throughout the IR tree.
// This must run before LetStmtNester so that every SeqStmt it encounters
// is already flat, preventing incorrect LetStmt/AttrStmt absorption across
// nested SeqStmt boundaries.
class SeqStmtFlattener : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    // First, recursively visit children.
    Array<Stmt> visited;
    for (const auto &s : op->seq) {
      visited.push_back(this->VisitStmt(s));
    }
    // Then flatten: if any child is itself a SeqStmt, inline its children.
    Array<Stmt> flat;
    std::function<void(const Stmt &)> Flatten = [&](const Stmt &s) {
      if (const auto *inner = s.as<SeqStmtNode>()) {
        for (const auto &inner_s : inner->seq) {
          Flatten(inner_s);
        }
      } else {
        flat.push_back(s);
      }
    };
    for (const auto &s : visited) {
      Flatten(s);
    }
    if (flat.empty())
      return Evaluate(0);
    if (flat.size() == 1)
      return flat[0];
    return SeqStmt(flat);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt) {
  SeqStmtFlattener flattener;
  Stmt flat = flattener(stmt);
  LetStmtNester nester;
  return nester(flat);
}

// StmtMutator to rewrite alloc_buffers in Block nodes for double buffering
class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const BlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    bool needs_update = false;
    Array<Buffer> new_alloc_buffers;
    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
    }
    return Stmt(new_block);
  }

  const std::vector<MultiVersionBufferInfo> &buffer_infos_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (buffer_infos.empty()) {
    return stmt;
  }
  AllocBufferRewriter rewriter(buffer_infos);
  return rewriter(stmt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
