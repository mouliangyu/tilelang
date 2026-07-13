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
#include <functional>
#include <iostream>
#include <limits>
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

#include "ascend/transform/auto_schedule/ir_structure.h"
#include "auto_schedule.h"
#include "backend/common/target_utils.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

// Mutator to extract the body of tilelang_root block and optional
// buffer version overrides annotation.
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;
  std::map<std::string, int64_t> buffer_versions_overrides;
  std::set<std::string> unlimit_memory_scopes;
  Optional<Var> sid;

  void VisitStmt_(const SBlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      body = op->body;
      auto ann_val = op->annotations.Get("tl.buffer_versions_map");
      if (ann_val.has_value()) {
        auto *map_obj = ann_val.value().as<ffi::MapObj>();
        if (map_obj) {
          for (const auto &it : *map_obj) {
            if (auto *var = it.first.as<VarNode>()) {
              if (auto imm = it.second.as<int>()) {
                buffer_versions_overrides[var->name_hint] = imm.value();
              }
            }
          }
        }
      }
      auto mem_ann = op->annotations.Get("tl.unlimit_memory_scopes");
      if (mem_ann.has_value()) {
        if (auto *arr = mem_ann.value().as<ffi::ArrayObj>()) {
          for (const auto &it : *arr) {
            if (auto str_opt = it.as<ffi::String>()) {
              unlimit_memory_scopes.insert(str_opt.value());
            }
          }
        }
      }
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "cthread") {
          sid = iv->var;
        }
      }
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

    // Peel through root block.
    if (const auto *block = inner->as<SBlockNode>()) {
      inner = &block->body;
    } else if (const auto *br = inner->as<SBlockRealizeNode>()) {
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
      if (attr->attr_key == tirx::attr::thread_extent) {
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

  Stmt VisitStmt_(const SBlockRealizeNode *op) override {
    auto new_block_body = this->VisitStmt(op->block->body);
    if (new_block_body.same_as(op->block->body))
      return GetRef<Stmt>(op);
    auto new_block =
        SBlock(op->block->iter_vars, op->block->reads, op->block->writes,
               op->block->name_hint, new_block_body, op->block->init,
               op->block->alloc_buffers, op->block->match_buffers,
               op->block->annotations);
    return SBlockRealize(op->iter_values, op->predicate, new_block);
  }

  Stmt VisitStmt_(const SBlockNode *op) override {
    auto new_body = this->VisitStmt(op->body);
    if (new_body.same_as(op->body))
      return GetRef<Stmt>(op);
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations, op->span);
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

  Stmt VisitStmt_(const SBlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      // Keep all block attributes but replace the body
      return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                    new_body_, op->init, op->alloc_buffers, op->match_buffers,
                    op->annotations, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
//
// Inherits `ConstrVisitor` so the constraint stack (For ranges, LetStmt
// bindings, AttrStmt thread_extent/virtual_thread/tilelang_assume,
// IfThenElse branch guards) is maintained automatically. Whenever a fresh
// TaskNode is constructed, `StampCtx` snapshots `GetConstrSet()` onto it
// for later use by RegionsMayConflict.
//
// Caveat: each VisitStmt_ override below replaces ConstrVisitor's default —
// for nodes that introduce a scope (For, LetStmt, AttrStmt, IfThenElse),
// the override must manually call `MakeGuard(...)` around the body
// recursion to keep the stack accurate.
class IRStructureBuilder : public ConstrVisitor {
public:
  // Build an ordered list of IRStructure nodes from a statement. The old
  // SequenceNode wrapper is gone: a SeqStmt simply contributes its children to
  // the current list inline.
  std::vector<std::shared_ptr<IRStructure>> Build(const Stmt &stmt,
                                                  Target target = Target()) {
    target_ = target;
    auto list = BuildList(stmt);
    if (list.empty()) {
      LOG(WARNING) << "IRStructureBuilder: produced empty list after visiting "
                      "statement. "
                   << "This may indicate an unhandled statement type.";
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = stmt;
      list.push_back(std::move(task_node));
    }
    return list;
  }

protected:
  // Visit `stmt` into a fresh accumulator list and return it. Nested visits
  // (loop bodies, if-branches) use this to collect their own ordered list
  // without disturbing the caller's.
  std::vector<std::shared_ptr<IRStructure>> BuildList(const Stmt &stmt) {
    std::vector<std::shared_ptr<IRStructure>> saved;
    saved.swap(result_);
    VisitStmt(stmt);
    std::vector<std::shared_ptr<IRStructure>> out;
    out.swap(result_);
    result_.swap(saved);
    return out;
  }

  // Stamp the current ConstrVisitor stack snapshot onto a freshly-built
  // TaskNode, so RegionsMayConflict later sees the symbol
  // environment (ancestor For ranges, Let bindings, thread_extents,
  // tilelang_assume, branch guards) the region was written under.
  void StampCtx(TaskNode *task) { task->outer_ctx = GetConstrSet(); }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled) {
      // Sequential For -> ControlNode
      auto control_node = std::make_shared<ControlNode>();
      control_node->control = GetRef<For>(op);
      control_node->task = std::make_shared<TaskNode>();
      control_node->task->stmt =
          For(op->loop_var, op->min, op->extent, op->kind, Evaluate(0),
              op->thread_binding, op->annotations, op->step, op->span);
      AnalyzeResourceUsage(Evaluate(op->min), control_node->task.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), control_node->task.get(),
                           true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             control_node->task.get(), true);
      // Stamp the ctx outside the loop scope onto the loop's own task.
      StampCtx(control_node->task.get());

      // Process the loop body inside the loop scope (push loop_var range).
      {
        auto guard_1 =
            MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        auto guard_2 = MakeGuard(op->extent > 0);
        control_node->children = BuildList(op->body);
      }

      result_.push_back(std::move(control_node));
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmt = GetRef<Stmt>(op);

      // Analyze the loop body for resource usage)
      AnalyzeResourceUsage(op->body, task_node.get());
      AnalyzeResourceUsage(Evaluate(op->min), task_node.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), task_node.get(), true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             task_node.get(), true);
      StampCtx(task_node.get());

      result_.push_back(std::move(task_node));
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmt = GetRef<Stmt>(op);

    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::loop_break())) {
        task_node->AddSpecialWrite(SpecialRegister::kLoopControl);
      } else if (call->op.same_as(tl::ascend_set_hf32_mode())) {
        task_node->AddSpecialWrite(SpecialRegister::kHf32Mode);
      } else if (call->op.same_as(tl::ascend_set_copy_pad_value())) {
        task_node->AddSpecialWrite(SpecialRegister::kPadValue);
      }
    }

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    StampCtx(task_node.get());

    result_.push_back(std::move(task_node));
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmt = GetRef<Stmt>(op);

    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    StampCtx(task_node.get());

    result_.push_back(std::move(task_node));
  }

  // Stamp a guard condition onto every schedule candidate produced for an
  // if-branch
  void StampConditionOnBranch(IRStructure *node, const PrimExpr &cond) {
    if (!node)
      return;
    auto stamp_task = [&](IRStructure *item, TaskNode *sink) {
      item->PrependCondition(cond);
      if (sink) {
        AnalyzeResourceUsage(Evaluate(cond), sink, /*only_variables=*/true);
      }
    };
    if (node->IsTask()) {
      stamp_task(node, static_cast<TaskNode *>(node));
    } else if (node->IsControl()) {
      stamp_task(node, static_cast<ControlNode *>(node)->task.get());
    } else {
      // Post-scheduling-only metadata does not appear pre-scheduling; stamp
      // defensively.
      node->PrependCondition(cond);
    }
  }

  // Stamp a `tl.assume` fact onto every schedule candidate produced for the
  // assume's body
  void StampAssumptionOnBranch(IRStructure *node, const PrimExpr &fact) {
    if (!node)
      return;
    auto stamp_task = [&](IRStructure *item, TaskNode *sink) {
      item->PrependAssumption(fact);
      if (sink) {
        AnalyzeResourceUsage(Evaluate(fact), sink, /*only_variables=*/true);
      }
    };
    if (node->IsTask()) {
      stamp_task(node, static_cast<TaskNode *>(node));
    } else if (node->IsControl()) {
      stamp_task(node, static_cast<ControlNode *>(node)->task.get());
    } else {
      node->PrependAssumption(fact);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // No dedicated IfNode: the guard is stamped onto the branch's schedule
    // candidates as a `conditions` delta (see StampConditionOnBranch).
    // then/else become ordinary schedule candidates distinguished only by their
    // conditions; both branches' candidates are spliced into the current list.
    std::vector<std::shared_ptr<IRStructure>> then_list;
    {
      auto guard = MakeGuard(op->condition);
      then_list = BuildList(op->then_case);
    }
    for (auto &node : then_list) {
      StampConditionOnBranch(node.get(), op->condition);
      result_.push_back(std::move(node));
    }

    if (op->else_case) {
      auto guard = MakeGuard(Not(op->condition));
      auto else_list = BuildList(op->else_case.value());
      for (auto &node : else_list) {
        StampConditionOnBranch(node.get(), Not(op->condition));
        result_.push_back(std::move(node));
      }
    }
  }

  void VisitStmt_(const BindNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmt = GetLetDecl(op);
    AnalyzeResourceUsage(GetLetDecl(op), task_node.get());
    StampCtx(task_node.get());
    result_.push_back(std::move(task_node));
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    ICHECK_EQ(op->attr_key, tirx::attr::tilelang_assume)
        << "AutoSchedule IRStructureBuilder only supports AttrStmt "
           "`tl.assume` inside the schedulable tilelang_root body. "
           "Unsupported attr_key: "
        << op->attr_key;

    // No dedicated assume TaskNode: the fact is stamped onto the body's
    // schedule candidates as an `is_assume` guard delta (see
    // StampAssumptionOnBranch), mirroring how IfThenElse stamps conditions.
    // Materializing it as its own task used to (a) inject spurious read-var
    // dependencies and a wasted sequence slot, and (b) scope the fact to an
    // empty body at codegen, hiding it from the loads/stores that actually
    // need it. Keep the fact in the prover environment for region proving via
    // MakeGuard while building the body.
    auto fact = Downcast<PrimExpr>(op->node);
    std::vector<std::shared_ptr<IRStructure>> body_list;
    {
      auto guard = MakeGuard(fact, /*is_assume=*/true);
      body_list = BuildList(op->body);
    }
    for (auto &node : body_list) {
      StampAssumptionOnBranch(node.get(), fact);
      result_.push_back(std::move(node));
    }
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmt = GetRef<Stmt>(op);

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());
    StampCtx(task_node.get());

    result_.push_back(std::move(task_node));
  }

  void VisitStmt_(const SBlockNode *op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by
    // TilelangRootBodyExtractor If we encounter it here, it means we're
    // processing the entire function body (not extracted), which should only
    // happen when there's no tilelang_root block
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmt = GetRef<Stmt>(op);
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    StampCtx(task_node.get());
    result_.push_back(std::move(task_node));
  }

private:
  std::vector<std::shared_ptr<IRStructure>> result_;
  Target target_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node,
                            bool only_variables = false) {
    // Recursively analyze statements to determine resource usage (Ascend pipe
    // model)
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      uint16_t pipe_mask{0};
      uint16_t hbm_mask{0};
      // A padded GM->UB copy (data_select / pad_value) reads the pad register.
      bool reads_pad_value{false};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        DataType input_dtype;
        DataType accum_dtype;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0,
                        DataType input_dtype = DataType::Float(16),
                        DataType accum_dtype = DataType::Float(16))
            : m(m), n(n), k(k), input_dtype(input_dtype),
              accum_dtype(accum_dtype) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;
      std::vector<TaskNode::CopyInfo> copy_infos;

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

      static int64_t CalculateCopyBytes(const CopyNode *copy) {
        int64_t elements = 1;
        for (size_t i = 0; i < copy->dst_range.size(); ++i) {
          const auto &range = copy->dst_range[i];
          if (const int64_t *extent = as_const_int(range->extent)) {
            elements *= *extent;
          } else if (i < copy->dst->shape.size()) {
            // TODO(ascend): use arith bounds from loop context for symbolic
            // extents. Buffer shape is a better task-level fallback than 1.
            if (const int64_t *shape_extent =
                    as_const_int(copy->dst->shape[i])) {
              elements *= *shape_extent;
            } else {
              elements *= 1;
            }
          } else {
            elements *= 1;
          }
        }
        return elements * DataType(copy->dst->dtype).bytes();
      }

      void VisitExpr_(const CallNode *op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");

        if (op->op.same_as(copy_op)) {
          Copy copy_obj(op->args, op->annotations);
          const CopyNode *copy = copy_obj.get();
          uint16_t copy_pipe = GetCopyPipeMask(copy->src, copy->dst);
          pipe_mask |= copy_pipe;
          hbm_mask |= GetHbmPortBit(copy->src, copy->dst);
          // A GM->UB copy with data_select (from pad_value= or an explicit
          // data_select) reads the pad-value register set by a preceding
          // ascend_set_copy_pad_value. Only these copies need to be ordered
          // against that scalar write — not every MTE2 copy.
          if (copy_obj->GetDataSelect() != 0)
            reads_pad_value = true;
          copy_infos.emplace_back(copy->src.scope(), copy->dst.scope(),
                                  DataType(copy->dst->dtype),
                                  CalculateCopyBytes(copy), copy_pipe);
        } else if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
          // SimdVF MicroAPI scatter writing into a UB tmp.
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
        } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
          // Per-D-group copy_ubuf_to_cbuf: UB -> L1 = MTE3.
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kMTE3);
        } else if (op->op.same_as(gemm_op)) {
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kCube);

          int64_t m = op->args[5].as<IntImmNode>()->value;
          int64_t n = op->args[6].as<IntImmNode>()->value;
          int64_t k = op->args[7].as<IntImmNode>()->value;
          Gemm gemm(op->args, op->annotations);
          tensor_core_shapes.emplace_back(m, n, k, gemm->a_->dtype,
                                          gemm->c_->dtype);
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

      void VisitStmt_(const SBlockNode *op) override {
        if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
          pipe_mask |= static_cast<uint16_t>(ResourcePipe::kVector);
        }
      }
    };

    // Specialized issue queues (DMA / Cube / Vector / Fixpipe). On Ascend these
    // are the only pipes a statement can occupy *other than* the scalar (S)
    // unit, which is the default issue queue for everything else.
    constexpr uint16_t kSpecializedPipes =
        static_cast<uint16_t>(ResourcePipe::kMTE1) |
        static_cast<uint16_t>(ResourcePipe::kMTE2) |
        static_cast<uint16_t>(ResourcePipe::kMTE3) |
        static_cast<uint16_t>(ResourcePipe::kCube) |
        static_cast<uint16_t>(ResourcePipe::kVector) |
        static_cast<uint16_t>(ResourcePipe::kFixpipe);

    auto resolve_pipe = [&](uint16_t mask) -> uint16_t {
      if ((mask & kSpecializedPipes) == 0) {
        return mask | static_cast<uint16_t>(ResourcePipe::kScalar);
      }
      return mask;
    };

    ResourceAnalyzer analyzer(task_node, target_);
    analyzer(stmt);

    if (!only_variables) {
      // Set task node pipe mask.
      task_node->SetPipeMask(resolve_pipe(analyzer.pipe_mask));
      task_node->SetHbmMask(analyzer.hbm_mask);
      if (analyzer.reads_pad_value)
        task_node->SetReadsPadValue(true);
      // Set Tensor Core shape information if available
      for (const auto &shape : analyzer.tensor_core_shapes) {
        if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
          task_node->AddTensorCoreShape(shape.m, shape.n, shape.k,
                                        shape.input_dtype, shape.accum_dtype);
        }
      }
      for (const auto &info : analyzer.copy_infos) {
        task_node->AddCopyInfo(info.src_scope, info.dst_scope, info.dtype,
                               info.bytes, info.pipe_mask);
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
};

Stmt ReNestLetStmts(const Stmt &stmt);

// Result of scheduling a single kernel segment
struct ScheduledKernelResult {
  Stmt scheduled_body;
  std::vector<MultiVersionBufferInfo> buffer_infos;
};

// Recursively check if any TaskNode has the given warpgroup_id.
static bool HasWarpgroupId(IRStructure *node, int wg_id) {
  if (!node)
    return false;
  if (node->IsTask())
    return static_cast<TaskNode *>(node)->GetWarpgroupId() == wg_id;
  if (node->IsControl()) {
    for (auto &c : static_cast<ControlNode *>(node)->children)
      if (HasWarpgroupId(c.get(), wg_id))
        return true;
  }
  return false;
}

static bool
HasWarpgroupId(const std::vector<std::shared_ptr<IRStructure>> &nodes,
               int wg_id) {
  for (const auto &n : nodes) {
    if (HasWarpgroupId(n.get(), wg_id))
      return true;
  }
  return false;
}

// Per-value warpgroup demand bitmask: bit0 = needed on wg0 (AIV), bit1 = needed
// on wg1 (AIC). A scalar/register value's demand is the union of the warpgroups
// of everything that consumes it (directly, or transitively through scalar Let
// chains and loop bounds / guard conditions).
static constexpr uint8_t kDemandWg0 = 0b01;
static constexpr uint8_t kDemandWg1 = 0b10;
static constexpr uint8_t kDemandBoth = 0b11;

// Assign every currently-broadcast scalar / LetDecl task to the warpgroup(s)
// that actually consume its output.
static void AssignScalarWarpgroupByDemand(
    const std::vector<std::shared_ptr<IRStructure>> &root) {
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  std::unordered_map<const VarNode *, uint8_t> var_demand;
  std::unordered_map<const BufferNode *, uint8_t> buf_demand;

  auto add_reads_demand = [&](const TaskNode *t, uint8_t mask) -> bool {
    if (!mask)
      return false;
    bool grew = false;
    for (const auto &v : t->GetReadVars()) {
      uint8_t &d = var_demand[v.get()];
      if ((d | mask) != d) {
        d |= mask;
        grew = true;
      }
    }
    for (const auto &r : t->GetReadRegions()) {
      if (!IsRegisterRegion(r))
        continue;
      uint8_t &d = buf_demand[r->buffer.get()];
      if ((d | mask) != d) {
        d |= mask;
        grew = true;
      }
    }
    return grew;
  };
  auto add_guard_demand = [&](const IRStructure *node, uint8_t mask) {
    if (!mask)
      return;
    for (const auto &g : node->GetGuards()) {
      if (g.is_assume)
        continue;
      for (const Var &v : UndefinedVars(g.expr))
        var_demand[v.get()] |= mask;
    }
  };
  // The warpgroup(s) a subtree's compute actually lands on: the union over its
  // concretely-assigned (wg 0/1) descendant tasks. Unassigned / broadcast
  // scalars are ignored here so they don't pollute the coverage (a broadcast
  // node's containWarpgroupId is true for every id). A loop whose body has
  // compute on both cores is replicated to both; a single-core loop only to
  // one.
  auto loop_body_coverage = [&](const IRStructure *node) -> uint8_t {
    uint8_t mask = 0;
    std::function<void(const IRStructure *)> walk = [&](const IRStructure *n) {
      if (!n)
        return;
      if (n->IsTask()) {
        int wg = static_cast<const TaskNode *>(n)->GetWarpgroupId();
        if (wg == 0)
          mask |= kDemandWg0;
        else if (wg == 1)
          mask |= kDemandWg1;
      } else if (n->IsControl()) {
        auto *c = static_cast<const ControlNode *>(n);
        for (const auto &child : c->children)
          walk(child.get());
      }
    };
    walk(node);
    return mask;
  };

  // Seed phase: each concretely-assigned compute/MTE task demands its reads on
  // its own warpgroup; a still-broadcast task (true serves-both scalar, or an
  // `m==0` task) demands on both. Loop headers and guards demand on whichever
  // cores the enclosing structure runs.
  std::function<void(const IRStructure *, uint8_t)> seed =
      [&](const IRStructure *node, uint8_t enclosing) {
        if (!node)
          return;
        if (node->IsTask()) {
          const auto *t = static_cast<const TaskNode *>(node);
          int wg = t->GetWarpgroupId();
          uint8_t mask = (wg == 0)   ? kDemandWg0
                         : (wg == 1) ? kDemandWg1
                                     : enclosing; // broadcast / unassigned
          // The task's guard runs on exactly the cores the task runs on.
          add_guard_demand(t, mask);
          add_reads_demand(t, mask);
        } else if (node->IsControl()) {
          auto *c = static_cast<const ControlNode *>(node);
          uint8_t cover = loop_body_coverage(node);
          // The loop's guard and header (bound: min/extent/step) are replicated
          // to exactly the cores the body runs on.
          add_guard_demand(c, cover);
          if (c->task)
            add_reads_demand(c->task.get(), cover);
          for (const auto &child : c->children)
            seed(child.get(), cover);
        }
      };
  for (const auto &node : root)
    seed(node.get(), kDemandBoth);

  // Backward fixpoint: a scalar / LetDecl producer whose output is demanded on
  // some cores demands its own (direct) reads on those same cores. Iterating to
  // a fixpoint closes the demand over the whole producer chain
  // (guard/extent <- advance <- init) and through scalar Let chains, so no
  // precomputed indirect-read closure is needed here.
  auto write_demand = [&](const TaskNode *t) -> uint8_t {
    uint8_t mask = 0;
    for (const auto &v : t->GetWriteVars()) {
      auto it = var_demand.find(v.get());
      if (it != var_demand.end())
        mask |= it->second;
    }
    for (const auto &r : t->GetWriteRegions()) {
      if (!IsRegisterRegion(r))
        continue;
      auto it = buf_demand.find(r->buffer.get());
      if (it != buf_demand.end())
        mask |= it->second;
    }
    return mask;
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &task_ctx : all_tasks) {
      TaskNode *t = task_ctx.task;
      if (!t->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(t))
        continue;
      uint8_t mask = write_demand(t);
      if (add_reads_demand(t, mask))
        changed = true;
    }
  }

  // Write-back phase: place each broadcast scalar / LetDecl into its consumer
  // warpgroup union. A value with no consumer keeps broadcast (harmless: a dead
  // LetDecl is DCE'd per warpgroup, a dead store is a no-op on both).
  for (auto &task_ctx : all_tasks) {
    TaskNode *t = task_ctx.task;
    if (!IsWarpgroupBroadcast(t->GetWarpgroupId()))
      continue;
    if (!t->UsesPipe(ResourcePipe::kScalar) && !IsLetDeclTask(t))
      continue;
    uint8_t mask = write_demand(t);
    if (mask == kDemandWg0)
      t->SetWarpgroupId(0);
    else if (mask == kDemandWg1)
      t->SetWarpgroupId(1);
    else if (mask == 0)
      t->SetWarpgroupId(kWarpgroupUnassigned);
    // kDemandBoth: leave broadcast.
  }
}

// Local enumerator for PropagateReadRegions. IRStructureBuilder emits BindNode
// and tl.assume AttrStmt as ordinary TaskNodes.
static void CollectTaskNodesForPropagation(IRStructure *node,
                                           std::vector<TaskNode *> &out) {
  if (!node)
    return;
  if (node->IsTask()) {
    out.push_back(static_cast<TaskNode *>(node));
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    for (const auto &child : ctrl->children)
      CollectTaskNodesForPropagation(child.get(), out);
  } else {
    LOG(FATAL) << "Unknown IRStructure kind in "
                  "CollectTaskNodesForPropagation";
  }
}

// Compute indirect read regions/vars for every task: the direct reads plus
// anything reached transitively through scalar Let bindings.
static void
PropagateReadRegions(const std::vector<std::shared_ptr<IRStructure>> &root) {
  std::vector<TaskNode *> all_tasks;
  for (const auto &node : root)
    CollectTaskNodesForPropagation(node.get(), all_tasks);

  // Seed every task's indirect set from its direct reads.
  for (TaskNode *task : all_tasks)
    task->SeedIndirectReads();

  // Build mapping: var → the LetDecl TaskNode that defines it
  std::unordered_map<const VarNode *, TaskNode *> var_to_let_decl;
  for (TaskNode *task : all_tasks) {
    if (IsLetDeclTask(task)) {
      const auto *bind = task->stmt.as<BindNode>();
      var_to_let_decl[bind->var.get()] = task;
    }
  }

  if (var_to_let_decl.empty())
    return;

  // Iterate until no new read regions or read vars are added.
  bool changed = true;
  while (changed) {
    changed = false;
    size_t count_before = 0;
    for (TaskNode *task : all_tasks)
      count_before += task->GetIndirectReadRegions().size() +
                      task->GetIndirectReadVars().size();

    for (TaskNode *task : all_tasks) {
      // Snapshot current indirect read vars (may grow during propagation)
      std::vector<Var> read_vars = task->GetIndirectReadVars();
      for (const auto &var : read_vars) {
        auto it = var_to_let_decl.find(var.get());
        if (it == var_to_let_decl.end())
          continue;
        TaskNode *def = it->second;
        if (def == task)
          continue;
        // Fold the defining Let's already-accumulated indirect closure into
        // this task's indirect set.
        for (const auto &region : def->GetIndirectReadRegions())
          task->AddIndirectReadRegion(region);
        for (const auto &v : def->GetIndirectReadVars())
          task->AddIndirectReadVar(v);
      }
    }

    size_t count_after = 0;
    for (TaskNode *task : all_tasks)
      count_after += task->GetIndirectReadRegions().size() +
                     task->GetIndirectReadVars().size();
    if (count_after > count_before)
      changed = true;
  }
}

// Handles IRStructure building, Z3 scheduling (with Ascend sync insertion),
// and Stmt conversion.
static ScheduledKernelResult ScheduleSingleKernel(
    const Stmt &kernel_body, Target target,
    const std::map<std::string, int64_t> &buffer_versions_overrides = {},
    const std::set<std::string> &unlimit_memory_scopes = {},
    Optional<Var> external_sid = {}) {
  // Build IRStructure from the body to schedule
  ScheduledKernelResult result;
  IRStructureBuilder builder;
  auto ir_structure = builder.Build(kernel_body, target);

  ICHECK(!ir_structure.empty()) << "IRStructure is empty (empty body?)";

  // Compute indirect read sets (direct reads + reads reached through scalar
  // Let bindings, e.g. m_tile → m_tile_3 → MTE copy) so warpgroup resolution
  // can see register buffers accessed indirectly. Direct read regions/vars are
  // left untouched for barrier/dependency analysis.
  PropagateReadRegions(ir_structure);

  // Schedule IRStructure in place (Z3 scheduling + Ascend sync)
  ScheduleBuilder unit_builder;
  int64_t shared_mem_limit = GetSharedMemoryLimit(target);
  if (TargetIsAscend(target) && !HasSimtVF(kernel_body)) {
    shared_mem_limit = 248 * 1024;
  }
  unit_builder.SetMemoryLimit("shared", shared_mem_limit);
  unit_builder.SetMemoryLimit("shared.l1", GetL1MemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0c", GetL0CMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0a", GetL0AMemoryLimit(target));
  unit_builder.SetMemoryLimit("shared.l0b", GetL0BMemoryLimit(target));
  constexpr int64_t kUnlimitedMemory = 1LL << 40;
  for (const auto &scope : unlimit_memory_scopes) {
    unit_builder.SetMemoryLimit(scope, kUnlimitedMemory);
  }
  unit_builder.SetBufferVersionsOverride(buffer_versions_overrides);
  unit_builder.ScheduleList(ir_structure);

  // Assign warpgroup IDs based on pipe and buffer scopes
  {
    std::vector<TaskNodeWithContext> all_tasks;
    CollectAllTaskNodesWithContext(ir_structure, all_tasks);
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
      } else if (t->WritesSpecialRegister(SpecialRegister::kHf32Mode)) {
        // HF32 mode configures the Cube (AIC) unit.
        t->SetWarpgroupId(1);
      } else if (t->WritesSpecialRegister(SpecialRegister::kPadValue)) {
        // set_copy_pad_value runs on the vector core (AIV).
        t->SetWarpgroupId(0);
      } else if (t->UsesPipe(ResourcePipe::kScalar) || m == 0) {
        // A scalar / mask-0 task issues on the S pipe, but its warpgroup is
        // still fixed by the on-chip memory it touches: UB is AIV-side (wg0),
        // L1/L0A/L0B/L0C are AIC-side (wg1).
        bool touches_ub = false, touches_cube = false;
        auto scan = [&](const Array<BufferRegion> &regions) {
          for (auto &r : regions) {
            if (IsSharedBuffer(r->buffer))
              touches_ub = true;
            else if (IsL1Buffer(r->buffer) || IsL0ABuffer(r->buffer) ||
                     IsL0BBuffer(r->buffer) || IsL0CBuffer(r->buffer))
              touches_cube = true;
          }
        };
        scan(t->GetReadRegions());
        scan(t->GetWriteRegions());
        if (touches_ub && !touches_cube) {
          t->SetWarpgroupId(0);
        } else if (touches_cube && !touches_ub) {
          t->SetWarpgroupId(1);
        } else {
          t->SetWarpgroupId(kWarpgroupBroadcast);
        }
      }
    }
  }

  // Assign each broadcast scalar / LetDecl to the warpgroup(s) that consume its
  // output (a {0}/{1}/{0,1} demand lattice); values needed on both cores stay
  // broadcast, one-sided values are demoted to their single consumer core.
  AssignScalarWarpgroupByDemand(ir_structure);

  // Fallback: pin every unresolved task to a single core: AIC if the kernel has
  // any concrete Cube-side (wg1) work, otherwise AIV.
  {
    std::vector<TaskNodeWithContext> all_tasks;
    CollectAllTaskNodesWithContext(ir_structure, all_tasks);
    std::vector<TaskNode *> unresolved;
    bool has_cube_wg = false;
    for (auto &task_ctx : all_tasks) {
      int wg = task_ctx.task->GetWarpgroupId();
      if (wg == 1)
        has_cube_wg = true;
      else if (wg == kWarpgroupUnassigned)
        unresolved.push_back(task_ctx.task);
    }
    int fallback_wg = has_cube_wg ? 1 : 0;
    for (auto *t : unresolved)
      t->SetWarpgroupId(fallback_wg);
  }

  EventIdMap event_id_map;
  FlagInfoList flag_infos;
  MultiVersionBufferMap buffer_info_map;
  AnalyzeAndInsertBarriers(ir_structure, event_id_map, flag_infos,
                           result.buffer_infos, buffer_info_map);

  Var sid;
  if (external_sid.has_value()) {
    sid = external_sid.value();
  } else {
    sid = Var("sid", DataType::Int(32));
  }

  Stmt wg0_body = Evaluate(0);
  Stmt wg1_body = Evaluate(0);

  bool has_wg0 = HasWarpgroupId(ir_structure, 0);
  bool has_wg1 = HasWarpgroupId(ir_structure, 1);

  if (has_wg0) {
    auto wg = CloneIRStructureWithWarpgroupFilter(ir_structure, 0);
    wg0_body = EmitWarpgroupBody(wg, sid);
  }
  if (has_wg1) {
    auto wg = CloneIRStructureWithWarpgroupFilter(ir_structure, 1);
    wg1_body = EmitWarpgroupBody(wg, sid);
  }

  if (!flag_infos.empty()) {
    std::vector<Stmt> stmts[2];
    for (const auto &info : flag_infos) {
      if (info.mode == 0) {
        stmts[info.warpgroup_id].push_back(
            MakeAscendSetFlag(info.hardevent, info.flag_var));
      } else if (info.mode == 1) {
        stmts[info.warpgroup_id].push_back(
            MakeAscendCrossCoreSetFlag(4, info.hardevent, info.flag_var));
      }
    }
    stmts[0].push_back(wg0_body);
    stmts[1].push_back(wg1_body);
    for (const auto &info : flag_infos) {
      if (info.mode == 0) {
        stmts[info.warpgroup_id].push_back(
            MakeAscendWaitFlag(info.hardevent, info.flag_var));
      } else if (info.mode == 2) {
        stmts[info.warpgroup_id].push_back(
            MakeAscendCrossCoreWaitFlag(4, info.hardevent, info.flag_var));
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
        SBlock({}, {}, {}, "CUBE", wg1_body, std::nullopt, {}, {}, {}));
    Map<String, Any> vec_annotations;
    vec_annotations.Set("vector_count", Integer(2));
    if (external_sid.has_value()) {
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", wg0_body, std::nullopt, {},
                              {}, vec_annotations));
    } else {
      IterVar sid_thread(Range(0, 2), sid, IterVarType::kThreadIndex,
                         "cthread");
      Stmt vec_body =
          AttrStmt(sid_thread, tirx::attr::thread_extent, Integer(2), wg0_body);
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", vec_body, std::nullopt, {},
                              {}, vec_annotations));
    }
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
  using namespace tirx::transform;
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
      auto kr =
          ScheduleSingleKernel(body_to_schedule, target, overrides,
                               extractor.unlimit_memory_scopes, extractor.sid);

      TilelangRootBodyReplacer replacer(kr.scheduled_body);
      Stmt final_body = replacer(func->body);
      if (!kr.buffer_infos.empty()) {
        final_body = RewriteAllocBuffers(final_body, kr.buffer_infos);
        // Rewrite multi-buffered accesses that live outside the annotated loop
        // (e.g. prologue fill): broadcast them to version 0 over the full
        // version dimension. In-loop accesses were already versioned.
        BroadcastMultiBufferRewriter broadcast(kr.buffer_infos);
        final_body = broadcast(final_body);
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
      auto result =
          ScheduleSingleKernel(body_to_schedule, target, overrides,
                               extractor.unlimit_memory_scopes, extractor.sid);
      Stmt scheduled_body = result.scheduled_body;

      TilelangRootBodyReplacer replacer(scheduled_body);
      Stmt scheduled_subtree = replacer(kernel_subtree);

      if (!result.buffer_infos.empty()) {
        scheduled_subtree =
            RewriteAllocBuffers(scheduled_subtree, result.buffer_infos);
        BroadcastMultiBufferRewriter broadcast(result.buffer_infos);
        scheduled_subtree = broadcast(scheduled_subtree);
      }

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

// Re-write empty AttrStmt markers to nest them properly.
// Example transformation:
//   SeqStmt {
//     Bind(x, 42)
//     Bind(y, x+1)
//     compute(x, y)                   // actual work
//     store(result)
//   }
// remains flat for Bind because tirx Bind is visible to subsequent statements
// in the same SeqStmt scope.
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
      if (const auto *attr = stmts[i].as<AttrStmtNode>()) {
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
// is already flat, preventing incorrect AttrStmt absorption across
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

Layout ExpandLayoutToBuffer(const Buffer &layout_buffer,
                            const Buffer &target_buffer, const Layout &layout) {
  if (!layout.defined() ||
      target_buffer->shape.size() < layout_buffer->shape.size()) {
    return Layout();
  }
  size_t leading_ndim =
      target_buffer->shape.size() - layout_buffer->shape.size();
  if (leading_ndim == 0) {
    return layout;
  }
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(target_buffer->shape[i]);
  }
  return layout->Expand(leading_shape);
}

Layout FindLayoutForBuffer(const Map<Buffer, Layout> &layout_map,
                           const Buffer &buffer) {
  if (layout_map.count(buffer)) {
    return layout_map[buffer];
  }
  for (const auto &[layout_buffer, layout] : layout_map) {
    if (layout_buffer->data.same_as(buffer->data) ||
        layout_buffer->name == buffer->name) {
      Layout expanded = ExpandLayoutToBuffer(layout_buffer, buffer, layout);
      if (expanded.defined()) {
        return expanded;
      }
    }
  }
  return Layout();
}

bool AddLayoutForBuffer(Map<Buffer, Layout> *layout_map, const Buffer &buffer) {
  if (!layout_map || layout_map->count(buffer)) {
    return false;
  }
  Layout layout = FindLayoutForBuffer(*layout_map, buffer);
  if (!layout.defined()) {
    return false;
  }
  layout_map->Set(buffer, layout);
  return true;
}

// StmtMutator to rewrite alloc_buffers in SBlock nodes for double buffering
class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    bool needs_update = false;
    bool layout_update = false;
    Array<Buffer> new_alloc_buffers;
    Map<String, Any> new_annotations = op->annotations;
    bool has_layout_map = false;
    Map<Buffer, Layout> layout_map;
    if (auto layout_map_ref = op->annotations.Get(attr::kLayoutMap)) {
      if (auto map = layout_map_ref.value().as<Map<Buffer, Layout>>()) {
        layout_map = map.value();
        has_layout_map = true;
      }
    }
    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }
    if (has_layout_map) {
      for (const auto &buffer : new_alloc_buffers) {
        layout_update |= AddLayoutForBuffer(&layout_map, buffer);
      }
    }
    if (layout_update) {
      new_annotations.Set(attr::kLayoutMap, layout_map);
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
    }
    if (layout_update) {
      new_block->annotations = new_annotations;
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
