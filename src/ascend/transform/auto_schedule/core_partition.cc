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
 * \file core_partition.cc
 * \brief Direct scheduled-TIR core partition and pipeline lowering.
 */

#include "./core_partition.h"
#include "./task_annotations.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>

#include "op/builtin.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectPtrEqual;
using ffi::ObjectPtrHash;
using ffi::Optional;
using ffi::String;

namespace {

Stmt MakeSequence(const Array<Stmt> &statements) {
  if (statements.empty())
    return Stmt();
  if (statements.size() == 1)
    return statements[0];
  return SeqStmt::Flatten(statements);
}

Array<Stmt> GetSequence(const Stmt &stmt) {
  if (const auto *sequence = stmt.as<SeqStmtNode>())
    return sequence->seq;
  return {stmt};
}

class LetVarSubstituter : public StmtExprMutator {
public:
  explicit LetVarSubstituter(const Map<Var, PrimExpr> &vmap) : vmap_(vmap) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = vmap_.find(GetRef<Var>(op));
    return it != vmap_.end() ? (*it).second : StmtExprMutator::VisitExpr_(op);
  }

  Buffer VisitBufferUse(const Buffer &buffer) final {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end())
      return (*it).second;
    auto var_it = vmap_.find(buffer->data);
    if (var_it == vmap_.end())
      return buffer;
    const auto *data = (*var_it).second.as<VarNode>();
    if (data == nullptr)
      return buffer;
    Buffer new_buffer = buffer;
    new_buffer.CopyOnWrite()->data = GetRef<Var>(data);
    buffer_remap_[buffer] = new_buffer;
    return new_buffer;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    ffi::Any node = op->node;
    bool node_changed = false;
    if (const auto *expr_node = op->node.as<PrimExprNode>()) {
      PrimExpr old_node = GetRef<PrimExpr>(expr_node);
      PrimExpr new_node = VisitExpr(old_node);
      node_changed = !new_node.same_as(old_node);
      node = std::move(new_node);
    }
    PrimExpr value = VisitExpr(op->value);
    Stmt body = VisitStmt(op->body);
    if (!node_changed && value.same_as(op->value) && body.same_as(op->body))
      return GetRef<Stmt>(op);
    return AttrStmt(node, op->attr_key, value, body, op->span);
  }

  const Map<Var, PrimExpr> &vmap_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

Stmt SubstituteLetVars(Stmt stmt, const Map<Var, PrimExpr> &vmap) {
  if (vmap.empty())
    return stmt;
  return LetVarSubstituter(vmap)(std::move(stmt));
}

std::optional<int64_t> GetScheduledInt(const AttrStmtNode *op,
                                       const char *key) {
  if (op->attr_key != attr::kAscendTask)
    return std::nullopt;
  const auto *metadata = op->node.as<ffi::MapObj>();
  if (metadata == nullptr)
    return std::nullopt;
  for (const auto &[metadata_key, value] : *metadata) {
    if (metadata_key.cast<ffi::String>() != key)
      continue;
    PrimExpr expr = value.cast<PrimExpr>();
    const auto *integer = expr.as<IntImmNode>();
    TVM_FFI_CHECK(integer != nullptr, ValueError)
        << "Scheduled metadata `" << key << "` must be an integer, got "
        << expr;
    return integer->value;
  }
  return std::nullopt;
}

int64_t RequireScheduledInt(const AttrStmtNode *op, const char *key) {
  std::optional<int64_t> value = GetScheduledInt(op, key);
  TVM_FFI_CHECK(value.has_value(), ValueError)
      << "Scheduled node is missing integer metadata `" << key << "`";
  return value.value();
}

bool IsScheduledUnitMarker(const AttrStmtNode *op) {
  return GetScheduledInt(op, kAscendScheduledStage).has_value();
}

bool IsScheduledLeafTask(const AttrStmtNode *op) {
  return IsScheduledUnitMarker(op) &&
         GetScheduledInt(op, kAscendScheduledCoreMask).has_value();
}

CoreMask GetTaskCoreMask(const AttrStmtNode *op) {
  int64_t value = RequireScheduledInt(op, kAscendScheduledCoreMask);
  TVM_FFI_CHECK_GE(value, 0, ValueError);
  TVM_FFI_CHECK_LE(value, std::numeric_limits<CoreMask>::max(), ValueError);
  CoreMask core_mask = static_cast<CoreMask>(value);
  TVM_FFI_CHECK(core_mask != kCoreUnassigned && IsValidCoreMask(core_mask),
                ValueError)
      << "Scheduled task has invalid core mask " << value;
  return core_mask;
}

int GetMarkerStage(const AttrStmtNode *op) {
  int64_t stage = RequireScheduledInt(op, kAscendScheduledStage);
  TVM_FFI_CHECK_GE(stage, std::numeric_limits<int>::min(), ValueError);
  TVM_FFI_CHECK_LE(stage, std::numeric_limits<int>::max(), ValueError);
  return static_cast<int>(stage);
}

Optional<Bind> FindTaskBind(const Stmt &task_body) {
  Stmt current = task_body;
  while (true) {
    if (const auto *bind = current.as<BindNode>())
      return GetRef<Bind>(bind);
    if (const auto *conditional = current.as<IfThenElseNode>()) {
      if (conditional->else_case.has_value())
        return std::nullopt;
      current = conditional->then_case;
      continue;
    }
    if (const auto *attribute = current.as<AttrStmtNode>()) {
      current = attribute->body;
      continue;
    }
    return std::nullopt;
  }
}

Stmt ReplaceTaskBind(const Stmt &task_body, const Stmt &replacement) {
  if (task_body.as<BindNode>())
    return replacement;
  if (const auto *conditional = task_body.as<IfThenElseNode>()) {
    ICHECK(!conditional->else_case.has_value());
    return IfThenElse(conditional->condition,
                      ReplaceTaskBind(conditional->then_case, replacement),
                      std::nullopt, conditional->span);
  }
  const auto *attribute = task_body.as<AttrStmtNode>();
  ICHECK(attribute != nullptr)
      << "Expected a native guard chain around Bind, got " << task_body;
  return AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                  ReplaceTaskBind(attribute->body, replacement),
                  attribute->span);
}

bool IsGuardedTaskBind(const AttrStmtNode *marker) {
  return FindTaskBind(marker->body).defined() &&
         marker->body.as<BindNode>() == nullptr;
}

class CoreExtractor : public StmtMutator {
public:
  static Optional<Stmt> Extract(const Stmt &stmt, CoreMask target_core) {
    ICHECK(IsConcreteCore(target_core));
    CoreExtractor extractor(target_core);
    Stmt result = extractor.VisitStmt(stmt);
    if (!extractor.found_core_)
      return std::nullopt;
    ICHECK(result.defined()) << "CoreExtractor removed detected core "
                             << static_cast<int>(target_core);
    return result;
  }

private:
  explicit CoreExtractor(CoreMask target_core) : target_core_(target_core) {}

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = VisitStmt(stmt);
      if (rewritten.defined())
        statements.push_back(std::move(rewritten));
    }
    return MakeSequence(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Map<Var, PrimExpr> outer_remap = var_remap_;
    PrimExpr min = Substitute(op->min, var_remap_);
    PrimExpr extent = Substitute(op->extent, var_remap_);
    Optional<PrimExpr> step = op->step;
    if (step.has_value())
      step = Substitute(step.value(), var_remap_);
    Var loop_var = op->loop_var.copy_with_suffix("");
    var_remap_.Set(op->loop_var, loop_var);
    Stmt body = VisitStmt(op->body);
    var_remap_ = std::move(outer_remap);
    if (!body.defined())
      return Stmt();
    return For(loop_var, min, extent, op->kind, std::move(body),
               op->thread_binding, op->annotations, step, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = Substitute(op->condition, var_remap_);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.has_value()) {
      Stmt rewritten_else = VisitStmt(op->else_case.value());
      if (rewritten_else.defined())
        else_case = std::move(rewritten_else);
    }
    if (!then_case.defined()) {
      if (!else_case.has_value())
        return Stmt();
      return IfThenElse(Not(condition), else_case.value(), std::nullopt,
                        op->span);
    }
    return IfThenElse(condition, std::move(then_case), std::move(else_case),
                      op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsScheduledUnitMarker(op)) {
      if (IsScheduledLeafTask(op))
        return ExtractTask(op);
      Stmt body = VisitStmt(op->body);
      if (!body.defined())
        return Stmt();
      return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                      op->span);
    }
    Stmt body = VisitStmt(op->body);
    if (!body.defined())
      return Stmt();
    return SubstituteLetVars(
        AttrStmt(op->node, op->attr_key, op->value, std::move(body), op->span),
        var_remap_);
  }

  Stmt ExtractTask(const AttrStmtNode *op) {
    CoreMask task_core_mask = GetTaskCoreMask(op);
    // Broadcast tasks follow an existing concrete stream, but must not create
    // an otherwise absent Vector or Cube stream by themselves.
    found_core_ |= task_core_mask == target_core_;
    if (!HasCore(task_core_mask, target_core_))
      return Stmt();

    Stmt body = SubstituteLetVars(op->body, var_remap_);
    Optional<Bind> bind = FindTaskBind(body);
    if (task_core_mask == kCoreBroadcast && bind.defined()) {
      Var source_var = bind.value()->var;
      Var output_var = source_var.copy_with_suffix("");
      body = ReplaceTaskBind(
          body, Bind(output_var, bind.value()->value, bind.value()->span));
      var_remap_.Set(source_var, output_var);
    }
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  CoreMask target_core_;
  bool found_core_{false};
  Map<Var, PrimExpr> var_remap_;
};

class ConditionalLetCollector : public StmtVisitor {
public:
  static Map<Var, Buffer> Collect(const Stmt &stmt) {
    ConditionalLetCollector collector;
    collector(stmt);
    return collector.buffers_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (!IsScheduledLeafTask(op)) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    Optional<Bind> bind = FindTaskBind(op->body);
    if (!bind.defined() || !IsGuardedTaskBind(op) ||
        bind.value()->value.dtype().is_handle())
      return;
    Var var = bind.value()->var;
    if (!buffers_.count(var)) {
      buffers_.Set(var, decl_buffer({IntImm(DataType::Int(32), 1)},
                                    bind.value()->value.dtype(), var->name_hint,
                                    "local.var"));
    }
  }

  Map<Var, Buffer> buffers_;
};

int GetScheduledStage(const Stmt &stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto *attribute = current.as<AttrStmtNode>()) {
      if (IsScheduledUnitMarker(attribute))
        return GetMarkerStage(attribute);
      current = attribute->body;
      continue;
    }
    if (const auto *conditional = current.as<IfThenElseNode>()) {
      TVM_FFI_CHECK(!conditional->else_case.has_value(), ValueError)
          << "A scheduled control guard may not contain an else branch";
      current = conditional->then_case;
      continue;
    }
    TVM_FFI_THROW(ValueError)
        << "Expected a tl.ascend_task scheduled unit, got " << current;
  }
}

bool IsScheduledLetTask(const Stmt &stmt) {
  const auto *attribute = stmt.as<AttrStmtNode>();
  return attribute != nullptr && IsScheduledLeafTask(attribute) &&
         FindTaskBind(attribute->body).defined();
}

Map<String, ffi::Any>
FilterSchedulingAnnotations(const Map<String, ffi::Any> &annotations) {
  Map<String, ffi::Any> result = annotations;
  result.erase("num_stages");
  result.erase("multi_buffer_eligible");
  result.erase("no_conflict");
  return result;
}

class PipelineLowerer : public StmtMutator {
public:
  static Stmt Lower(const Stmt &stmt) {
    Map<Var, Buffer> conditional_lets = ConditionalLetCollector::Collect(stmt);
    Map<Var, PrimExpr> substitutions;
    for (const auto &[var, buffer] : conditional_lets) {
      substitutions.Set(var,
                        BufferLoad(buffer, {IntImm(DataType::Int(32), 0)}));
    }
    Stmt rewritten = SubstituteLetVars(stmt, substitutions);
    return PipelineLowerer(std::move(conditional_lets))(rewritten);
  }

private:
  explicit PipelineLowerer(Map<Var, Buffer> conditional_lets)
      : conditional_lets_(std::move(conditional_lets)) {}

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &stmt : op->seq) {
      Stmt lowered = VisitStmt(stmt);
      if (lowered.defined())
        statements.push_back(std::move(lowered));
    }
    if (statements.empty())
      return Evaluate(0);
    return MakeSequence(statements);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (!IsScheduledUnitMarker(op))
      return StmtMutator::VisitStmt_(op);
    if (!IsScheduledLeafTask(op))
      return VisitStmt(op->body);
    return LowerTask(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Array<Stmt> children = GetSequence(op->body);
    TVM_FFI_CHECK(!children.empty(), ValueError)
        << "Scheduled control may not have an empty body";
    int min_stage = std::numeric_limits<int>::max();
    int max_stage = std::numeric_limits<int>::min();
    for (const Stmt &child : children) {
      int stage = GetScheduledStage(child);
      min_stage = std::min(min_stage, stage);
      max_stage = std::max(max_stage, stage);
    }

    PrimExpr loop_step =
        op->step.has_value() ? op->step.value() : IntImm(DataType::Int(32), 1);
    Array<Stmt> body;
    for (const Stmt &child : children) {
      int stage = GetScheduledStage(child);
      bool is_let = IsScheduledLetTask(child);
      Stmt lowered = VisitStmt(child);
      ICHECK(lowered.defined());
      if (min_stage == max_stage) {
        body.push_back(std::move(lowered));
        continue;
      }

      Map<Var, PrimExpr> substitution;
      substitution.Set(op->loop_var,
                       op->loop_var -
                           loop_step * (max_stage - static_cast<int>(stage)));
      if (is_let) {
        Map<Var, PrimExpr> clamped_substitution;
        clamped_substitution.Set(
            op->loop_var,
            Max(op->min,
                Min(op->min + op->extent - loop_step,
                    op->loop_var -
                        loop_step * (max_stage - static_cast<int>(stage)))));
        body.push_back(Substitute(std::move(lowered), clamped_substitution));
        continue;
      }

      PrimExpr condition =
          And(op->loop_var < op->min + op->extent, op->loop_var >= op->min);
      if (stage == min_stage)
        condition = op->loop_var >= op->min;
      if (stage == max_stage)
        condition = op->loop_var < op->min + op->extent;
      body.push_back(
          Substitute(IfThenElse(condition, std::move(lowered)), substitution));
    }

    PrimExpr extent = op->extent;
    if (min_stage != max_stage)
      extent = extent + loop_step * (max_stage - min_stage);
    return For(op->loop_var, op->min, extent, op->kind, MakeSequence(body),
               op->thread_binding, FilterSchedulingAnnotations(op->annotations),
               op->step, op->span);
  }

  Stmt LowerTask(const AttrStmtNode *op) {
    Optional<Bind> bind = FindTaskBind(op->body);
    if (!bind.defined())
      return op->body;

    Var var = bind.value()->var;
    if (IsGuardedTaskBind(op) && !bind.value()->value.dtype().is_handle()) {
      auto buffer = conditional_lets_.Get(var);
      ICHECK(buffer.has_value())
          << "Missing local.var buffer for guarded scalar Bind " << var;
      Map<String, ffi::Any> annotations;
      annotations.Set(attr::kLocalVarInit, make_zero(buffer.value()->dtype));
      Stmt allocation = AllocBuffer(buffer.value(), annotations);
      Stmt store = ReplaceTaskBind(
          op->body, BufferStore(buffer.value(), bind.value()->value,
                                {IntImm(DataType::Int(32), 0)}));
      return SeqStmt::Flatten(Array<Stmt>{allocation, std::move(store)});
    }

    // Guarded handle bindings must remain unconditional so that their pointer
    // is in scope for later buffer uses. Unguarded bindings are already native.
    return Bind(var, bind.value()->value, bind.value()->span);
  }

  Map<Var, Buffer> conditional_lets_;
};

Optional<Stmt> LowerCore(const Stmt &scheduled_body, CoreMask target_core) {
  Optional<Stmt> extracted =
      CoreExtractor::Extract(scheduled_body, target_core);
  if (!extracted.has_value())
    return std::nullopt;
  return PipelineLowerer::Lower(extracted.value());
}

// Replace every scheduled tilelang_root independently. Walking the full
// PrimFunc naturally handles both single- and multi-kernel functions while the
// scoped cthread state preserves an externally supplied vector-core sid.
class ScheduledTIRPartitioner : public StmtMutator {
public:
  Stmt Partition(const Stmt &stmt) { return VisitStmt(stmt); }
  int NumPartitionedKernels() const { return num_partitioned_kernels_; }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    Optional<Var> outer_sid = sid_;
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iter_var = op->node.as<IterVarNode>()) {
        if (iter_var->thread_tag == "cthread")
          sid_ = iter_var->var;
      }
    }
    Stmt body = VisitStmt(op->body);
    sid_ = std::move(outer_sid);
    if (body.same_as(op->body))
      return GetRef<Stmt>(op);
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root")
      return StmtMutator::VisitStmt_(op);
    ++num_partitioned_kernels_;
    SBlock block = GetRef<SBlock>(op);
    block.CopyOnWrite()->body = LowerScheduledTIRToCores(op->body, sid_);
    return block;
  }

  Optional<Var> sid_;
  int num_partitioned_kernels_{0};
};

// Re-write empty AttrStmt markers to nest them properly. A flat Bind remains
// visible to subsequent statements in the same SeqStmt scope.
class AttrStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> statements;
    for (const Stmt &statement : op->seq)
      statements.push_back(VisitStmt(statement));

    Array<Stmt> flattened;
    for (const Stmt &statement : statements) {
      if (const auto *sequence = statement.as<SeqStmtNode>()) {
        for (const Stmt &inner : sequence->seq)
          flattened.push_back(inner);
      } else {
        flattened.push_back(statement);
      }
    }
    statements = std::move(flattened);

    for (int index = static_cast<int>(statements.size()) - 2; index >= 0;
         --index) {
      if (const auto *attribute = statements[index].as<AttrStmtNode>()) {
        if (IsEmptyBody(attribute->body)) {
          Stmt body = CollectRemaining(statements, index + 1);
          statements =
              TruncateAndReplace(statements, index,
                                 AttrStmt(attribute->node, attribute->attr_key,
                                          attribute->value, std::move(body)));
        }
      }
    }

    if (statements.empty())
      return Evaluate(0);
    return MakeSequence(statements);
  }

private:
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (const auto *integer = evaluate->value.as<IntImmNode>())
        return integer->value == 0;
    }
    return false;
  }

  static Stmt CollectRemaining(const Array<Stmt> &statements, int start) {
    if (start >= static_cast<int>(statements.size()))
      return Evaluate(0);
    Array<Stmt> remaining;
    for (int index = start; index < static_cast<int>(statements.size());
         ++index) {
      remaining.push_back(statements[index]);
    }
    return MakeSequence(remaining);
  }

  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &statements,
                                        int index, Stmt replacement) {
    Array<Stmt> result;
    for (int current = 0; current < index; ++current)
      result.push_back(statements[current]);
    result.push_back(std::move(replacement));
    return result;
  }
};

// Recursively flatten all nested SeqStmt nodes before AttrStmtNester absorbs
// empty AttrStmt markers, preventing absorption across nested boundaries.
class SeqStmtFlattener : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    for (const Stmt &statement : op->seq)
      visited.push_back(VisitStmt(statement));

    Array<Stmt> flattened;
    std::function<void(const Stmt &)> flatten = [&](const Stmt &statement) {
      if (const auto *sequence = statement.as<SeqStmtNode>()) {
        for (const Stmt &inner : sequence->seq)
          flatten(inner);
      } else {
        flattened.push_back(statement);
      }
    };
    for (const Stmt &statement : visited)
      flatten(statement);
    if (flattened.empty())
      return Evaluate(0);
    return MakeSequence(flattened);
  }
};

Stmt ReNestAttrStmts(const Stmt &stmt) {
  Stmt flattened = SeqStmtFlattener()(stmt);
  return AttrStmtNester()(flattened);
}

} // namespace

Stmt LowerScheduledTIRToCores(const Stmt &scheduled_body,
                              Optional<Var> external_sid) {
  Optional<Stmt> vector_body = LowerCore(scheduled_body, kCoreVector);
  Optional<Stmt> cube_body = LowerCore(scheduled_body, kCoreCube);
  if (vector_body.has_value() && cube_body.has_value()) {
    Array<Stmt> blocks;
    blocks.push_back(SBlock({}, {}, {}, "CUBE", cube_body.value(), std::nullopt,
                            {}, {}, {}));
    Map<String, ffi::Any> vector_annotations;
    vector_annotations.Set("vector_count", Integer(2));
    if (external_sid.has_value()) {
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", vector_body.value(),
                              std::nullopt, {}, {}, vector_annotations));
    } else {
      Var sid("sid", DataType::Int(32));
      IterVar sid_thread(Range(0, 2), sid, IterVarType::kThreadIndex,
                         "cthread");
      Stmt threaded_vector_body =
          AttrStmt(sid_thread, tirx::attr::thread_extent, Integer(2),
                   vector_body.value());
      blocks.push_back(SBlock({}, {}, {}, "VECTOR", threaded_vector_body,
                              std::nullopt, {}, {}, vector_annotations));
    }
    return SeqStmt(blocks);
  }
  if (vector_body.has_value())
    return vector_body.value();
  if (cube_body.has_value())
    return cube_body.value();
  return Evaluate(0);
}

tvm::transform::Pass CorePartition() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    ScheduledTIRPartitioner partitioner;
    Stmt body = partitioner.Partition(func->body);
    TVM_FFI_CHECK_GT(partitioner.NumPartitionedKernels(), 0, ValueError)
        << "CorePartition could not find the scheduled tilelang_root";
    body = ReNestAttrStmts(body);
    return PrimFunc(func->params, std::move(body), func->ret_type,
                    func->buffer_map, func->attrs);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CorePartition", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CorePartition", CorePartition);
}

} // namespace tl
} // namespace tvm
