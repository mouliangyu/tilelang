#pragma once
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "./ir_structure.h"
#include "op/builtin.h"
#include "op/utils.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::Map;

// Calculate the size in bytes of a buffer. Returns 0 if the size cannot be
// determined at compile time.
inline size_t GetBufferSize(const Buffer &buffer) {
  size_t elem_size = buffer->dtype.bits() * buffer->dtype.lanes() / 8;
  PrimExpr size = IntImm(DataType::Int(64), elem_size);
  for (const auto &dim : buffer->shape) {
    size *= dim;
  }
  arith::Analyzer analyzer;
  auto size_val = analyzer.Simplify(size);
  if (const auto *int_imm = size_val.as<IntImmNode>()) {
    return int_imm->value;
  } else {
    return 0;
  }
}

// Helper function to rewrite alloc_buffer for multi-version support
inline Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
  // Create a copy of the buffer
  ObjectPtr<BufferNode> new_buffer =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));

  // Add num_versions as first dimension
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));

  // Update strides if they exist
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }

  return Buffer(new_buffer);
}

// Structure to store loop nesting information
struct LoopNestingInfo {
  std::vector<Var> loop_vars;
  std::vector<PrimExpr> loop_starts;
  std::vector<PrimExpr> loop_steps;
  std::vector<PrimExpr> loop_extents;

  // Add a loop to the nesting info
  void AddLoop(const ForNode *for_node) {
    loop_vars.push_back(for_node->loop_var);
    loop_starts.push_back(for_node->min);
    loop_steps.push_back(for_node->step.has_value()
                             ? for_node->step.value()
                             : IntImm(DataType::Int(32), 1));
    loop_extents.push_back(for_node->extent);
  }

  // Remove the innermost loop
  void PopLoop() {
    if (!loop_vars.empty()) {
      loop_vars.pop_back();
      loop_starts.pop_back();
      loop_steps.pop_back();
      loop_extents.pop_back();
    }
  }

  PrimExpr CalculateIterationCount() const {
    PrimExpr total_iter = IntImm(DataType::Int(32), 0);
    PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);
    for (size_t i = loop_vars.size(); i-- > 0;) {
      PrimExpr normalized_iter =
          indexdiv(loop_vars[i] - loop_starts[i], loop_steps[i]);
      if (i == static_cast<int>(loop_vars.size()) - 1) {
        total_iter = normalized_iter;
      } else {
        total_iter = total_iter + normalized_iter * total_multiplier;
      }
      total_multiplier = total_multiplier * loop_extents[i];
    }
    return total_iter;
  }
};

// Structure to store multi-version buffer information
struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_versions;
  Buffer new_buffer;

  MultiVersionBufferInfo(Buffer buffer, int num_versions, Buffer new_buffer)
      : buffer(buffer), num_versions(num_versions), new_buffer(new_buffer) {}
};

// Synchronization point types
enum class SyncType { PipeBarrier, CrossCore, IntraCore };

// Describes a synchronization point between two ScheduleUnits.
// Collected during analysis and inserted in a separate pass so that
// future optimizations can inspect, deduplicate, or eliminate sync points.
struct SyncPoint {
  SyncType type;
  ScheduleUnit *prod_unit;
  int prod_wg;
  ScheduleUnit *prod_insertion_unit;
  PrimExpr prod_guard;
  ScheduleUnit *cons_unit;
  int cons_wg;
  ScheduleUnit *cons_insertion_unit;
  PrimExpr cons_guard;
  int distance;
  int num_versions;
  std::vector<DepInfo> deps;
  std::string hard_event;
  std::string prod_pipe;
  std::string cons_pipe;
  // CrossCore-specific
  int mode_id;
};

// Barrier dependency analysis function declarations
// Event ID map: per-hard_event counter for independent event ID allocation
using EventIdMap = std::unordered_map<std::string, int>;

// Per-hard_event metadata for prologue/epilogue generation
struct FlagInfo {
  int warpgroup_id; // which side (0=Vector, 1=Cube)
  PrimExpr flag_var;
  std::string hardevent;
  int mode;
};
using FlagInfoList = std::vector<FlagInfo>;

static void
AnalyzeAndInsertBarriers(IRStructure *node, EventIdMap &event_id_map,
                         FlagInfoList &flag_infos, Var sid,
                         LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos);
static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, EventIdMap &event_id_map,
                            FlagInfoList &flag_infos, Var sid,
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos);
static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, EventIdMap &event_id_map,
                           FlagInfoList &flag_infos, Var sid,
                           LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos);
static std::vector<SyncPoint>
CollectSyncPoints(const std::vector<ScheduleUnit *> &units, int num_wgs,
                  std::optional<For> current_loop,
                  const std::map<Buffer, int64_t> &buffer_num_versions = {},
                  bool is_loop = false);
static void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                             FlagInfoList &flag_infos, EventIdMap &event_id_map,
                             LoopNestingInfo &loop_info);
static std::vector<SyncPoint>
OptimizeSyncPoints(const std::vector<SyncPoint> &sync_points);

// StmtExprMutator to rewrite BufferLoad/BufferStore for multi-version buffers
class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &multi_buffer,
      PrimExpr iteration)
      : multi_buffer_(multi_buffer), iteration_(iteration) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(load->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = load.CopyOnWrite();
      n->buffer = multi_buffer_.at(load->buffer);
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
    }

    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(store->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = store.CopyOnWrite();
      n->buffer = multi_buffer_.at(store->buffer);
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
    }

    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    // Check if this is a tl.tileop.region call
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for multi-version buffers
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
      // write, 3: read/write) args[2..] = extents
      if (op->args.size() >= 2) {
        // Check if the buffer in BufferLoad needs multi-version support
        if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
          auto it = multi_buffer_.find(buffer_load->buffer);
          if (it != multi_buffer_.end()) {
            // This buffer needs multi-version support
            // Create new arguments array
            Array<PrimExpr> new_args;

            // Add the updated BufferLoad (already processed by VisitExpr)
            new_args.push_back(VisitExpr(op->args[0]));

            // Add access_type (unchanged)
            new_args.push_back(VisitExpr(op->args[1]));

            // Add extent for version dimension (value = 1)
            new_args.push_back(IntImm(DataType::Int(32), 1));

            // Add existing extents (if any)
            for (size_t i = 2; i < op->args.size(); i++) {
              new_args.push_back(VisitExpr(op->args[i]));
            }

            // Create new Call node with updated arguments
            return Call(op->dtype, op->op, new_args, op->annotations, op->span);
          }
        }
      }
    }

    // For other Call nodes, use default processing
    return StmtExprMutator::VisitExpr_(op);
  }

  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      &multi_buffer_;
  PrimExpr iteration_;
};

// Rewrites accesses to a multi-buffered buffer that live OUTSIDE the annotated
// loop (e.g. a prologue T.fill before the pipelined loop). Unlike the in-loop
// MultiBufferAccessRewriter — which targets a single version per iteration
// (start = indexmod(iteration, N), version-dim extent = 1) — these outside
// accesses have no loop variable in scope, so they are broadcast to version 0
// with the version dimension fully covered: index 0, region version-dim extent
// = num_versions. This makes a prologue init touch all N versions at once.
//
// Keyed on the ORIGINAL (un-expanded) buffer objects: in-loop accesses were
// already swapped to the expanded buffer by RewriteTaskNodeBuffers, so keying
// on the original buffer naturally hits only the outside-loop occurrences and
// never double-rewrites the in-loop ones.
class BroadcastMultiBufferRewriter : public StmtExprMutator {
public:
  explicit BroadcastMultiBufferRewriter(
      const std::vector<MultiVersionBufferInfo> &buffer_infos) {
    for (const auto &info : buffer_infos) {
      multi_buffer_[info.buffer] = info.new_buffer;
    }
  }

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = multi_buffer_.find(load->buffer);
    if (it != multi_buffer_.end()) {
      auto *n = load.CopyOnWrite();
      n->buffer = it->second;
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = multi_buffer_.find(store->buffer);
    if (it != multi_buffer_.end()) {
      auto *n = store.CopyOnWrite();
      n->buffer = it->second;
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        auto it = multi_buffer_.find(buffer_load->buffer);
        if (it != multi_buffer_.end()) {
          // Cover the full version dimension: extent = num_versions, not 1.
          auto num_versions = it->second->shape[0];
          Array<PrimExpr> new_args;
          // args[0]: BufferLoad rewritten by VisitExpr_ above (swap + index 0).
          new_args.push_back(VisitExpr(op->args[0]));
          // args[1]: access_type, unchanged.
          new_args.push_back(VisitExpr(op->args[1]));
          // version-dim extent.
          new_args.push_back(num_versions);
          for (size_t i = 2; i < op->args.size(); i++) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          return Call(op->dtype, op->op, new_args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      multi_buffer_;
};

// Recursive function to rewrite BufferLoad/BufferStore in TaskNode stmts
static void RewriteTaskNodeBuffers(
    IRStructure *node,
    const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        &multi_buffer,
    PrimExpr iteration) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // Apply MultiBufferAccessRewriter to all stmts in the task
    MultiBufferAccessRewriter rewriter(multi_buffer, iteration);
    for (auto &stmt : task->stmts) {
      stmt = rewriter(stmt);
    }
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      RewriteTaskNodeBuffers(child.get(), multi_buffer, iteration);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    RewriteTaskNodeBuffers(ctrl->child.get(), multi_buffer, iteration);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    RewriteTaskNodeBuffers(wrapper->child.get(), multi_buffer, iteration);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    RewriteTaskNodeBuffers(unit->child.get(), multi_buffer, iteration);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child)
      RewriteTaskNodeBuffers(if_node->then_child.get(), multi_buffer,
                             iteration);
    if (if_node->else_child)
      RewriteTaskNodeBuffers(if_node->else_child.get(), multi_buffer,
                             iteration);
  }
}

// Helper function to insert a statement into ScheduleUnit's stmts.
// If `guard` is defined, wraps the stmt in an IfThenElse before inserting.
static void
InsertStatementIntoScheduleUnit(ScheduleUnit *task, Stmt stmt,
                                bool at_beginning, int warpgroup_id,
                                const PrimExpr &guard = PrimExpr()) {
  if (guard.defined()) {
    stmt = IfThenElse(guard, stmt);
  }
  if (at_beginning) {
    task->before[warpgroup_id].insert(task->before[warpgroup_id].begin(), stmt);
  } else {
    task->after[warpgroup_id].push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void
AnalyzeAndInsertBarriers(IRStructure *node, EventIdMap &event_id_map,
                         FlagInfoList &flag_infos, Var sid,
                         LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node), event_id_map,
                                flag_infos, sid, loop_info, buffer_infos);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), event_id_map, flag_infos,
                             sid, loop_info, buffer_infos);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child) {
      AnalyzeAndInsertBarriers(if_node->then_child.get(), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos);
    }
    if (if_node->else_child) {
      AnalyzeAndInsertBarriers(if_node->else_child.get(), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos);
    }
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static TaskNode *GetInnerTask(ScheduleUnit *unit) {
  std::vector<TaskNodeWithContext> task_contexts;
  CollectAllTaskNodesWithContext(unit, task_contexts);
  if (task_contexts.size() == 1 && task_contexts[0].control_node == nullptr) {
    return task_contexts[0].task;
  } else {
    return nullptr;
  }
}

// Check if a TaskNode's operation has unit_flag_ctl/ctrl != 0.
static bool HasUnitFlagCtrl(const TaskNode *task) {
  if (!task || task->stmts.size() != 1)
    return false;
  auto *eval = task->stmts[0].as<EvaluateNode>();
  if (!eval)
    return false;
  auto *call = eval->value.as<CallNode>();
  if (!call)
    return false;

  if (auto ann = call->annotations.Get("unit_flag_ctrl")) {
    if (auto imm = ann->as<IntImmNode>())
      return imm->value != 0;
    return true; // non-IntImm → dynamic → enabled
  }
  return false;
}

// --- Ascend set_flag / wait_flag helpers ---

static Stmt makeAscendSetFlag(const std::string &hard_event,
                              PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_set_flag(),
                       {StringImm(hard_event), event_id}));
}

static Stmt makeAscendWaitFlag(const std::string &hard_event,
                               PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_wait_flag(),
                       {StringImm(hard_event), event_id}));
}

static Stmt makeAscendPipeBarrier(const std::string &pipe) {
  return Evaluate(
      Call(DataType::Handle(), ascend_pipe_barrier(), {StringImm(pipe)}));
}

static Stmt makeAscendCrossCoreSetFlag(int mode_id, const std::string &pipe,
                                       PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_set_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

static Stmt makeAscendCrossCoreWaitFlag(int mode_id, const std::string &pipe,
                                        PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_wait_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

static std::string GetPipeName(uint16_t mask) {
  ICHECK((mask & (mask - 1)) == 0) << "Multiple pipes in one pipe: " << mask;
  if (mask & static_cast<uint16_t>(ResourcePipe::kMTE1))
    return "MTE1";
  if (mask & static_cast<uint16_t>(ResourcePipe::kMTE2))
    return "MTE2";
  if (mask & static_cast<uint16_t>(ResourcePipe::kMTE3))
    return "MTE3";
  if (mask & static_cast<uint16_t>(ResourcePipe::kCube))
    return "M";
  if (mask & static_cast<uint16_t>(ResourcePipe::kVector))
    return "V";
  if (mask & static_cast<uint16_t>(ResourcePipe::kFixpipe))
    return "FIX";
  if (mask & static_cast<uint16_t>(ResourcePipe::kScalar))
    return "S";
  return "UNKNOWN";
}

static std::string GetHardEvent(const TaskNode *producer,
                                const TaskNode *consumer) {
  auto producer_pipe = GetPipeName(producer->GetPipeMask());
  ICHECK(producer_pipe != "UNKNOWN") << producer->stmts[0];
  auto consumer_pipe = GetPipeName(consumer->GetPipeMask());
  ICHECK(consumer_pipe != "UNKNOWN") << consumer->stmts[0];
  auto event = producer_pipe + "_" + consumer_pipe;
  if (event == "MTE2_M")
    return "MTE2_MTE1";
  if (event == "M_MTE2")
    return "MTE1_MTE2";
  if (event == "MTE3_M")
    return "MTE3_MTE1";
  if (event == "M_MTE3")
    return "MTE1_MTE3";
  return event;
}

using InsertionSite = std::pair<ScheduleUnit *, PrimExpr>;

static std::vector<InsertionSite>
CollectInsertCandidates(const BufferAccessPoint &pt, bool at_beginning) {
  const auto &path = pt.path;
  std::vector<InsertionSite> candidates;
  PrimExpr accumulated_guard;

  auto append_guard = [&](PrimExpr term) {
    if (!accumulated_guard.defined()) {
      accumulated_guard = term;
    } else {
      accumulated_guard = accumulated_guard && term;
    }
  };

  for (auto node : path) {
    if (node->IsScheduleUnit()) {
      candidates.emplace_back(
          const_cast<ScheduleUnit *>(static_cast<const ScheduleUnit *>(node)),
          accumulated_guard);
    } else if (node->IsIf()) {
      return candidates;
    } else if (node->IsControl()) {
      const ControlNode *ctrl = static_cast<const ControlNode *>(node);
      const ForNode *for_node = ctrl->control.get();
      if (ctrl->child->ContainsLoopBreak()) {
        return candidates;
      }
      PrimExpr loop_var = for_node->loop_var;
      PrimExpr min = for_node->min;
      PrimExpr extent = for_node->extent;
      PrimExpr step = for_node->step.has_value() ? for_node->step.value()
                                                 : IntImm(DataType::Int(32), 1);
      PrimExpr term;
      if (at_beginning) {
        term = (loop_var == min);
      } else {
        term = (((loop_var - min) / step + 1) * step >= extent);
      }
      append_guard(term);
    }
  }
  return candidates;
}

static InsertionSite
CommonInsertionSite(const std::vector<std::vector<InsertionSite>> &chains) {
  if (chains.empty()) {
    return {nullptr, PrimExpr()};
  }
  size_t max_common = chains[0].size();
  for (const auto &c : chains) {
    max_common = std::min(max_common, c.size());
  }
  size_t common = 0;
  for (size_t i = 0; i < max_common; ++i) {
    ScheduleUnit *u = chains[0][i].first;
    bool all_match = true;
    for (const auto &c : chains) {
      if (c[i].first != u) {
        all_match = false;
        break;
      }
    }
    if (!all_match)
      break;
    common = i + 1;
  }
  ICHECK_GT(common, 0u);
  return chains[0][common - 1];
}

static std::vector<SyncPoint>
CollectSyncPoints(const std::vector<ScheduleUnit *> &units, int num_wgs,
                  std::optional<For> current_loop,
                  const std::map<Buffer, int64_t> &buffer_num_versions,
                  bool is_loop) {
  auto deps = AnalyzeDependencies(units, is_loop ? current_loop : std::nullopt);

  std::map<std::pair<ScheduleUnit *, int>,
           std::map<std::pair<ScheduleUnit *, int>, std::vector<DepInfo>>>
      sync_infos;

  // Expand a task's warpgroup id into the concrete wg ids it participates in.
  auto expand_wgs = [&](int task_wg) {
    std::vector<int> out;
    if (IsWarpgroupBroadcast(task_wg)) {
      for (int w = 0; w < num_wgs; ++w) {
        out.push_back(w);
      }
    } else if (task_wg >= 0 && task_wg < num_wgs) {
      out.push_back(task_wg);
    }
    return out;
  };

  for (const auto &dep : deps) {
    if (IsLocalBuffer(dep.buffer, true))
      continue;

    // Bucket producer / consumer access points by warpgroup.
    std::map<int, std::vector<BufferAccessPoint>> prod_by_wg, cons_by_wg;
    for (const auto &pt : dep.producer_pts) {
      for (int wg : expand_wgs(pt.task()->GetWarpgroupId()))
        prod_by_wg[wg].push_back(pt);
    }
    for (const auto &pt : dep.consumer_pts) {
      for (int wg : expand_wgs(pt.task()->GetWarpgroupId()))
        cons_by_wg[wg].push_back(pt);
    }

    for (auto &[pwg, ppts] : prod_by_wg) {
      for (auto &[cwg, cpts] : cons_by_wg) {
        DepInfo d;
        d.prod_unit = dep.prod_unit;
        d.cons_unit = dep.cons_unit;
        d.buffer = dep.buffer;
        d.producer_pts = ppts;
        d.consumer_pts = cpts;
        d.cross_iter = dep.cross_iter;
        d.latency = dep.latency;
        sync_infos[{dep.prod_unit, pwg}][{dep.cons_unit, cwg}].push_back(
            std::move(d));
      }
    }
  }

  std::vector<SyncPoint> sync_points;

  auto get_versions = [&](const Buffer &b) -> int {
    auto it = buffer_num_versions.find(b);
    return it != buffer_num_versions.end() ? static_cast<int>(it->second) : 1;
  };

  auto get_distance = [&](const DepInfo &d) -> int {
    return d.cross_iter ? get_versions(d.buffer) : 0;
  };

  for (auto prod_unit : units) {
    for (int prod_wg = 0; prod_wg < 2; ++prod_wg) {
      auto sync_it = sync_infos.find({prod_unit, prod_wg});
      if (sync_it == sync_infos.end())
        continue;
      const auto &wait_map = sync_it->second;
      for (const auto &[cons_unit_info, dep_list] : wait_map) {
        auto [cons_unit, cons_wg] = cons_unit_info;
        if (dep_list.empty())
          continue;
        if (prod_unit == cons_unit)
          continue;

        // Keep only the dependencies at the minimum distance (same-iter wins
        // over the loop back-edge when both are present).
        int distance = std::numeric_limits<int>::max();
        for (const auto &d : dep_list) {
          distance = std::min(distance, get_distance(d));
        }
        std::vector<DepInfo> kept_deps;
        for (const auto &d : dep_list) {
          if (get_distance(d) == distance)
            kept_deps.push_back(d);
        }

        int num_versions = 1;
        std::set<std::string> all_event;
        std::vector<std::vector<InsertionSite>> prod_chains;
        std::vector<std::vector<InsertionSite>> cons_chains;
        for (const auto &d : kept_deps) {
          num_versions = std::max(num_versions, get_versions(d.buffer));
          for (const auto &ppt : d.producer_pts) {
            for (const auto &cpt : d.consumer_pts) {
              all_event.insert(GetHardEvent(ppt.task(), cpt.task()));
            }
          }
          for (const auto &ppt : d.producer_pts) {
            prod_chains.push_back(CollectInsertCandidates(ppt, false));
          }
          for (const auto &cpt : d.consumer_pts) {
            cons_chains.push_back(CollectInsertCandidates(cpt, true));
          }
        }
        InsertionSite prod_site = CommonInsertionSite(prod_chains);
        InsertionSite cons_site = CommonInsertionSite(cons_chains);
        ICHECK(prod_site.first != nullptr);
        ICHECK(cons_site.first != nullptr);

        if (all_event.size() > 1) {
          ICHECK(prod_wg == cons_wg);
          sync_points.push_back({SyncType::PipeBarrier, prod_unit, prod_wg,
                                 prod_unit, PrimExpr(), cons_unit, cons_wg,
                                 cons_unit, PrimExpr(), distance, num_versions,
                                 kept_deps});
          continue;
        }

        std::string hard_event = *all_event.begin();
        size_t pos = hard_event.find('_');
        std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
        std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);

        if (prod_wg != cons_wg) {
          int mode_id = 4;

          sync_points.push_back({SyncType::CrossCore, prod_unit, prod_wg,
                                 prod_site.first, prod_site.second, cons_unit,
                                 cons_wg, cons_site.first, cons_site.second,
                                 distance, num_versions, kept_deps, hard_event,
                                 prod_pipe, cons_pipe, mode_id});
          continue;
        }

        // Intra-core sync: use ascend_set_flag / wait_flag. prod_pipe/cons_pipe
        // are filled (same PIPE_ encoding as cross-core) so grouping and the
        // transitive reduction can treat both sync types uniformly.
        sync_points.push_back({SyncType::IntraCore, prod_unit, prod_wg,
                               prod_site.first, prod_site.second, cons_unit,
                               cons_wg, cons_site.first, cons_site.second,
                               distance, num_versions, kept_deps, hard_event,
                               prod_pipe, cons_pipe});
      }
    }
  }
  return sync_points;
}

using UnitChain =
    std::vector<std::pair<const SequenceNode *, const ScheduleUnit *>>;

static UnitChain
BuildChainFromPath(const std::vector<const IRStructure *> &path,
                   const ScheduleUnit *target_unit) {
  UnitChain out;
  const SequenceNode *cur_parent = nullptr;
  for (const IRStructure *node : path) {
    if (!node)
      continue;
    if (node->IsSequence()) {
      cur_parent = static_cast<const SequenceNode *>(node);
    } else if (node->IsScheduleUnit()) {
      const ScheduleUnit *u = static_cast<const ScheduleUnit *>(node);
      out.emplace_back(cur_parent, u);
      if (u == target_unit)
        break;
    }
  }
  return out;
}

static int CompareChainPos(const UnitChain &a, const UnitChain &b,
                           bool at_beginning) {
  auto get_idx =
      [](std::pair<const SequenceNode *, const ScheduleUnit *> p) -> size_t {
    const SequenceNode *parent = p.first;
    const ScheduleUnit *child = p.second;
    for (size_t i = 0; i < parent->children.size(); ++i) {
      if (parent->children[i].get() == child)
        return i;
    }
    LOG(FATAL) << "Child not found in parent";
    return 0;
  };
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    if (a[i].first != b[i].first) {
      return 0;
    }
    if (a[i].second != b[i].second) {
      int64_t ta = a[i].second->GetStartTime();
      int64_t tb = b[i].second->GetStartTime();
      if (ta != tb)
        return ta < tb ? -1 : +1;
      return get_idx(a[i]) < get_idx(b[i]) ? -1 : +1;
    }
  }
  if (a.size() == b.size())
    return 0;
  if (at_beginning) {
    return a.size() < b.size() ? -1 : +1;
  } else {
    return a.size() < b.size() ? +1 : -1;
  }
}

static UnitChain ChainForSync(const SyncPoint &sp, const ScheduleUnit *unit,
                              bool is_producer) {
  if (sp.deps.empty() || unit == nullptr)
    return {};
  const DepInfo &rep = sp.deps.front();
  const auto &pts = is_producer ? rep.producer_pts : rep.consumer_pts;
  if (pts.empty())
    return {};
  return BuildChainFromPath(pts.front().path, unit);
}

static bool SyncDominates(const SyncPoint &x, const SyncPoint &y) {
  if (x.type != y.type)
    return false;
  if (x.type != SyncType::CrossCore && x.type != SyncType::IntraCore)
    return false;
  if (x.prod_pipe != y.prod_pipe || x.cons_pipe != y.cons_pipe ||
      x.prod_wg != y.prod_wg || x.cons_wg != y.cons_wg)
    return false;
  UnitChain xp = ChainForSync(x, x.prod_insertion_unit, true);
  UnitChain xc = ChainForSync(x, x.cons_insertion_unit, false);
  UnitChain yp = ChainForSync(y, y.prod_insertion_unit, true);
  UnitChain yc = ChainForSync(y, y.cons_insertion_unit, false);
  if (xp.empty() || xc.empty() || yp.empty() || yc.empty())
    return false;
  int D = y.distance - x.distance;
  if (D < 0)
    return false;
  if (D >= 2)
    return true;
  bool c0 = CompareChainPos(xp, yp, false) >= 0;
  bool cD = CompareChainPos(xc, yc, true) <= 0;
  if (D == 1)
    return c0 || cD;
  return c0 && cD;
}

static std::vector<SyncPoint>
FilterUnitFlagFixSyncs(const std::vector<SyncPoint> &sync_points) {
  std::vector<SyncPoint> result;
  result.reserve(sync_points.size());
  for (const auto &sp : sync_points) {
    if (sp.type == SyncType::IntraCore &&
        sp.hard_event.find("FIX") != std::string::npos) {
      auto *prod_task = GetInnerTask(sp.prod_unit);
      auto *cons_task = GetInnerTask(sp.cons_unit);
      if ((prod_task && HasUnitFlagCtrl(prod_task)) ||
          (cons_task && HasUnitFlagCtrl(cons_task)))
        continue;
    }
    result.push_back(sp);
  }
  return result;
}

static std::vector<SyncPoint>
OptimizeSyncPoints(const std::vector<SyncPoint> &raw_sync_points) {
  std::vector<SyncPoint> sync_points = FilterUnitFlagFixSyncs(raw_sync_points);

  std::vector<bool> keep(sync_points.size(), true);

  // Pairwise domination within a group
  for (size_t i = 0; i < sync_points.size(); ++i) {
    if (!keep[i])
      continue;
    for (size_t j = i + 1; j < sync_points.size(); ++j) {
      if (!keep[j])
        continue;
      if (SyncDominates(sync_points[i], sync_points[j])) {
        keep[j] = false;
      } else if (SyncDominates(sync_points[j], sync_points[i])) {
        keep[i] = false;
        break;
      }
    }
  }

  // Cross-pipe transitive reduction
  for (size_t i = 0; i < sync_points.size(); ++i) {
    const auto &ab = sync_points[i];
    if (ab.type != SyncType::IntraCore && ab.type != SyncType::CrossCore)
      continue;
    for (size_t j = 0; j < sync_points.size(); ++j) {
      const auto &bc = sync_points[j];
      if (bc.type != SyncType::IntraCore && bc.type != SyncType::CrossCore)
        continue;
      // ab and bc must compose at the intermediate unit: same unit, same
      // in-order pipe + warpgroup.
      if (ab.cons_unit != bc.prod_unit || ab.cons_pipe != bc.prod_pipe ||
          ab.cons_wg != bc.prod_wg)
        continue;
      // Synthetic edge: producer side from ab, consumer side from bc, distances
      // summed.
      SyncPoint chain = ab;
      chain.cons_unit = bc.cons_unit;
      chain.cons_insertion_unit = bc.cons_insertion_unit;
      chain.cons_guard = bc.cons_guard;
      chain.cons_wg = bc.cons_wg;
      chain.cons_pipe = bc.cons_pipe;
      chain.distance = ab.distance + bc.distance;
      chain.type = (chain.prod_wg != chain.cons_wg) ? SyncType::CrossCore
                                                    : SyncType::IntraCore;
      if (!chain.deps.empty() && !bc.deps.empty())
        chain.deps.front().consumer_pts = bc.deps.front().consumer_pts;
      for (size_t k = 0; k < sync_points.size(); ++k) {
        if (!keep[k] || k == i || k == j)
          continue;
        if (SyncDominates(chain, sync_points[k]))
          keep[k] = false;
      }
    }
  }

  std::vector<SyncPoint> result;
  for (size_t i = 0; i < sync_points.size(); ++i) {
    if (keep[i])
      result.push_back(sync_points[i]);
  }
  return result;
}

static void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                             FlagInfoList &flag_infos, EventIdMap &event_id_map,
                             LoopNestingInfo &loop_info) {
  for (const auto &sp : sync_points) {
    if (sp.type == SyncType::PipeBarrier) {
      Stmt barrier = makeAscendPipeBarrier("PIPE_ALL");
      InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, barrier, true,
                                      sp.cons_wg, sp.cons_guard);
    } else if (sp.type == SyncType::CrossCore) {
      auto cc_id = event_id_map.find("Cube-Vector");
      if (cc_id == event_id_map.end()) {
        event_id_map["Cube-Vector"] = 0;
        cc_id = event_id_map.find("Cube-Vector");
      }
      int cc_base_id = cc_id->second;
      cc_id->second += sp.num_versions;

      if (sp.distance > 0) {
        if (!sp.prod_wg) {
          for (int base = 0; base < sp.num_versions; ++base) {
            flag_infos.push_back(
                FlagInfo{0, cc_base_id + base, sp.prod_pipe, 1});
          }
          for (unsigned s = 0; s != 2; ++s) {
            for (int base = 0; base < sp.num_versions; ++base) {
              flag_infos.push_back(FlagInfo{
                  1, Integer(cc_base_id + s * 16 + base), sp.prod_pipe, 2});
            }
          }
        } else {
          for (unsigned s = 0; s != 2; ++s) {
            for (int base = 0; base < sp.num_versions; ++base) {
              flag_infos.push_back(FlagInfo{
                  1, Integer(cc_base_id + s * 16 + base), sp.prod_pipe, 1});
            }
          }
          for (int base = 0; base < sp.num_versions; ++base) {
            flag_infos.push_back(
                FlagInfo{0, cc_base_id + base, sp.prod_pipe, 2});
          }
        }
      }

      PrimExpr iteration =
          loop_info.CalculateIterationCount() - Integer(sp.distance);
      PrimExpr cc_event_id =
          indexmod(iteration, Integer(sp.num_versions)) + cc_base_id;

      if (!sp.prod_wg) {
        Stmt set_stmt =
            makeAscendCrossCoreSetFlag(sp.mode_id, sp.prod_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.prod_insertion_unit, set_stmt, false,
                                        sp.prod_wg, sp.prod_guard);
        for (unsigned s = 0; s != 2; ++s) {
          Stmt wait_stmt = makeAscendCrossCoreWaitFlag(
              sp.mode_id, sp.cons_pipe,
              indexmod(iteration, Integer(sp.num_versions)) +
                  Integer(cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, wait_stmt,
                                          true, sp.cons_wg, sp.cons_guard);
        }
      } else {
        for (unsigned s = 0; s != 2; ++s) {
          Stmt set_stmt = makeAscendCrossCoreSetFlag(
              sp.mode_id, sp.prod_pipe,
              indexmod(iteration, Integer(sp.num_versions)) +
                  Integer(cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.prod_insertion_unit, set_stmt,
                                          false, sp.prod_wg, sp.prod_guard);
        }
        Stmt wait_stmt =
            makeAscendCrossCoreWaitFlag(sp.mode_id, sp.cons_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, wait_stmt, true,
                                        sp.cons_wg, sp.cons_guard);
      }
    } else if (sp.type == SyncType::IntraCore) {
      // When producer and consumer share the same pipe, insert a pipe_barrier
      // before the consumer to ensure pipeline ordering.
      if (sp.prod_pipe == sp.cons_pipe) {
        Stmt barrier = makeAscendPipeBarrier(sp.prod_pipe);
        InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, barrier, true,
                                        sp.cons_wg, sp.cons_guard);
        continue;
      }
      auto base_id = event_id_map.find(sp.hard_event);
      if (base_id == event_id_map.end()) {
        event_id_map[sp.hard_event] = 0;
        base_id = event_id_map.find(sp.hard_event);
      }
      int base_id_val = base_id->second;
      base_id->second += sp.num_versions;

      PrimExpr iteration =
          loop_info.CalculateIterationCount() - Integer(sp.distance);
      PrimExpr event_id =
          indexmod(iteration, Integer(sp.num_versions)) + base_id_val;

      if (sp.distance > 0) {
        for (int base = 0; base < sp.num_versions; ++base) {
          flag_infos.push_back(FlagInfo{sp.prod_wg, Integer(base_id_val + base),
                                        sp.hard_event, 0});
        }
      }

      Stmt set_stmt = makeAscendSetFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.prod_insertion_unit, set_stmt, false,
                                      sp.prod_wg, sp.prod_guard);

      Stmt wait_stmt = makeAscendWaitFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, wait_stmt, true,
                                      sp.cons_wg, sp.cons_guard);
    }
  }
}

static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, EventIdMap &event_id_map,
                            FlagInfoList &flag_infos, Var sid,
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (!seq)
    return;

  // Collect all units from the sequence
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    if (GetInnerTask(unit) != nullptr) {
      // We will handle these units specially in
      // CollectSyncPoints/InsertSyncPoints, so we skip it here.
      continue;
    }
    if (unit->child->IsSequence() || unit->child->IsControl() ||
        unit->child->IsIf()) {
      // If child is SequenceNode, ControlNode, or IfNode, recursively analyze
      // it
      AnalyzeAndInsertBarriers(unit->child.get(), event_id_map, flag_infos, sid,
                               loop_info, buffer_infos);
    }
  }

  // Analyze dependencies with the shared AnalyzeDependencies, no enclosing
  // loop at the straight-line level (hence no multi-buffering).
  auto sync_points = CollectSyncPoints(units, 2, /*current_loop=*/std::nullopt);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, event_id_map, loop_info);
}

static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, EventIdMap &event_id_map,
                           FlagInfoList &flag_infos, Var sid,
                           LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (!ctrl || !ctrl->child)
    return;

  // Get loop information
  const ForNode *for_node = ctrl->control.get();
  if (!for_node)
    return;

  // Add this loop to nesting info
  loop_info.AddLoop(for_node);

  // Collect all units from the sequence
  ICHECK(ctrl->child->IsSequence());
  auto seq = static_cast<SequenceNode *>(ctrl->child.get());
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    if (GetInnerTask(unit) != nullptr) {
      // We will handle these units specially in
      // CollectSyncPoints/InsertSyncPoints, so we skip it here.
      continue;
    }
    if (unit->child->IsSequence() || unit->child->IsControl() ||
        unit->child->IsIf()) {
      // If child is SequenceNode, ControlNode, or IfNode, recursively analyze
      // it
      AnalyzeAndInsertBarriers(unit->child.get(), event_id_map, flag_infos, sid,
                               loop_info, buffer_infos);
    }
  }

  // Sort units by stage (largest first) to match SW-pipelining order — the
  // latest stage produces a value first across iterations.
  std::stable_sort(
      units.begin(), units.end(),
      [](ScheduleUnit *a, ScheduleUnit *b) { return a->stage > b->stage; });

  // Create new buffers for multi-buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      multi_buffer;
  for (auto &region : ctrl->GetWriteRegions()) {
    auto &buffer = region.get()->buffer;
    if (multi_buffer.find(buffer) != multi_buffer.end())
      continue;
    auto it = ctrl->buffer_num_versions.find(buffer);
    if (it == ctrl->buffer_num_versions.end())
      continue;
    int num_versions = it->second;
    if (num_versions == 1)
      continue;
    auto new_buffer = RewriteAllocBuffer(buffer, num_versions);
    multi_buffer[buffer] = new_buffer;
    buffer_infos.emplace_back(buffer, num_versions, new_buffer);
  }

  // Rewrite BufferLoad/BufferStore in TaskNode stmts for multi-version
  // buffers
  PrimExpr iteration = loop_info.CalculateIterationCount();

  // Recursively rewrite all TaskNode stmts
  RewriteTaskNodeBuffers(ctrl, multi_buffer, iteration);

  // Analyze dependencies with the shared AnalyzeDependencies, passing the
  // enclosing loop so cross-iteration (loop back-edge) deps are detected.
  auto sync_points = CollectSyncPoints(units, 2, ctrl->control,
                                       ctrl->buffer_num_versions, true);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, event_id_map, loop_info);

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}

} // namespace tl
} // namespace tvm
