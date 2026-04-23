#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../op/builtin.h"
#include "../../op/utils.h"
#include "./ir_structure.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
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

// Create a barrier buffer allocation and return the Buffer object.
// shape = (num_versions,) so that each pipeline stage has its own barrier slot,
// accessed via BufferLoad(buffer, {stage_index}).
static Buffer makeBarrierBuffer(const std::string &name, int num_versions,
                                std::vector<Buffer> &barrier_buffers,
                                Map<ObjectRef, ObjectRef> &barrier_map) {
  Array<PrimExpr> shape = {num_versions};
  DataType dtype = DataType::UInt(64);
  Type ptr_type = PointerType(PrimType(dtype), "shared.barrier");
  Var handle(name, ptr_type);
  Buffer buffer =
      Buffer(handle, dtype, shape, {}, PrimExpr(), name, 0, 0, kDefault);
  barrier_buffers.push_back(buffer);
  return buffer;
}

inline bool IsEvaluateZero(const tvm::tir::Stmt &stmt) {
  if (const EvaluateNode *eval_node = stmt.as<EvaluateNode>()) {
    if (is_const_int(eval_node->value, 0)) {
      return true;
    }
  }
  return false;
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
  int prod_wg_id;
  ScheduleUnit *cons_unit;
  int cons_wg_id;
  // CrossCore-specific
  std::string prod_pipe;
  std::string cons_pipe;
  int mode_id;
  int cc_base_id;
  // Shared
  int num_versions;
  int prod_idx;
  int cons_idx;
  int prod_wg;
  // IntraCore-specific
  std::string hard_event;
  int base_id;
  int distance;
  PrimExpr event_id;
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

static void AnalyzeAndInsertBarriers(
    IRStructure *node, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos);
static void AnalyzeSequenceNodeBarriers(
    SequenceNode *seq, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos);
static void AnalyzeControlNodeBarriers(
    ControlNode *ctrl, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos);
static std::vector<SyncPoint> CollectSyncPoints(
    const std::vector<ScheduleUnit *> &units,
    const std::map<
        std::pair<ScheduleUnit *, int>,
        std::pair<int, std::map<std::pair<ScheduleUnit *, int>, int>>>
        &sync_infos);
static void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                             FlagInfoList &flag_infos, Var sid,
                             EventIdMap &event_id_map,
                             LoopNestingInfo &loop_info);
static std::vector<SyncPoint>
OptimizeSyncPoints(const std::vector<SyncPoint> &sync_points);

static bool IsRegularSharedScope(const String &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

static bool IsTmemScope(const String &scope) { return scope == "shared.tmem"; }

static bool HasWriteReadDependencyByScope(
    const IRStructure *producer, const IRStructure *consumer,
    const std::function<bool(const String &)> &scope_matcher) {
  if (!producer || !consumer) {
    return false;
  }

  auto producer_writes = producer->GetWriteRegions();
  auto consumer_reads = consumer->GetReadRegions();

  for (const auto &write_region : producer_writes) {
    const Buffer &write_buffer = write_region->buffer;
    if (!scope_matcher(write_buffer.scope())) {
      continue;
    }
    for (const auto &read_region : consumer_reads) {
      if (write_buffer.same_as(read_region->buffer)) {
        return true;
      }
    }
  }
  return false;
}

static bool HasSharedWriteReadDependency(const IRStructure *producer,
                                         const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsRegularSharedScope(scope); });
}

static bool HasTmemWriteReadDependency(const IRStructure *producer,
                                       const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsTmemScope(scope); });
}

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

// Helper function to insert a statement into ScheduleUnit's stmts
static void InsertStatementIntoScheduleUnit(ScheduleUnit *task,
                                            const Stmt &stmt, bool at_beginning,
                                            int warpgroup_id) {
  if (at_beginning) {
    task->before[warpgroup_id].insert(task->before[warpgroup_id].begin(), stmt);
  } else {
    task->after[warpgroup_id].push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void AnalyzeAndInsertBarriers(
    IRStructure *node, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node), event_id_map,
                                flag_infos, sid, barrier_buffers, barrier_map,
                                loop_info, buffer_infos);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node), event_id_map,
                               flag_infos, sid, barrier_buffers, barrier_map,
                               loop_info, buffer_infos);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), event_id_map, flag_infos,
                             sid, barrier_buffers, barrier_map, loop_info,
                             buffer_infos);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child) {
      AnalyzeAndInsertBarriers(if_node->then_child.get(), event_id_map,
                               flag_infos, sid, barrier_buffers, barrier_map,
                               loop_info, buffer_infos);
    }
    if (if_node->else_child) {
      AnalyzeAndInsertBarriers(if_node->else_child.get(), event_id_map,
                               flag_infos, sid, barrier_buffers, barrier_map,
                               loop_info, buffer_infos);
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

static auto
GetSyncInfos(const std::vector<ScheduleUnit *> &units, int num_wgs,
             const std::unordered_map<Buffer, int, ObjectPtrHash,
                                      ObjectPtrEqual> &buffer_num_versions = {},
             bool is_loop = false) {
  std::set<Buffer> buffers;
  for (auto *unit : units) {
    for (const auto &buffer_access : unit->GetBufferAccessInfo(num_wgs)) {
      if (IsLocalBuffer(buffer_access.buffer, true))
        continue;
      buffers.insert(buffer_access.buffer);
    }
  }
  std::map<std::pair<ScheduleUnit *, int>,
           std::pair<int, std::map<std::pair<ScheduleUnit *, int>, int>>>
      sync_infos;
  for (const auto &buffer : buffers) {
    int num_versions = 1;
    auto it = buffer_num_versions.find(buffer);
    if (it != buffer_num_versions.end()) {
      num_versions = it->second;
    }
    std::vector<ScheduleUnit *> last_read_unit(num_wgs, nullptr);
    ScheduleUnit *last_write_unit = nullptr;
    int last_write_wg_id = -1;
    std::vector<bool> waited_write_wgs(num_wgs, false);
    for (int iter = 0; iter < (is_loop ? 2 : 1); ++iter) {
      for (ScheduleUnit *unit : units) {
        for (const auto &buffer_access : unit->GetBufferAccessInfo(num_wgs)) {
          int wg_id = buffer_access.warpgroup_id;
          ICHECK(0 <= wg_id && wg_id < num_wgs);
          if (buffer_access.buffer != buffer)
            continue;
          auto add_sync = [&](ScheduleUnit *wait_unit, int wait_wg_id) {
            int distance = iter ? num_versions : 0;
            auto &[barrier_versions, wait_map] =
                sync_infos[{wait_unit, wait_wg_id}];
            barrier_versions = std::max(barrier_versions, num_versions);
            auto it = wait_map.find({unit, wg_id});
            if (it == wait_map.end()) {
              wait_map[{unit, wg_id}] = distance;
            } else {
              it->second = std::min(it->second, distance);
            }
          };
          if (!buffer_access.is_write) {
            if (last_write_unit == nullptr || last_write_unit == unit)
              continue;
            if (waited_write_wgs[wg_id])
              continue;
            add_sync(last_write_unit, last_write_wg_id);
          } else {
            for (int last_wg_id = 0; last_wg_id < num_wgs; ++last_wg_id) {
              if (last_read_unit[last_wg_id] == nullptr ||
                  last_read_unit[last_wg_id] == unit)
                continue;
              add_sync(last_read_unit[last_wg_id], last_wg_id);
            }
          }
        }
        if (iter == 0) {
          for (const auto &buffer_access : unit->GetBufferAccessInfo(num_wgs)) {
            int wg_id = buffer_access.warpgroup_id;
            if (buffer_access.buffer != buffer)
              continue;
            if (!buffer_access.is_write) {
              waited_write_wgs[wg_id] = true;
            } else {
              for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
                last_read_unit[wg_id] = nullptr;
              }
            }
          }
          for (const auto &buffer_access : unit->GetBufferAccessInfo(num_wgs)) {
            int wg_id = buffer_access.warpgroup_id;
            if (buffer_access.buffer != buffer)
              continue;
            if (!buffer_access.is_write) {
              last_read_unit[wg_id] = unit;
            } else {
              last_write_unit = unit;
              last_write_wg_id = wg_id;
              for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
                waited_write_wgs[wg_id] = false;
              }
            }
          }
        }
      }
    }
  }
  return sync_infos;
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
  LOG(FATAL) << "Unknown pipe";
}

static std::string GetHardEvent(uint16_t producer_mask,
                                uint16_t consumer_mask) {
  auto event = GetPipeName(producer_mask) + "_" + GetPipeName(consumer_mask);
  if (event == "MTE2_M")
    return "MTE2_MTE1";
  if (event == "M_MTE2")
    return "MTE1_MTE2";
  return event;
}

static std::vector<SyncPoint> CollectSyncPoints(
    const std::vector<ScheduleUnit *> &units,
    const std::map<
        std::pair<ScheduleUnit *, int>,
        std::pair<int, std::map<std::pair<ScheduleUnit *, int>, int>>>
        &sync_infos) {
  // Build unit→index map for order comparison
  std::unordered_map<ScheduleUnit *, int> unit_index;
  for (size_t i = 0; i < units.size(); ++i)
    unit_index[units[i]] = static_cast<int>(i);

  std::vector<SyncPoint> sync_points;

  for (auto unit : units) {
    for (int wg_id = 0; wg_id < 2; ++wg_id) {
      auto sync_it = sync_infos.find({unit, wg_id});
      if (sync_it == sync_infos.end())
        continue;
      const auto &wait_map = sync_it->second.second;
      int num_versions = std::max(sync_it->second.first, 1);

      for (const auto &[waiting_unit_info, distance] : wait_map) {
        auto [waiting_unit, waiting_wg_id] = waiting_unit_info;

        int prod_wg =
            static_cast<TaskNode *>(unit->child.get())->GetWarpgroupId();
        int cons_wg = static_cast<TaskNode *>(waiting_unit->child.get())
                          ->GetWarpgroupId();

        uint16_t producer_mask = unit->GetPipeMask();
        uint16_t consumer_mask = waiting_unit->GetPipeMask();

        if (!unit->isInnerTask() || !waiting_unit->isInnerTask()) {
          sync_points.push_back({SyncType::PipeBarrier, unit, wg_id,
                                 waiting_unit, waiting_wg_id});
          continue;
        }

        std::string hard_event = GetHardEvent(producer_mask, consumer_mask);
        int prod_idx = unit_index[unit];
        int cons_idx = unit_index[waiting_unit];

        if (prod_wg != cons_wg) {
          // Cross-core sync: use ascend_cross_core_set_flag / wait_flag
          std::string prod_pipe = "PIPE_" + GetPipeName(producer_mask);
          std::string cons_pipe = "PIPE_" + GetPipeName(consumer_mask);
          int mode_id = 4;

          sync_points.push_back({SyncType::CrossCore, unit, wg_id, waiting_unit,
                                 waiting_wg_id, prod_pipe, cons_pipe, mode_id,
                                 0, num_versions, prod_idx, cons_idx, prod_wg});
          continue;
        }

        // Intra-core sync: use ascend_set_flag / wait_flag
        sync_points.push_back({SyncType::IntraCore,
                               unit,
                               wg_id,
                               waiting_unit,
                               waiting_wg_id,
                               {},
                               {},
                               0,
                               0,
                               num_versions,
                               prod_idx,
                               cons_idx,
                               prod_wg,
                               hard_event,
                               0,
                               distance,
                               PrimExpr()});
      }
    }
  }
  return sync_points;
}

static std::vector<SyncPoint>
OptimizeSyncPoints(const std::vector<SyncPoint> &sync_points) {
  std::vector<bool> keep(sync_points.size(), true);

  for (size_t i = 0; i < sync_points.size(); ++i) {
    if (!keep[i])
      continue;
    for (size_t j = i + 1; j < sync_points.size(); ++j) {
      if (!keep[j])
        continue;
      const auto &a = sync_points[i];
      const auto &b = sync_points[j];

      if (a.type != b.type)
        continue;

      bool same_group = false;
      if (a.type == SyncType::CrossCore) {
        same_group =
            (a.prod_pipe == b.prod_pipe && a.cons_pipe == b.cons_pipe &&
             a.num_versions == b.num_versions &&
             (a.cons_idx < a.prod_idx) == (b.cons_idx < b.prod_idx));
      } else if (a.type == SyncType::IntraCore) {
        same_group =
            (a.hard_event == b.hard_event && a.num_versions == b.num_versions &&
             (a.cons_idx < a.prod_idx) == (b.cons_idx < b.prod_idx));
      } else {
        continue;
      }

      if (!same_group)
        continue;

      if (a.prod_idx >= b.prod_idx && a.cons_idx <= b.cons_idx) {
        keep[j] = false;
      } else if (b.prod_idx >= a.prod_idx && b.cons_idx <= a.cons_idx) {
        keep[i] = false;
        break;
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
                             FlagInfoList &flag_infos, Var sid,
                             EventIdMap &event_id_map,
                             LoopNestingInfo &loop_info) {
  for (const auto &sp : sync_points) {
    if (sp.type == SyncType::PipeBarrier) {
      Stmt barrier = makeAscendPipeBarrier("PIPE_ALL");
      InsertStatementIntoScheduleUnit(sp.prod_unit, barrier, false,
                                      sp.prod_wg_id);
      InsertStatementIntoScheduleUnit(sp.cons_unit, barrier, true,
                                      sp.cons_wg_id);
    } else if (sp.type == SyncType::CrossCore) {
      auto cc_id = event_id_map.find("Cube-Vector");
      if (cc_id == event_id_map.end()) {
        event_id_map["Cube-Vector"] = 0;
        cc_id = event_id_map.find("Cube-Vector");
      }
      int cc_base_id = cc_id->second;
      cc_id->second += sp.num_versions;

      PrimExpr cc_event_id = cc_base_id + sid * 16;

      if (sp.cons_idx < sp.prod_idx) {
        if (!sp.prod_wg) {
          for (int base = 0; base < sp.num_versions; ++base) {
            flag_infos.push_back(
                FlagInfo{0, cc_event_id + base, sp.prod_pipe, 1});
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
                FlagInfo{0, cc_event_id + base, sp.prod_pipe, 2});
          }
        }
      }
      if (!sp.prod_wg) {
        Stmt set_stmt =
            makeAscendCrossCoreSetFlag(sp.mode_id, sp.prod_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.prod_unit, set_stmt, false,
                                        sp.prod_wg_id);
        for (unsigned s = 0; s != 2; ++s) {
          Stmt wait_stmt = makeAscendCrossCoreWaitFlag(
              sp.mode_id, sp.cons_pipe,
              IntImm(DataType::Int(64), cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.cons_unit, wait_stmt, true,
                                          sp.cons_wg_id);
        }
      } else {
        for (unsigned s = 0; s != 2; ++s) {
          Stmt set_stmt = makeAscendCrossCoreSetFlag(
              sp.mode_id, sp.prod_pipe,
              IntImm(DataType::Int(64), cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.prod_unit, set_stmt, false,
                                          sp.prod_wg_id);
        }
        Stmt wait_stmt =
            makeAscendCrossCoreWaitFlag(sp.mode_id, sp.cons_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.cons_unit, wait_stmt, true,
                                        sp.cons_wg_id);
      }
    } else if (sp.type == SyncType::IntraCore) {
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

      if (sp.cons_idx < sp.prod_idx) {
        for (int base = 0; base < sp.num_versions; ++base) {
          flag_infos.push_back(FlagInfo{sp.prod_wg, Integer(base_id_val + base),
                                        sp.hard_event, 0});
        }
      }

      Stmt set_stmt = makeAscendSetFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.prod_unit, set_stmt, false,
                                      sp.prod_wg_id);

      Stmt wait_stmt = makeAscendWaitFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.cons_unit, wait_stmt, true,
                                      sp.cons_wg_id);
    }
  }
}

static void AnalyzeSequenceNodeBarriers(
    SequenceNode *seq, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
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
                               barrier_buffers, barrier_map, loop_info,
                               buffer_infos);
    }
  }

  // Analyze dependencies and insert synchronization
  auto sync_infos = GetSyncInfos(units, 2);
  auto sync_points = CollectSyncPoints(units, sync_infos);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, sid, event_id_map, loop_info);
}

static void AnalyzeControlNodeBarriers(
    ControlNode *ctrl, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, LoopNestingInfo &loop_info,
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
                               barrier_buffers, barrier_map, loop_info,
                               buffer_infos);
    }
  }

  // Detect multi-version buffers and create new buffers for them
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      multi_buffer;
  std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>
      buffer_num_versions;
  int num_wgs = 2;
  for (const auto &unit : units) {
    for (const auto &buffer_access : unit->GetBufferAccessInfo(num_wgs)) {
      auto &buffer = buffer_access.buffer;
      if (!ctrl->multi_buffering_buffers.count(buffer))
        continue;
      for (const auto &other_unit : units) {
        if (unit == other_unit)
          continue;
        int distance = unit->child->GetStartTime() + unit->child->GetLatency() -
                       other_unit->child->GetStartTime();
        if (distance <= 0)
          continue;
        distance = (distance - 1) / ctrl->GetIIperIter() + 1;
        for (const auto &other_buffer_access :
             other_unit->GetBufferAccessInfo(num_wgs)) {
          auto &other_buffer = other_buffer_access.buffer;
          if (!buffer.same_as(other_buffer))
            continue;
          if (buffer_access.is_write || other_buffer_access.is_write) {
            auto &num_versions = buffer_num_versions[buffer];
            num_versions = std::max(num_versions, distance);
          }
        }
      }
    }
  }
  for (auto &region : ctrl->GetWriteRegions()) {
    auto &buffer = region.get()->buffer;
    if (!ctrl->multi_buffering_buffers.count(buffer))
      continue;
    if (multi_buffer.find(buffer) != multi_buffer.end())
      continue;
    auto it = buffer_num_versions.find(buffer);
    if (it == buffer_num_versions.end())
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

  // Analyze dependencies and insert synchronization
  auto sync_infos = GetSyncInfos(units, 2, buffer_num_versions, true);
  auto sync_points = CollectSyncPoints(units, sync_infos);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, sid, event_id_map, loop_info);

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}

} // namespace tl
} // namespace tvm
