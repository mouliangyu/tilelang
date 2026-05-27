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

// Structure to store synchronization dependency information
struct SyncInfo {
  int distance;
  Buffer buffer;
  BufferAccessPoint producer_pt;
  BufferAccessPoint consumer_pt;
  int num_versions;

  SyncInfo(int distance, Buffer buffer, BufferAccessPoint producer_pt,
           BufferAccessPoint consumer_pt, int num_versions)
      : distance(distance), buffer(buffer), producer_pt(std::move(producer_pt)),
        consumer_pt(std::move(consumer_pt)), num_versions(num_versions) {}

  bool operator<(const SyncInfo &other) const {
    if (distance != other.distance)
      return distance < other.distance;
    if (buffer != other.buffer)
      return buffer.get() < other.buffer.get();
    if (producer_pt != other.producer_pt)
      return producer_pt < other.producer_pt;
    if (consumer_pt != other.consumer_pt)
      return consumer_pt < other.consumer_pt;
    return num_versions < other.num_versions;
  }
};

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
  int prod_idx;
  ScheduleUnit *cons_unit;
  int cons_wg;
  ScheduleUnit *cons_insertion_unit;
  PrimExpr cons_guard;
  int cons_idx;
  int distance;
  int num_versions;
  std::set<SyncInfo> sync_info_set;
  std::string hard_event;
  // CrossCore-specific
  std::string prod_pipe;
  std::string cons_pipe;
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
static std::vector<SyncPoint> CollectSyncPoints(
    const std::vector<ScheduleUnit *> &units,
    const std::map<std::pair<ScheduleUnit *, int>,
                   std::map<std::pair<ScheduleUnit *, int>, std::set<SyncInfo>>>
        &sync_infos);
static void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                             FlagInfoList &flag_infos, Var sid,
                             EventIdMap &event_id_map,
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

// Enumerate (region_group_id, is_write, wg_id) accesses by every TaskNode
// under the given `unit`, derived from the partition's entry table.
// Broadcast tasks are expanded across [0, num_wgs).
static std::set<std::tuple<int, bool, int>>
GetUnitRegionGroupAccesses(IRStructure *unit, const RegionPartition &partition,
                           int num_wgs) {
  std::set<std::tuple<int, bool, int>> result;
  const auto &entries = partition.GetEntries();
  const auto &gids = partition.GetEntryGroupIds();
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i].unit != unit)
      continue;
    int rg = gids[i];
    int task_wg = entries[i].task->GetWarpgroupId();
    if (IsWarpgroupBroadcast(task_wg)) {
      for (int w = 0; w < num_wgs; ++w)
        result.emplace(rg, entries[i].is_write, w);
    } else {
      result.emplace(rg, entries[i].is_write, task_wg);
    }
  }
  return result;
}

static auto
GetSyncInfos(const std::vector<ScheduleUnit *> &units, int num_wgs,
             const RegionPartition &partition,
             const std::map<Buffer, int64_t> &buffer_num_versions = {},
             bool is_loop = false) {
  // Collect the region groups we need to track. Skip groups whose
  // representative buffer is register/local — sync between those is handled
  // by the warpgroup-broadcast mechanism, not by this pass.
  std::set<int> rg_set;
  for (auto *unit : units) {
    const auto &entries = partition.GetEntries();
    const auto &gids = partition.GetEntryGroupIds();
    for (size_t i = 0; i < entries.size(); ++i) {
      if (entries[i].unit != unit)
        continue;
      int rg = gids[i];
      if (IsLocalBuffer(partition.GetRepresentativeBuffer(rg), true))
        continue;
      rg_set.insert(rg);
    }
  }
  std::map<std::pair<ScheduleUnit *, int>,
           std::map<std::pair<ScheduleUnit *, int>, std::set<SyncInfo>>>
      sync_infos;
  for (int rg : rg_set) {
    Buffer buffer = partition.GetRepresentativeBuffer(rg);
    int num_versions = 1;
    auto it = buffer_num_versions.find(buffer);
    if (it != buffer_num_versions.end()) {
      num_versions = it->second;
    }
    std::vector<ScheduleUnit *> last_read_unit(num_wgs, nullptr);
    std::vector<std::set<BufferAccessPoint>> last_read_unit_tasks(num_wgs);
    ScheduleUnit *last_write_unit = nullptr;
    std::vector<std::pair<int, std::set<BufferAccessPoint>>>
        last_write_unit_wg_tasks;
    std::vector<bool> waited_write_wgs(num_wgs, false);
    for (int iter = 0; iter < (is_loop ? 2 : 1); ++iter) {
      for (ScheduleUnit *unit : units) {
        // Region-group accesses of this unit (rg, is_write, wg_id).
        auto unit_accesses =
            GetUnitRegionGroupAccesses(unit, partition, num_wgs);
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          int distance = iter ? num_versions : 0;
          // RAW: unit reads rg, wait for last writer
          auto first_reads = unit->GetFirstAccessPoints(
              rg, partition, /*is_write=*/false, wg_id);
          if (!first_reads.empty() && last_write_unit != nullptr &&
              !waited_write_wgs[wg_id]) {
            if (last_write_unit == unit)
              continue;
            for (const auto &consumer_pt : first_reads) {
              for (const auto &[last_write_wg_id, last_write_unit_tasks] :
                   last_write_unit_wg_tasks) {
                for (const auto &producer_pt : last_write_unit_tasks) {
                  sync_infos[{last_write_unit, last_write_wg_id}][{unit, wg_id}]
                      .emplace(distance, buffer, producer_pt, consumer_pt,
                               num_versions);
                }
              }
            }
          }
          // WAR/WAW: unit writes rg, wait for all last readers
          // if there are no last readers, wait for last writer
          auto first_writes = unit->GetFirstAccessPoints(
              rg, partition, /*is_write=*/true, wg_id);
          if (!first_writes.empty()) {
            bool has_last_read = false;
            for (int last_wg = 0; last_wg < num_wgs; ++last_wg) {
              if (last_read_unit[last_wg] == nullptr)
                continue;
              has_last_read = true;
              if (last_read_unit[last_wg] == unit)
                continue;
              for (const auto &consumer_pt : first_writes) {
                for (const auto &producer_pt : last_read_unit_tasks[last_wg]) {
                  sync_infos[{last_read_unit[last_wg], last_wg}][{unit, wg_id}]
                      .emplace(distance, buffer, producer_pt, consumer_pt,
                               num_versions);
                }
              }
            }
            if (!has_last_read && last_write_unit != nullptr &&
                !waited_write_wgs[wg_id] && last_write_unit != unit) {
              for (const auto &consumer_pt : first_writes) {
                for (const auto &[last_write_wg_id, last_write_unit_tasks] :
                     last_write_unit_wg_tasks) {
                  for (const auto &producer_pt : last_write_unit_tasks) {
                    sync_infos[{last_write_unit, last_write_wg_id}]
                              [{unit, wg_id}]
                                  .emplace(distance, buffer, producer_pt,
                                           consumer_pt, num_versions);
                  }
                }
              }
            }
          }
        }
        // Update state machine from this unit's accesses to `rg`.
        for (const auto &[a_rg, a_is_write, a_wg] : unit_accesses) {
          if (a_rg != rg)
            continue;
          if (!a_is_write) {
            waited_write_wgs[a_wg] = true;
            last_read_unit[a_wg] = nullptr;
          } else {
            for (int w = 0; w < num_wgs; ++w) {
              last_read_unit[w] = nullptr;
            }
            last_write_unit = nullptr;
          }
        }
        if (iter == 0) {
          for (int wg = 0; wg < num_wgs; ++wg) {
            auto last_reads = unit->GetLastAccessPoints(rg, partition,
                                                        /*is_write=*/false, wg);
            if (!last_reads.empty()) {
              last_read_unit[wg] = unit;
              last_read_unit_tasks[wg] = std::move(last_reads);
            }
          }
          {
            std::vector<int> write_wg_ids;
            for (const auto &[a_rg, a_is_write, a_wg] : unit_accesses) {
              if (a_rg == rg && a_is_write) {
                if (std::find(write_wg_ids.begin(), write_wg_ids.end(), a_wg) ==
                    write_wg_ids.end())
                  write_wg_ids.push_back(a_wg);
              }
            }
            if (!write_wg_ids.empty()) {
              last_write_unit = unit;
              last_write_unit_wg_tasks.clear();
              for (int wg : write_wg_ids) {
                auto last_writes = unit->GetLastAccessPoints(
                    rg, partition, /*is_write=*/true, wg);
                if (!last_writes.empty())
                  last_write_unit_wg_tasks.emplace_back(wg,
                                                        std::move(last_writes));
              }
              for (int wg = 0; wg < num_wgs; ++wg)
                waited_write_wgs[wg] = false;
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
  return "UNKNOWN";
}

static std::string GetHardEvent(uint16_t producer_mask,
                                uint16_t consumer_mask) {
  auto event = GetPipeName(producer_mask) + "_" + GetPipeName(consumer_mask);
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

static std::vector<SyncPoint> CollectSyncPoints(
    const std::vector<ScheduleUnit *> &units,
    const std::map<std::pair<ScheduleUnit *, int>,
                   std::map<std::pair<ScheduleUnit *, int>, std::set<SyncInfo>>>
        &sync_infos) {
  // Build unit→index map for order comparison
  std::unordered_map<ScheduleUnit *, int> unit_index;
  for (size_t i = 0; i < units.size(); ++i)
    unit_index[units[i]] = static_cast<int>(i);

  std::vector<SyncPoint> sync_points;

  for (auto prod_unit : units) {
    for (int prod_wg = 0; prod_wg < 2; ++prod_wg) {
      auto sync_it = sync_infos.find({prod_unit, prod_wg});
      if (sync_it == sync_infos.end())
        continue;
      const auto &wait_map = sync_it->second;
      for (const auto &[cons_unit_info, sync_info_set] : wait_map) {
        auto [cons_unit, cons_wg] = cons_unit_info;

        if (sync_info_set.empty())
          continue;

        int min_distance = std::numeric_limits<int>::max();
        for (const auto &si : sync_info_set) {
          min_distance = std::min(min_distance, si.distance);
        }
        std::set<SyncInfo> kept;
        for (const auto &si : sync_info_set) {
          if (si.distance == min_distance) {
            kept.insert(si);
          }
        }

        int num_versions = 1;
        std::set<std::string> all_event;
        std::vector<std::vector<InsertionSite>> prod_chains;
        std::vector<std::vector<InsertionSite>> cons_chains;
        for (const auto &si : kept) {
          const TaskNode *producer = si.producer_pt.task();
          const TaskNode *consumer = si.consumer_pt.task();
          num_versions = std::max(num_versions, si.num_versions);
          all_event.insert(
              GetHardEvent(producer->GetPipeMask(), consumer->GetPipeMask()));
          prod_chains.push_back(CollectInsertCandidates(si.producer_pt, false));
          cons_chains.push_back(CollectInsertCandidates(si.consumer_pt, true));
        }

        InsertionSite prod_site = CommonInsertionSite(prod_chains);
        InsertionSite cons_site = CommonInsertionSite(cons_chains);
        ICHECK(prod_site.first != nullptr);
        ICHECK(cons_site.first != nullptr);

        int prod_idx = unit_index[prod_unit];
        int cons_idx = unit_index[cons_unit];
        int distance = min_distance;

        if (all_event.size() > 1) {
          ICHECK(prod_wg == cons_wg);
          sync_points.push_back({SyncType::PipeBarrier, prod_unit, prod_wg,
                                 prod_unit, PrimExpr(), prod_idx, cons_unit,
                                 cons_wg, cons_unit, PrimExpr(), cons_idx,
                                 distance, num_versions, kept});
          continue;
        }

        std::string hard_event = *all_event.begin();

        if (prod_wg != cons_wg) {
          size_t pos = hard_event.find('_');
          std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
          std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
          int mode_id = 4;

          sync_points.push_back(
              {SyncType::CrossCore, prod_unit, prod_wg, prod_site.first,
               prod_site.second, prod_idx, cons_unit, cons_wg, cons_site.first,
               cons_site.second, cons_idx, distance, num_versions, kept,
               hard_event, prod_pipe, cons_pipe, mode_id});
          continue;
        }

        // Intra-core sync: use ascend_set_flag / wait_flag
        sync_points.push_back({SyncType::IntraCore, prod_unit, prod_wg,
                               prod_site.first, prod_site.second, prod_idx,
                               cons_unit, cons_wg, cons_site.first,
                               cons_site.second, cons_idx, distance,
                               num_versions, kept, hard_event});
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

static std::vector<SyncPoint>
OptimizeSyncPoints(const std::vector<SyncPoint> &sync_points) {
  // Derive insertion chains from the SyncPoint's own sync_info_set — any
  // representative SyncInfo works because the insertion_unit is on the LCA
  // common prefix of every SyncInfo's path in the group.
  auto chain_for = [](const SyncPoint &sp, const ScheduleUnit *unit,
                      bool is_producer) -> UnitChain {
    if (sp.sync_info_set.empty() || unit == nullptr)
      return {};
    const SyncInfo &rep = *sp.sync_info_set.begin();
    const auto &path =
        is_producer ? rep.producer_pt.path : rep.consumer_pt.path;
    return BuildChainFromPath(path, unit);
  };

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

      UnitChain a_prod = chain_for(a, a.prod_insertion_unit, true);
      UnitChain a_cons = chain_for(a, a.cons_insertion_unit, false);
      UnitChain b_prod = chain_for(b, b.prod_insertion_unit, true);
      UnitChain b_cons = chain_for(b, b.cons_insertion_unit, false);
      if (a_prod.empty() || a_cons.empty() || b_prod.empty() || b_cons.empty())
        continue;

      auto dominates = [&](const SyncPoint &x, const SyncPoint &y,
                           const UnitChain &xp, const UnitChain &yp,
                           const UnitChain &xc, const UnitChain &yc) {
        if (x.distance < y.distance)
          return false;
        if (CompareChainPos(xp, yp, false) < 0)
          return false;
        if (CompareChainPos(xc, yc, true) > 0)
          return false;
        return true;
      };

      if (dominates(a, b, a_prod, b_prod, a_cons, b_cons)) {
        keep[j] = false;
      } else if (dominates(b, a, b_prod, a_prod, b_cons, a_cons)) {
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

      if (sp.cons_idx < sp.prod_idx) {
        if (!sp.prod_wg) {
          for (int base = 0; base < sp.num_versions; ++base) {
            flag_infos.push_back(
                FlagInfo{0, cc_base_id + sid * 16 + base, sp.prod_pipe, 1});
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
                FlagInfo{0, cc_base_id + sid * 16 + base, sp.prod_pipe, 2});
          }
        }
      }

      PrimExpr iteration =
          loop_info.CalculateIterationCount() - Integer(sp.distance);
      PrimExpr cc_event_id =
          indexmod(iteration, Integer(sp.num_versions)) + cc_base_id + sid * 16;

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
      size_t pos = sp.hard_event.find('_');
      if (pos != std::string::npos) {
        std::string prod_pipe = sp.hard_event.substr(0, pos);
        std::string cons_pipe = sp.hard_event.substr(pos + 1);
        if (prod_pipe == cons_pipe) {
          Stmt barrier = makeAscendPipeBarrier("PIPE_" + prod_pipe);
          InsertStatementIntoScheduleUnit(sp.cons_insertion_unit, barrier, true,
                                          sp.cons_wg, sp.cons_guard);
          continue;
        }
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

      if (sp.cons_idx < sp.prod_idx) {
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

  // Reuse the RegionPartition built during scheduling. Z3SchedulePython is
  // guaranteed to have stashed it on the SequenceNode.
  ICHECK(seq->region_partition)
      << "SequenceNode missing region_partition cached by schedule_builder";
  const RegionPartition &partition = *seq->region_partition;

  // Analyze dependencies and insert synchronization
  auto sync_infos = GetSyncInfos(units, 2, partition);
  auto sync_points = CollectSyncPoints(units, sync_infos);
  // sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, sid, event_id_map, loop_info);
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

  // Reuse the RegionPartition built during scheduling. Z3SchedulePythonLoop
  // is guaranteed to have stashed it on the body SequenceNode.
  ICHECK(seq->region_partition)
      << "SequenceNode missing region_partition cached by schedule_builder";
  const RegionPartition &partition = *seq->region_partition;

  // Analyze dependencies and insert synchronization
  auto sync_infos =
      GetSyncInfos(units, 2, partition, ctrl->buffer_num_versions, true);
  auto sync_points = CollectSyncPoints(units, sync_infos);
  // sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, sid, event_id_map, loop_info);

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}

} // namespace tl
} // namespace tvm
