#include "./barrier.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

#include "op/utils.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

Buffer CreateNewBuffer(const Buffer &buffer, int num_versions) {
  ObjectPtr<BufferNode> new_buffer =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }
  return Buffer(new_buffer);
}

void InsertStatementIntoScheduleUnit(ScheduleUnit *task, Stmt stmt,
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

TaskNode *GetInnerTask(ScheduleUnit *unit) {
  std::vector<TaskNodeWithContext> task_contexts;
  CollectAllTaskNodesWithContext(unit, task_contexts);
  if (task_contexts.size() == 1 && task_contexts[0].control_node == nullptr) {
    return task_contexts[0].task;
  } else {
    return nullptr;
  }
}

// Whether a same-unit (prod_unit == cons_unit) self-dependency can be skipped.
// A self-dependency can be skipped only in two provably-safe shapes:
//   (a) the unit uses ONLY the Cube (M) pipe, including the L1-gemm.
//   (b) the unit's child is directly a For loop without a loop break.
// Any other shape may carry a real dependency.
bool CanSkipSelfDependency(ScheduleUnit *unit) {
  if (unit->UsesOnlyPipe(ResourcePipe::kCube))
    return true;
  if (unit->child && unit->child->IsControl() &&
      !unit->child->ContainsLoopBreak())
    return true;
  return false;
}

// Check if a TaskNode's operation has unit_flag_ctl/ctrl != 0.
bool HasUnitFlagCtrl(const TaskNode *task) {
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

std::string GetPipeName(uint16_t mask) {
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

std::string GetHardEvent(const TaskNode *producer, const TaskNode *consumer) {
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

enum class SyncType { CrossCore, IntraCore };

struct SyncInsertionSite {
  ScheduleUnit *unit;
  PrimExpr guard;
  bool at_beginning;
  std::vector<std::pair<const SequenceNode *, const ScheduleUnit *>> chain;
};

struct SyncPoint {
  SyncType type;
  ScheduleUnit *prod_unit;
  int prod_wg;
  SyncInsertionSite prod_site;
  std::string prod_pipe;
  ScheduleUnit *cons_unit;
  int cons_wg;
  SyncInsertionSite cons_site;
  std::string cons_pipe;
  int distance;
  int num_versions;
  PrimExpr flag_iteration;
  std::string hard_event;
  // CrossCore-specific
  int mode_id;
};

SyncInsertionSite GetInsertionSite(const std::set<BufferAccessPoint> &pts,
                                   bool at_beginning) {
  SyncInsertionSite site = {nullptr, PrimExpr(), at_beginning, {}};
  if (pts.empty()) {
    LOG(FATAL) << "No access points provided for insertion site.";
    return site;
  }
  PrimExpr accumulated_guard;
  auto append_guard = [&](PrimExpr term) {
    if (!accumulated_guard.defined()) {
      accumulated_guard = term;
    } else {
      accumulated_guard = accumulated_guard && term;
    }
  };
  const SequenceNode *parent = nullptr;
  for (size_t idx = 0; idx < pts.begin()->path.size(); ++idx) {
    const auto &node = pts.begin()->path[idx];
    bool all_match = true;
    for (const auto &pt : pts) {
      if (idx >= pt.path.size() || pt.path[idx] != node) {
        all_match = false;
        break;
      }
    }
    if (!all_match) {
      break;
    }
    if (node->IsScheduleUnit()) {
      site.unit =
          const_cast<ScheduleUnit *>(static_cast<const ScheduleUnit *>(node));
      site.guard = accumulated_guard;
      site.chain.emplace_back(parent, site.unit);
    } else if (node->IsIf()) {
      break;
    } else if (node->IsSequence()) {
      parent = static_cast<const SequenceNode *>(node);
    } else if (node->IsControl()) {
      const ControlNode *ctrl = static_cast<const ControlNode *>(node);
      const ForNode *for_node = ctrl->control.get();
      if (ctrl->child->ContainsLoopBreak()) {
        break;
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
  if (!site.unit) {
    LOG(FATAL) << "Failed to find insertion ScheduleUnit for access points.";
  }
  return site;
}

std::vector<SyncPoint>
CollectSyncPoints(const std::vector<ScheduleUnit *> &units, int num_wgs,
                  const MultiVersionBufferMap &buffer_info_map,
                  const std::map<Buffer, int64_t> &buffer_num_versions = {},
                  std::optional<For> current_loop = std::nullopt) {
  auto deps = AnalyzeDependencies(units, current_loop);

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
    if (IsLocalBuffer(dep.buffer, true) || IsFragmentBuffer(dep.buffer))
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

  // Collect synchronization points from the dependency information.
  std::vector<SyncPoint> sync_points;

  auto get_versions = [&](const Buffer &b) -> int {
    auto it = buffer_num_versions.find(b);
    return it != buffer_num_versions.end() ? static_cast<int>(it->second) : 1;
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
        if (prod_unit == cons_unit && CanSkipSelfDependency(prod_unit))
          continue;

        // Bucket the producer/consumer access points by distance and the
        // hardware event (PRODPIPE_CONSPIPE) of each (producer, consumer) pair.
        struct EventPts {
          std::set<BufferAccessPoint> ppts;
          std::set<BufferAccessPoint> cpts;
          bool has_flag_source = false;
          size_t flag_depth = 0;
          int num_versions = 1;
          PrimExpr flag_iteration = IntImm(DataType::Int(32), 0);
        };
        std::map<std::pair<int, std::string>, EventPts> group_by_distance_event;
        for (const auto &d : dep_list) {
          int num_versions = get_versions(d.buffer);
          int distance = d.cross_iter ? num_versions : 0;
          size_t candidate_depth = 0;
          int candidate_num_versions = 1;
          PrimExpr candidate_iteration = IntImm(DataType::Int(32), 0);
          auto buffer_info_it = buffer_info_map.find(d.buffer);
          if (buffer_info_it != buffer_info_map.end()) {
            const auto &info = buffer_info_it->second;
            candidate_depth = info.claim_loop_info.loop_vars.size();
            candidate_num_versions = info.num_versions;
            candidate_iteration =
                info.claim_loop_info.CalculateIterationCount();
          }
          for (const auto &ppt : d.producer_pts) {
            for (const auto &cpt : d.consumer_pts) {
              std::string event = GetHardEvent(ppt.task(), cpt.task());
              auto &ev = group_by_distance_event[{distance, event}];
              ev.ppts.insert(ppt);
              ev.cpts.insert(cpt);
              if (!ev.has_flag_source || candidate_depth < ev.flag_depth) {
                ev.has_flag_source = true;
                ev.flag_depth = candidate_depth;
                ev.num_versions = candidate_num_versions;
                ev.flag_iteration = candidate_iteration;
              } else if (candidate_depth == ev.flag_depth) {
                ev.num_versions =
                    std::max(ev.num_versions, candidate_num_versions);
              }
            }
          }
        }

        for (const auto &[distance_event, pts] : group_by_distance_event) {
          const auto &[distance, hard_event] = distance_event;
          auto prod_site = GetInsertionSite(pts.ppts, false);
          auto cons_site = GetInsertionSite(pts.cpts, true);
          size_t pos = hard_event.find('_');
          std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
          std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
          SyncType sync_type =
              (prod_wg == cons_wg) ? SyncType::IntraCore : SyncType::CrossCore;
          int mode_id = 4; // Default mode for cross-core syncs
          sync_points.push_back({sync_type, prod_unit, prod_wg, prod_site,
                                 prod_pipe, cons_unit, cons_wg, cons_site,
                                 cons_pipe, distance, pts.num_versions,
                                 pts.flag_iteration, hard_event, mode_id});
        }
      }
    }
  }
  return sync_points;
}

int CompareInsertionSites(const SyncInsertionSite &a_site,
                          const SyncInsertionSite &b_site) {
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
  const auto &a = a_site.chain;
  const auto &b = b_site.chain;
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
  if (a.size() < b.size()) {
    return a_site.at_beginning ? -1 : +1;
  } else {
    return b_site.at_beginning ? +1 : -1;
  }
}

bool SyncDominates(const SyncPoint &x, const SyncPoint &y) {
  if (x.type != y.type)
    return false;
  if (x.prod_pipe != y.prod_pipe || x.cons_pipe != y.cons_pipe ||
      x.prod_wg != y.prod_wg || x.cons_wg != y.cons_wg)
    return false;
  int D = y.distance - x.distance;
  if (D < 0)
    return false;
  if (D >= 2)
    return true;
  bool c0 = CompareInsertionSites(x.prod_site, y.prod_site) >= 0;
  bool cD = CompareInsertionSites(x.cons_site, y.cons_site) <= 0;
  if (D == 1)
    return c0 || cD;
  return c0 && cD;
}

std::vector<SyncPoint>
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

std::vector<SyncPoint>
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
    for (size_t j = 0; j < sync_points.size(); ++j) {
      const auto &bc = sync_points[j];
      // ab and bc must compose at the intermediate unit: same unit, same
      // in-order pipe + warpgroup, and B's consume-of-A must be at-or-before
      // B's produce-for-C on that shared in-order pipe.
      // TODO: this is conservative — it only handles the same-pipe-in-B case.
      // When ab.cons and bc.prod are on *different* pipes in B but B internally
      // synchronizes them, the chain is still valid; detecting that needs
      // looking inside B and is deferred.
      if (ab.cons_unit != bc.prod_unit || ab.cons_pipe != bc.prod_pipe ||
          ab.cons_wg != bc.prod_wg ||
          CompareInsertionSites(ab.cons_site, bc.prod_site) > 0)
        continue;
      // Synthetic edge: producer side from ab, consumer side from bc, distances
      // summed.
      SyncPoint chain = ab;
      chain.cons_unit = bc.cons_unit;
      chain.cons_wg = bc.cons_wg;
      chain.cons_pipe = bc.cons_pipe;
      chain.cons_site = bc.cons_site;
      chain.distance = ab.distance + bc.distance;
      chain.type = (chain.prod_wg != chain.cons_wg) ? SyncType::CrossCore
                                                    : SyncType::IntraCore;
      chain.hard_event =
          chain.prod_pipe.substr(5) + "_" + chain.cons_pipe.substr(5);
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

void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                      FlagInfoList &flag_infos, EventIdMap &event_id_map) {
  for (const auto &sp : sync_points) {
    if (sp.type == SyncType::CrossCore) {
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

      PrimExpr cc_event_id =
          indexmod(sp.flag_iteration, Integer(sp.num_versions)) + cc_base_id;

      if (!sp.prod_wg) {
        Stmt set_stmt =
            MakeAscendCrossCoreSetFlag(sp.mode_id, sp.prod_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.prod_site.unit, set_stmt, false,
                                        sp.prod_wg, sp.prod_site.guard);
        for (unsigned s = 0; s != 2; ++s) {
          Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(
              sp.mode_id, sp.cons_pipe,
              indexmod(sp.flag_iteration, Integer(sp.num_versions)) +
                  Integer(cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.cons_site.unit, wait_stmt, true,
                                          sp.cons_wg, sp.cons_site.guard);
        }
      } else {
        for (unsigned s = 0; s != 2; ++s) {
          Stmt set_stmt = MakeAscendCrossCoreSetFlag(
              sp.mode_id, sp.prod_pipe,
              indexmod(sp.flag_iteration, Integer(sp.num_versions)) +
                  Integer(cc_base_id + 16 * s));
          InsertStatementIntoScheduleUnit(sp.prod_site.unit, set_stmt, false,
                                          sp.prod_wg, sp.prod_site.guard);
        }
        Stmt wait_stmt =
            MakeAscendCrossCoreWaitFlag(sp.mode_id, sp.cons_pipe, cc_event_id);
        InsertStatementIntoScheduleUnit(sp.cons_site.unit, wait_stmt, true,
                                        sp.cons_wg, sp.cons_site.guard);
      }
    } else if (sp.type == SyncType::IntraCore) {
      // When producer and consumer share the same pipe, insert a pipe_barrier
      // before the consumer to ensure pipeline ordering.
      if (sp.prod_pipe == sp.cons_pipe) {
        // The scalar pipe (PIPE_S) issues instructions in program order
        if (sp.prod_pipe == "PIPE_S") {
          continue;
        }
        Stmt barrier = MakeAscendPipeBarrier(sp.prod_pipe);
        InsertStatementIntoScheduleUnit(sp.cons_site.unit, barrier, true,
                                        sp.cons_wg, sp.cons_site.guard);
        continue;
      }
      auto base_id = event_id_map.find(sp.hard_event);
      if (base_id == event_id_map.end()) {
        event_id_map[sp.hard_event] = 0;
        base_id = event_id_map.find(sp.hard_event);
      }
      int base_id_val = base_id->second;
      base_id->second += sp.num_versions;

      PrimExpr event_id =
          indexmod(sp.flag_iteration, Integer(sp.num_versions)) + base_id_val;

      if (sp.distance > 0) {
        for (int base = 0; base < sp.num_versions; ++base) {
          flag_infos.push_back(FlagInfo{sp.prod_wg, Integer(base_id_val + base),
                                        sp.hard_event, 0});
        }
      }

      Stmt set_stmt = MakeAscendSetFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.prod_site.unit, set_stmt, false,
                                      sp.prod_wg, sp.prod_site.guard);

      Stmt wait_stmt = MakeAscendWaitFlag(sp.hard_event, event_id);
      InsertStatementIntoScheduleUnit(sp.cons_site.unit, wait_stmt, true,
                                      sp.cons_wg, sp.cons_site.guard);
    }
  }
}

void AnalyzeSequenceNodeBarriers(
    SequenceNode *seq, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    MultiVersionBufferMap &buffer_info_map) {
  if (!seq)
    return;
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    AnalyzeAndInsertBarriers(unit->child.get(), event_id_map, flag_infos, sid,
                             loop_info, buffer_infos, buffer_info_map);
  }
  auto sync_points = CollectSyncPoints(units, 2, buffer_info_map);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, event_id_map);
}

void AnalyzeControlNodeBarriers(
    ControlNode *ctrl, EventIdMap &event_id_map, FlagInfoList &flag_infos,
    Var sid, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    MultiVersionBufferMap &buffer_info_map) {
  if (!ctrl || !ctrl->child)
    return;
  const ForNode *for_node = ctrl->control.get();
  if (!for_node)
    return;
  loop_info.AddLoop(for_node);

  // Create new buffers for multi-buffer before recursing so nested accesses
  // can use the claim-level iteration for both data and flags.
  std::vector<Buffer> scoped_buffers;
  for (auto &region : ctrl->GetWriteRegions()) {
    auto &buffer = region.get()->buffer;
    if (buffer_info_map.find(buffer) != buffer_info_map.end())
      continue;
    auto it = ctrl->buffer_num_versions.find(buffer);
    if (it == ctrl->buffer_num_versions.end())
      continue;
    int num_versions = it->second;
    if (num_versions == 1)
      continue;
    auto new_buffer = CreateNewBuffer(buffer, num_versions);
    MultiVersionBufferInfo info(buffer, num_versions, new_buffer, loop_info);
    buffer_info_map.emplace(buffer, info);
    buffer_infos.push_back(std::move(info));
    scoped_buffers.push_back(buffer);
  }

  ICHECK(ctrl->child->IsSequence());
  auto seq = static_cast<SequenceNode *>(ctrl->child.get());
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    AnalyzeAndInsertBarriers(unit->child.get(), event_id_map, flag_infos, sid,
                             loop_info, buffer_infos, buffer_info_map);
  }

  // Sort units by stage (largest first) to match SW-pipelining order — the
  // latest stage produces a value first across iterations.
  std::stable_sort(
      units.begin(), units.end(),
      [](ScheduleUnit *a, ScheduleUnit *b) { return a->stage > b->stage; });

  auto sync_points = CollectSyncPoints(
      units, 2, buffer_info_map, ctrl->buffer_num_versions, ctrl->control);
  sync_points = OptimizeSyncPoints(sync_points);
  InsertSyncPoints(sync_points, flag_infos, event_id_map);

  for (const auto &buffer : scoped_buffers) {
    buffer_info_map.erase(buffer);
  }
  loop_info.PopLoop();
}

// Barrier dependency analysis implementation
void AnalyzeAndInsertBarriers(IRStructure *node, EventIdMap &event_id_map,
                              FlagInfoList &flag_infos, Var sid,
                              LoopNestingInfo &loop_info,
                              std::vector<MultiVersionBufferInfo> &buffer_infos,
                              MultiVersionBufferMap &buffer_info_map) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node), event_id_map,
                                flag_infos, sid, loop_info, buffer_infos,
                                buffer_info_map);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos,
                               buffer_info_map);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(wrapper->child.get(), event_id_map, flag_infos,
                             sid, loop_info, buffer_infos, buffer_info_map);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child) {
      AnalyzeAndInsertBarriers(if_node->then_child.get(), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos,
                               buffer_info_map);
    }
    if (if_node->else_child) {
      AnalyzeAndInsertBarriers(if_node->else_child.get(), event_id_map,
                               flag_infos, sid, loop_info, buffer_infos,
                               buffer_info_map);
    }
  } else if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    MultiBufferAccessRewriter rewriter(buffer_info_map);
    for (auto &stmt : task->stmts) {
      stmt = rewriter(stmt);
    }
  } else {
    LOG(FATAL);
  }
}

} // namespace tl
} // namespace tvm
