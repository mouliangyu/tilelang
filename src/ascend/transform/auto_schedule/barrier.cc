#include "./barrier.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

#include "op/utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

// ============================================================================
// Helper functions
// ============================================================================

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

void InsertStatementIntoScheduledNode(IRStructure *node, Stmt stmt,
                                      bool at_beginning, int warpgroup_id,
                                      const PrimExpr &guard = PrimExpr()) {
  if (guard.defined()) {
    stmt = IfThenElse(guard, stmt);
  }
  if (at_beginning) {
    node->before[warpgroup_id].insert(node->before[warpgroup_id].begin(), stmt);
  } else {
    node->after[warpgroup_id].push_back(stmt);
  }
}

// Check if any operation in `stmt` has unit_flag_ctl/ctrl != 0. This walks the
// whole statement rather than shape-matching a bare Evaluate(Call), because a
// T.PerCoreTask leaf holds a compound body.
bool HasUnitFlagCtrl(const Stmt &stmt) {
  class Detector : public StmtExprVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (auto ann = op->annotations.Get("unit_flag_ctrl")) {
        if (const auto *imm = ann->as<IntImmNode>()) {
          found = imm->value != 0;
        } else {
          found = true; // non-IntImm means dynamically enabled
        }
      }
      if (!found)
        StmtExprVisitor::VisitExpr_(op);
    }
  } detector;
  detector(stmt);
  return detector.found;
}

std::string GetResourcePipeName(uint16_t mask) {
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

std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe) {
  ICHECK(producer_pipe != "UNKNOWN" && consumer_pipe != "UNKNOWN");
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

void BarrierBuilder::CollectInsertionSites(
    IRStructure *node, std::vector<const IRStructure *> &path,
    SyncInsertionSiteList &sites) const {
  if (!node)
    return;
  if (node->IsTask()) {
    std::vector<const IRStructure *> task_path = path;
    task_path.push_back(node);
    BufferAccessPoint pt(0, std::move(task_path));
    auto *task = static_cast<const TaskNode *>(node);
    int wg = task->GetWarpgroupId();
    std::string pipe = GetResourcePipeName(task->GetPipeMask());
    std::vector<int> wgs;
    if (IsWarpgroupBroadcast(wg)) {
      for (int i = 0; i < kNumWarpgroups; ++i)
        wgs.push_back(i);
    } else if (wg >= 0 && wg < kNumWarpgroups) {
      wgs.push_back(wg);
    }
    for (bool at_beginning : {false, true}) {
      for (int w : wgs) {
        sites.emplace_back(pt, at_beginning, w, pipe);
      }
    }
    return;
  }
  auto *ctrl = static_cast<ControlNode *>(node);
  path.push_back(node);
  for (const auto &child : ctrl->children) {
    CollectInsertionSites(child.get(), path, sites);
  }
  path.pop_back();
}

void BarrierBuilder::CollectAllInsertionSites(
    const std::vector<IRStructure *> &nodes) {
  std::vector<const IRStructure *> path;
  for (IRStructure *node : nodes) {
    CollectInsertionSites(node, path, sites_);
  }
  for (IRStructure *node : nodes) {
    if (!node || !node->IsControl())
      continue;
    auto *ctrl = static_cast<ControlNode *>(node);
    std::vector<IRStructure *> children;
    children.reserve(ctrl->children.size());
    for (const auto &child : ctrl->children)
      children.push_back(child.get());
    CollectAllInsertionSites(children);
  }
}

std::vector<SyncPoint> BarrierBuilder::CollectSyncPoints(
    const std::vector<IRStructure *> &nodes, const ControlNode *cache_key,
    const std::map<Buffer, int64_t> &buffer_num_versions) {
  auto deps = AnalyzeDependencies(nodes, loop_stack_, manual_,
                                  &dependency_cache_, cache_key);

  // Expand a task's warpgroup id into the concrete wg ids it participates in.
  auto expand_wgs = [&](int task_wg) {
    std::vector<int> out;
    if (IsWarpgroupBroadcast(task_wg)) {
      for (int w = 0; w < kNumWarpgroups; ++w) {
        out.push_back(w);
      }
    } else if (task_wg >= 0 && task_wg < kNumWarpgroups) {
      out.push_back(task_wg);
    }
    return out;
  };

  std::vector<SyncPoint> sync_points;
  for (const auto &dep : deps) {
    if (IsLocalBuffer(dep.buffer, true) || IsFragmentBuffer(dep.buffer)) {
      continue;
    }

    int distance;
    if (dep.distance >= 0) {
      distance = dep.distance;
    } else {
      auto it = buffer_num_versions.find(dep.buffer);
      if (it != buffer_num_versions.end()) {
        distance = it->second;
      } else {
        distance = 1;
      }
    }
    int num_versions = 1;
    std::vector<For> flag_loops;
    auto buffer_info_it = buffer_info_map_.find(dep.buffer);
    if (buffer_info_it != buffer_info_map_.end()) {
      const auto &info = buffer_info_it->second;
      num_versions = info.num_versions;
      flag_loops = info.claim_loops;
    } else {
      auto it = manual_.find(dep.buffer->data.get());
      if (it != manual_.end()) {
        if (dep.distance > 0) {
          num_versions = dep.distance;
        } else {
          num_versions = it->second;
        }
        flag_loops = loop_stack_;
      }
    }

    for (const auto &[ppt, cpt] : dep.task_pairs) {
      std::string prod_pipe = GetResourcePipeName(ppt.task()->GetPipeMask());
      std::string cons_pipe = GetResourcePipeName(cpt.task()->GetPipeMask());
      std::string hard_event = GetHardEvent(prod_pipe, cons_pipe);
      bool fix_unit_flag = hard_event.find("FIX") != std::string::npos &&
                           (HasUnitFlagCtrl(ppt.task()->stmt) ||
                            HasUnitFlagCtrl(cpt.task()->stmt));
      for (int prod_wg : expand_wgs(ppt.task()->GetWarpgroupId())) {
        for (int cons_wg : expand_wgs(cpt.task()->GetWarpgroupId())) {
          // Intra-core FIX unit-flag syncs are handled by the hardware.
          if (prod_wg == cons_wg && fix_unit_flag)
            continue;
          size_t prod_site =
              FindSiteId(SyncInsertionSite(ppt, false, prod_wg, prod_pipe));
          size_t cons_site =
              FindSiteId(SyncInsertionSite(cpt, true, cons_wg, cons_pipe));
          sync_points.emplace_back(
              DepEdge(prod_site, cons_site, distance, true), num_versions,
              flag_loops);
        }
      }
    }
  }
  return sync_points;
}

void BarrierBuilder::AddPromotedChildClosure(
    DepClosure &out, const IRStructure *child,
    const DepClosure &child_closure) const {
  if (child->HasConditions() ||
      (child->IsControl() &&
       static_cast<const ControlNode *>(child)->BodyHasLoopBreak()))
    return;
  auto promote = [&](const SyncInsertionSite &site) {
    SyncInsertionSite promoted = site;
    promoted.chain.insert(promoted.chain.begin(), child);
    return promoted;
  };

  for (const DepEdge &edge : child_closure.Edges()) {
    size_t src = FindSiteId(promote(edge.Src(sites_)));
    size_t dst = FindSiteId(promote(edge.Dst(sites_)));
    out.Insert(DepEdge(src, dst, edge.distance, edge.strict));
  }
}

std::vector<size_t>
BarrierBuilder::CollectSiteIds(const std::vector<IRStructure *> &nodes) const {
  SyncInsertionSiteList level_sites;
  std::vector<const IRStructure *> path;
  for (IRStructure *node : nodes) {
    CollectInsertionSites(node, path, level_sites);
  }
  SortAndUniqueSites(level_sites);

  std::vector<size_t> result;
  result.reserve(level_sites.size());
  for (const SyncInsertionSite &site : level_sites)
    result.push_back(FindSiteId(site));
  return result;
}

void BarrierBuilder::AddSamePipeEdges(DepClosure &closure,
                                      const std::vector<size_t> &task_sites,
                                      int max_distance) const {
  std::map<std::pair<int, std::string>, std::vector<size_t>> by_pipe;
  for (size_t site_id : task_sites) {
    const SyncInsertionSite &site = sites_[site_id];
    by_pipe[{site.wg, site.pipe}].push_back(site_id);
  }
  // This level's iteration stage of a site (the outermost node in its chain is
  // the one carried by the loop being pipelined at this level).
  auto level_stage = [](const SyncInsertionSite &s) {
    return s.chain.empty() ? 0 : s.chain.front()->GetStage();
  };
  for (auto &[_, pipe_sites] : by_pipe) {
    // A single pipe issues its sites in physical (iteration, position) order.
    // For each ordered pair a->b, the happens-before edge a(k) < b(k+d) holds
    // for all d >= min_d, where a node at stage s in physical iteration p runs
    // logical iteration k = p + s. Solving the in-order constraint gives:
    //   min_d = (s_b - s_a) + (a physically before b ? 0 : 1),  clamped to >=0.
    // Adding every valid distance in [min_d, max_distance] seeds the closure
    // with the true pipe ordering, without assuming iterations run serially.
    for (size_t a_id : pipe_sites) {
      for (size_t b_id : pipe_sites) {
        if (a_id == b_id)
          continue;
        const SyncInsertionSite &a = sites_[a_id];
        const SyncInsertionSite &b = sites_[b_id];
        bool a_before_b = a_id < b_id;
        int min_d = (level_stage(b) - level_stage(a)) + (a_before_b ? 0 : 1);
        if (min_d < 0)
          min_d = 0;
        for (int d = min_d; d <= max_distance; ++d) {
          closure.Saturate(DepEdge(a_id, b_id, d, false), max_distance);
        }
      }
    }
  }
}

void BarrierBuilder::OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                                        DepClosure &closure, int max_distance) {
  std::stable_sort(sync_points.begin(), sync_points.end());
  std::vector<SyncPoint> emit;
  for (auto &sp : sync_points) {
    if (!closure.Contains(sp.edge)) {
      closure.Saturate(sp.edge, max_distance);
      emit.push_back(std::move(sp));
    }
  }
  sync_points = std::move(emit);
}

void BarrierBuilder::OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                                          const DepClosure &closure) const {
  for (auto &sp : sync_points) {
    const SyncInsertionSite &src = sp.edge.Src(sites_);
    const SyncInsertionSite &dst = sp.edge.Dst(sites_);
    if (src.pipe == dst.pipe)
      continue;
    bool is_innermost = HasSameLoopNest(loop_stack_, sp.flag_loops);
    int innermost_num_versions = is_innermost ? sp.num_versions : 1;
    int min_distance = closure.MinDistance(sp.edge.dst, sp.edge.src);
    // If the loop trip count is no larger than the number of flag versions, the
    // flag is never reused across iterations, so an absent/loose reverse
    // ordering is harmless — no version bound is needed.
    if (!loop_stack_.empty()) {
      const ForNode *inner = loop_stack_.back().get();
      PrimExpr step = inner->step.has_value()
                          ? inner->step.value()
                          : IntImm(inner->extent.dtype(), 1);
      arith::Analyzer ana;
      if (ana.CanProve(inner->extent <=
                       IntImm(inner->extent.dtype(), innermost_num_versions) *
                           step)) {
        continue;
      }
    }
    if (sp.edge.distance > 0) {
      if (sp.edge.distance != innermost_num_versions || min_distance > 0) {
        LOG(WARNING) << "cross-iteration flag " << src.pipe << "->" << dst.pipe
                     << ": the synchronization ring is malformed.";
      }
      continue;
    }
    if (min_distance > innermost_num_versions) {
      LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                   << ": reverse ordering too loose to bound flag reuse.";
      continue;
    }
    if (min_distance <= 0) {
      LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                   << ": dependency cycle (reverse distance <= 0).";
      continue;
    }
    if (is_innermost) {
      sp.num_versions = min_distance;
    }
  }
}

bool BarrierBuilder::CanReuse(const SyncPoint &a, const SyncPoint &b,
                              const DepClosure &closure, bool allow_cross_iter,
                              bool allow_mixed_iter) const {
  if (!UsesFlag(a) || !UsesFlag(b) || !HasCompatibleFlagKind(a, b))
    return false;

  bool a_cross_iter = a.edge.distance > 0;
  bool b_cross_iter = b.edge.distance > 0;
  if (a_cross_iter != b_cross_iter) {
    // A mixed same/cross-iteration handshake alternates on one flag ring. Its
    // local lifetime proof is valid only when both points use the same ring.
    if (!allow_mixed_iter || !IsCrossCore(a) || !IsCrossCore(b) ||
        a.num_versions != b.num_versions ||
        !HasSameLoopNest(a.flag_loops, b.flag_loops))
      return false;
    const SyncPoint &cross_iter = a_cross_iter ? a : b;
    const SyncPoint &same_iter = a_cross_iter ? b : a;
    return closure.HasEdge(cross_iter.reuse_dst, same_iter.reuse_src, 0) &&
           closure.HasEdge(same_iter.reuse_dst, cross_iter.reuse_src, 0);
  }

  if (!a_cross_iter) {
    bool a_before_b = closure.HasEdge(a.reuse_dst, b.reuse_src, 0) &&
                      closure.HasEdge(b.reuse_dst, a.reuse_src, 1);
    bool b_before_a = closure.HasEdge(b.reuse_dst, a.reuse_src, 0) &&
                      closure.HasEdge(a.reuse_dst, b.reuse_src, 1);
    return a_before_b != b_before_a;
  }

  if (!allow_cross_iter)
    return false;

  bool a_before_b = closure.HasEdge(a.reuse_src, b.reuse_dst, 0) &&
                    closure.HasEdge(b.reuse_src, a.reuse_dst, 1);
  bool b_before_a = closure.HasEdge(b.reuse_src, a.reuse_dst, 0) &&
                    closure.HasEdge(a.reuse_src, b.reuse_dst, 1);
  return a_before_b != b_before_a;
}

void BarrierBuilder::PromoteReuseSites(SyncPoint &sp, IRStructure *node) const {
  SyncInsertionSite src = sites_[sp.reuse_src];
  SyncInsertionSite dst = sites_[sp.reuse_dst];
  if (!node->HasConditions()) {
    src.chain.insert(src.chain.begin(), node);
    dst.chain.insert(dst.chain.begin(), node);
  } else {
    bool cross_iter = sp.edge.distance > 0;
    auto collapse = [&](SyncInsertionSite &site, bool at_beginning) {
      site.node = node;
      site.at_beginning = at_beginning;
      site.chain = {node};
    };
    collapse(src, !cross_iter);
    collapse(dst, cross_iter);
  }

  sp.reuse_src = FindSiteId(src);
  sp.reuse_dst = FindSiteId(dst);
}

void BarrierBuilder::RecordReusablePairs(const std::vector<size_t> &a_points,
                                         const std::vector<size_t> &b_points,
                                         const DepClosure &closure,
                                         bool allow_cross_iter,
                                         bool allow_mixed_iter) {
  for (const auto &a : a_points) {
    for (const auto &b : b_points) {
      if (a == b || !CanReuse(sync_points_[a], sync_points_[b], closure,
                              allow_cross_iter, allow_mixed_iter))
        continue;
      reusable_pairs_.emplace(std::min(a, b), std::max(a, b));
    }
  }
}

void BarrierBuilder::PlanFlagAllocations() {
  allocations_.clear();
  std::vector<SyncPoint> &sync_points = sync_points_;
  std::vector<FlagAllocation> &allocations = allocations_;
  const auto &reusable_pairs = reusable_pairs_;
  const auto &cross_core_flag_reservation = cross_core_flag_reservation_;
  std::map<std::string, int> required_ids;
  std::map<std::string, int> next_id;
  bool needs_cross_core_flag = false;
  for (const auto &sp : sync_points) {
    if (!UsesFlag(sp))
      continue;
    std::string domain = FlagDomain(sp);
    required_ids[domain] += sp.num_versions;
    needs_cross_core_flag |= IsCrossCore(sp);
  }

  ICHECK(!needs_cross_core_flag ||
         !cross_core_flag_reservation.has_unbounded_id)
      << "Ascend AutoSchedule cannot safely allocate cross-core flag IDs when "
         "an explicit mode-0 CrossCoreSetFlag/WaitFlag uses a flag_id whose "
         "integer range cannot be bounded; constrain the flag_id with a "
         "constant-range loop or disable AutoSchedule";

  // Mode-4 synchronization addresses the same 16 base slots as mode 0. The
  // generated Cube side may additionally address slot + 16 for the second
  // Vector core, so normalize explicit IDs and reserve only the actual slots.
  std::array<bool, kCrossCoreFlagLimit> cross_core_slots{};
  for (const auto &[lo, hi] : cross_core_flag_reservation.explicit_id_ranges) {
    ICHECK_LE(lo, hi);
    uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    if (span >= static_cast<uint64_t>(kCrossCoreFlagLimit - 1)) {
      cross_core_slots.fill(true);
      continue;
    }
    for (uint64_t offset = 0; offset <= span; ++offset) {
      int64_t explicit_id = lo + static_cast<int64_t>(offset);
      int normalized_id = static_cast<int>(explicit_id % kCrossCoreFlagLimit);
      if (normalized_id < 0)
        normalized_id += kCrossCoreFlagLimit;
      cross_core_slots[normalized_id] = true;
    }
  }
  const int explicit_cross_core_slots =
      std::count(cross_core_slots.begin(), cross_core_slots.end(), true);
  auto find_cross_core_range = [&](int width) {
    for (int base = 0; base + width <= kCrossCoreFlagLimit; ++base) {
      bool available = true;
      for (int offset = 0; offset < width; ++offset)
        available &= !cross_core_slots[base + offset];
      if (available)
        return base;
    }
    return -1;
  };

  for (bool cross_iter : {true, false}) {
    std::vector<size_t> candidates;
    for (size_t i = 0; i < sync_points.size(); ++i) {
      if (UsesFlag(sync_points[i]) &&
          (sync_points[i].edge.distance > 0) == cross_iter) {
        candidates.push_back(i);
      }
    }
    std::stable_sort(
        candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
          return sync_points[a].num_versions > sync_points[b].num_versions;
        });

    for (size_t index : candidates) {
      SyncPoint &sp = sync_points[index];
      bool compatible = false;
      std::string domain = FlagDomain(sp);
      int limit = IsCrossCore(sp) ? kCrossCoreFlagLimit : kIntraCoreFlagLimit;
      int cross_core_base =
          IsCrossCore(sp) ? find_cross_core_range(sp.num_versions) : -1;
      bool needs_reuse =
          IsCrossCore(sp) ? required_ids[domain] + explicit_cross_core_slots >
                                    kCrossCoreFlagLimit ||
                                cross_core_base < 0
                          : required_ids[domain] > limit;
      if (needs_reuse) {
        for (size_t allocation_index = 0; allocation_index < allocations.size();
             ++allocation_index) {
          FlagAllocation &allocation = allocations[allocation_index];
          const SyncPoint &representative =
              sync_points[allocation.sync_points.front()];
          if (!HasCompatibleFlagKind(sp, representative))
            continue;
          if (allocation.cross_iter != cross_iter &&
              (!IsCrossCore(sp) || !IsCrossCore(representative)))
            continue;
          compatible = true;
          for (size_t member : allocation.sync_points) {
            if (!reusable_pairs.count(
                    {std::min(index, member), std::max(index, member)})) {
              compatible = false;
              break;
            }
          }
          if (compatible) {
            ICHECK_GE(allocation.num_versions, sp.num_versions);
            required_ids[domain] -= sp.num_versions;
            allocation.sync_points.push_back(index);
            sp.allocation_index = allocation_index;
            break;
          }
        }
      }
      if (compatible)
        continue;
      sp.allocation_index = allocations.size();
      int base_id = next_id[domain];
      if (IsCrossCore(sp)) {
        ICHECK_GE(cross_core_base, 0)
            << "Ascend AutoSchedule cannot allocate " << sp.num_versions
            << " contiguous cross-core flag slots without overlapping explicit "
               "mode-0 flags or safely reusable auto-scheduled flags";
        base_id = cross_core_base;
        for (int offset = 0; offset < sp.num_versions; ++offset)
          cross_core_slots[base_id + offset] = true;
      }
      allocations.emplace_back(index, sp.num_versions, cross_iter, base_id);
      if (!IsCrossCore(sp))
        next_id[domain] += sp.num_versions;
      if (!IsCrossCore(sp) && base_id <= limit && next_id[domain] > limit) {
        LOG(WARNING) << "Ascend auto-schedule allocated " << next_id[domain]
                     << " flag ids for " << domain << " after safe reuse, "
                     << "exceeding the dav-3510 limit of " << limit << ".";
      }
    }
  }
}

void BarrierBuilder::AppendCrossIterFlagInfo(const FlagAllocation &allocation) {
  const SyncPoint &sp = sync_points_[allocation.sync_points.front()];
  const SyncInsertionSite &src = sp.edge.Src(sites_);
  const SyncInsertionSite &dst = sp.edge.Dst(sites_);
  std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
  size_t pos = hard_event.find('_');
  std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
  std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
  int base_id = allocation.base_id;

  if (!IsCrossCore(sp)) {
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos_.push_back(
          FlagInfo{src.wg, Integer(base_id + version), hard_event, 0});
    }
    return;
  }

  // Cross-core set uses the producer pipe; wait uses the consumer pipe.
  if (!src.wg) {
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos_.push_back(
          FlagInfo{0, Integer(base_id + version), prod_pipe, 1});
    }
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(
            FlagInfo{1, Integer(base_id + sid * 16 + version), cons_pipe, 2});
      }
    }
  } else {
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(
            FlagInfo{1, Integer(base_id + sid * 16 + version), prod_pipe, 1});
      }
    }
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos_.push_back(
          FlagInfo{0, Integer(base_id + version), cons_pipe, 2});
    }
  }
}

void BarrierBuilder::InsertSyncPoints() {
  for (const auto &allocation : allocations_) {
    if (allocation.cross_iter)
      AppendCrossIterFlagInfo(allocation);
  }

  for (const auto &sp : sync_points_) {
    const SyncInsertionSite &src = sp.edge.Src(sites_);
    const SyncInsertionSite &dst = sp.edge.Dst(sites_);
    std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
    size_t pos = hard_event.find('_');
    std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
    std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
    PrimExpr prod_guard = src.GetGuard();
    PrimExpr cons_guard = dst.GetGuard();
    PrimExpr flag_iteration = sp.FlagIteration();

    if (!IsCrossCore(sp) && prod_pipe == cons_pipe) {
      // The scalar, vector, and cube pipes already execute in program order.
      if (prod_pipe == "PIPE_S" || prod_pipe == "PIPE_V" ||
          prod_pipe == "PIPE_M") {
        continue;
      }
      Stmt barrier = MakeAscendPipeBarrier(prod_pipe);
      InsertStatementIntoScheduledNode(dst.node, barrier, true, dst.wg,
                                       cons_guard);
      continue;
    }

    const FlagAllocation &allocation = allocations_[sp.allocation_index];
    int base_id = allocation.base_id;
    PrimExpr event_id =
        indexmod(flag_iteration, Integer(sp.num_versions)) + base_id;

    if (IsCrossCore(sp)) {
      if (!src.wg) {
        Stmt set_stmt = MakeAscendCrossCoreSetFlag(4, prod_pipe, event_id);
        InsertStatementIntoScheduledNode(src.node, set_stmt, false, src.wg,
                                         prod_guard);
        for (int sid = 0; sid < 2; ++sid) {
          Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(
              4, cons_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatementIntoScheduledNode(dst.node, wait_stmt, true, dst.wg,
                                           cons_guard);
        }
      } else {
        for (int sid = 0; sid < 2; ++sid) {
          Stmt set_stmt = MakeAscendCrossCoreSetFlag(
              4, prod_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatementIntoScheduledNode(src.node, set_stmt, false, src.wg,
                                           prod_guard);
        }
        Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(4, cons_pipe, event_id);
        InsertStatementIntoScheduledNode(dst.node, wait_stmt, true, dst.wg,
                                         cons_guard);
      }
      continue;
    }

    Stmt set_stmt = MakeAscendSetFlag(hard_event, event_id);
    InsertStatementIntoScheduledNode(src.node, set_stmt, false, src.wg,
                                     prod_guard);
    Stmt wait_stmt = MakeAscendWaitFlag(hard_event, event_id);
    InsertStatementIntoScheduledNode(dst.node, wait_stmt, true, dst.wg,
                                     cons_guard);
  }
}

BarrierAnalysisResult BarrierBuilder::AnalyzeNodeList(
    std::vector<IRStructure *> nodes, const ControlNode *cache_key,
    bool is_loop_body, const std::map<Buffer, int64_t> &buffer_num_versions) {
  BarrierAnalysisResult result{DepClosure(sites_.size()), {}};
  std::vector<size_t> task_site_ids = CollectSiteIds(nodes);

  std::vector<std::vector<size_t>> child_sync_points;
  for (IRStructure *node : nodes) {
    BarrierAnalysisResult child_result = AnalyzeNode(node);
    AddPromotedChildClosure(result.closure, node, child_result.closure);
    child_sync_points.push_back(std::move(child_result.visible_sync_points));
  }

  if (is_loop_body) {
    // Sort nodes by stage (largest first) to match SW-pipelining order — the
    // latest stage produces a value first across iterations.
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](IRStructure *a, IRStructure *b) {
                       return a->GetStage() > b->GetStage();
                     });
  }

  std::vector<SyncPoint> local_sync_points =
      CollectSyncPoints(nodes, cache_key, buffer_num_versions);
  int max_distance = loop_stack_.empty() ? 0 : 1;
  for (const auto &sp : local_sync_points) {
    max_distance = std::max(max_distance, sp.edge.distance);
  }
  AddSamePipeEdges(result.closure, task_site_ids, max_distance);
  OptimizeSyncPoints(local_sync_points, result.closure, max_distance);
  if (is_loop_body)
    OptimizeFlagVersions(local_sync_points, result.closure);

  std::vector<size_t> local_points;
  local_points.reserve(local_sync_points.size());
  for (auto &sp : local_sync_points) {
    size_t index = sync_points_.size();
    local_points.push_back(index);
    sync_points_.push_back(std::move(sp));
  }

  for (size_t i = 0; i < child_sync_points.size(); ++i) {
    for (size_t j = i + 1; j < child_sync_points.size(); ++j) {
      RecordReusablePairs(child_sync_points[i], child_sync_points[j],
                          result.closure,
                          /*allow_cross_iter=*/true,
                          /*allow_mixed_iter=*/false);
    }
    RecordReusablePairs(child_sync_points[i], local_points, result.closure,
                        /*allow_cross_iter=*/false,
                        /*allow_mixed_iter=*/false);
    result.visible_sync_points.insert(result.visible_sync_points.end(),
                                      child_sync_points[i].begin(),
                                      child_sync_points[i].end());
  }
  // Mixed same/cross-iteration reuse is checked only at the level where both
  // sync points are collected; promoted child points have lost that context.
  RecordReusablePairs(local_points, local_points, result.closure,
                      /*allow_cross_iter=*/false,
                      /*allow_mixed_iter=*/true);
  result.visible_sync_points.insert(result.visible_sync_points.end(),
                                    local_points.begin(), local_points.end());
  result.closure = result.closure.DistanceZero();
  return result;
}

BarrierAnalysisResult BarrierBuilder::AnalyzeNode(IRStructure *node) {
  if (!node)
    return {DepClosure(sites_.size()), {}};

  if (node->IsControl()) {
    auto *ctrl = static_cast<ControlNode *>(node);
    if (ctrl->children.empty() || !ctrl->control.defined())
      return {DepClosure(sites_.size()), {}};
    loop_stack_.push_back(ctrl->control);

    // Create new buffers for multi-buffer before recursing so nested accesses
    // can use the claim-level iteration for both data and flags.
    std::vector<Buffer> scoped_buffers;
    std::set<const VarNode *> scoped_storages;
    for (auto &region : ctrl->GetWriteRegions()) {
      auto &buffer = region.get()->buffer;
      if (buffer_info_map_.find(buffer) != buffer_info_map_.end())
        continue;
      auto it = ctrl->buffer_num_versions.find(buffer);
      if (it == ctrl->buffer_num_versions.end())
        continue;
      int num_versions = it->second;
      if (num_versions == 1)
        continue;
      auto new_buffer = CreateNewBuffer(buffer, num_versions);
      MultiVersionBufferInfo info(buffer, num_versions, new_buffer,
                                  loop_stack_);
      buffer_info_map_.emplace(buffer, info);
      buffer_infos_.push_back(std::move(info));
      scoped_buffers.push_back(buffer);
      scoped_storages.insert(buffer->data.get());
    }
    // View/reshape aliases share storage (->data) with a multi-buffered write
    // buffer; give them the same versioning so their accesses stay consistent.
    for (auto &region : ctrl->GetReadRegions()) {
      const auto &buffer = region.get()->buffer;
      if (buffer_info_map_.find(buffer) != buffer_info_map_.end())
        continue;
      if (!scoped_storages.count(buffer->data.get()))
        continue;
      auto it = ctrl->buffer_num_versions.find(buffer);
      if (it == ctrl->buffer_num_versions.end() || it->second == 1)
        continue;
      MultiVersionBufferInfo info(
          buffer, it->second, CreateNewBuffer(buffer, it->second), loop_stack_);
      buffer_info_map_.emplace(buffer, info);
      buffer_infos_.push_back(std::move(info));
      scoped_buffers.push_back(buffer);
    }

    std::vector<IRStructure *> nodes;
    for (auto &child : ctrl->children)
      nodes.push_back(child.get());
    BarrierAnalysisResult result =
        AnalyzeNodeList(std::move(nodes), ctrl, /*is_loop_body=*/true,
                        ctrl->buffer_num_versions);

    for (const auto &buffer : scoped_buffers)
      buffer_info_map_.erase(buffer);
    loop_stack_.pop_back();
    if (!ctrl->BodyHasLoopBreak()) {
      for (size_t index : result.visible_sync_points) {
        PromoteReuseSites(sync_points_[index], node);
      }
    } else {
      result.visible_sync_points.clear();
    }
    return result;
  }

  if (node->IsTask()) {
    auto *task = static_cast<TaskNode *>(node);
    MultiBufferAccessRewriter rewriter(buffer_info_map_);
    task->stmt = rewriter(task->stmt);
    return {DepClosure(sites_.size()), {}};
  }

  LOG(FATAL) << "Unknown IRStructure kind in BarrierBuilder";
  return {DepClosure(sites_.size()), {}};
}

void BarrierBuilder::Build(std::vector<std::shared_ptr<IRStructure>> &root) {
  std::vector<IRStructure *> nodes;
  nodes.reserve(root.size());
  for (auto &child : root)
    nodes.push_back(child.get());

  CollectAllInsertionSites(nodes);
  SortAndUniqueSites(sites_);
  AnalyzeNodeList(std::move(nodes), /*cache_key=*/nullptr,
                  /*is_loop_body=*/false, {});
  PlanFlagAllocations();
  InsertSyncPoints();

  ICHECK(loop_stack_.empty());
  ICHECK(buffer_info_map_.empty());
}

} // namespace tl
} // namespace tvm
