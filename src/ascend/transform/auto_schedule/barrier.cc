#include "./barrier.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>

#include "op/utils.h"

#include <algorithm>
#include <array>
#include <limits>
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

// ============================================================================
// Helper structs
// ============================================================================

struct SyncInsertionSite {
  IRStructure *node;
  bool at_beginning;
  int wg;
  std::string pipe;
  std::vector<const IRStructure *> chain;

  SyncInsertionSite(const BufferAccessPoint &pt, bool at_beginning, int wg,
                    const std::string &pipe)
      : node(), at_beginning(at_beginning), wg(wg), pipe(pipe), chain() {
    for (size_t idx = 0; idx < pt.path.size(); ++idx) {
      node = const_cast<IRStructure *>(pt.path[idx]);
      chain.push_back(node);
      if (node->HasConditions()) {
        break;
      }
      if (node->IsControl()) {
        const ControlNode *ctrl = static_cast<const ControlNode *>(node);
        if (ctrl->BodyHasLoopBreak()) {
          break;
        }
      }
    }
    if (!node) {
      LOG(FATAL) << "Failed to find insertion node for access points.";
    }
  }

  friend int CompareInsertionSites(const SyncInsertionSite &a_site,
                                   const SyncInsertionSite &b_site) {
    const auto &a = a_site.chain;
    const auto &b = b_site.chain;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
      if (a[i] != b[i]) {
        if (a[i]->GetStage() != b[i]->GetStage())
          return a[i]->GetStage() > b[i]->GetStage() ? -1 : +1;
        if (a[i]->GetIndex() != b[i]->GetIndex())
          return a[i]->GetIndex() < b[i]->GetIndex() ? -1 : +1;
        return a[i] < b[i] ? -1 : +1;
      }
    }
    if (a.size() < b.size()) {
      return a_site.at_beginning ? -1 : +1;
    } else if (a.size() > b.size()) {
      return b_site.at_beginning ? +1 : -1;
    } else {
      return b_site.at_beginning - a_site.at_beginning;
    }
  }

  bool operator<(const SyncInsertionSite &other) const {
    int cmp = CompareInsertionSites(*this, other);
    if (cmp != 0)
      return cmp < 0;
    if (wg != other.wg)
      return wg < other.wg;
    return pipe < other.pipe;
  }

  bool operator==(const SyncInsertionSite &other) const {
    return CompareInsertionSites(*this, other) == 0 && wg == other.wg &&
           pipe == other.pipe;
  }

  PrimExpr GetGuard() const {
    PrimExpr guard;
    auto append = [&](PrimExpr term) {
      guard = guard.defined() ? guard && term : term;
    };
    // The final chain node is the insertion node itself. Guards come only from
    // enclosing loops that we descended through to reach it.
    for (size_t i = 0; i + 1 < chain.size(); ++i) {
      const auto *node = chain[i];
      if (!node || !node->IsControl() || node->HasConditions())
        continue;
      const auto *ctrl = static_cast<const ControlNode *>(node);
      if (ctrl->BodyHasLoopBreak())
        continue;
      const ForNode *for_node = ctrl->control.get();
      PrimExpr loop_var = for_node->loop_var;
      PrimExpr min = for_node->min;
      PrimExpr extent = for_node->extent;
      PrimExpr step = for_node->step.has_value() ? for_node->step.value()
                                                 : IntImm(DataType::Int(32), 1);
      if (at_beginning) {
        append(loop_var == min);
      } else {
        append(((loop_var - min) / step + 1) * step >= extent);
      }
    }
    return guard;
  }
};

struct DepEdge {
  SyncInsertionSite src;
  SyncInsertionSite dst;
  int distance;
  bool strict;

  DepEdge(SyncInsertionSite src, SyncInsertionSite dst, int distance,
          bool strict)
      : src(std::move(src)), dst(std::move(dst)), distance(distance),
        strict(strict) {}

  bool operator<(const DepEdge &other) const {
    // Order by "tightness" so OptimizeSyncPoints processes tighter edges first:
    // smaller distance, then later producer, then earlier consumer. If a tight
    // B->C is in the closure before a wide A->D (A<=B<=C<=D), in-order pipe
    // edges derive A->D and the wide sync is dropped. SyncInsertionSite's own
    // operator< (chain position, then wg, then pipe) is the tiebreaker, keeping
    // this a total order — sites differing only in wg/pipe stay distinct.
    if (distance != other.distance)
      return distance < other.distance;
    if (other.src < src)
      return true; // later producer first
    if (src < other.src)
      return false;
    if (dst < other.dst)
      return true; // earlier consumer first
    if (other.dst < dst)
      return false;
    return strict < other.strict;
  }
};

struct SyncPoint {
  DepEdge edge;
  int num_versions;
  std::vector<For> flag_loops;
  SyncInsertionSite reuse_src;
  SyncInsertionSite reuse_dst;
  size_t allocation_index;

  SyncPoint(DepEdge edge, int num_versions, std::vector<For> flag_loops)
      : edge(std::move(edge)), num_versions(num_versions),
        flag_loops(std::move(flag_loops)), reuse_src(this->edge.src),
        reuse_dst(this->edge.dst) {}

  PrimExpr FlagIteration() const { return CalculateIterationCount(flag_loops); }

  bool operator<(const SyncPoint &other) const {
    if (edge < other.edge)
      return true;
    if (other.edge < edge)
      return false;
    if (flag_loops.size() != other.flag_loops.size())
      return flag_loops.size() < other.flag_loops.size();
    for (size_t i = 0; i < flag_loops.size(); ++i) {
      if (flag_loops[i].get() != other.flag_loops[i].get())
        return flag_loops[i].get() < other.flag_loops[i].get();
    }
    return num_versions < other.num_versions;
  }
};

using DepClosure = std::set<DepEdge>;

struct FlagAllocation {
  std::vector<size_t> sync_points;
  int num_versions;
  bool cross_iter;
  int base_id;

  FlagAllocation(size_t sync_point, int num_versions, bool cross_iter,
                 int base_id)
      : sync_points{sync_point}, num_versions(num_versions),
        cross_iter(cross_iter), base_id(base_id) {}
};

constexpr int kIntraCoreFlagLimit = 8;
constexpr int kCrossCoreFlagLimit = 16;

std::vector<SyncPoint>
CollectSyncPoints(const std::vector<IRStructure *> &nodes, int num_wgs,
                  const MultiVersionBufferMap &buffer_info_map,
                  const ManualMultiBufferTable &manual = {},
                  const std::map<Buffer, int64_t> &buffer_num_versions = {},
                  const std::vector<For> &loop_stack = {}) {
  auto deps = AnalyzeDependencies(nodes, loop_stack, manual);

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
    auto buffer_info_it = buffer_info_map.find(dep.buffer);
    if (buffer_info_it != buffer_info_map.end()) {
      const auto &info = buffer_info_it->second;
      num_versions = info.num_versions;
      flag_loops = info.claim_loops;
    } else {
      auto it = manual.find(dep.buffer->data.get());
      if (it != manual.end()) {
        if (dep.distance > 0) {
          num_versions = dep.distance;
        } else {
          num_versions = it->second;
        }
        flag_loops = loop_stack;
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
          SyncInsertionSite prod_site(ppt, false, prod_wg, prod_pipe);
          SyncInsertionSite cons_site(cpt, true, cons_wg, cons_pipe);
          sync_points.emplace_back(DepEdge(std::move(prod_site),
                                           std::move(cons_site), distance,
                                           true),
                                   num_versions, flag_loops);
        }
      }
    }
  }
  return sync_points;
}

static void AddPromotedChildClosure(DepClosure &out, const IRStructure *child,
                                    const DepClosure &child_closure) {
  if (child->HasConditions())
    return;
  auto promote = [&](const SyncInsertionSite &site) {
    SyncInsertionSite promoted = site;
    promoted.chain.insert(promoted.chain.begin(), child);
    return promoted;
  };
  for (const auto &edge : child_closure) {
    out.emplace(promote(edge.src), promote(edge.dst), edge.distance,
                edge.strict);
  }
}

static void SaturateEdge(DepClosure &closure, const DepEdge &start,
                         int max_distance) {
  std::vector<DepEdge> work{start};
  while (!work.empty()) {
    DepEdge edge = work.back();
    work.pop_back();
    if (edge.distance > max_distance || closure.count(edge))
      continue;

    std::vector<DepEdge> existing(closure.begin(), closure.end());
    closure.insert(edge);
    for (const auto &other : existing) {
      if (edge.dst == other.src) {
        work.emplace_back(edge.src, other.dst, edge.distance + other.distance,
                          edge.strict || other.strict);
      }
      if (other.dst == edge.src) {
        work.emplace_back(other.src, edge.dst, other.distance + edge.distance,
                          other.strict || edge.strict);
      }
    }
  }
}

static void CollectTaskSites(IRStructure *node, int num_wgs,
                             std::vector<const IRStructure *> &path,
                             std::set<SyncInsertionSite> &sites) {
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
      for (int i = 0; i < num_wgs; ++i)
        wgs.push_back(i);
    } else if (wg >= 0 && wg < num_wgs) {
      wgs.push_back(wg);
    }
    for (bool at_beginning : {false, true}) {
      for (int w : wgs) {
        sites.emplace(pt, at_beginning, w, pipe);
      }
    }
    return;
  }
  auto *ctrl = static_cast<ControlNode *>(node);
  path.push_back(node);
  for (const auto &child : ctrl->children) {
    CollectTaskSites(child.get(), num_wgs, path, sites);
  }
  path.pop_back();
}

static void AddSamePipeEdges(DepClosure &closure,
                             const std::vector<IRStructure *> &nodes,
                             int num_wgs, int max_distance) {
  std::set<SyncInsertionSite> sites;
  std::vector<const IRStructure *> path;
  for (auto *node : nodes) {
    CollectTaskSites(node, num_wgs, path, sites);
  }
  std::map<std::pair<int, std::string>, std::vector<SyncInsertionSite>> by_pipe;
  for (const auto &site : sites) {
    by_pipe[{site.wg, site.pipe}].push_back(site);
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
    for (const auto &a : pipe_sites) {
      for (const auto &b : pipe_sites) {
        int cmp = CompareInsertionSites(a, b);
        if (cmp == 0)
          continue;
        bool a_before_b = cmp < 0;
        int min_d = (level_stage(b) - level_stage(a)) + (a_before_b ? 0 : 1);
        if (min_d < 0)
          min_d = 0;
        for (int d = min_d; d <= max_distance; ++d) {
          SaturateEdge(closure, DepEdge(a, b, d, false), max_distance);
        }
      }
    }
  }
}

static void OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                               DepClosure &closure, int max_distance) {
  std::stable_sort(sync_points.begin(), sync_points.end());
  std::vector<SyncPoint> emit;
  for (auto &sp : sync_points) {
    if (!closure.count(sp.edge)) {
      SaturateEdge(closure, sp.edge, max_distance);
      emit.push_back(std::move(sp));
    }
  }
  sync_points = std::move(emit);
}

static void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                                 const DepClosure &closure,
                                 const std::vector<For> &loop_stack) {
  auto same_loop_nest = [&](const std::vector<For> &loops) {
    if (loops.size() != loop_stack.size())
      return false;
    for (size_t i = 0; i < loops.size(); ++i) {
      if (loops[i].get() != loop_stack[i].get())
        return false;
    }
    return true;
  };
  for (auto &sp : sync_points) {
    if (sp.edge.src.pipe == sp.edge.dst.pipe)
      continue;
    bool is_innermost = same_loop_nest(sp.flag_loops);
    int innermost_num_versions = is_innermost ? sp.num_versions : 1;
    int min_distance = std::numeric_limits<int>::max();
    for (const auto &edge : closure) {
      if (edge.src == sp.edge.dst && edge.dst == sp.edge.src) {
        min_distance = std::min(min_distance, edge.distance);
      }
    }
    // If the loop trip count is no larger than the number of flag versions, the
    // flag is never reused across iterations, so an absent/loose reverse
    // ordering is harmless — no version bound is needed.
    if (!loop_stack.empty()) {
      const ForNode *inner = loop_stack.back().get();
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
        LOG(WARNING) << "cross-iteration flag " << sp.edge.src.pipe << "->"
                     << sp.edge.dst.pipe
                     << ": the synchronization ring is malformed.";
      }
      continue;
    }
    if (min_distance > innermost_num_versions) {
      LOG(WARNING) << "flag " << sp.edge.src.pipe << "->" << sp.edge.dst.pipe
                   << ": reverse ordering too loose to bound flag reuse.";
      continue;
    }
    if (min_distance <= 0) {
      LOG(WARNING) << "flag " << sp.edge.src.pipe << "->" << sp.edge.dst.pipe
                   << ": dependency cycle (reverse distance <= 0).";
      continue;
    }
    if (is_innermost) {
      sp.num_versions = min_distance;
    }
  }
}

static bool IsCrossCore(const SyncPoint &sp) {
  return sp.edge.src.wg != sp.edge.dst.wg;
}

static bool UsesFlag(const SyncPoint &sp) {
  return IsCrossCore(sp) || sp.edge.src.pipe != sp.edge.dst.pipe;
}

static bool HasSameFlagKind(const SyncPoint &a, const SyncPoint &b) {
  return a.edge.src.wg == b.edge.src.wg && a.edge.src.pipe == b.edge.src.pipe &&
         a.edge.dst.wg == b.edge.dst.wg && a.edge.dst.pipe == b.edge.dst.pipe;
}

static std::string FlagDomain(const SyncPoint &sp) {
  if (IsCrossCore(sp))
    return "Cube-Vector";
  return GetHardEvent(sp.edge.src.pipe, sp.edge.dst.pipe);
}

static bool HasClosureEdge(const DepClosure &closure,
                           const SyncInsertionSite &src,
                           const SyncInsertionSite &dst, int distance) {
  return closure.count(DepEdge(src, dst, distance, false)) ||
         closure.count(DepEdge(src, dst, distance, true));
}

static bool CanReuse(const SyncPoint &a, const SyncPoint &b,
                     const DepClosure &closure, bool allow_cross_iter) {
  if (!UsesFlag(a) || !UsesFlag(b) || !HasSameFlagKind(a, b))
    return false;

  bool a_cross_iter = a.edge.distance > 0;
  bool b_cross_iter = b.edge.distance > 0;
  if (a_cross_iter != b_cross_iter)
    return false;

  if (!a_cross_iter) {
    bool a_before_b = HasClosureEdge(closure, a.reuse_dst, b.reuse_src, 0) &&
                      HasClosureEdge(closure, b.reuse_dst, a.reuse_src, 1);
    bool b_before_a = HasClosureEdge(closure, b.reuse_dst, a.reuse_src, 0) &&
                      HasClosureEdge(closure, a.reuse_dst, b.reuse_src, 1);
    return a_before_b != b_before_a;
  }

  if (!allow_cross_iter)
    return false;

  bool a_before_b = HasClosureEdge(closure, a.reuse_src, b.reuse_dst, 0) &&
                    HasClosureEdge(closure, b.reuse_src, a.reuse_dst, 1);
  bool b_before_a = HasClosureEdge(closure, b.reuse_src, a.reuse_dst, 0) &&
                    HasClosureEdge(closure, a.reuse_src, b.reuse_dst, 1);
  return a_before_b != b_before_a;
}

static void PromoteReuseSites(SyncPoint &sp, IRStructure *node) {
  if (!node->HasConditions()) {
    sp.reuse_src.chain.insert(sp.reuse_src.chain.begin(), node);
    sp.reuse_dst.chain.insert(sp.reuse_dst.chain.begin(), node);
    return;
  }

  bool cross_iter = sp.edge.distance > 0;
  auto collapse = [&](SyncInsertionSite &site, bool at_beginning) {
    site.node = node;
    site.at_beginning = at_beginning;
    site.chain = {node};
  };
  collapse(sp.reuse_src, !cross_iter);
  collapse(sp.reuse_dst, cross_iter);
}

static void RecordReusablePairs(
    const std::vector<size_t> &a_points, const std::vector<size_t> &b_points,
    const DepClosure &closure, const std::vector<SyncPoint> &sync_points,
    bool allow_cross_iter,
    std::set<std::pair<size_t, size_t>> &reusable_pairs) {
  for (const auto &a : a_points) {
    for (const auto &b : b_points) {
      if (a == b ||
          !CanReuse(sync_points[a], sync_points[b], closure, allow_cross_iter))
        continue;
      reusable_pairs.emplace(std::min(a, b), std::max(a, b));
    }
  }
}

static std::vector<FlagAllocation> PlanFlagAllocations(
    std::vector<SyncPoint> &sync_points,
    const std::set<std::pair<size_t, size_t>> &reusable_pairs,
    const CrossCoreFlagReservation &cross_core_flag_reservation) {
  std::vector<FlagAllocation> allocations;
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

  ICHECK(!needs_cross_core_flag || !cross_core_flag_reservation.has_dynamic_id)
      << "Ascend AutoSchedule cannot safely allocate cross-core flag IDs when "
         "an explicit mode-0 CrossCoreSetFlag/WaitFlag uses a dynamic flag_id; "
         "use a constant flag_id or disable AutoSchedule";

  // Mode-4 synchronization addresses the same 16 base slots as mode 0. The
  // generated Cube side may additionally address slot + 16 for the second
  // Vector core, so normalize explicit IDs and reserve only the actual slots.
  std::array<bool, kCrossCoreFlagLimit> cross_core_slots{};
  for (int64_t explicit_id : cross_core_flag_reservation.explicit_ids) {
    int normalized_id = static_cast<int>(explicit_id % kCrossCoreFlagLimit);
    if (normalized_id < 0)
      normalized_id += kCrossCoreFlagLimit;
    cross_core_slots[normalized_id] = true;
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
          if (allocation.cross_iter != cross_iter ||
              !HasSameFlagKind(sp, sync_points[allocation.sync_points.front()]))
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
  return allocations;
}

static void AppendCrossIterFlagInfo(const std::vector<SyncPoint> &sync_points,
                                    const FlagAllocation &allocation,
                                    FlagInfoList &flag_infos) {
  const SyncPoint &sp = sync_points[allocation.sync_points.front()];
  std::string hard_event = GetHardEvent(sp.edge.src.pipe, sp.edge.dst.pipe);
  size_t pos = hard_event.find('_');
  std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
  std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
  int base_id = allocation.base_id;

  if (!IsCrossCore(sp)) {
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos.push_back(
          FlagInfo{sp.edge.src.wg, Integer(base_id + version), hard_event, 0});
    }
    return;
  }

  // Cross-core set uses the producer pipe; wait uses the consumer pipe.
  if (!sp.edge.src.wg) {
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos.push_back(
          FlagInfo{0, Integer(base_id + version), prod_pipe, 1});
    }
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos.push_back(
            FlagInfo{1, Integer(base_id + sid * 16 + version), cons_pipe, 2});
      }
    }
  } else {
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos.push_back(
            FlagInfo{1, Integer(base_id + sid * 16 + version), prod_pipe, 1});
      }
    }
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos.push_back(
          FlagInfo{0, Integer(base_id + version), cons_pipe, 2});
    }
  }
}

static void InsertSyncPoints(const std::vector<SyncPoint> &sync_points,
                             const std::vector<FlagAllocation> &allocations,
                             FlagInfoList &flag_infos) {
  for (const auto &allocation : allocations) {
    if (allocation.cross_iter)
      AppendCrossIterFlagInfo(sync_points, allocation, flag_infos);
  }

  for (const auto &sp : sync_points) {
    std::string hard_event = GetHardEvent(sp.edge.src.pipe, sp.edge.dst.pipe);
    size_t pos = hard_event.find('_');
    std::string prod_pipe = "PIPE_" + hard_event.substr(0, pos);
    std::string cons_pipe = "PIPE_" + hard_event.substr(pos + 1);
    PrimExpr prod_guard = sp.edge.src.GetGuard();
    PrimExpr cons_guard = sp.edge.dst.GetGuard();
    PrimExpr flag_iteration = sp.FlagIteration();

    if (!IsCrossCore(sp) && prod_pipe == cons_pipe) {
      // The scalar, vector, and cube pipes already execute in program order.
      if (prod_pipe == "PIPE_S" || prod_pipe == "PIPE_V" ||
          prod_pipe == "PIPE_M") {
        continue;
      }
      Stmt barrier = MakeAscendPipeBarrier(prod_pipe);
      InsertStatementIntoScheduledNode(sp.edge.dst.node, barrier, true,
                                       sp.edge.dst.wg, cons_guard);
      continue;
    }

    const FlagAllocation &allocation = allocations[sp.allocation_index];
    int base_id = allocation.base_id;
    PrimExpr event_id =
        indexmod(flag_iteration, Integer(sp.num_versions)) + base_id;

    if (IsCrossCore(sp)) {
      if (!sp.edge.src.wg) {
        Stmt set_stmt = MakeAscendCrossCoreSetFlag(4, prod_pipe, event_id);
        InsertStatementIntoScheduledNode(sp.edge.src.node, set_stmt, false,
                                         sp.edge.src.wg, prod_guard);
        for (int sid = 0; sid < 2; ++sid) {
          Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(
              4, cons_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatementIntoScheduledNode(sp.edge.dst.node, wait_stmt, true,
                                           sp.edge.dst.wg, cons_guard);
        }
      } else {
        for (int sid = 0; sid < 2; ++sid) {
          Stmt set_stmt = MakeAscendCrossCoreSetFlag(
              4, prod_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatementIntoScheduledNode(sp.edge.src.node, set_stmt, false,
                                           sp.edge.src.wg, prod_guard);
        }
        Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(4, cons_pipe, event_id);
        InsertStatementIntoScheduledNode(sp.edge.dst.node, wait_stmt, true,
                                         sp.edge.dst.wg, cons_guard);
      }
      continue;
    }

    Stmt set_stmt = MakeAscendSetFlag(hard_event, event_id);
    InsertStatementIntoScheduledNode(sp.edge.src.node, set_stmt, false,
                                     sp.edge.src.wg, prod_guard);
    Stmt wait_stmt = MakeAscendWaitFlag(hard_event, event_id);
    InsertStatementIntoScheduledNode(sp.edge.dst.node, wait_stmt, true,
                                     sp.edge.dst.wg, cons_guard);
  }
}

static DepClosure DistanceZeroClosure(const DepClosure &closure) {
  DepClosure result;
  for (const auto &edge : closure) {
    if (edge.distance == 0)
      result.insert(edge);
  }
  return result;
}

static DepClosure
AnalyzeNode(IRStructure *node, std::vector<SyncPoint> &sync_points,
            std::set<std::pair<size_t, size_t>> &reusable_pairs,
            std::vector<For> &loop_stack,
            std::vector<MultiVersionBufferInfo> &buffer_infos,
            MultiVersionBufferMap &buffer_info_map,
            const ManualMultiBufferTable &manual,
            std::vector<size_t> &visible_sync_points);

static DepClosure
AnalyzeNodeList(std::vector<IRStructure *> &nodes,
                std::vector<SyncPoint> &sync_points,
                std::set<std::pair<size_t, size_t>> &reusable_pairs,
                std::vector<For> &loop_stack,
                std::vector<MultiVersionBufferInfo> &buffer_infos,
                MultiVersionBufferMap &buffer_info_map,
                const ManualMultiBufferTable &manual, bool is_loop_body,
                const std::map<Buffer, int64_t> &buffer_num_versions,
                std::vector<size_t> &visible_sync_points) {
  DepClosure closure;
  std::vector<std::vector<size_t>> child_sync_points;
  for (auto *node : nodes) {
    std::vector<size_t> child_points;
    DepClosure child_closure =
        AnalyzeNode(node, sync_points, reusable_pairs, loop_stack, buffer_infos,
                    buffer_info_map, manual, child_points);
    AddPromotedChildClosure(closure, node, child_closure);
    child_sync_points.push_back(std::move(child_points));
  }

  if (is_loop_body) {
    // Sort nodes by stage (largest first) to match SW-pipelining order — the
    // latest stage produces a value first across iterations.
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](IRStructure *a, IRStructure *b) {
                       return a->GetStage() > b->GetStage();
                     });
  }

  auto local_sync_points = CollectSyncPoints(nodes, 2, buffer_info_map, manual,
                                             buffer_num_versions, loop_stack);
  int max_distance = loop_stack.empty() ? 0 : 1;
  for (const auto &sp : local_sync_points) {
    max_distance = std::max(max_distance, sp.edge.distance);
  }
  AddSamePipeEdges(closure, nodes, 2, max_distance);
  OptimizeSyncPoints(local_sync_points, closure, max_distance);
  if (is_loop_body)
    OptimizeFlagVersions(local_sync_points, closure, loop_stack);

  std::vector<size_t> local_points;
  local_points.reserve(local_sync_points.size());
  for (auto &sp : local_sync_points) {
    size_t index = sync_points.size();
    local_points.push_back(index);
    sync_points.push_back(std::move(sp));
  }

  for (size_t i = 0; i < child_sync_points.size(); ++i) {
    for (size_t j = i + 1; j < child_sync_points.size(); ++j) {
      RecordReusablePairs(child_sync_points[i], child_sync_points[j], closure,
                          sync_points, true, reusable_pairs);
    }
    RecordReusablePairs(child_sync_points[i], local_points, closure,
                        sync_points, false, reusable_pairs);
    visible_sync_points.insert(visible_sync_points.end(),
                               child_sync_points[i].begin(),
                               child_sync_points[i].end());
  }
  RecordReusablePairs(local_points, local_points, closure, sync_points, false,
                      reusable_pairs);
  visible_sync_points.insert(visible_sync_points.end(), local_points.begin(),
                             local_points.end());
  return DistanceZeroClosure(closure);
}

static DepClosure
AnalyzeNode(IRStructure *node, std::vector<SyncPoint> &sync_points,
            std::set<std::pair<size_t, size_t>> &reusable_pairs,
            std::vector<For> &loop_stack,
            std::vector<MultiVersionBufferInfo> &buffer_infos,
            MultiVersionBufferMap &buffer_info_map,
            const ManualMultiBufferTable &manual,
            std::vector<size_t> &visible_sync_points) {
  if (!node)
    return {};

  if (node->IsControl()) {
    auto *ctrl = static_cast<ControlNode *>(node);
    if (ctrl->children.empty())
      return {};
    if (!ctrl->control.defined())
      return {};
    loop_stack.push_back(ctrl->control);

    // Create new buffers for multi-buffer before recursing so nested accesses
    // can use the claim-level iteration for both data and flags.
    std::vector<Buffer> scoped_buffers;
    std::set<const VarNode *> scoped_storages;
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
      MultiVersionBufferInfo info(buffer, num_versions, new_buffer, loop_stack);
      buffer_info_map.emplace(buffer, info);
      buffer_infos.push_back(std::move(info));
      scoped_buffers.push_back(buffer);
      scoped_storages.insert(buffer->data.get());
    }
    // View/reshape aliases share storage (->data) with a multi-buffered write
    // buffer; give them the same versioning so their accesses stay consistent.
    for (auto &region : ctrl->GetReadRegions()) {
      const auto &buffer = region.get()->buffer;
      if (buffer_info_map.find(buffer) != buffer_info_map.end())
        continue;
      if (!scoped_storages.count(buffer->data.get()))
        continue;
      auto it = ctrl->buffer_num_versions.find(buffer);
      if (it == ctrl->buffer_num_versions.end() || it->second == 1)
        continue;
      MultiVersionBufferInfo info(
          buffer, it->second, CreateNewBuffer(buffer, it->second), loop_stack);
      buffer_info_map.emplace(buffer, info);
      buffer_infos.push_back(std::move(info));
      scoped_buffers.push_back(buffer);
    }

    std::vector<IRStructure *> nodes;
    for (auto &child : ctrl->children)
      nodes.push_back(child.get());
    std::vector<size_t> child_points;
    DepClosure closure = AnalyzeNodeList(
        nodes, sync_points, reusable_pairs, loop_stack, buffer_infos,
        buffer_info_map, manual,
        /*is_loop_body=*/true, ctrl->buffer_num_versions, child_points);

    for (const auto &buffer : scoped_buffers)
      buffer_info_map.erase(buffer);
    loop_stack.pop_back();
    if (!ctrl->BodyHasLoopBreak()) {
      for (size_t index : child_points) {
        SyncPoint &sp = sync_points[index];
        PromoteReuseSites(sp, node);
      }
      visible_sync_points = std::move(child_points);
    }
    return closure;
  }

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    MultiBufferAccessRewriter rewriter(buffer_info_map);
    task->stmt = rewriter(task->stmt);
    return {};
  }

  LOG(FATAL);
  return {};
}

void AnalyzeAndInsertBarriers(
    std::vector<std::shared_ptr<IRStructure>> &root, FlagInfoList &flag_infos,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    MultiVersionBufferMap &buffer_info_map,
    const ManualMultiBufferTable &manual,
    const CrossCoreFlagReservation &cross_core_flag_reservation) {
  std::vector<SyncPoint> sync_points;
  std::set<std::pair<size_t, size_t>> reusable_pairs;
  std::vector<For> loop_stack;
  std::vector<IRStructure *> nodes;
  for (auto &child : root)
    nodes.push_back(child.get());
  std::vector<size_t> visible_sync_points;
  AnalyzeNodeList(nodes, sync_points, reusable_pairs, loop_stack, buffer_infos,
                  buffer_info_map, manual, /*is_loop_body=*/false, {},
                  visible_sync_points);
  std::vector<FlagAllocation> allocations = PlanFlagAllocations(
      sync_points, reusable_pairs, cross_core_flag_reservation);
  InsertSyncPoints(sync_points, allocations, flag_infos);
}

} // namespace tl
} // namespace tvm
