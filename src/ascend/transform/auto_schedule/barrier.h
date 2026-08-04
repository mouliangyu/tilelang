#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "./ir_structure.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;

struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_versions;
  Buffer new_buffer;
  std::vector<For> claim_loops;

  MultiVersionBufferInfo(Buffer buffer, int num_versions, Buffer new_buffer,
                         std::vector<For> claim_loops = {})
      : buffer(buffer), num_versions(num_versions), new_buffer(new_buffer),
        claim_loops(std::move(claim_loops)) {}
};

using MultiVersionBufferMap = std::unordered_map<Buffer, MultiVersionBufferInfo,
                                                 ObjectPtrHash, ObjectPtrEqual>;

struct FlagInfo {
  int warpgroup_id;
  PrimExpr flag_var;
  std::string hardevent;
  int mode;
};

using FlagInfoList = std::vector<FlagInfo>;

struct CrossCoreFlagReservation {
  // Inclusive ranges of mode-0 flag IDs used explicitly by the source kernel.
  // Dynamic expressions contribute the ConstIntBound inferred from their
  // enclosing loops.
  std::vector<std::pair<int64_t, int64_t>> explicit_id_ranges;
  bool has_unbounded_id{false};
};

std::string GetResourcePipeName(uint16_t mask);
std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe);

struct SyncInsertionSite {
  IRStructure *node;
  bool at_beginning;
  int wg;
  std::string pipe;
  std::vector<const IRStructure *> chain;

  SyncInsertionSite(const BufferAccessPoint &pt, bool at_beginning, int wg,
                    std::string pipe)
      : node(nullptr), at_beginning(at_beginning), wg(wg),
        pipe(std::move(pipe)) {
    for (const IRStructure *path_node : pt.path) {
      node = const_cast<IRStructure *>(path_node);
      chain.push_back(node);
      if (node->HasConditions())
        break;
      if (node->IsControl() &&
          static_cast<const ControlNode *>(node)->BodyHasLoopBreak()) {
        break;
      }
    }
    if (!node)
      LOG(FATAL) << "Failed to find insertion node for access points.";
  }

  int Compare(const SyncInsertionSite &other) const {
    size_t n = std::min(chain.size(), other.chain.size());
    for (size_t i = 0; i < n; ++i) {
      if (chain[i] != other.chain[i]) {
        if (chain[i]->GetStage() != other.chain[i]->GetStage())
          return chain[i]->GetStage() > other.chain[i]->GetStage() ? -1 : +1;
        if (chain[i]->GetIndex() != other.chain[i]->GetIndex())
          return chain[i]->GetIndex() < other.chain[i]->GetIndex() ? -1 : +1;
        return chain[i] < other.chain[i] ? -1 : +1;
      }
    }
    if (chain.size() < other.chain.size())
      return at_beginning ? -1 : +1;
    if (chain.size() > other.chain.size())
      return other.at_beginning ? +1 : -1;
    return other.at_beginning - at_beginning;
  }

  bool operator<(const SyncInsertionSite &other) const {
    int cmp = Compare(other);
    if (cmp != 0)
      return cmp < 0;
    if (wg != other.wg)
      return wg < other.wg;
    return pipe < other.pipe;
  }

  bool operator==(const SyncInsertionSite &other) const {
    return Compare(other) == 0 && wg == other.wg && pipe == other.pipe;
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

using SyncInsertionSiteList = std::vector<SyncInsertionSite>;

struct DepEdge {
  size_t src;
  size_t dst;
  int distance;
  bool strict;

  DepEdge(size_t src, size_t dst, int distance, bool strict)
      : src(src), dst(dst), distance(distance), strict(strict) {}

  const SyncInsertionSite &Src(const SyncInsertionSiteList &sites) const {
    return sites[src];
  }

  const SyncInsertionSite &Dst(const SyncInsertionSiteList &sites) const {
    return sites[dst];
  }

  bool operator<(const DepEdge &other) const {
    // Process tighter edges first: smaller distance, later producer, then
    // earlier consumer. Site IDs preserve SyncInsertionSite's total order.
    if (distance != other.distance)
      return distance < other.distance;
    if (src != other.src)
      return src > other.src;
    if (dst != other.dst)
      return dst < other.dst;
    return strict < other.strict;
  }
};

struct SyncPoint {
  DepEdge edge;
  int num_versions;
  std::vector<For> flag_loops;
  size_t reuse_src;
  size_t reuse_dst;
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
      if (!flag_loops[i].same_as(other.flag_loops[i]))
        return flag_loops[i].get() < other.flag_loops[i].get();
    }
    return num_versions < other.num_versions;
  }
};

class DepClosure {
public:
  DepClosure() = default;

  explicit DepClosure(size_t num_sites)
      : outgoing_(num_sites), incoming_(num_sites) {}

  bool Contains(const DepEdge &edge) const { return present_.count(edge); }

  bool HasEdge(size_t src, size_t dst, int distance) const {
    return Contains(DepEdge(src, dst, distance, false)) ||
           Contains(DepEdge(src, dst, distance, true));
  }

  int MinDistance(size_t src, size_t dst) const {
    int result = std::numeric_limits<int>::max();
    if (src >= outgoing_.size())
      return result;
    for (const DepEdge &edge : outgoing_[src]) {
      if (edge.dst == dst)
        result = std::min(result, edge.distance);
    }
    return result;
  }

  void Insert(const DepEdge &edge) {
    if (!present_.insert(edge).second)
      return;
    ICHECK_LT(std::max(edge.src, edge.dst), outgoing_.size());
    edges_.push_back(edge);
    outgoing_[edge.src].push_back(edge);
    incoming_[edge.dst].push_back(edge);
  }

  void Saturate(const DepEdge &start, int max_distance) {
    if (start.distance > max_distance)
      return;
    std::vector<DepEdge> work;
    std::set<DepEdge> pending;
    auto enqueue = [&](const DepEdge &edge) {
      if (edge.distance <= max_distance && !Contains(edge) &&
          pending.insert(edge).second) {
        work.push_back(edge);
      }
    };
    enqueue(start);
    while (!work.empty()) {
      DepEdge edge = work.back();
      work.pop_back();
      if (Contains(edge)) {
        pending.erase(edge);
        continue;
      }
      for (const DepEdge &other : outgoing_[edge.dst]) {
        int distance = edge.distance + other.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(edge.src, other.dst, distance,
                          edge.strict || other.strict));
        }
      }
      for (const DepEdge &other : incoming_[edge.src]) {
        int distance = other.distance + edge.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(other.src, edge.dst, distance,
                          other.strict || edge.strict));
        }
      }
      pending.erase(edge);
      Insert(edge);
    }
  }

  DepClosure DistanceZero() const {
    DepClosure result(outgoing_.size());
    for (const DepEdge &edge : edges_) {
      if (edge.distance == 0)
        result.Insert(edge);
    }
    return result;
  }

  const std::vector<DepEdge> &Edges() const { return edges_; }

private:
  std::vector<DepEdge> edges_;
  std::set<DepEdge> present_;
  std::vector<std::vector<DepEdge>> outgoing_;
  std::vector<std::vector<DepEdge>> incoming_;
};

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

struct BarrierAnalysisResult {
  DepClosure closure;
  std::vector<size_t> visible_sync_points;
};

// Builds synchronization and multi-buffer metadata for a scheduled IR tree.
// Each builder instance is used for one barrier-analysis phase.
class BarrierBuilder {
public:
  explicit BarrierBuilder(
      ManualMultiBufferTable manual = {},
      CrossCoreFlagReservation cross_core_flag_reservation = {})
      : manual_(std::move(manual)),
        cross_core_flag_reservation_(std::move(cross_core_flag_reservation)) {}

  BarrierBuilder(const BarrierBuilder &) = delete;
  BarrierBuilder &operator=(const BarrierBuilder &) = delete;

  void Build(std::vector<std::shared_ptr<IRStructure>> &root);

  const FlagInfoList &GetFlagInfos() const { return flag_infos_; }
  const std::vector<MultiVersionBufferInfo> &GetBufferInfos() const {
    return buffer_infos_;
  }

private:
  void CollectInsertionSites(IRStructure *node,
                             std::vector<const IRStructure *> &path,
                             SyncInsertionSiteList &sites) const;
  void CollectAllInsertionSites(const std::vector<IRStructure *> &nodes);

  static void SortAndUniqueSites(SyncInsertionSiteList &sites) {
    std::sort(sites.begin(), sites.end());
    sites.erase(std::unique(sites.begin(), sites.end()), sites.end());
  }

  size_t FindSiteId(const SyncInsertionSite &site) const {
    auto it = std::lower_bound(sites_.begin(), sites_.end(), site);
    ICHECK(it != sites_.end() && *it == site)
        << "Synchronization insertion site was not collected";
    return std::distance(sites_.begin(), it);
  }

  std::vector<SyncPoint>
  CollectSyncPoints(const std::vector<IRStructure *> &nodes,
                    const ControlNode *cache_key,
                    const std::map<Buffer, int64_t> &buffer_num_versions);

  void AddPromotedChildClosure(DepClosure &out, const IRStructure *child,
                               const DepClosure &child_closure) const;

  std::vector<size_t>
  CollectSiteIds(const std::vector<IRStructure *> &nodes) const;

  void AddSamePipeEdges(DepClosure &closure,
                        const std::vector<size_t> &task_sites,
                        int max_distance) const;

  static void OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                                 DepClosure &closure, int max_distance);

  static bool HasSameLoopNest(const std::vector<For> &a,
                              const std::vector<For> &b) {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i].same_as(b[i]))
        return false;
    }
    return true;
  }

  void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                            const DepClosure &closure) const;

  bool IsCrossCore(const SyncPoint &sp) const {
    return sp.edge.Src(sites_).wg != sp.edge.Dst(sites_).wg;
  }

  bool UsesFlag(const SyncPoint &sp) const {
    return IsCrossCore(sp) ||
           sp.edge.Src(sites_).pipe != sp.edge.Dst(sites_).pipe;
  }

  bool HasSameFlagKind(const SyncPoint &a, const SyncPoint &b) const {
    const SyncInsertionSite &a_src = a.edge.Src(sites_);
    const SyncInsertionSite &a_dst = a.edge.Dst(sites_);
    const SyncInsertionSite &b_src = b.edge.Src(sites_);
    const SyncInsertionSite &b_dst = b.edge.Dst(sites_);
    return a_src.wg == b_src.wg && a_src.pipe == b_src.pipe &&
           a_dst.wg == b_dst.wg && a_dst.pipe == b_dst.pipe;
  }

  bool HasCompatibleFlagKind(const SyncPoint &a, const SyncPoint &b) const {
    bool a_cross_core = IsCrossCore(a);
    bool b_cross_core = IsCrossCore(b);
    if (a_cross_core || b_cross_core)
      return a_cross_core && b_cross_core;
    return HasSameFlagKind(a, b);
  }

  std::string FlagDomain(const SyncPoint &sp) const {
    if (IsCrossCore(sp))
      return "Cube-Vector";

    return GetHardEvent(sp.edge.Src(sites_).pipe, sp.edge.Dst(sites_).pipe);
  }

  bool CanReuse(const SyncPoint &a, const SyncPoint &b,
                const DepClosure &closure, bool allow_cross_iter,
                bool allow_mixed_iter) const;

  void PromoteReuseSites(SyncPoint &sp, IRStructure *node) const;

  void RecordReusablePairs(const std::vector<size_t> &a_points,
                           const std::vector<size_t> &b_points,
                           const DepClosure &closure, bool allow_cross_iter,
                           bool allow_mixed_iter);

  void PlanFlagAllocations();
  void AppendCrossIterFlagInfo(const FlagAllocation &allocation);
  void InsertSyncPoints();

  BarrierAnalysisResult AnalyzeNode(IRStructure *node);

  BarrierAnalysisResult
  AnalyzeNodeList(std::vector<IRStructure *> nodes,
                  const ControlNode *cache_key, bool is_loop_body,
                  const std::map<Buffer, int64_t> &buffer_num_versions);

  static constexpr int kNumWarpgroups = 2;
  static constexpr int kIntraCoreFlagLimit = 8;
  static constexpr int kCrossCoreFlagLimit = 16;

  ManualMultiBufferTable manual_;
  CrossCoreFlagReservation cross_core_flag_reservation_;

  FlagInfoList flag_infos_;
  std::vector<MultiVersionBufferInfo> buffer_infos_;

  MultiVersionBufferMap buffer_info_map_;
  std::vector<SyncPoint> sync_points_;
  std::set<std::pair<size_t, size_t>> reusable_pairs_;
  DependencyCache dependency_cache_;
  std::vector<For> loop_stack_;
  SyncInsertionSiteList sites_;
  std::vector<FlagAllocation> allocations_;
};

inline Stmt MakeAscendSetFlag(const std::string &hard_event,
                              PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_set_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendWaitFlag(const std::string &hard_event,
                               PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_wait_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendPipeBarrier(const std::string &pipe) {
  return Evaluate(
      Call(DataType::Handle(), ascend_pipe_barrier(), {StringImm(pipe)}));
}

inline Stmt MakeAscendCrossCoreSetFlag(int mode_id, const std::string &pipe,
                                       PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_set_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

inline Stmt MakeAscendCrossCoreWaitFlag(int mode_id, const std::string &pipe,
                                        PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_wait_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  explicit MultiBufferAccessRewriter(const MultiVersionBufferMap &multi_buffer)
      : multi_buffer_(multi_buffer) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = multi_buffer_.find(load->buffer);
    if (it != multi_buffer_.end()) {
      const auto &info = it->second;
      auto *n = load.CopyOnWrite();
      n->buffer = info.new_buffer;
      PrimExpr iteration = CalculateIterationCount(info.claim_loops);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info.num_versions)));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = multi_buffer_.find(store->buffer);
    if (it != multi_buffer_.end()) {
      const auto &info = it->second;
      auto *n = store.CopyOnWrite();
      n->buffer = info.new_buffer;
      PrimExpr iteration = CalculateIterationCount(info.claim_loops);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info.num_versions)));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        auto it = multi_buffer_.find(buffer_load->buffer);
        if (it != multi_buffer_.end()) {
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(VisitExpr(op->args[0]));
          new_args.push_back(VisitExpr(op->args[1]));
          new_args.push_back(IntImm(DataType::Int(32), 1));
          for (size_t i = 2; i < op->args.size(); i++) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          return Call(op->dtype, op->op, new_args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const MultiVersionBufferMap &multi_buffer_;
};

// Rewrites multi-buffered accesses that are outside the owning loop.
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
          auto num_versions = it->second->shape[0];
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(VisitExpr(op->args[0]));
          new_args.push_back(VisitExpr(op->args[1]));
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

} // namespace tl
} // namespace tvm
