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
 * \file insert_sync.cc
 * \brief Synchronization analysis, insertion, and buffer
 * materialization.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/dependency_analysis.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "buffer_version.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;
using ffi::Map;

namespace {

using SyncInsertionMap = std::map<CoreMask, std::vector<Stmt>>;

// InsertSync records synchronization here while dependency analysis still
// refers to the original tree. Build() materializes the lists as adjacent
// TaskNodes after all sync points have been planned and inserted. Same-core
// T.PerCoreTask synchronization is instead migrated beside its candidate tasks.
class SyncInsertionExtraInfo final : public IRExtraInfo {
public:
  std::unique_ptr<IRExtraInfo> Clone() const final {
    return std::make_unique<SyncInsertionExtraInfo>(*this);
  }

  SyncInsertionMap &Before() { return before_; }
  const SyncInsertionMap &Before() const { return before_; }
  SyncInsertionMap &After() { return after_; }
  const SyncInsertionMap &After() const { return after_; }

  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    for (auto &[_, statements] : before_) {
      for (Stmt &statement : statements)
        statement = tirx::Substitute(statement, {{old_var, new_var}});
    }
    for (auto &[_, statements] : after_) {
      for (Stmt &statement : statements)
        statement = tirx::Substitute(statement, {{old_var, new_var}});
    }
  }

private:
  SyncInsertionMap before_;
  SyncInsertionMap after_;
};

SyncInsertionExtraInfo *GetSyncInsertionExtraInfo(IRStructure *node) {
  return node->GetExtraInfo<SyncInsertionExtraInfo>();
}

std::unique_ptr<IRExtraInfo> MakeSyncInsertionExtraInfo() {
  return std::make_unique<SyncInsertionExtraInfo>();
}

Stmt MakeSyncTask(const Stmt &body, CoreMask core_mask) {
  ICHECK(IsConcreteCore(core_mask));
  TaskMetadata metadata;
  metadata.core_mask = core_mask;
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  attr::kAscendTask, Integer(1), body, body->span);
}

std::shared_ptr<TaskNode> MakeSyncTaskNode(const Stmt &body, CoreMask core_mask,
                                           int stage) {
  auto task = std::make_shared<TaskNode>();
  task->stmt = MakeSyncTask(body, core_mask);
  task->SetStage(stage);
  task->SetCoreMask(core_mask);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  return task;
}

class PerCoreTaskSyncMigrator : public StmtMutator {
public:
  PerCoreTaskSyncMigrator(std::vector<Stmt> before, std::vector<Stmt> after,
                          CoreMask core_mask)
      : before_(std::move(before)), after_(std::move(after)),
        core_mask_(core_mask) {
    ICHECK(IsConcreteCore(core_mask_));
  }

  Stmt Rewrite(Stmt body) {
    Stmt result = VisitStmt(std::move(body));
    ICHECK_GT(num_candidates_, 0)
        << "Cannot find a T.Task marker in normalized T.PerCoreTask";
    return result;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kAscendTask)
      return StmtMutator::VisitStmt_(op);
    Stmt candidate_body = VisitStmt(op->body);
    Stmt candidate =
        AttrStmt(op->node, op->attr_key, op->value, candidate_body, op->span);
    if (!HasTaskBufferAccess(candidate_body))
      return candidate;
    ++num_candidates_;
    Array<Stmt> sequence;
    sequence.reserve(before_.size() + 1 + after_.size());
    for (const Stmt &stmt : before_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    sequence.push_back(std::move(candidate));
    for (const Stmt &stmt : after_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    return SeqStmt::Flatten(sequence);
  }

  std::vector<Stmt> before_;
  std::vector<Stmt> after_;
  CoreMask core_mask_;
  int num_candidates_{0};
};

std::vector<Stmt> GetCoreStatements(const SyncInsertionMap &insertions,
                                    CoreMask core_mask) {
  auto it = insertions.find(core_mask);
  return it == insertions.end() ? std::vector<Stmt>() : it->second;
}

void AppendSyncTaskNodes(std::vector<std::shared_ptr<IRStructure>> *result,
                         const SyncInsertionMap &insertions, int stage,
                         std::optional<CoreMask> migrated_core = std::nullopt) {
  for (const auto &[core_mask, statements] : insertions) {
    if (migrated_core.has_value() && core_mask == migrated_core.value())
      continue;
    for (const Stmt &statement : statements)
      result->push_back(MakeSyncTaskNode(statement, core_mask, stage));
  }
}

struct MultiBufferInfo {
  Var storage;
  int num_versions;
  ControlNode *claim_loop;

  MultiBufferInfo(Var storage, int num_versions, ControlNode *claim_loop)
      : storage(std::move(storage)), num_versions(num_versions),
        claim_loop(claim_loop) {}
};

using MultiBufferInfoMap =
    std::unordered_map<Var, MultiBufferInfo, ObjectPtrHash, ObjectPtrEqual>;
using VersionedBufferMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

struct FlagInfo {
  CoreMask core_mask;
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

std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe);

struct SyncInsertionSite {
  IRStructure *path_begin;
  IRStructure *path_end;
  bool at_beginning;
  CoreMask core;
  std::string pipe;

  SyncInsertionSite(IRStructure *path_begin, TaskNode *task, bool at_beginning,
                    CoreMask core)
      : path_begin(path_begin), path_end(nullptr), at_beginning(at_beginning),
        core(core) {
    ICHECK(path_begin != nullptr);
    ICHECK(task != nullptr);
    ICHECK(IsConcreteCore(core));
    pipe = GetResourcePipeName(task->GetPipeMask());
    for (IRStructure *path_node : task->PathFrom(path_begin)) {
      path_end = path_node;
      if (path_end->HasConditions())
        break;
      if (path_end->IsControl() &&
          static_cast<ControlNode *>(path_end)->BodyHasLoopBreak()) {
        break;
      }
    }
    ICHECK(path_end != nullptr) << "Failed to find a sync insertion node";
  }

  std::vector<IRStructure *> Path() const {
    return path_end->PathFrom(path_begin);
  }

  bool operator<(const SyncInsertionSite &other) const {
    int parent_order = CompareIRStructure(path_begin->GetParent(),
                                          other.path_begin->GetParent());
    if (parent_order != 0)
      return parent_order < 0;

    int path_end_order = CompareIRStructure(path_end, other.path_end);
    if (path_end_order == -2)
      return at_beginning;
    if (path_end_order == 2)
      return !other.at_beginning;
    if (path_end_order != 0)
      return path_end_order < 0;
    if (at_beginning != other.at_beginning)
      return at_beginning;
    if (core != other.core)
      return core < other.core;
    return pipe < other.pipe;
  }

  bool operator==(const SyncInsertionSite &other) const {
    return path_begin == other.path_begin && path_end == other.path_end &&
           at_beginning == other.at_beginning && core == other.core &&
           pipe == other.pipe;
  }

  PrimExpr GetGuard() const {
    PrimExpr guard;
    auto append = [&](PrimExpr term) {
      guard = guard.defined() ? guard && term : term;
    };
    // The path end is the insertion node itself. Guards come only from
    // enclosing loops that we descended through to reach it.
    std::vector<IRStructure *> path = Path();
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      IRStructure *node = path[i];
      if (!node || !node->IsControl() || node->HasConditions())
        continue;
      auto *ctrl = static_cast<ControlNode *>(node);
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
  ControlNode *flag_loop;
  size_t reuse_src;
  size_t reuse_dst;
  size_t allocation_index;

  SyncPoint(DepEdge edge, int num_versions, ControlNode *flag_loop)
      : edge(std::move(edge)), num_versions(num_versions), flag_loop(flag_loop),
        reuse_src(this->edge.src), reuse_dst(this->edge.dst) {}

  PrimExpr FlagIteration() const { return CalculateIterationCount(flag_loop); }

  bool operator<(const SyncPoint &other) const {
    if (edge < other.edge)
      return true;
    if (other.edge < edge)
      return false;
    int loop_order = CompareIRStructure(flag_loop, other.flag_loop);
    if (loop_order != 0)
      return loop_order < 0;
    if (num_versions != other.num_versions)
      return num_versions < other.num_versions;
    return false;
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

struct SyncAnalysisResult {
  DepClosure closure;
  std::vector<size_t> visible_sync_points;
};

// Builds synchronization and multi-buffer metadata for a scheduled IR tree.
// Each inserter instance is used for one synchronization-analysis phase.
class SyncInserter {
public:
  explicit SyncInserter(
      BufferVersionMap buffer_versions = {},
      BufferVersionMap manual_buffer_versions = {},
      CrossCoreFlagReservation cross_core_flag_reservation = {})
      : buffer_versions_(std::move(buffer_versions)),
        manual_buffer_versions_(std::move(manual_buffer_versions)),
        cross_core_flag_reservation_(std::move(cross_core_flag_reservation)) {}

  SyncInserter(const SyncInserter &) = delete;
  SyncInserter &operator=(const SyncInserter &) = delete;

  void Build(std::vector<std::shared_ptr<IRStructure>> &root);

  const FlagInfoList &GetFlagInfos() const { return flag_infos_; }
  const std::vector<MultiBufferInfo> &GetMultiBufferInfos() const {
    return multi_buffer_infos_;
  }
  const VersionedBufferMap &GetVersionedBuffers() const {
    return versioned_buffers_;
  }

private:
  void AddInsertionSites(IRStructure *path_begin, TaskNode *task,
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
  CollectSyncPoints(const std::vector<IRStructure *> &nodes, ControlNode *loop);

  std::optional<int> GetClaimedBufferVersion(ControlNode *control,
                                             const Var &storage) const;

  void AddPromotedChildClosure(DepClosure &out, IRStructure *child,
                               const DepClosure &child_closure) const;

  std::vector<size_t>
  CollectSiteIds(const std::vector<IRStructure *> &nodes) const;

  void AddSamePipeEdges(DepClosure &closure,
                        const std::vector<size_t> &task_sites,
                        int max_distance) const;

  static void OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                                 DepClosure &closure, int max_distance);

  void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                            const DepClosure &closure, ControlNode *loop) const;

  bool IsCrossCore(const SyncPoint &sp) const {
    return sp.edge.Src(sites_).core != sp.edge.Dst(sites_).core;
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
    return a_src.core == b_src.core && a_src.pipe == b_src.pipe &&
           a_dst.core == b_dst.core && a_dst.pipe == b_dst.pipe;
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
  void InsertStatement(IRStructure *node, Stmt stmt, bool at_beginning,
                       CoreMask core, const PrimExpr &guard = PrimExpr());
  void MaterializeInsertions(std::vector<std::shared_ptr<IRStructure>> &nodes,
                             ControlNode *parent);

  SyncAnalysisResult AnalyzeNode(IRStructure *node);

  SyncAnalysisResult AnalyzeNodeList(std::vector<IRStructure *> nodes,
                                     ControlNode *loop);

  static constexpr int kIntraCoreFlagLimit = 8;
  static constexpr int kCrossCoreFlagLimit = 16;

  BufferVersionMap buffer_versions_;
  BufferVersionMap manual_buffer_versions_;
  CrossCoreFlagReservation cross_core_flag_reservation_;

  FlagInfoList flag_infos_;
  std::vector<MultiBufferInfo> multi_buffer_infos_;

  MultiBufferInfoMap active_multi_buffers_;
  VersionedBufferMap versioned_buffers_;
  std::vector<SyncPoint> sync_points_;
  std::set<std::pair<size_t, size_t>> reusable_pairs_;
  DependencyCache dependency_cache_;
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

Buffer CreateVersionedBuffer(const Buffer &buffer, int num_versions) {
  ObjectPtr<BufferNode> versioned =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));
  versioned->shape.insert(versioned->shape.begin(), PrimExpr(num_versions));
  if (!versioned->strides.empty()) {
    ICHECK(versioned->strides.size() + 1 == versioned->shape.size());
    PrimExpr stride_0 = versioned->strides[0] * versioned->shape[1];
    versioned->strides.insert(versioned->strides.begin(), stride_0);
  }
  return Buffer(versioned);
}

Buffer GetOrCreateVersionedBuffer(const Buffer &buffer, int num_versions,
                                  VersionedBufferMap *versioned_buffers) {
  auto existing = versioned_buffers->find(buffer);
  if (existing != versioned_buffers->end())
    return existing->second;
  Buffer versioned = CreateVersionedBuffer(buffer, num_versions);
  versioned_buffers->emplace(buffer, versioned);
  return versioned;
}

class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(const MultiBufferInfoMap &multi_buffer_infos,
                            VersionedBufferMap *versioned_buffers)
      : multi_buffer_infos_(multi_buffer_infos),
        versioned_buffers_(versioned_buffers) {}

private:
  const MultiBufferInfo *GetMultiBufferInfo(const Buffer &buffer) const {
    auto it = multi_buffer_infos_.find(buffer->data);
    return it == multi_buffer_infos_.end() ? nullptr : &it->second;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (const MultiBufferInfo *info = GetMultiBufferInfo(load->buffer)) {
      auto *n = load.CopyOnWrite();
      n->buffer = GetOrCreateVersionedBuffer(n->buffer, info->num_versions,
                                             versioned_buffers_);
      PrimExpr iteration = CalculateIterationCount(info->claim_loop);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info->num_versions)));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (const MultiBufferInfo *info = GetMultiBufferInfo(store->buffer)) {
      auto *n = store.CopyOnWrite();
      n->buffer = GetOrCreateVersionedBuffer(n->buffer, info->num_versions,
                                             versioned_buffers_);
      PrimExpr iteration = CalculateIterationCount(info->claim_loop);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info->num_versions)));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        if (GetMultiBufferInfo(buffer_load->buffer)) {
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

  const MultiBufferInfoMap &multi_buffer_infos_;
  VersionedBufferMap *versioned_buffers_;
};

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
  AllocBufferRewriter(const MultiBufferInfoMap &multi_buffer_infos,
                      VersionedBufferMap *versioned_buffers)
      : multi_buffer_infos_(multi_buffer_infos),
        versioned_buffers_(versioned_buffers) {}

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
      auto info = multi_buffer_infos_.find(buffer->data);
      if (info != multi_buffer_infos_.end()) {
        new_alloc_buffers.push_back(GetOrCreateVersionedBuffer(
            buffer, info->second.num_versions, versioned_buffers_));
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

  const MultiBufferInfoMap &multi_buffer_infos_;
  VersionedBufferMap *versioned_buffers_;
};

class BroadcastMultiBufferRewriter : public StmtExprMutator {
public:
  BroadcastMultiBufferRewriter(const MultiBufferInfoMap &multi_buffer_infos,
                               VersionedBufferMap versioned_buffers)
      : multi_buffer_infos_(multi_buffer_infos),
        versioned_buffers_(std::move(versioned_buffers)) {
    for (const auto &[_, versioned] : versioned_buffers_)
      already_versioned_.insert(versioned);
  }

private:
  const MultiBufferInfo *GetMultiBufferInfo(const Buffer &buffer) const {
    if (already_versioned_.count(buffer))
      return nullptr;
    auto it = multi_buffer_infos_.find(buffer->data);
    return it == multi_buffer_infos_.end() ? nullptr : &it->second;
  }

  Buffer GetVersionedBuffer(const Buffer &buffer, const MultiBufferInfo &info) {
    Buffer versioned = GetOrCreateVersionedBuffer(buffer, info.num_versions,
                                                  &versioned_buffers_);
    already_versioned_.insert(versioned);
    return versioned;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (const MultiBufferInfo *info = GetMultiBufferInfo(load->buffer)) {
      auto *n = load.CopyOnWrite();
      n->buffer = GetVersionedBuffer(n->buffer, *info);
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (const MultiBufferInfo *info = GetMultiBufferInfo(store->buffer)) {
      auto *n = store.CopyOnWrite();
      n->buffer = GetVersionedBuffer(n->buffer, *info);
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        if (const MultiBufferInfo *info =
                GetMultiBufferInfo(buffer_load->buffer)) {
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(VisitExpr(op->args[0]));
          new_args.push_back(VisitExpr(op->args[1]));
          new_args.push_back(Integer(info->num_versions));
          for (size_t i = 2; i < op->args.size(); i++) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          return Call(op->dtype, op->op, new_args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const MultiBufferInfoMap &multi_buffer_infos_;
  VersionedBufferMap versioned_buffers_;
  BufferSet already_versioned_;
};

MultiBufferInfoMap BuildMultiBufferInfoMap(
    const std::vector<MultiBufferInfo> &multi_buffer_infos) {
  MultiBufferInfoMap result;
  for (const MultiBufferInfo &info : multi_buffer_infos) {
    bool inserted = result.emplace(info.storage, info).second;
    ICHECK(inserted) << "Automatic multi-buffer storage " << info.storage
                     << " is claimed by more than one loop";
  }
  return result;
}

static Stmt RewriteAllocBuffers(Stmt stmt,
                                const MultiBufferInfoMap &multi_buffer_infos,
                                VersionedBufferMap *versioned_buffers) {
  if (multi_buffer_infos.empty()) {
    return stmt;
  }
  AllocBufferRewriter rewriter(multi_buffer_infos, versioned_buffers);
  return rewriter(std::move(stmt));
}

Stmt MaterializeMultiVersionBuffers(
    Stmt stmt, const std::vector<MultiBufferInfo> &multi_buffer_infos,
    VersionedBufferMap versioned_buffers) {
  if (multi_buffer_infos.empty())
    return stmt;
  MultiBufferInfoMap multi_buffer_info_map =
      BuildMultiBufferInfoMap(multi_buffer_infos);
  stmt = RewriteAllocBuffers(std::move(stmt), multi_buffer_info_map,
                             &versioned_buffers);
  return BroadcastMultiBufferRewriter(
      multi_buffer_info_map, std::move(versioned_buffers))(std::move(stmt));
}

// Collect every mode-0 CrossCore flag range used explicitly by the source
// kernel. Track enclosing loops, flat Bind declarations, branch guards, and
// assumptions so ConstIntBound can resolve dynamic flag expressions.
class ExplicitCrossCoreFlagCollector : public TaskAwareConstrVisitor {
public:
  CrossCoreFlagReservation Collect(const Stmt &stmt) {
    operator()(stmt);
    return std::move(reservation_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects exactly 3 "
             "arguments";
      const auto *mode = op->args[0].as<IntImmNode>();
      ICHECK(mode)
          << "Ascend AutoSchedule requires cross-core mode_id to be constant 0";
      ICHECK_EQ(mode->value, 0)
          << "Ascend AutoSchedule only supports cross-core synchronization "
             "with mode_id=0; disable AutoSchedule for manually scheduled "
             "mode_id="
          << mode->value;

      const PrimExpr &flag_id = op->args[2];
      arith::Analyzer analyzer;
      GetConstrSet().Populate(analyzer);
      arith::ConstIntBound bound = analyzer.const_int_bound(flag_id);
      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_flag_id("unconstrained_cross_core_flag_id",
                                flag_id.dtype());
      arith::ConstIntBound unconstrained_bound =
          unconstrained_analyzer.const_int_bound(unconstrained_flag_id);
      if (bound->min_value == arith::ConstIntBound::kNegInf ||
          bound->max_value == arith::ConstIntBound::kPosInf ||
          (bound->min_value == unconstrained_bound->min_value &&
           bound->max_value == unconstrained_bound->max_value)) {
        reservation_.has_unbounded_id = true;
      } else {
        ICHECK_LE(bound->min_value, bound->max_value)
            << "Ascend AutoSchedule inferred an invalid explicit cross-core "
               "flag range ["
            << bound->min_value << ", " << bound->max_value << "]";
        reservation_.explicit_id_ranges.emplace_back(bound->min_value,
                                                     bound->max_value);
      }
    }
    ConstrVisitor::VisitExpr_(op);
  }

  CrossCoreFlagReservation reservation_;
};

CrossCoreFlagReservation
CollectExplicitCrossCoreFlags(const Stmt &kernel_body) {
  ExplicitCrossCoreFlagCollector collector;
  return collector.Collect(kernel_body);
}

static void AddBufferVersion(BufferVersionMap *buffer_versions, const Var &data,
                             int num_versions) {
  ICHECK_GT(num_versions, 0);
  if (num_versions == 1) {
    return;
  }
  auto existing = buffer_versions->find(data);
  if (existing != buffer_versions->end()) {
    ICHECK_EQ((*existing).second, num_versions)
        << "Conflicting manual and automatic buffer version counts for storage "
        << data << ": " << (*existing).second << " vs " << num_versions;
    return;
  }
  buffer_versions->Set(data, num_versions);
}

static BufferVersionMap BuildPhysicalBufferVersions(
    const BufferVersionMap &manual_buffer_versions,
    const std::vector<MultiBufferInfo> &automatic_multi_buffers) {
  BufferVersionMap result;
  for (const auto &[data, num_versions] : manual_buffer_versions) {
    AddBufferVersion(&result, data, num_versions);
  }
  for (const MultiBufferInfo &info : automatic_multi_buffers) {
    AddBufferVersion(&result, info.storage, info.num_versions);
  }
  return result;
}

struct SyncKernelUpdate {
  ScheduledTIR scheduled_tir;
  std::vector<MultiBufferInfo> multi_buffer_infos;
  VersionedBufferMap versioned_buffers;
};

static SyncKernelUpdate InsertKernelSync(const Stmt &kernel_body,
                                         ScheduledTIR scheduled_tir) {
  std::vector<std::shared_ptr<IRStructure>> &sync_root = scheduled_tir.tree;
  ScheduledTIRMetadata &metadata = scheduled_tir.metadata;
  CrossCoreFlagReservation cross_core_flag_reservation =
      CollectExplicitCrossCoreFlags(kernel_body);
  std::vector<TaskNode *> scheduled_tasks;
  CollectAllTaskNodes(sync_root, scheduled_tasks);
  for (TaskNode *task : scheduled_tasks) {
    ICHECK_NE(task->GetCoreMask(), kCoreUnassigned)
        << "InsertSync found a scheduled T.Task without core_mask. Run "
           "tl.transform.AssignCore immediately after AutoSchedule.";
  }
  SyncInserter inserter(metadata.buffer_versions,
                        metadata.manual_buffer_versions,
                        std::move(cross_core_flag_reservation));
  inserter.Build(sync_root);

  std::vector<std::shared_ptr<IRStructure>> prefix;
  std::vector<std::shared_ptr<IRStructure>> suffix;
  for (const FlagInfo &info : inserter.GetFlagInfos()) {
    if (info.mode == 0) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendSetFlag(info.hardevent, info.flag_var), info.core_mask, 0));
      suffix.push_back(
          MakeSyncTaskNode(MakeAscendWaitFlag(info.hardevent, info.flag_var),
                           info.core_mask, 0));
    } else if (info.mode == 1) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreSetFlag(4, info.hardevent, info.flag_var),
          info.core_mask, 0));
    } else if (info.mode == 2) {
      suffix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreWaitFlag(4, info.hardevent, info.flag_var),
          info.core_mask, 0));
    }
  }
  sync_root.insert(sync_root.begin(), prefix.begin(), prefix.end());
  sync_root.insert(sync_root.end(), suffix.begin(), suffix.end());
  for (size_t i = 0; i < sync_root.size(); ++i) {
    sync_root[i]->SetIndex(i);
    sync_root[i]->SetParent(nullptr);
  }
  const std::vector<MultiBufferInfo> &multi_buffer_infos =
      inserter.GetMultiBufferInfos();
  BufferVersionMap buffer_versions = BuildPhysicalBufferVersions(
      metadata.manual_buffer_versions, multi_buffer_infos);
  metadata.buffer_versions = {};
  metadata.manual_buffer_versions = std::move(buffer_versions);
  SyncKernelUpdate update;
  update.scheduled_tir = std::move(scheduled_tir);
  update.multi_buffer_infos = multi_buffer_infos;
  update.versioned_buffers = inserter.GetVersionedBuffers();
  return update;
}

// Synchronization analysis and insertion.

void SyncInserter::InsertStatement(IRStructure *node, Stmt stmt,
                                   bool at_beginning, CoreMask core,
                                   const PrimExpr &guard) {
  ICHECK(node != nullptr);
  ICHECK(IsConcreteCore(core));
  if (guard.defined())
    stmt = IfThenElse(guard, stmt);
  SyncInsertionExtraInfo *insertions = GetSyncInsertionExtraInfo(node);
  if (at_beginning) {
    auto &before = insertions->Before()[core];
    before.insert(before.begin(), std::move(stmt));
  } else {
    insertions->After()[core].push_back(std::move(stmt));
  }
}

void SyncInserter::MaterializeInsertions(
    std::vector<std::shared_ptr<IRStructure>> &nodes, ControlNode *parent) {
  std::vector<std::shared_ptr<IRStructure>> materialized;
  materialized.reserve(nodes.size());
  for (const std::shared_ptr<IRStructure> &node : nodes) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node.get());
      MaterializeInsertions(control->children, control);
    }

    const SyncInsertionExtraInfo *insertions =
        GetSyncInsertionExtraInfo(node.get());
    const SyncInsertionMap &before_insertions = insertions->Before();
    const SyncInsertionMap &after_insertions = insertions->After();

    std::optional<CoreMask> migrated_core;
    if (node->IsTask()) {
      auto *task = static_cast<TaskNode *>(node.get());
      if (task->IsPerCoreTask()) {
        CoreMask core_mask = task->GetCoreMask();
        ICHECK(IsConcreteCore(core_mask))
            << "Scheduled T.PerCoreTask must belong to one concrete core";
        migrated_core = core_mask;
        std::vector<Stmt> before =
            GetCoreStatements(before_insertions, core_mask);
        std::vector<Stmt> after =
            GetCoreStatements(after_insertions, core_mask);
        if (!before.empty() || !after.empty()) {
          task->stmt = PerCoreTaskSyncMigrator(std::move(before),
                                               std::move(after), core_mask)
                           .Rewrite(std::move(task->stmt));
        }
      }
    }

    AppendSyncTaskNodes(&materialized, before_insertions, node->GetStage(),
                        migrated_core);
    materialized.push_back(node);
    AppendSyncTaskNodes(&materialized, after_insertions, node->GetStage(),
                        migrated_core);
  }

  for (size_t i = 0; i < materialized.size(); ++i) {
    materialized[i]->SetIndex(i);
    materialized[i]->SetParent(parent);
  }
  nodes = std::move(materialized);
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

void SyncInserter::AddInsertionSites(IRStructure *path_begin, TaskNode *task,
                                     SyncInsertionSiteList &sites) const {
  CoreMask core_mask = task->GetCoreMask();
  for (bool at_beginning : {false, true}) {
    for (CoreMask core : kConcreteCores) {
      if (HasCore(core_mask, core))
        sites.emplace_back(path_begin, task, at_beginning, core);
    }
  }
}

void SyncInserter::CollectAllInsertionSites(
    const std::vector<IRStructure *> &nodes) {
  std::vector<TaskNode *> tasks;
  for (IRStructure *node : nodes) {
    CollectAllTaskNodes(node, tasks);
  }
  for (TaskNode *task : tasks) {
    for (IRStructure *path_begin = task; path_begin != nullptr;
         path_begin = path_begin->GetParent()) {
      AddInsertionSites(path_begin, task, sites_);
    }
  }
}

bool IsRegisterStorage(const Var &storage) {
  std::string scope = GetPtrStorageScope(storage);
  return scope == "local" || scope == "local.var" || scope == "local.fragment";
}

std::vector<SyncPoint>
SyncInserter::CollectSyncPoints(const std::vector<IRStructure *> &nodes,
                                ControlNode *loop) {
  auto deps = AnalyzeDependencies(nodes, loop, manual_buffer_versions_,
                                  &dependency_cache_);

  // Expand a task mask into the concrete cores on which it participates.
  auto expand_cores = [](CoreMask task_mask) {
    std::vector<CoreMask> out;
    for (CoreMask core : kConcreteCores) {
      if (HasCore(task_mask, core))
        out.push_back(core);
    }
    return out;
  };

  std::vector<SyncPoint> sync_points;
  for (const auto &dep : deps) {
    if (IsRegisterStorage(dep.storage)) {
      continue;
    }

    int distance;
    if (dep.distance >= 0) {
      distance = dep.distance;
    } else if (std::optional<int> num_versions =
                   GetClaimedBufferVersion(loop, dep.storage)) {
      distance = num_versions.value();
    } else {
      distance = 1;
    }
    int num_versions = 1;
    ControlNode *flag_loop = nullptr;
    auto multi_buffer = active_multi_buffers_.find(dep.storage);
    if (multi_buffer != active_multi_buffers_.end()) {
      const MultiBufferInfo &info = multi_buffer->second;
      num_versions = info.num_versions;
      flag_loop = info.claim_loop;
    } else {
      auto it = manual_buffer_versions_.find(dep.storage);
      if (it != manual_buffer_versions_.end()) {
        if (dep.distance > 0) {
          num_versions = dep.distance;
        } else {
          num_versions = static_cast<int>((*it).second);
        }
        flag_loop = loop;
      }
    }

    for (const auto &[prod_task, cons_task] : dep.task_pairs) {
      std::string prod_pipe = GetResourcePipeName(prod_task->GetPipeMask());
      std::string cons_pipe = GetResourcePipeName(cons_task->GetPipeMask());
      std::string hard_event = GetHardEvent(prod_pipe, cons_pipe);
      bool fix_unit_flag = hard_event.find("FIX") != std::string::npos &&
                           (HasUnitFlagCtrl(prod_task->stmt) ||
                            HasUnitFlagCtrl(cons_task->stmt));
      for (CoreMask prod_core : expand_cores(prod_task->GetCoreMask())) {
        for (CoreMask cons_core : expand_cores(cons_task->GetCoreMask())) {
          // Intra-core FIX unit-flag syncs are handled by the hardware.
          if (prod_core == cons_core && fix_unit_flag)
            continue;
          size_t prod_site = FindSiteId(
              SyncInsertionSite(dep.prod_node, prod_task, false, prod_core));
          size_t cons_site = FindSiteId(
              SyncInsertionSite(dep.cons_node, cons_task, true, cons_core));
          sync_points.emplace_back(
              DepEdge(prod_site, cons_site, distance, true), num_versions,
              flag_loop);
        }
      }
    }
  }
  return sync_points;
}

std::optional<int>
SyncInserter::GetClaimedBufferVersion(ControlNode *control,
                                      const Var &storage) const {
  if (control == nullptr || !control->IsMultiBufferEligible(storage))
    return std::nullopt;
  auto selected = buffer_versions_.find(storage);
  if (selected == buffer_versions_.end())
    return std::nullopt;
  return (*selected).second;
}

void SyncInserter::AddPromotedChildClosure(
    DepClosure &out, IRStructure *child,
    const DepClosure &child_closure) const {
  if (child->HasConditions() ||
      (child->IsControl() &&
       static_cast<ControlNode *>(child)->BodyHasLoopBreak()))
    return;
  auto promote = [&](const SyncInsertionSite &site) {
    SyncInsertionSite promoted = site;
    promoted.path_begin = child;
    return promoted;
  };

  for (const DepEdge &edge : child_closure.Edges()) {
    size_t src = FindSiteId(promote(edge.Src(sites_)));
    size_t dst = FindSiteId(promote(edge.Dst(sites_)));
    out.Insert(DepEdge(src, dst, edge.distance, edge.strict));
  }
}

std::vector<size_t>
SyncInserter::CollectSiteIds(const std::vector<IRStructure *> &nodes) const {
  SyncInsertionSiteList level_sites;
  for (IRStructure *node : nodes) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(node, tasks);
    for (TaskNode *task : tasks)
      AddInsertionSites(node, task, level_sites);
  }
  SortAndUniqueSites(level_sites);

  std::vector<size_t> result;
  result.reserve(level_sites.size());
  for (const SyncInsertionSite &site : level_sites)
    result.push_back(FindSiteId(site));
  return result;
}

void SyncInserter::AddSamePipeEdges(DepClosure &closure,
                                    const std::vector<size_t> &task_sites,
                                    int max_distance) const {
  std::map<std::pair<CoreMask, std::string>, std::vector<size_t>> by_pipe;
  for (size_t site_id : task_sites) {
    const SyncInsertionSite &site = sites_[site_id];
    by_pipe[{site.core, site.pipe}].push_back(site_id);
  }
  // The path start is the node carried by the loop being pipelined at this
  // level, so its stage is the site's logical iteration stage.
  auto level_stage = [](const SyncInsertionSite &s) {
    return s.path_begin->GetStage();
  };
  for (auto &[_, pipe_sites] : by_pipe) {
    // A single pipe issues its sites in physical (iteration, position) order.
    // For each ordered pair a->b, the happens-before edge a(k) < b(k+d) holds
    // for all d >= min_d, where a node at stage s in physical iteration p runs
    // logical iteration k = p - s. Solving the in-order constraint gives:
    //   min_d = (s_a - s_b) + (a physically before b ? 0 : 1),  clamped to >=0.
    // Adding every valid distance in [min_d, max_distance] seeds the closure
    // with the true pipe ordering, without assuming iterations run serially.
    for (size_t a_id : pipe_sites) {
      for (size_t b_id : pipe_sites) {
        if (a_id == b_id)
          continue;
        const SyncInsertionSite &a = sites_[a_id];
        const SyncInsertionSite &b = sites_[b_id];
        bool a_before_b = a_id < b_id;
        int min_d = (level_stage(a) - level_stage(b)) + (a_before_b ? 0 : 1);
        if (min_d < 0)
          min_d = 0;
        for (int d = min_d; d <= max_distance; ++d) {
          closure.Saturate(DepEdge(a_id, b_id, d, false), max_distance);
        }
      }
    }
  }
}

void SyncInserter::OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
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

void SyncInserter::OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                                        const DepClosure &closure,
                                        ControlNode *loop) const {
  for (auto &sp : sync_points) {
    const SyncInsertionSite &src = sp.edge.Src(sites_);
    const SyncInsertionSite &dst = sp.edge.Dst(sites_);
    if (src.pipe == dst.pipe)
      continue;
    bool is_innermost = loop == sp.flag_loop;
    int innermost_num_versions = is_innermost ? sp.num_versions : 1;
    int min_distance = closure.MinDistance(sp.edge.dst, sp.edge.src);
    // If the loop trip count is no larger than the number of flag versions, the
    // flag is never reused across iterations, so an absent/loose reverse
    // ordering is harmless — no version bound is needed.
    if (loop != nullptr) {
      const ForNode *inner = loop->control.get();
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

bool SyncInserter::CanReuse(const SyncPoint &a, const SyncPoint &b,
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
        a.num_versions != b.num_versions || a.flag_loop != b.flag_loop)
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

void SyncInserter::PromoteReuseSites(SyncPoint &sp, IRStructure *node) const {
  SyncInsertionSite src = sites_[sp.reuse_src];
  SyncInsertionSite dst = sites_[sp.reuse_dst];
  if (!node->HasConditions()) {
    src.path_begin = node;
    dst.path_begin = node;
  } else {
    bool cross_iter = sp.edge.distance > 0;
    auto collapse = [&](SyncInsertionSite &site, bool at_beginning) {
      site.path_begin = node;
      site.path_end = node;
      site.at_beginning = at_beginning;
    };
    collapse(src, !cross_iter);
    collapse(dst, cross_iter);
  }

  sp.reuse_src = FindSiteId(src);
  sp.reuse_dst = FindSiteId(dst);
}

void SyncInserter::RecordReusablePairs(const std::vector<size_t> &a_points,
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

void SyncInserter::PlanFlagAllocations() {
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

void SyncInserter::AppendCrossIterFlagInfo(const FlagAllocation &allocation) {
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
          FlagInfo{src.core, Integer(base_id + version), hard_event, 0});
    }
    return;
  }

  // Cross-core set uses the producer pipe; wait uses the consumer pipe.
  if (src.core == kCoreVector) {
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos_.push_back(
          FlagInfo{kCoreVector, Integer(base_id + version), prod_pipe, 1});
    }
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{
            kCoreCube, Integer(base_id + sid * 16 + version), cons_pipe, 2});
      }
    }
  } else {
    for (int sid = 0; sid < 2; ++sid) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{
            kCoreCube, Integer(base_id + sid * 16 + version), prod_pipe, 1});
      }
    }
    for (int version = 0; version < allocation.num_versions; ++version) {
      flag_infos_.push_back(
          FlagInfo{kCoreVector, Integer(base_id + version), cons_pipe, 2});
    }
  }
}

void SyncInserter::InsertSyncPoints() {
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
      Stmt pipe_barrier = MakeAscendPipeBarrier(prod_pipe);
      InsertStatement(dst.path_end, pipe_barrier, true, dst.core, cons_guard);
      continue;
    }

    const FlagAllocation &allocation = allocations_[sp.allocation_index];
    int base_id = allocation.base_id;
    PrimExpr event_id =
        indexmod(flag_iteration, Integer(sp.num_versions)) + base_id;

    if (IsCrossCore(sp)) {
      if (src.core == kCoreVector) {
        Stmt set_stmt = MakeAscendCrossCoreSetFlag(4, prod_pipe, event_id);
        InsertStatement(src.path_end, set_stmt, false, src.core, prod_guard);
        for (int sid = 0; sid < 2; ++sid) {
          Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(
              4, cons_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatement(dst.path_end, wait_stmt, true, dst.core, cons_guard);
        }
      } else {
        for (int sid = 0; sid < 2; ++sid) {
          Stmt set_stmt = MakeAscendCrossCoreSetFlag(
              4, prod_pipe,
              indexmod(flag_iteration, Integer(sp.num_versions)) +
                  Integer(base_id + 16 * sid));
          InsertStatement(src.path_end, set_stmt, false, src.core, prod_guard);
        }
        Stmt wait_stmt = MakeAscendCrossCoreWaitFlag(4, cons_pipe, event_id);
        InsertStatement(dst.path_end, wait_stmt, true, dst.core, cons_guard);
      }
      continue;
    }

    Stmt set_stmt = MakeAscendSetFlag(hard_event, event_id);
    InsertStatement(src.path_end, set_stmt, false, src.core, prod_guard);
    Stmt wait_stmt = MakeAscendWaitFlag(hard_event, event_id);
    InsertStatement(dst.path_end, wait_stmt, true, dst.core, cons_guard);
  }
}

SyncAnalysisResult
SyncInserter::AnalyzeNodeList(std::vector<IRStructure *> nodes,
                              ControlNode *loop) {
  SyncAnalysisResult result{DepClosure(sites_.size()), {}};
  std::vector<size_t> task_site_ids = CollectSiteIds(nodes);

  std::vector<std::vector<size_t>> child_sync_points;
  for (IRStructure *node : nodes) {
    SyncAnalysisResult child_result = AnalyzeNode(node);
    AddPromotedChildClosure(result.closure, node, child_result.closure);
    child_sync_points.push_back(std::move(child_result.visible_sync_points));
  }

  if (loop != nullptr) {
    // Sort nodes by stage (smallest first) to match SW-pipelining order — the
    // earliest stage produces a value first across iterations.
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](IRStructure *a, IRStructure *b) {
                       return a->GetStage() < b->GetStage();
                     });
  }

  std::vector<SyncPoint> local_sync_points = CollectSyncPoints(nodes, loop);
  int max_distance = loop == nullptr ? 0 : 1;
  for (const auto &sp : local_sync_points) {
    max_distance = std::max(max_distance, sp.edge.distance);
  }
  AddSamePipeEdges(result.closure, task_site_ids, max_distance);
  OptimizeSyncPoints(local_sync_points, result.closure, max_distance);
  if (loop != nullptr)
    OptimizeFlagVersions(local_sync_points, result.closure, loop);

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

SyncAnalysisResult SyncInserter::AnalyzeNode(IRStructure *node) {
  if (!node)
    return {DepClosure(sites_.size()), {}};

  if (node->IsControl()) {
    auto *ctrl = static_cast<ControlNode *>(node);
    if (ctrl->children.empty() || !ctrl->control.defined())
      return {DepClosure(sites_.size()), {}};

    // Activate one versioning record per claimed storage before recursing so
    // every Buffer alias uses the same claim-level iteration for data and
    // flags.
    std::vector<Var> scoped_storages;
    for (const BufferRegion &region : ctrl->GetWriteRegions()) {
      const Var &storage = region->buffer->data;
      if (active_multi_buffers_.count(storage))
        continue;
      std::optional<int> num_versions = GetClaimedBufferVersion(ctrl, storage);
      if (!num_versions.has_value())
        continue;
      MultiBufferInfo info(storage, num_versions.value(), ctrl);
      active_multi_buffers_.emplace(storage, info);
      multi_buffer_infos_.push_back(std::move(info));
      scoped_storages.push_back(storage);
    }

    std::vector<IRStructure *> nodes;
    for (auto &child : ctrl->children)
      nodes.push_back(child.get());
    SyncAnalysisResult result = AnalyzeNodeList(std::move(nodes), ctrl);

    for (const Var &storage : scoped_storages)
      active_multi_buffers_.erase(storage);
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
    MultiBufferAccessRewriter rewriter(active_multi_buffers_,
                                       &versioned_buffers_);
    task->stmt = rewriter(task->stmt);
    return {DepClosure(sites_.size()), {}};
  }

  LOG(FATAL) << "Unknown IRStructure kind in SyncInserter";
  return {DepClosure(sites_.size()), {}};
}

void SyncInserter::Build(std::vector<std::shared_ptr<IRStructure>> &root) {
  std::vector<IRStructure *> nodes;
  nodes.reserve(root.size());
  for (auto &child : root)
    nodes.push_back(child.get());

  CollectAllInsertionSites(nodes);
  SortAndUniqueSites(sites_);
  AnalyzeNodeList(std::move(nodes), /*loop=*/nullptr);
  PlanFlagAllocations();
  InsertSyncPoints();
  MaterializeInsertions(root, nullptr);

  ICHECK(active_multi_buffers_.empty());
}

} // namespace

// Consume AutoSchedule's scheduled TIR, insert synchronization, and materialize
// the buffer versions selected by the scheduler.
tvm::transform::Pass InsertSync() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    return RewriteTilelangKernels(
        std::move(func), "InsertSync",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.root, context.outer_ctx, MakeSyncInsertionExtraInfo);
          SyncKernelUpdate update =
              InsertKernelSync(context.root->body, std::move(scheduled_tir));
          // MultiBufferInfo keeps claim-loop pointers into this tree. Keep the
          // owning ScheduledTIR alive until buffer materialization completes.
          Stmt serialized = EncodeScheduledTIR(update.scheduled_tir);
          serialized = MaterializeMultiVersionBuffers(
              std::move(serialized), update.multi_buffer_infos,
              std::move(update.versioned_buffers));
          const auto *rewritten_root = serialized.as<SBlockNode>();
          ICHECK(rewritten_root)
              << "Multi-buffer materialization must preserve tilelang_root";
          return GetRef<SBlock>(rewritten_root);
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertSync", InsertSync);
}

} // namespace tl
} // namespace tvm
