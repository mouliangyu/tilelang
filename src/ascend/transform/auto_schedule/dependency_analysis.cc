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
 * \file dependency_analysis.cc
 * \brief Shared schedule and synchronization dependency analysis.
 */

#include "./dependency_analysis.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;

using TaskAccessPredicate = std::function<bool(const TaskNode *task)>;
using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

using CoveredTaskPair = std::pair<const TaskNode *, const TaskNode *>;
using CoveredDependencyMap =
    std::unordered_map<Var, std::set<CoveredTaskPair>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

PrimExpr CalculateIterationCount(const std::vector<For> &loops) {
  PrimExpr total_iter = IntImm(DataType::Int(32), 0);
  PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);
  for (size_t i = loops.size(); i-- > 0;) {
    const ForNode *for_node = loops[i].get();
    PrimExpr step = for_node->step.has_value() ? for_node->step.value()
                                               : IntImm(DataType::Int(32), 1);
    total_iter =
        total_iter +
        indexdiv(for_node->loop_var - for_node->min, step) * total_multiplier;
    total_multiplier = total_multiplier * for_node->extent;
  }
  return total_iter;
}

// BufferAccessPoint::task — defined here so the inline `static_cast` sees the
// full TaskNode definition via ir_structure.h.
const TaskNode *BufferAccessPoint::task() const {
  ICHECK(!path.empty() && path.back() && path.back()->IsTask());
  return static_cast<const TaskNode *>(path.back());
}

// ============================================================================
// Dependency analysis implementation
// ============================================================================

inline bool RegionUsesStorage(const BufferRegion &region, const Var &storage) {
  return region->buffer->data.same_as(storage);
}

TaskAccessPredicate MakeStorageAccessPredicate(const Var &storage,
                                               bool is_write) {
  return [storage, is_write](const TaskNode *task) -> bool {
    const auto &regions =
        is_write ? task->GetWriteRegions() : task->GetReadRegions();
    for (const auto &region : regions) {
      if (RegionUsesStorage(region, storage))
        return true;
    }
    return false;
  };
}

std::vector<BufferRegion>
TaskRegionsForStorage(const TaskNode *task, const Var &storage, bool is_write) {
  std::vector<BufferRegion> result;
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  for (const auto &region : regions) {
    if (RegionUsesStorage(region, storage)) {
      result.push_back(region);
    }
  }
  return result;
}

// Collect access points for every task touching `storage`. Unlike the old
// first/last-only collection, this gathers ALL matching points; the caller
// pairs producer/consumer points and lets the closure prune redundancy.
// `producer_side` shifts a loop body's timing to its last iteration so a
// producer's latency is measured from the final write.
static void CollectAllAccessPointsImpl(const IRStructure *node,
                                       const TaskAccessPredicate &pred,
                                       std::set<BufferAccessPoint> &result,
                                       std::vector<const IRStructure *> &path,
                                       int64_t time_offset, bool producer_side,
                                       const DependencyTiming *timing) {
  if (!node)
    return;

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    if (pred(task)) {
      std::vector<const IRStructure *> point_path = path;
      point_path.push_back(node);
      result.emplace(time_offset, std::move(point_path));
    }
    return;
  }

  const ControlNode *ctrl = static_cast<const ControlNode *>(node);
  path.push_back(node);

  int64_t body_offset = time_offset;
  if (producer_side) {
    int64_t trip_count = ctrl->GetTripCount();
    int64_t ii_per_iter = timing ? timing->IIperIter(ctrl) : 0;
    body_offset += (trip_count > 0 ? (trip_count - 1) : 0) * ii_per_iter;
  }

  for (const auto &child : ctrl->children) {
    int64_t start_time = timing ? timing->StartTime(child.get()) : 0;
    CollectAllAccessPointsImpl(child.get(), pred, result, path,
                               body_offset + start_time, producer_side, timing);
  }

  path.pop_back();
}

std::set<BufferAccessPoint>
CollectAccessPoints(const IRStructure *node, const Var &storage, bool is_write,
                    bool producer_side, const DependencyTiming *timing) {
  std::set<BufferAccessPoint> result;
  std::vector<const IRStructure *> path;
  CollectAllAccessPointsImpl(node,
                             MakeStorageAccessPredicate(storage, is_write),
                             result, path, 0, producer_side, timing);
  return result;
}

// Check user `T.assume_no_conflict` hints carried on the current (innermost)
// loop's `no_conflict` annotation. Each entry is a triple
// [a, b, IntImm(cross_code)] with cross_code -1=any / 1=cross / 0=same, where a
// and b are each a bare buffer's data Var (whole buffer, matched by storage
// key) or a concrete BufferRegion (matched by region equality). A hint matches
// when its two operands match the two queried regions (order-insensitive, since
// non-overlap is symmetric) and the cross mode agrees.
static bool DeclaredNoConflict(const BufferRegion &a_region,
                               const BufferRegion &b_region,
                               const std::vector<For> &loop_stack, bool cross) {
  if (loop_stack.empty())
    return false;
  auto ann = loop_stack.back()->annotations.Get("no_conflict");
  if (!ann.has_value())
    return false;

  // A hint operand matches a queried region: a bare buffer's data Var matches
  // any region of the same storage; a BufferRegion must match the same buffer
  // and region.
  auto operand_matches = [](const Any &hint, const BufferRegion &q) {
    if (auto var = hint.as<Var>())
      return var.value().same_as(q->buffer->data);
    auto hr = Downcast<BufferRegion>(hint);
    return hr->buffer.same_as(q->buffer) && RegionsEqual(hr->region, q->region);
  };

  for (const auto &entry : Downcast<Array<Any>>(ann.value())) {
    auto triple = Downcast<Array<Any>>(entry);
    const Any &ha = triple[0];
    const Any &hb = triple[1];
    int64_t cross_code = Downcast<IntImm>(triple[2])->value;
    if (cross_code != -1 && (cross_code == 1) != cross)
      continue;
    if ((operand_matches(ha, a_region) && operand_matches(hb, b_region)) ||
        (operand_matches(ha, b_region) && operand_matches(hb, a_region)))
      return true;
  }
  return false;
}

static bool
IsCurrentLoopMultiBufferEligible(const Var &storage,
                                 const std::vector<For> &loop_stack) {
  if (loop_stack.empty())
    return false;
  auto ann = loop_stack.back()->annotations.Get(kMultiBufferEligible);
  if (!ann.has_value())
    return false;
  for (const Var &eligible : ann.value().cast<Array<Var>>()) {
    if (eligible.same_as(storage))
      return true;
  }
  return false;
}

// Check whether a_region and b_region are provably disjoint under their
// per-side outer contexts. Cross-iteration checks normally relate the
// innermost loop variable. For a buffer claimed by the current loop's
// kMultiBufferEligible annotation, they instead relate the flattened complete
// loop nest so an outer-loop change also counts. `offset`: <0 constrains
// consumer > producer; ==0 is same iteration; >0 constrains consumer ==
// producer + offset.
bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        const std::vector<For> &loop_stack, int offset) {
  if (!a_region->buffer->data.same_as(b_region->buffer->data))
    return false;

  // Same storage but different logical buffer (T.view/T.reshape): regions may
  // differ in rank/dtype/strides, so keep the dependency conservatively.
  if (!a_region->buffer.same_as(b_region->buffer))
    return true;

  const bool cross = offset != 0;
  if (cross && loop_stack.empty())
    return false;
  if (DeclaredNoConflict(a_region, b_region, loop_stack, cross))
    return false;

  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  ana.z3_prover.SetRLimit(50000);
  Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet a_local_ctx = a_ctx;
  ConstrSet b_local_ctx = b_ctx;
  ConstrSet loop_ctx;
  if (cross) {
    bool use_complete_loop_nest =
        IsCurrentLoopMultiBufferEligible(a_region->buffer->data, loop_stack);
    Var pivot_lv = use_complete_loop_nest ? loop_stack.front()->loop_var
                                          : loop_stack.back()->loop_var;
    a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot_lv);
    b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot_lv);
    PrimExpr iteration = use_complete_loop_nest
                             ? CalculateIterationCount(loop_stack)
                             : PrimExpr(loop_stack.back()->loop_var);
    PrimExpr p = Substitute(iteration, a_sub);
    PrimExpr c = Substitute(iteration, b_sub);
    loop_ctx.AddConstr(
        offset < 0 ? (c > p) : (c == p + IntImm(iteration.dtype(), offset)));
  }
  loop_ctx.Merge(a_local_ctx).Merge(b_local_ctx).Populate(ana);

  // Freshen mutable reads (BufferLoads etc.) inline in the region bounds, per
  // side: a read written verbatim in both bounds is not provably equal across
  // the two access points (a store may sit between them), so it must become an
  // independent unknown on each side.
  FreshenMutableReads freshen_a, freshen_b;
  PrimExpr any_dim_disjoint = Bool(false);
  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr a_min = freshen_a(Substitute(a_rng[i]->min, a_sub));
    PrimExpr a_max =
        freshen_a(Substitute(a_rng[i]->min + a_rng[i]->extent - 1, a_sub));
    PrimExpr b_min = freshen_b(Substitute(b_rng[i]->min, b_sub));
    PrimExpr b_max =
        freshen_b(Substitute(b_rng[i]->min + b_rng[i]->extent - 1, b_sub));
    any_dim_disjoint = any_dim_disjoint || (a_max < b_min) || (b_max < a_min);
  }
  any_dim_disjoint = ana.Simplify(any_dim_disjoint);
  // Raw Z3 as a fallback catches the infeasible-context case that Analyzer's
  // constant-fold short-circuit misses.
  return !(ana.CanProve(any_dim_disjoint) ||
           ana.z3_prover.CanProve(any_dim_disjoint));
}

static std::vector<DepInfo> AnalyzeDependenciesImpl(
    const std::vector<IRStructure *> &nodes, const std::vector<For> &loop_stack,
    const BufferVersionMap &manual_buffer_versions,
    DependencyCache &dependency_cache, const ControlNode *cache_key,
    const DependencyTiming *timing) {
  if (cache_key) {
    auto cache_it = dependency_cache.find(cache_key);
    if (cache_it != dependency_cache.end())
      return cache_it->second;
  }

  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  std::vector<Var> storages;
  StorageSet storages_seen;
  auto add_storage = [&](const Var &storage) {
    if (storages_seen.insert(storage).second)
      storages.push_back(storage);
  };
  for (size_t i = 0; i < n; ++i) {
    for (const auto &region : nodes[i]->GetReadRegions()) {
      add_storage(region->buffer->data);
    }
    for (const auto &region : nodes[i]->GetWriteRegions()) {
      add_storage(region->buffer->data);
    }
  }

  std::function<void(IRStructure *, std::vector<For> &, CoveredDependencyMap &)>
      collect_subtree_deps = [&](IRStructure *node, std::vector<For> &stack,
                                 CoveredDependencyMap &out) {
        if (!node || !node->IsControl())
          return;
        auto *ctrl = static_cast<ControlNode *>(node);
        std::vector<IRStructure *> children;
        for (const auto &child : ctrl->children)
          children.push_back(child.get());
        stack.push_back(ctrl->control);
        auto child_deps =
            AnalyzeDependenciesImpl(children, stack, manual_buffer_versions,
                                    dependency_cache, ctrl, timing);
        for (const auto &cdep : child_deps) {
          auto &covered_pairs = out[cdep.storage];
          for (const auto &[cprod, ccons] : cdep.task_pairs) {
            covered_pairs.emplace(cprod.task(), ccons.task());
          }
        }
        for (auto *child : children)
          collect_subtree_deps(child, stack, out);
        stack.pop_back();
      };
  auto subtree_covered = [&](IRStructure *node) {
    CoveredDependencyMap covered;
    std::vector<For> stack = loop_stack;
    collect_subtree_deps(node, stack, covered);
    return covered;
  };

  for (const Var &storage : storages) {
    auto it = manual_buffer_versions.find(storage);
    int manual_versions = it == manual_buffer_versions.end() ? 0 : (*it).second;

    auto touches = [&](size_t u) {
      for (const auto &region : nodes[u]->GetReadRegions()) {
        if (RegionUsesStorage(region, storage))
          return true;
      }
      for (const auto &region : nodes[u]->GetWriteRegions()) {
        if (RegionUsesStorage(region, storage))
          return true;
      }
      return false;
    };

    for (size_t i = 0; i < n; ++i) {
      if (!touches(i))
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (i == j && loop_stack.empty())
          continue;
        if (!touches(j))
          continue;
        if (!manual_versions && loop_stack.empty() && i >= j)
          continue;

        auto a_w = CollectAccessPoints(nodes[i], storage, true,
                                       /*producer_side=*/true, timing);
        auto a_r = CollectAccessPoints(nodes[i], storage, false,
                                       /*producer_side=*/true, timing);
        auto b_r = CollectAccessPoints(nodes[j], storage, false,
                                       /*producer_side=*/false, timing);
        auto b_w = CollectAccessPoints(nodes[j], storage, true,
                                       /*producer_side=*/false, timing);

        std::set<std::pair<BufferAccessPoint, BufferAccessPoint>> pairs;
        int64_t latency = -1;
        int min_manual_dist = std::numeric_limits<int>::max();

        auto consider = [&](const std::set<BufferAccessPoint> &a_pts,
                            bool a_write,
                            const std::set<BufferAccessPoint> &b_pts,
                            bool b_write) {
          for (const auto &a : a_pts) {
            std::vector<BufferRegion> a_regions =
                TaskRegionsForStorage(a.task(), storage, a_write);
            for (const auto &b : b_pts) {
              std::vector<BufferRegion> b_regions =
                  TaskRegionsForStorage(b.task(), storage, b_write);
              for (const auto &rA : a_regions) {
                for (const auto &rB : b_regions) {
                  if (manual_versions) {
                    // Find the smallest distance d in [0, N] at which this
                    // access pair collides.
                    int dist = -1;
                    for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                      if (RegionsMayConflict(a.task()->outer_ctx, rA,
                                             b.task()->outer_ctx, rB,
                                             loop_stack, d)) {
                        dist = d;
                        break;
                      }
                    }
                    if (dist < 0)
                      continue;
                    min_manual_dist = std::min(min_manual_dist, dist);
                  } else if (!RegionsMayConflict(a.task()->outer_ctx, rA,
                                                 b.task()->outer_ctx, rB,
                                                 loop_stack, i >= j ? -1 : 0)) {
                    continue;
                  }
                  pairs.emplace(a, b);
                  int64_t l =
                      a.timestamp + a.task()->GetLatency() - b.timestamp;
                  if (l < 0)
                    l = 0;
                  if (l > latency)
                    latency = l;
                }
              }
            }
          }
        };
        consider(a_w, /*a_write=*/true, b_r, /*b_write=*/false);
        consider(a_w, /*a_write=*/true, b_w, /*b_write=*/true);
        consider(a_r, /*a_write=*/false, b_w, /*b_write=*/true);

        // Self-dependency on a control node: drop pairs already covered by the
        // node's own subtree.
        if (i == j && nodes[i]->IsControl()) {
          const auto &covered = subtree_covered(nodes[i]);
          auto covered_storage = covered.find(storage);
          for (auto it = pairs.begin(); it != pairs.end();) {
            if (covered_storage != covered.end() &&
                covered_storage->second.count(
                    {it->first.task(), it->second.task()})) {
              it = pairs.erase(it);
            } else {
              ++it;
            }
          }
        }

        if (!pairs.empty() && latency >= 0) {
          int distance = manual_versions ? min_manual_dist : (i >= j ? -1 : 0);
          deps.push_back(
              {nodes[i], nodes[j], storage,
               std::vector<std::pair<BufferAccessPoint, BufferAccessPoint>>(
                   pairs.begin(), pairs.end()),
               distance, latency});
        }
      }
    }
  }
  if (cache_key)
    dependency_cache.emplace(cache_key, deps);
  return deps;
}

std::vector<DepInfo> AnalyzeDependencies(
    const std::vector<IRStructure *> &nodes, const std::vector<For> &loop_stack,
    const BufferVersionMap &manual_buffer_versions,
    DependencyCache *dependency_cache, const ControlNode *cache_key,
    const DependencyTiming *timing) {
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  return AnalyzeDependenciesImpl(nodes, loop_stack, manual_buffer_versions,
                                 cache, cache_key, timing);
}

} // namespace tl
} // namespace tvm
