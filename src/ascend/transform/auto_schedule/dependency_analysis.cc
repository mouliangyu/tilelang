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

using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

using CoveredDependencyMap =
    std::unordered_map<Var, std::set<DependencyTaskPair>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

inline bool RegionUsesStorage(const BufferRegion &region, const Var &storage) {
  return region->buffer->data.same_as(storage);
}

bool TaskAccessesStorage(TaskNode *task, const Var &storage, bool is_write) {
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  return std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
    return RegionUsesStorage(region, storage);
  });
}

std::vector<BufferRegion>
TaskRegionsForStorage(TaskNode *task, const Var &storage, bool is_write) {
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

// Collect every leaf task touching `storage`. Dependency consumers can recover
// paths and schedule-local timestamps from the task's parent chain.
std::vector<TaskNode *> CollectAccessTasks(IRStructure *node,
                                           const Var &storage, bool is_write) {
  std::vector<TaskNode *> result;
  CollectAllTaskNodes(node, result);
  result.erase(std::remove_if(result.begin(), result.end(),
                              [&](auto *task) {
                                return !TaskAccessesStorage(task, storage,
                                                            is_write);
                              }),
               result.end());
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
                               const BufferRegion &b_region, ControlNode *loop,
                               bool cross) {
  if (loop == nullptr)
    return false;
  auto ann = loop->control->annotations.Get("no_conflict");
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

// Check whether a_region and b_region are provably disjoint under their
// per-side outer contexts. Cross-iteration checks normally relate the
// innermost loop variable. For a buffer claimed by the current loop's
// kMultiBufferEligible annotation, they instead relate the flattened complete
// loop nest so an outer-loop change also counts. `offset`: <0 constrains
// consumer > producer; ==0 is same iteration; >0 constrains consumer ==
// producer + offset.
bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        ControlNode *loop, int offset) {
  if (!a_region->buffer->data.same_as(b_region->buffer->data))
    return false;

  // Same storage but different logical buffer (T.view/T.reshape): regions may
  // differ in rank/dtype/strides, so keep the dependency conservatively.
  if (!a_region->buffer.same_as(b_region->buffer))
    return true;

  const bool cross = offset != 0;
  if (cross && loop == nullptr)
    return false;
  if (DeclaredNoConflict(a_region, b_region, loop, cross))
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
        loop->IsMultiBufferEligible(a_region->buffer->data);
    ControlNode *pivot_loop =
        use_complete_loop_nest ? loop->GetOutermostControl() : loop;
    Var pivot_lv = pivot_loop->control->loop_var;
    a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot_lv);
    b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot_lv);
    PrimExpr iteration = use_complete_loop_nest
                             ? CalculateIterationCount(loop)
                             : PrimExpr(loop->control->loop_var);
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

std::vector<DepInfo>
AnalyzeDependencies(const std::vector<IRStructure *> &nodes, ControlNode *loop,
                    const BufferVersionMap &manual_buffer_versions,
                    DependencyCache *dependency_cache) {
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  if (loop) {
    auto cache_it = cache.find(loop);
    if (cache_it != cache.end())
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

  std::function<void(IRStructure *, CoveredDependencyMap &)>
      collect_subtree_deps = [&](IRStructure *node, CoveredDependencyMap &out) {
        if (!node || !node->IsControl())
          return;
        auto *ctrl = static_cast<ControlNode *>(node);
        std::vector<IRStructure *> children;
        for (const auto &child : ctrl->children)
          children.push_back(child.get());
        auto child_deps =
            AnalyzeDependencies(children, ctrl, manual_buffer_versions, &cache);
        for (const auto &cdep : child_deps) {
          auto &covered_pairs = out[cdep.storage];
          for (const auto &[cprod, ccons] : cdep.task_pairs)
            covered_pairs.emplace(cprod, ccons);
        }
        for (auto *child : children)
          collect_subtree_deps(child, out);
      };
  auto subtree_covered = [&](IRStructure *node) {
    CoveredDependencyMap covered;
    collect_subtree_deps(node, covered);
    return covered;
  };

  for (const Var &storage : storages) {
    auto it = manual_buffer_versions.find(storage);
    int manual_versions = it == manual_buffer_versions.end() ? 0 : (*it).second;

    struct AccessTasks {
      std::vector<TaskNode *> reads;
      std::vector<TaskNode *> writes;

      bool TouchesStorage() const { return !reads.empty() || !writes.empty(); }
    };
    std::vector<AccessTasks> accesses(n);
    for (size_t i = 0; i < n; ++i) {
      accesses[i].reads = CollectAccessTasks(nodes[i], storage, false);
      accesses[i].writes = CollectAccessTasks(nodes[i], storage, true);
    }

    for (size_t i = 0; i < n; ++i) {
      if (!accesses[i].TouchesStorage())
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (i == j && loop == nullptr)
          continue;
        if (!accesses[j].TouchesStorage())
          continue;
        if (!manual_versions && loop == nullptr && i >= j)
          continue;

        std::set<DependencyTaskPair> pairs;
        int min_manual_dist = std::numeric_limits<int>::max();

        auto consider = [&](const std::vector<TaskNode *> &a_tasks,
                            bool a_write,
                            const std::vector<TaskNode *> &b_tasks,
                            bool b_write) {
          for (TaskNode *a : a_tasks) {
            std::vector<BufferRegion> a_regions =
                TaskRegionsForStorage(a, storage, a_write);
            for (TaskNode *b : b_tasks) {
              std::vector<BufferRegion> b_regions =
                  TaskRegionsForStorage(b, storage, b_write);
              for (const auto &rA : a_regions) {
                for (const auto &rB : b_regions) {
                  if (manual_versions) {
                    // Find the smallest distance d in [0, N] at which this
                    // access pair collides.
                    int dist = -1;
                    for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                      if (RegionsMayConflict(a->outer_ctx, rA, b->outer_ctx, rB,
                                             loop, d)) {
                        dist = d;
                        break;
                      }
                    }
                    if (dist < 0)
                      continue;
                    min_manual_dist = std::min(min_manual_dist, dist);
                  } else if (!RegionsMayConflict(a->outer_ctx, rA, b->outer_ctx,
                                                 rB, loop, i >= j ? -1 : 0)) {
                    continue;
                  }
                  pairs.emplace(a, b);
                }
              }
            }
          }
        };
        consider(accesses[i].writes, /*a_write=*/true, accesses[j].reads,
                 /*b_write=*/false);
        consider(accesses[i].writes, /*a_write=*/true, accesses[j].writes,
                 /*b_write=*/true);
        consider(accesses[i].reads, /*a_write=*/false, accesses[j].writes,
                 /*b_write=*/true);

        // Self-dependency on a control node: drop pairs already covered by the
        // node's own subtree.
        if (i == j && nodes[i]->IsControl()) {
          const auto &covered = subtree_covered(nodes[i]);
          auto covered_storage = covered.find(storage);
          for (auto it = pairs.begin(); it != pairs.end();) {
            if (covered_storage != covered.end() &&
                covered_storage->second.count(*it)) {
              it = pairs.erase(it);
            } else {
              ++it;
            }
          }
        }

        if (!pairs.empty()) {
          int distance = manual_versions ? min_manual_dist : (i >= j ? -1 : 0);
          deps.push_back(
              {nodes[i], nodes[j], storage,
               std::vector<DependencyTaskPair>(pairs.begin(), pairs.end()),
               distance});
        }
      }
    }
  }
  if (loop)
    cache.emplace(loop, deps);
  return deps;
}

} // namespace tl
} // namespace tvm
