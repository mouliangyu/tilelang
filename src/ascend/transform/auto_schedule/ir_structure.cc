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
 * \file ir_structure.cc
 * \brief IR structure analysis for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./ir_structure.h"
#include "op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/collector.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

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

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>();
  // Copy statement
  new_task->stmt = stmt;
  // Copy resource pipe mask
  new_task->SetPipeMask(GetPipeMask());
  new_task->SetHbmMask(GetHbmMask());
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  // Copy indirect read sets (propagated through scalar Let bindings)
  new_task->SetIndirectReads(indirect_read_regions_, indirect_read_vars_,
                             indirect_computed_);
  // Copy latency and II
  new_task->SetLatency(GetLatency());
  new_task->SetII(GetII());
  for (const auto &shape : tensor_core_shapes_) {
    new_task->AddTensorCoreShape(shape.m, shape.n, shape.k, shape.input_dtype,
                                 shape.accum_dtype);
  }
  for (const auto &info : copy_infos_) {
    new_task->AddCopyInfo(info.src_scope, info.dst_scope, info.dtype,
                          info.bytes, info.pipe_mask);
  }
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  // Copy loop_break cache
  new_task->contains_loop_break_cache_ = contains_loop_break_cache_;
  // Copy hf32_mode cache
  new_task->contains_hf32_mode_cache_ = contains_hf32_mode_cache_;
  new_task->CopyBaseMetadataFrom(*this);
  return new_task;
}

bool TaskNode::ContainsLoopBreak() const {
  // Return cached result if available
  if (contains_loop_break_cache_.has_value()) {
    return contains_loop_break_cache_.value();
  }

  // Check if this task's statement contains a loop_break call
  bool found_loop_break = false;
  PostOrderVisit(stmt, [&found_loop_break](const ObjectRef &node) {
    if (found_loop_break)
      return;
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::loop_break())) {
        found_loop_break = true;
      }
    }
  });

  contains_loop_break_cache_ = found_loop_break;
  return found_loop_break;
}

bool TaskNode::ContainsHf32Mode() const {
  // Return cached result if available
  if (contains_hf32_mode_cache_.has_value()) {
    return contains_hf32_mode_cache_.value();
  }

  // Check if this task's statement contains a set_hf32_mode call
  bool found_hf32_mode = false;
  PostOrderVisit(stmt, [&found_hf32_mode](const ObjectRef &node) {
    if (found_hf32_mode)
      return;
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::ascend_set_hf32_mode())) {
        found_hf32_mode = true;
      }
    }
  });

  contains_hf32_mode_cache_ = found_hf32_mode;
  return found_hf32_mode;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>();
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone children
  new_ctrl->children.reserve(children.size());
  for (const auto &child : children) {
    new_ctrl->children.push_back(child ? child->Clone() : nullptr);
  }
  // Copy latency and II
  new_ctrl->SetLatency(GetLatency());
  new_ctrl->SetII(GetII());
  new_ctrl->CopyBaseMetadataFrom(*this);
  return new_ctrl;
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(IRStructure *node,
                                    std::vector<TaskNodeWithContext> &all_tasks,
                                    ControlNode *current_control_node) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    TaskNodeWithContext task_ctx;
    task_ctx.task = task;
    task_ctx.control_node = current_control_node;

    all_tasks.push_back(task_ctx);
  } else if (node->IsControl()) {
    auto control = static_cast<const ControlNode *>(node);
    // When entering a control node, update the current control context
    for (const auto &child : control->children) {
      CollectAllTaskNodesWithContext(child.get(), all_tasks,
                                     const_cast<ControlNode *>(control));
    }
  } else {
    LOG(FATAL);
  }
}

inline bool WarpgroupMatches(const TaskNode *task, int wg_id) {
  int task_wg = task->GetWarpgroupId();
  return IsWarpgroupBroadcast(wg_id) || IsWarpgroupBroadcast(task_wg) ||
         wg_id == task_wg;
}

inline const VarNode *StorageKey(const Buffer &buffer) {
  return buffer->data.get();
}

inline bool RegionUsesStorage(const BufferRegion &region,
                              const VarNode *storage_key) {
  return StorageKey(region->buffer) == storage_key;
}

TaskAccessPredicate MakeStorageAccessPredicate(const VarNode *storage_key,
                                               bool is_write, int wg_id) {
  return [storage_key, is_write, wg_id](const TaskNode *task) -> bool {
    if (!WarpgroupMatches(task, wg_id))
      return false;
    const auto &regions =
        is_write ? task->GetWriteRegions() : task->GetReadRegions();
    for (const auto &region : regions) {
      if (RegionUsesStorage(region, storage_key))
        return true;
    }
    return false;
  };
}

std::vector<BufferRegion> TaskRegionsForStorage(const TaskNode *task,
                                                const VarNode *storage_key,
                                                bool is_write) {
  std::vector<BufferRegion> result;
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  for (const auto &region : regions) {
    if (RegionUsesStorage(region, storage_key)) {
      result.push_back(region);
    }
  }
  return result;
}

// Collect access points for every task touching `storage_key`. Unlike the old
// first/last-only collection, this gathers ALL matching points; the caller
// pairs producer/consumer points and lets the closure prune redundancy.
// `producer_side` shifts a loop body's timing to its last iteration so a
// producer's latency is measured from the final write.
static void CollectAllAccessPointsImpl(const IRStructure *node,
                                       const TaskAccessPredicate &pred,
                                       std::set<BufferAccessPoint> &result,
                                       std::vector<const IRStructure *> &path,
                                       int64_t time_offset,
                                       bool producer_side) {
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
    int64_t ii_per_iter = ctrl->GetIIperIter();
    body_offset += (trip_count > 0 ? (trip_count - 1) : 0) * ii_per_iter;
  }

  for (const auto &child : ctrl->children) {
    CollectAllAccessPointsImpl(child.get(), pred, result, path,
                               body_offset + child->GetStartTime(),
                               producer_side);
  }

  path.pop_back();
}

std::set<BufferAccessPoint> CollectAccessPoints(const IRStructure *node,
                                                const VarNode *storage_key,
                                                bool is_write,
                                                bool producer_side) {
  std::set<BufferAccessPoint> result;
  std::vector<const IRStructure *> path;
  CollectAllAccessPointsImpl(
      node,
      MakeStorageAccessPredicate(storage_key, is_write, kWarpgroupBroadcast),
      result, path, 0, producer_side);
  return result;
}

// ============================================================================
// Dependency analysis implementation
// ============================================================================

// Check user `T.assume_no_conflict` hints carried on the current (innermost)
// loop's `tl.no_conflict` annotation. Each entry is a triple
// [a, b, IntImm(cross_code)] with cross_code -1=any / 1=cross / 0=same, where a
// and b are each a Buffer (whole buffer, matched by storage key) or a concrete
// BufferRegion (matched by region equality). A hint matches when its two
// operands match the two queried regions (order-insensitive, since non-overlap
// is symmetric) and the cross mode agrees.
static bool DeclaredNoConflict(const BufferRegion &a_region,
                               const BufferRegion &b_region,
                               const std::vector<For> &loop_stack, bool cross) {
  if (loop_stack.empty())
    return false;
  auto ann = loop_stack.back()->annotations.Get("tl.no_conflict");
  if (!ann.has_value())
    return false;

  // A hint operand matches a queried region: a bare Buffer matches any region
  // of the same storage; a BufferRegion must match the same buffer and region.
  auto operand_matches = [](const Any &hint, const BufferRegion &q) {
    if (auto buf = hint.as<Buffer>())
      return StorageKey(buf.value()) == StorageKey(q->buffer);
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

bool RegionsMayConflict(const TaskNode *a_task, const BufferRegion &a_region,
                        const TaskNode *b_task, const BufferRegion &b_region,
                        const std::vector<For> &loop_stack, bool cross) {
  if (!a_region->buffer->data.same_as(b_region->buffer->data))
    return false;

  // Same data Var but different logical Buffer, e.g. T.view/T.reshape.  Their
  // regions may have different ranks/dtypes/strides, so do not compare them
  // dimension-by-dimension.  Conservatively keep the dependency.
  bool same_logical_buffer = a_region->buffer.same_as(b_region->buffer);
  if (!same_logical_buffer)
    return true;

  // User-declared disjointness (T.assume_no_conflict) overrides analysis.
  if (DeclaredNoConflict(a_region, b_region, loop_stack, cross))
    return false;

  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet loop_ctx;
  // Only the innermost loop (loop_stack.back()) is treated as cross-iteration:
  // the consumer runs a later iteration of that loop while all outer loops are
  // held fixed (shared var). Outer-loop reuse is analyzed at those outer
  // levels.
  const size_t last = loop_stack.empty() ? 0 : loop_stack.size() - 1;
  for (size_t k = 0; k < loop_stack.size(); ++k) {
    const ForNode *fn = loop_stack[k].get();
    Var lv = fn->loop_var;
    // A loop that never executes carries no dependency, so we may assume its
    // body runs (extent > 0).
    loop_ctx.AddConstr(fn->extent > make_const(fn->extent.dtype(), 0));
    if (!cross || k < last) {
      Var lv_s(lv->name_hint + "_s", lv.dtype());
      loop_ctx.AddConstr(lv_s, Range::FromMinExtent(fn->min, fn->extent));
      a_sub.Set(lv, lv_s);
      b_sub.Set(lv, lv_s);
    } else {
      // Loop back-edge: consumer runs d>=1 iterations after the producer.
      Var lv_p(lv->name_hint + "_p", lv.dtype());
      Var lv_c(lv->name_hint + "_c", lv.dtype());
      loop_ctx.AddConstr(lv_p, Range::FromMinExtent(fn->min, fn->extent));
      loop_ctx.AddConstr(lv_c, Range::FromMinExtent(fn->min, fn->extent));
      loop_ctx.AddConstr(lv_c > lv_p);
      a_sub.Set(lv, lv_p);
      b_sub.Set(lv, lv_c);
    }
  }
  loop_ctx.Populate(ana);
  auto rename_bindings = [](const TaskNode *task, const char *suffix,
                            Map<Var, PrimExpr> &sub) {
    if (!task)
      return;
    for (const auto &c : task->outer_ctx.constrs_) {
      if (c.kind == Constr::kBindValue || c.kind == Constr::kBindRange) {
        const Var &v = c.var;
        if (!sub.count(v)) {
          sub.Set(v, Var(v->name_hint + suffix, v.dtype()));
        }
      }
    }
  };
  rename_bindings(a_task, "_p", a_sub);
  rename_bindings(b_task, "_c", b_sub);
  if (a_task) {
    a_task->outer_ctx.Substitute(a_sub).Populate(ana);
  }
  if (b_task) {
    b_task->outer_ctx.Substitute(b_sub).Populate(ana);
  }

  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr a_min = Substitute(a_rng[i]->min, a_sub);
    PrimExpr a_max = Substitute(a_rng[i]->min + a_rng[i]->extent - 1, a_sub);
    PrimExpr b_min = Substitute(b_rng[i]->min, b_sub);
    PrimExpr b_max = Substitute(b_rng[i]->min + b_rng[i]->extent - 1, b_sub);
    try {
      a_min = ana.Simplify(a_min);
      a_max = ana.Simplify(a_max);
      b_min = ana.Simplify(b_min);
      b_max = ana.Simplify(b_max);
      if (ana.CanProve(a_max < b_min || b_max < a_min)) {
        return false; // disjoint on this dim ⇒ regions disjoint
      }
    } catch (...) {
      // conservative — treat this dim as non-disjoint
    }
  }
  return true; // no dim proved disjoint ⇒ conservatively overlap
}

std::vector<DepInfo>
AnalyzeDependencies(const std::vector<IRStructure *> &nodes,
                    const std::vector<For> &loop_stack) {
  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  std::set<const VarNode *> storage_keys;
  for (size_t i = 0; i < n; ++i) {
    for (const auto &region : nodes[i]->GetReadRegions()) {
      storage_keys.insert(StorageKey(region->buffer));
    }
    for (const auto &region : nodes[i]->GetWriteRegions()) {
      storage_keys.insert(StorageKey(region->buffer));
    }
  }

  using CoveredKey =
      std::tuple<const VarNode *, const TaskNode *, const TaskNode *>;
  std::map<IRStructure *, std::set<CoveredKey>> covered_cache;
  std::function<void(IRStructure *, std::vector<For> &, std::set<CoveredKey> &)>
      collect_subtree_deps = [&](IRStructure *node, std::vector<For> &stack,
                                 std::set<CoveredKey> &out) {
        if (!node || !node->IsControl())
          return;
        auto *ctrl = static_cast<ControlNode *>(node);
        std::vector<IRStructure *> children;
        for (const auto &child : ctrl->children)
          children.push_back(child.get());
        stack.push_back(ctrl->control);
        for (const auto &cdep : AnalyzeDependencies(children, stack)) {
          const VarNode *ckey = StorageKey(cdep.buffer);
          for (const auto &[cprod, ccons] : cdep.task_pairs) {
            out.emplace(ckey, cprod.task(), ccons.task());
          }
        }
        for (auto *child : children)
          collect_subtree_deps(child, stack, out);
        stack.pop_back();
      };
  auto subtree_covered =
      [&](IRStructure *node) -> const std::set<CoveredKey> & {
    auto cache_it = covered_cache.find(node);
    if (cache_it != covered_cache.end())
      return cache_it->second;
    std::set<CoveredKey> covered;
    std::vector<For> stack = loop_stack;
    collect_subtree_deps(node, stack, covered);
    return covered_cache.emplace(node, std::move(covered)).first->second;
  };

  for (const auto *storage_key : storage_keys) {
    auto touches = [&](size_t u) {
      for (const auto &region : nodes[u]->GetReadRegions()) {
        if (RegionUsesStorage(region, storage_key))
          return true;
      }
      for (const auto &region : nodes[u]->GetWriteRegions()) {
        if (RegionUsesStorage(region, storage_key))
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
        bool same_iter = (i < j);
        bool cross = !loop_stack.empty() && (i >= j);
        if (!same_iter && !cross)
          continue;

        auto a_w = CollectAccessPoints(nodes[i], storage_key, true,
                                       /*producer_side=*/true);
        auto a_r = CollectAccessPoints(nodes[i], storage_key, false,
                                       /*producer_side=*/true);
        auto b_r = CollectAccessPoints(nodes[j], storage_key, false,
                                       /*producer_side=*/false);
        auto b_w = CollectAccessPoints(nodes[j], storage_key, true,
                                       /*producer_side=*/false);

        std::set<std::pair<BufferAccessPoint, BufferAccessPoint>> pairs;
        std::optional<Buffer> dep_buffer;
        int64_t latency = -1;

        auto consider =
            [&](const std::set<BufferAccessPoint> &a_pts, bool a_write,
                const std::set<BufferAccessPoint> &b_pts, bool b_write) {
              for (const auto &a : a_pts) {
                std::vector<BufferRegion> a_regions =
                    TaskRegionsForStorage(a.task(), storage_key, a_write);
                for (const auto &b : b_pts) {
                  std::vector<BufferRegion> b_regions =
                      TaskRegionsForStorage(b.task(), storage_key, b_write);
                  for (const auto &rA : a_regions) {
                    for (const auto &rB : b_regions) {
                      if (!RegionsMayConflict(a.task(), rA, b.task(), rB,
                                              loop_stack, cross))
                        continue;
                      // For alias buffers that share one data Var, use the
                      // write-side logical buffer as the dependency buffer.
                      if (!dep_buffer.has_value()) {
                        dep_buffer = a_write ? rA->buffer : rB->buffer;
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
          for (auto it = pairs.begin(); it != pairs.end();) {
            if (covered.count(
                    {storage_key, it->first.task(), it->second.task()})) {
              it = pairs.erase(it);
            } else {
              ++it;
            }
          }
        }

        if (!pairs.empty() && latency >= 0) {
          ICHECK(dep_buffer.has_value())
              << "Dependency found without a representative write buffer for "
              << storage_key->name_hint;
          deps.push_back(
              {nodes[i], nodes[j], dep_buffer.value(),
               std::vector<std::pair<BufferAccessPoint, BufferAccessPoint>>(
                   pairs.begin(), pairs.end()),
               cross, latency});
        }
      }
    }
  }
  return deps;
}

} // namespace tl
} // namespace tvm
