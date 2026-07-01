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

// ============================================================================
// CollectFirstAccessPoints / CollectLastAccessPoints implementations
// ============================================================================

static bool LoopMustExecute(const ControlNode *ctrl) {
  if (!ctrl->control.defined())
    return false;
  const ForNode *for_node = ctrl->control.get();
  const int64_t *extent_ptr = as_const_int(for_node->extent);
  if (extent_ptr && *extent_ptr >= 1) {
    if (for_node->step.has_value()) {
      const int64_t *step_ptr = as_const_int(for_node->step.value());
      if (step_ptr && *step_ptr >= 1)
        return true;
      if (!step_ptr)
        return false;
    } else {
      return true;
    }
  }
  return false;
}

// An entry in the unified sequence helper. `node` is the IR-tree node to
// recurse into; `start_time` is the entry's offset relative to the enclosing
// sequence's start.
//
// Conditional execution is carried by each node's `conditions` delta rather
// than by a dedicated IfNode, so the helper dispatches entries in order;
// conditional accesses are recorded as candidates but do not terminate the
// first/last scan.
struct SequenceEntry {
  const IRStructure *node;
  int64_t start_time;
};

// A buffer access is "must happen" at its sequence position only if neither the
// accessing node nor any structural ancestor on its path is conditional. A
// conditional access (some node carries a `conditions` delta) is recorded as a
// candidate access point but must NOT terminate the first/last scan, because a
// guarded statement may be skipped at runtime — the conservative choice is to
// keep scanning so sibling / earlier / later accesses are also collected (more
// sync points, never fewer).
static bool AnyConditionalOnPath(const std::vector<const IRStructure *> &path,
                                 const IRStructure *leaf) {
  if (leaf && leaf->HasConditions())
    return true;
  for (const auto *p : path) {
    if (p && p->HasConditions())
      return true;
  }
  return false;
}

static bool SequenceCollectFirstAccessPoints(
    const std::vector<SequenceEntry> &entries, const TaskAccessPredicate &pred,
    std::set<BufferAccessPoint> &result, std::vector<const IRStructure *> &path,
    int64_t base_offset) {
  int n = static_cast<int>(entries.size());

  auto dispatch = [&](const SequenceEntry &e) -> bool {
    bool m = e.node->CollectFirstAccessPoints(pred, result, path,
                                              base_offset + e.start_time);
    return m;
  };

  int break_idx = -1;
  for (int j = 0; j < n; ++j) {
    if (entries[j].node->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx > 0) {
      std::vector<SequenceEntry> before(entries.begin(),
                                        entries.begin() + break_idx);
      if (SequenceCollectFirstAccessPoints(before, pred, result, path,
                                           base_offset))
        return true;
    }
    if (dispatch(entries[break_idx]))
      return true;
    if (break_idx + 1 < n) {
      std::vector<SequenceEntry> after(entries.begin() + break_idx + 1,
                                       entries.end());
      SequenceCollectFirstAccessPoints(after, pred, result, path, base_offset);
    }
    return false;
  }
  for (int i = 0; i < n; ++i) {
    // Conditional entries record their point but return false, so the scan
    // naturally continues past them to find unconditional first accesses.
    if (dispatch(entries[i]))
      return true;
  }
  return false;
}

static bool SequenceCollectLastAccessPoints(
    const std::vector<SequenceEntry> &entries, const TaskAccessPredicate &pred,
    std::set<BufferAccessPoint> &result, std::vector<const IRStructure *> &path,
    int64_t base_offset) {
  int n = static_cast<int>(entries.size());

  auto dispatch = [&](const SequenceEntry &e) -> bool {
    bool m = e.node->CollectLastAccessPoints(pred, result, path,
                                             base_offset + e.start_time);
    return m;
  };

  int break_idx = -1;
  for (int j = n - 1; j >= 0; --j) {
    if (entries[j].node->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx + 1 < n) {
      std::vector<SequenceEntry> after(entries.begin() + break_idx + 1,
                                       entries.end());
      SequenceCollectLastAccessPoints(after, pred, result, path, base_offset);
    }
    dispatch(entries[break_idx]);
    if (break_idx > 0) {
      std::vector<SequenceEntry> before(entries.begin(),
                                        entries.begin() + break_idx);
      SequenceCollectLastAccessPoints(before, pred, result, path, base_offset);
    }
    return false;
  }
  for (int i = n - 1; i >= 0; --i) {
    // Conditional entries record their point but return false, so the scan
    // continues past them to find unconditional last accesses.
    if (dispatch(entries[i]))
      return true;
  }
  return false;
}

// Build a SequenceEntry list from an ordered list of scheduled children.
static std::vector<SequenceEntry>
MakeSequenceEntries(const std::vector<std::shared_ptr<IRStructure>> &children) {
  std::vector<SequenceEntry> entries;
  entries.reserve(children.size());
  for (const auto &c : children) {
    SequenceEntry e;
    e.node = c.get();
    e.start_time = c->GetStartTime();
    entries.push_back(std::move(e));
  }
  return entries;
}

bool TaskNode::CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                        std::set<BufferAccessPoint> &result,
                                        std::vector<const IRStructure *> &path,
                                        int64_t time_offset) const {
  if (pred(this)) {
    std::vector<const IRStructure *> point_path = path;
    point_path.push_back(this);
    result.emplace(time_offset, std::move(point_path));
    // A conditional access may be skipped at runtime, so it is a candidate
    // but does not establish a definite first access — keep scanning.
    return !AnyConditionalOnPath(path, this);
  }
  return false;
}

bool TaskNode::CollectLastAccessPoints(const TaskAccessPredicate &pred,
                                       std::set<BufferAccessPoint> &result,
                                       std::vector<const IRStructure *> &path,
                                       int64_t time_offset) const {
  if (pred(this)) {
    std::vector<const IRStructure *> point_path = path;
    point_path.push_back(this);
    result.emplace(time_offset, std::move(point_path));
    // A conditional access may be skipped at runtime, so it is a candidate
    // but does not establish a definite last access — keep scanning.
    return !AnyConditionalOnPath(path, this);
  }
  return false;
}

bool ControlNode::CollectFirstAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  if (task) {
    if (task->CollectFirstAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  auto ordered = children;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto &a, const auto &b) {
                     return a->GetStage() > b->GetStage();
                   });
  bool body_must = SequenceCollectFirstAccessPoints(
      MakeSequenceEntries(ordered), pred, result, path, time_offset);
  path.pop_back();
  if (body_must && LoopMustExecute(this))
    return true;
  return false;
}

bool ControlNode::CollectLastAccessPoints(
    const TaskAccessPredicate &pred, std::set<BufferAccessPoint> &result,
    std::vector<const IRStructure *> &path, int64_t time_offset) const {
  path.push_back(this);
  int64_t trip_count = GetTripCount();
  int64_t ii_per_iter = GetIIperIter();
  int64_t last_iter_offset =
      (trip_count > 0 ? (trip_count - 1) : 0) * ii_per_iter;
  int64_t body_offset = time_offset + last_iter_offset;
  auto ordered = children;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto &a, const auto &b) {
                     return a->GetStage() > b->GetStage();
                   });
  bool body_must = SequenceCollectLastAccessPoints(
      MakeSequenceEntries(ordered), pred, result, path, body_offset);
  if (body_must && LoopMustExecute(this)) {
    path.pop_back();
    return true;
  }
  if (task) {
    if (task->CollectLastAccessPoints(pred, result, path, time_offset)) {
      path.pop_back();
      return true;
    }
  }
  path.pop_back();
  return false;
}

inline bool WarpgroupMatches(const TaskNode *task, int wg_id) {
  int task_wg = task->GetWarpgroupId();
  return IsWarpgroupBroadcast(wg_id) || IsWarpgroupBroadcast(task_wg) ||
         wg_id == task_wg;
}

TaskAccessPredicate MakeBufferAccessPredicate(const Buffer &buffer,
                                              bool is_write, int wg_id) {
  return [buffer, is_write, wg_id](const TaskNode *task) -> bool {
    if (!WarpgroupMatches(task, wg_id))
      return false;
    const auto &regions =
        is_write ? task->GetWriteRegions() : task->GetReadRegions();
    for (const auto &region : regions) {
      if (region->buffer == buffer)
        return true;
    }
    return false;
  };
}

// ============================================================================
// Dependency analysis implementation
// ============================================================================

bool RegionsMayConflict(const TaskNode *a_task, const BufferRegion &a_region,
                        const TaskNode *b_task, const BufferRegion &b_region,
                        std::optional<For> current_loop, bool cross) {
  if (!a_region->buffer.same_as(b_region->buffer))
    return false;
  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  Map<Var, PrimExpr> a_sub, b_sub;
  if (current_loop.has_value()) {
    const ForNode *fn = current_loop->get();
    Var lv = fn->loop_var;
    if (!cross) {
      // Same iteration: producer and consumer share one loop-var value.
      Var lv_s(lv->name_hint + "_s", lv.dtype());
      ana.Bind(lv_s, Range::FromMinExtent(fn->min, fn->extent));
      a_sub.Set(lv, lv_s);
      b_sub.Set(lv, lv_s);
    } else {
      // Loop back-edge: consumer runs d>=1 iterations after the producer.
      // lv_p ranges the loop; lv_c = lv_p + d with d in [1, extent-1].
      Var lv_p(lv->name_hint + "_p", lv.dtype());
      Var d(lv->name_hint + "_d", lv.dtype());
      ana.Bind(lv_p, Range::FromMinExtent(fn->min, fn->extent));
      // d in [1, extent-1]: at least one later iteration exists.
      ana.Bind(
          d, Range::FromMinExtent(IntImm(lv.dtype(), 1),
                                  max(fn->extent - 1, IntImm(lv.dtype(), 1))));
      a_sub.Set(lv, lv_p);
      b_sub.Set(lv, lv_p + d);
    }
  }
  // Feed each side's surrounding-scope ConstrSet to the prover, after
  // substituting the current_loop var with that side's loop-var map.
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

BufferRegion TaskRegionFor(const TaskNode *task, const Buffer &buffer,
                           bool is_write) {
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  std::optional<BufferRegion> found;
  for (const auto &r : regions) {
    if (r->buffer.same_as(buffer)) {
      ICHECK(!found) << "Buffer " << buffer->name
                     << " appears in more than one region of the task";
      found = r;
    }
  }
  ICHECK(found) << "Buffer " << buffer->name << " not found in task regions";
  return *found;
}

std::vector<DepInfo>
AnalyzeDependencies(const std::vector<IRStructure *> &nodes,
                    std::optional<For> current_loop) {
  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  std::set<Buffer> buffers;
  for (size_t i = 0; i < n; ++i) {
    for (const auto &region : nodes[i]->GetReadRegions()) {
      buffers.insert(region->buffer);
    }
    for (const auto &region : nodes[i]->GetWriteRegions()) {
      buffers.insert(region->buffer);
    }
  }

  for (const auto &buffer : buffers) {
    auto touches = [&](size_t u) {
      for (const auto &region : nodes[u]->GetReadRegions()) {
        if (region->buffer.same_as(buffer))
          return true;
      }
      for (const auto &region : nodes[u]->GetWriteRegions()) {
        if (region->buffer.same_as(buffer))
          return true;
      }
      return false;
    };

    for (size_t i = 0; i < n; ++i) {
      if (!touches(i))
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (i == j && !current_loop.has_value())
          continue;
        if (!touches(j))
          continue;
        bool same_iter = (i < j);
        bool cross = current_loop.has_value() && (i >= j);
        if (!same_iter && !cross)
          continue;

        auto a_w = nodes[i]->GetLastAccessPoints(buffer, /*is_write=*/true);
        auto a_r = nodes[i]->GetLastAccessPoints(buffer, /*is_write=*/false);
        auto b_r = nodes[j]->GetFirstAccessPoints(buffer, /*is_write=*/false);
        auto b_w = nodes[j]->GetFirstAccessPoints(buffer, /*is_write=*/true);

        std::set<BufferAccessPoint> prod_pts, cons_pts;
        int64_t latency = -1;

        auto consider = [&](const std::set<BufferAccessPoint> &a_pts,
                            bool a_write,
                            const std::set<BufferAccessPoint> &b_pts,
                            bool b_write) {
          for (const auto &a : a_pts) {
            BufferRegion rA = TaskRegionFor(a.task(), buffer, a_write);
            for (const auto &b : b_pts) {
              BufferRegion rB = TaskRegionFor(b.task(), buffer, b_write);
              if (!RegionsMayConflict(a.task(), rA, b.task(), rB, current_loop,
                                      cross))
                continue;
              prod_pts.insert(a);
              cons_pts.insert(b);
              int64_t l = a.timestamp + a.task()->GetLatency() - b.timestamp;
              if (l < 0)
                l = 0;
              if (l > latency)
                latency = l;
            }
          }
        };
        consider(a_w, /*a_write=*/true, b_r, /*b_write=*/false);
        consider(a_w, /*a_write=*/true, b_w, /*b_write=*/true);
        consider(a_r, /*a_write=*/false, b_w, /*b_write=*/true);

        if (latency >= 0) {
          deps.push_back(
              {nodes[i], nodes[j], buffer,
               std::vector<BufferAccessPoint>(prod_pts.begin(), prod_pts.end()),
               std::vector<BufferAccessPoint>(cons_pts.begin(), cons_pts.end()),
               cross, latency});
        }
      }
    }
  }
  return deps;
}

} // namespace tl
} // namespace tvm
