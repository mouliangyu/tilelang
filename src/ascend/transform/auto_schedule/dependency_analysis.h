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
 * \file dependency_analysis.h
 * \brief Shared schedule and synchronization dependency analysis.
 */

#pragma once

#include <map>
#include <utility>
#include <vector>

#include "../buffer_version.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tirx;

PrimExpr CalculateIterationCount(const std::vector<For> &loops);

struct BufferAccessPoint {
  int64_t timestamp; // The time at which the access occurs
  std::vector<const IRStructure *> path;

  BufferAccessPoint(int64_t timestamp,
                    std::vector<const IRStructure *> path = {})
      : timestamp(timestamp), path(std::move(path)) {}

  // The task where the access occurs (= path.back()).
  const TaskNode *task() const;

  bool operator<(const BufferAccessPoint &other) const {
    if (timestamp != other.timestamp) {
      return timestamp < other.timestamp;
    }
    return path < other.path;
  }

  bool operator==(const BufferAccessPoint &other) const {
    return timestamp == other.timestamp && path == other.path;
  }

  bool operator!=(const BufferAccessPoint &other) const {
    return !(*this == other);
  }
};
struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  // Storage identity shared by every Buffer alias participating in this
  // dependency.
  Var storage;
  // Conflicting (producer access, consumer access) pairs. Each pair is a
  // concrete producer→consumer ordering that needs a sync.
  std::vector<std::pair<BufferAccessPoint, BufferAccessPoint>> task_pairs;
  // Cross-iteration distance of this dependency:
  //   0   : same-iteration dependency.
  //   -1  : cross-iteration dependency on an auto-versioned buffer; the true
  //         distance equals the buffer's num_versions.
  //   >=1 : dependency on a manually multi-buffered buffer, at the physical
  //         iteration distance solved per access pair.
  int distance;
  int64_t latency;
};

// Per-phase dependency-analysis cache. A ControlNode key represents the
// dependency result for its ordered child list at that phase.
using DependencyCache = std::map<const ControlNode *, std::vector<DepInfo>>;

// Optional timing projection supplied by AutoSchedule. InsertSync analyzes
// the same structural dependencies with zero start-time offsets and therefore
// leaves this unset.
struct DependencyTiming {
  int64_t (*start_time)(const IRStructure *){nullptr};
  int64_t (*ii_per_iter)(const ControlNode *){nullptr};

  int64_t StartTime(const IRStructure *node) const {
    return start_time ? start_time(node) : 0;
  }
  int64_t IIperIter(const ControlNode *node) const {
    return ii_per_iter ? ii_per_iter(node) : 0;
  }
};

// Analyze data dependencies among scheduled TaskNode/ControlNode nodes.
// `loop_stack` is the full enclosing serial loop stack at this level
// (outermost-first). Empty means root/list level. Non-empty means the last loop
// is the current loop body, enabling cross-iteration dependencies.
// `manual_buffer_versions` maps user-declared manual multi-buffer data Vars to
// their version count; empty means no manual buffers.
// `dependency_cache` is shared for one scheduling or synchronization phase. A
// non-null `cache_key` identifies `nodes` as that ControlNode's ordered child
// list.
// An `i == j` control-node self-dependency whose producer→consumer pairs are
// already established anywhere inside that node's own subtree is dropped.
std::vector<DepInfo>
AnalyzeDependencies(const std::vector<IRStructure *> &nodes,
                    const std::vector<For> &loop_stack,
                    const BufferVersionMap &manual_buffer_versions = {},
                    DependencyCache *dependency_cache = nullptr,
                    const ControlNode *cache_key = nullptr,
                    const DependencyTiming *timing = nullptr);

// Check whether two task regions may overlap under their respective symbolic
// contexts. Shared by scheduling-time IRStructure analysis and the scheduled
// TIR synchronization pass.
bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        const std::vector<For> &loop_stack, int offset);

} // namespace tl
} // namespace tvm
