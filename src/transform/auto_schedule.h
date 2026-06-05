/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

#include "../target/utils.h"
#include "./auto_schedule/barrier.h"
#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/latency_estimator.h"
#include "./auto_schedule/memory_detector.h"
#include "./auto_schedule/schedule_builder.h"
#include "./auto_schedule/warpgroup_partition.h"
#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// Simple Union-Find (Disjoint Set Union) for task grouping
class TaskUnionFind {
public:
  TaskUnionFind(int n) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; i++) {
      parent[i] = i;
    }
  }

  int find(int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]); // path compression
    }
    return parent[x];
  }

  int find(int x) const {
    int root = x;
    while (parent[root] != root) {
      root = parent[root];
    }
    return root;
  }

  void unite(int x, int y) {
    int root_x = find(x);
    int root_y = find(y);
    if (root_x == root_y)
      return;

    // union by rank
    if (rank[root_x] < rank[root_y]) {
      parent[root_x] = root_y;
    } else if (rank[root_x] > rank[root_y]) {
      parent[root_y] = root_x;
    } else {
      parent[root_y] = root_x;
      rank[root_x]++;
    }
  }

private:
  std::vector<int> parent;
  std::vector<int> rank;
};

// Structure for component information used in warpgroup assignment
struct ComponentInfo {
  int root;
  int64_t weighted_latency; // total weighted latency in this component
  std::vector<int> task_indices;
  uint16_t pipe_mask{0};
};

inline int64_t GetSharedMemoryLimit(Target target) {
  if (TargetIsHopper(target)) {
    return 228 * 1024;
  } else if (TargetIsSm100(target)) {
    return 228 * 1024;
  } else if (TargetIsAscend(target)) {
    // Physical UB = 248 KB, but SimtVF (asc_vf_call) reserves at least 32 KB
    // for thread context. Usable shared buffer = 248 - 32 = 216 KB.
    return 216 * 1024;
  } else {
    return 48 * 1024;
  }
}

inline int64_t GetL1MemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 512 * 1024; // 512 KB L1 cache on Ascend
  }
  return 0;
}

inline int64_t GetL0CMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 256 * 1024; // 256 KB L0C buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0AMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0A buffer on Ascend
  }
  return 0;
}

inline int64_t GetL0BMemoryLimit(Target target) {
  if (TargetIsAscend(target)) {
    return 64 * 1024; // 64 KB L0B buffer on Ascend
  }
  return 0;
}

// Function to rewrite alloc_buffers for multi-version (double buffer) support
Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos);

} // namespace tl
} // namespace tvm
