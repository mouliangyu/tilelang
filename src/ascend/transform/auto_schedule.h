/*!
 * \file auto_schedule.h
 * \brief AutoSchedule pass structures and declarations for TileLang
 */

#pragma once

#include "./auto_schedule/barrier.h"
#include "./auto_schedule/core_partition.h"
#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/memory_detector.h"
#include "./auto_schedule/schedule_builder.h"
#include "backend/common/target_utils.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

struct AutoScheduleAttrSpec {
  const char *key;
  bool is_assumption;
};

// Returns the propagation semantics for an AttrStmt key supported by
// AutoSchedule, or nullptr when the attribute is not in the allowlist.
const AutoScheduleAttrSpec *FindAutoScheduleAttr(const ffi::String &key);

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

// Check whether a kernel body contains any SIMT_VF blocks. SIMT_VF
// (asc_vf_call) reserves 32 KB of the physical 248 KB Unified Buffer for
// thread context, so the usable shared memory depends on its presence.
class SimtVFDetector : public tirx::StmtVisitor {
public:
  bool found = false;

  void VisitStmt_(const tirx::SBlockNode *op) override {
    if (op->name_hint == "SIMT_VF") {
      found = true;
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

inline bool HasSimtVF(const tirx::Stmt &stmt) {
  SimtVFDetector detector;
  detector(stmt);
  return detector.found;
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
