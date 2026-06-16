#pragma once
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include "op/gemm.h"
#include "transform/common/constr_visitor.h"
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::String;

// Forward declarations
class IRStructure;
class TaskNode;
class ControlNode;
class SequenceNode;
class WrapperNode;
class IfNode;

// Hardware execution pipe bitmask for resource dependency analysis.
// Two operations conflict (cannot run concurrently) if they share any pipe bit.
enum class ResourcePipe : uint16_t {
  kNone = 0,
  kMTE1 = 1 << 0,    // L1 -> L0A / L1 -> L0B
  kMTE2 = 1 << 1,    // GM -> UB / GM -> L1
  kMTE3 = 1 << 2,    // UB -> GM / UB -> L1
  kCube = 1 << 3,    // M pipe: MAD matrix multiply
  kVector = 1 << 4,  // V pipe: SimtVF / SimdVF / Reduce / Fill / arithmetic
  kFixpipe = 1 << 5, // L0C -> UB / L0C -> GM
  kScalar = 1 << 6,  // S pipe: scalar ops, set_flag / wait_flag
};

// HBM bandwidth-contention resource bits. These are tracked separately from
// ResourcePipe because a load and a store use different issue pipes but can
// still contend on the same HBM port.
enum class HbmPort : uint16_t {
  kNone = 0,
  kAiv = 1 << 0, // AIV HBM port: GM<->UB
  kAic = 1 << 1, // AIC HBM port: GM->L1 / L0C->GM
};

// Structure to represent a specific access to a buffer (used for ordering
// accesses). `path` records the chain of IRStructure nodes traversed from the
// queried subtree's root down to (and including) the task — only real IR-tree
// nodes; synthetic helpers introduced during traversal are excluded. The task
// itself is always the last element of `path`.
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

// Helper function to compare if two regions are equal
inline bool RegionsEqual(const Region &a, const Region &b) {
  if (a.size() != b.size())
    return false;

  arith::Analyzer analyzer;
  for (size_t i = 0; i < a.size(); ++i) {
    // Check if min values are equal
    if (!analyzer.CanProveEqual(a[i]->min, b[i]->min)) {
      return false;
    }
    // Check if extent values are equal
    if (!analyzer.CanProveEqual(a[i]->extent, b[i]->extent)) {
      return false;
    }
  }
  return true;
}

// Special warpgroup id constants
constexpr int kWarpgroupUnassigned = -1; // Not yet assigned (initial state)
constexpr int kWarpgroupBroadcast = -2;  // Broadcast: the statement is cloned
                                         // into every warp group; each wg
                                         // operates on its own register copies.
                                         // No cross-wg sync is needed for
                                         // register (local.fragment) buffers.

// Helper: check if a warpgroup id represents a broadcast task
inline bool IsWarpgroupBroadcast(int wg_id) {
  return wg_id == kWarpgroupBroadcast;
}

// Predicate the Collect*AccessPoints walks use to decide whether a leaf
// TaskNode "matches": pred(task) returns true if the task has a matching
// access. The (is_write, wg_id) filter is baked in by the factory below.
//
// Concrete predicates are produced by:
//   - MakeBufferAccessPredicate(buffer, is_write, wg_id)
class TaskNode;
using TaskAccessPredicate = std::function<bool(const TaskNode *task)>;

TaskAccessPredicate MakeBufferAccessPredicate(const Buffer &buffer,
                                              bool is_write, int wg_id);

// Base class for all IR nodes in scheduling
class IRStructure {
public:
  enum class Kind { kTask, kControl, kSequence, kWrapper, kSchedule, kIf };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }
  bool IsSequence() const { return GetKind() == Kind::kSequence; }
  bool IsWrapper() const { return GetKind() == Kind::kWrapper; }
  bool IsScheduleUnit() const { return GetKind() == Kind::kSchedule; }
  bool IsIf() const { return GetKind() == Kind::kIf; }

  // Resource pipe bitmask (accessible by all IR nodes)
  virtual uint16_t GetPipeMask() const = 0;
  bool UsesPipe(ResourcePipe pipe) const {
    return (GetPipeMask() & static_cast<uint16_t>(pipe)) != 0;
  }
  bool UsesOnlyPipe(ResourcePipe pipe) const {
    return GetPipeMask() == static_cast<uint16_t>(pipe);
  }

  // HBM port-contention bitmask. Composite nodes OR-aggregate descendants;
  // only TaskNode stores a settable mask.
  virtual uint16_t GetHbmMask() const = 0;

  // Memory access regions (collected during analysis)
  virtual std::vector<BufferRegion> GetReadRegions() const = 0;
  virtual std::vector<BufferRegion> GetWriteRegions() const = 0;

  // Variable access (used for dependency analysis)
  virtual std::vector<Var> GetReadVars() const = 0;
  virtual std::vector<Var> GetWriteVars() const = 0;

  // Latency estimation
  virtual int64_t GetLatency() const = 0; // Estimated latency in cycles
  virtual int64_t GetII() const = 0;      // Initiation interval in cycles

  // Setters (for analysis passes to update these values)
  virtual void SetPipeMask(uint16_t mask) = 0;
  virtual void SetReadRegions(const std::vector<BufferRegion> &regions) = 0;
  virtual void SetWriteRegions(const std::vector<BufferRegion> &regions) = 0;
  virtual void SetLatency(int64_t latency) = 0;
  virtual void SetII(int64_t ii) = 0;

  // Collect tasks that could possibly be the first/last to access something
  // matching `pred` within this IR subtree, together with the earliest
  // (First) / latest (Last) possible task start time relative to this
  // subtree's root (in cycles). `path` is used as a stack: each impl pushes
  // `this` before recursing into children and pops it on return; TaskNode
  // copies the path (with itself appended) into the BufferAccessPoint.
  //
  // `pred(task)` returns true if the task has a matching access (the factory
  // bakes is_write / wg_id into the predicate). Use
  // `MakeBufferAccessPredicate(buffer, is_write, wg_id)` for buffer-keyed
  // queries.
  virtual bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                        std::set<BufferAccessPoint> &result,
                                        std::vector<const IRStructure *> &path,
                                        int64_t time_offset = 0) const = 0;
  virtual bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                                       std::set<BufferAccessPoint> &result,
                                       std::vector<const IRStructure *> &path,
                                       int64_t time_offset = 0) const = 0;

  // Convenience wrappers — buffer-keyed.
  std::set<BufferAccessPoint>
  GetFirstAccessPoints(const Buffer &buffer, bool is_write,
                       int wg_id = kWarpgroupBroadcast) const {
    std::set<BufferAccessPoint> result;
    std::vector<const IRStructure *> path;
    CollectFirstAccessPoints(MakeBufferAccessPredicate(buffer, is_write, wg_id),
                             result, path);
    return result;
  }
  std::set<BufferAccessPoint>
  GetLastAccessPoints(const Buffer &buffer, bool is_write,
                      int wg_id = kWarpgroupBroadcast) const {
    std::set<BufferAccessPoint> result;
    std::vector<const IRStructure *> path;
    CollectLastAccessPoints(MakeBufferAccessPredicate(buffer, is_write, wg_id),
                            result, path);
    return result;
  }

  // Substitute a variable throughout this IR node
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;

  // Get warpgroup id for this node (kWarpgroupUnassigned if not applicable)
  virtual int GetWarpgroupId() const { return kWarpgroupUnassigned; }

  virtual bool containWarpgroupId(int id) const = 0;

  // Check if this node (or any descendant) contains a loop_break call
  virtual bool ContainsLoopBreak() const = 0;
};

// Task node: contains a vector of statements
class TaskNode : public IRStructure {
public:
  std::vector<Stmt> stmts;

  // Snapshot of the surrounding-scope ConstrSet at the moment this task was
  // built. Captures ancestor For ranges, LetStmt bindings, kernel-block
  // thread_extent / virtual_thread iter ranges, tilelang_assume, and
  // IfThenElse / Select branch guards. Populated by IRStructureBuilder in
  // auto_schedule.cc when the TaskNode is constructed; consumed by
  // RegionsMayConflict so the prover sees the symbol environment
  // each region was written under.
  ConstrSet outer_ctx;

  Kind GetKind() const override { return Kind::kTask; }

  // Resource pipe mask
  uint16_t GetPipeMask() const override { return pipe_mask_; }
  uint16_t GetHbmMask() const override { return hbm_mask_; }

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> GetReadRegions() const override {
    return read_regions_;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return write_regions_;
  }
  std::vector<Var> GetReadVars() const override { return read_vars_; }
  std::vector<Var> GetWriteVars() const override { return write_vars_; }
  // Latency estimation
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters
  void SetPipeMask(uint16_t mask) override { pipe_mask_ = mask; }
  void SetHbmMask(uint16_t mask) { hbm_mask_ = mask; }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    read_regions_ = regions;
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    write_regions_ = regions;
  }
  void SetReadVars(const std::vector<Var> &vars) { read_vars_ = vars; }
  void SetWriteVars(const std::vector<Var> &vars) { write_vars_ = vars; }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    for (auto &var : read_vars_) {
      if (var.same_as(old_var)) {
        var = new_var;
      }
    }
    for (auto &var : write_vars_) {
      if (var.same_as(old_var)) {
        var = new_var;
      }
    }
    for (size_t i = 0; i < stmts.size(); ++i) {
      stmts[i] = Substitute(stmts[i], {{old_var, new_var}});
    }
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  // Warpgroup id for warpgroup specialization
  void SetWarpgroupId(int warpgroup_id) { warpgroup_id_ = warpgroup_id; }
  int GetWarpgroupId() const override { return warpgroup_id_; }

  // Tensor Core shape information structure
  struct TensorCoreShape {
    int64_t m;
    int64_t n;
    int64_t k;
    DataType input_dtype;
    DataType accum_dtype;
    TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0,
                    DataType input_dtype = DataType::Float(16),
                    DataType accum_dtype = DataType::Float(16))
        : m(m), n(n), k(k), input_dtype(input_dtype), accum_dtype(accum_dtype) {
    }
  };
  void AddTensorCoreShape(int64_t m, int64_t n, int64_t k) {
    tensor_core_shapes_.emplace_back(m, n, k);
  }
  void AddTensorCoreShape(int64_t m, int64_t n, int64_t k, DataType input_dtype,
                          DataType accum_dtype) {
    tensor_core_shapes_.emplace_back(m, n, k, input_dtype, accum_dtype);
  }

  // Get shape information
  size_t GetTensorCoreShapeCount() const { return tensor_core_shapes_.size(); }
  const TensorCoreShape &GetTensorCoreShape(size_t index) const {
    return tensor_core_shapes_[index];
  }

  // Helper methods for single shape (backward compatibility)
  int64_t GetTensorCoreM() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].m;
  }
  int64_t GetTensorCoreN() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].n;
  }
  int64_t GetTensorCoreK() const {
    return tensor_core_shapes_.empty() ? 0 : tensor_core_shapes_[0].k;
  }
  bool HasTensorCoreShape() const { return !tensor_core_shapes_.empty(); }

  // Task-level copy information collected from tl.copy calls. Ascend latency
  // estimation uses this instead of re-deriving copy direction from read/write
  // regions, which avoids double-counting source and destination regions.
  struct CopyInfo {
    String src_scope;
    String dst_scope;
    DataType dtype;
    int64_t bytes;
    uint16_t pipe_mask;
    CopyInfo(String src_scope = String(""), String dst_scope = String(""),
             DataType dtype = DataType::Float(16), int64_t bytes = 0,
             uint16_t pipe_mask = 0)
        : src_scope(src_scope), dst_scope(dst_scope), dtype(dtype),
          bytes(bytes), pipe_mask(pipe_mask) {}
  };
  void AddCopyInfo(String src_scope, String dst_scope, DataType dtype,
                   int64_t bytes, uint16_t pipe_mask) {
    copy_infos_.emplace_back(src_scope, dst_scope, dtype, bytes, pipe_mask);
  }
  const std::vector<CopyInfo> &GetCopyInfos() const { return copy_infos_; }
  bool HasCopyInfo() const { return !copy_infos_.empty(); }

  // Get aggregated shape information for II estimation
  int64_t GetTotalTensorCoreOps() const {
    int64_t total_ops = 0;
    for (const auto &shape : tensor_core_shapes_) {
      total_ops += shape.m * shape.n * shape.k * 2; // 2 ops per element
    }
    return total_ops;
  }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  // Helper methods to add regions (for incremental analysis)
  void AddReadRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    read_regions_.push_back(region);
  }

  void AddWriteRegion(const BufferRegion &region) {
    // Check for duplicate regions
    for (const auto &existing : write_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return; // Region already exists
      }
    }
    write_regions_.push_back(region);
  }

  void AddReadVar(const Var &var) {
    for (const auto &existing : read_vars_) {
      if (existing.same_as(var))
        return;
    }
    read_vars_.push_back(var);
  }
  void AddWriteVar(const Var &var) {
    for (const auto &existing : write_vars_) {
      if (existing.same_as(var))
        return;
    }
    write_vars_.push_back(var);
  }

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  bool containWarpgroupId(int id) const override {
    return ContainsLoopBreak() || IsWarpgroupBroadcast(warpgroup_id_) ||
           warpgroup_id_ == id;
  }

  // Check if this task contains loop_break call
  bool ContainsLoopBreak() const override;

private:
  // Resource pipe mask (bitmask of ResourcePipe values)
  uint16_t pipe_mask_{0};
  uint16_t hbm_mask_{0};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions_;
  std::vector<BufferRegion> write_regions_;

  std::vector<Var> read_vars_;
  std::vector<Var> write_vars_;

  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
  int warpgroup_id_{
      kWarpgroupUnassigned}; // Warpgroup id for warpgroup specialization

  // Tensor Core shape information (M, N, K dimensions)
  // A task may contain multiple Tensor Core operations with different shapes
  std::vector<TensorCoreShape> tensor_core_shapes_;

  // Selected tensor core instruction key for this task.
  std::vector<CopyInfo> copy_infos_;

  // Cached flag for loop_break detection
  mutable std::optional<bool> contains_loop_break_cache_;
};

// Control node: contains a For operation and a child IRStructure
class ControlNode : public IRStructure {
public:
  For control; // The For operation
  std::shared_ptr<TaskNode> task;
  std::shared_ptr<IRStructure> child;

  Kind GetKind() const override { return Kind::kControl; }

  // Resource pipe mask (aggregate from child)
  uint16_t GetPipeMask() const override {
    return child ? child->GetPipeMask() : 0;
  }
  uint16_t GetHbmMask() const override {
    return child ? child->GetHbmMask() : 0;
  }

  // Memory access regions (aggregate from child & task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetReadRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  // Variable access (aggregate from child & task)
  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars = child ? child->GetReadVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars = child ? child->GetWriteVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetWriteVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    if (task)
      task->SubstituteVar(old_var, new_var);
    // Also substitute in the For statement's min/extent/step
    For new_for = control;
    new_for.CopyOnWrite()->min = Substitute(control->min, {{old_var, new_var}});
    new_for.CopyOnWrite()->extent =
        Substitute(control->extent, {{old_var, new_var}});
    if (control->step.has_value()) {
      new_for.CopyOnWrite()->step =
          Substitute(control->step.value(), {{old_var, new_var}});
    }
    control = new_for;
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to child if exists)
  void SetPipeMask(uint16_t mask) override {
    if (child)
      child->SetPipeMask(mask);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetReadRegions(regions);
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetWriteRegions(regions);
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  void SetIIperIter(int64_t ii) { ii_per_iter_ = ii; }

  int64_t GetIIperIter() const { return ii_per_iter_; }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child && child->containWarpgroupId(id);
  }

  bool ContainsLoopBreak() const override {
    return false; // Loop does not contain loop break
  }

  int64_t GetTripCount() const {
    PrimExpr loop_extent = control->extent;
    PrimExpr loop_step = control->step.has_value()
                             ? control->step.value()
                             : IntImm(DataType::Int(32), 1);
    // Try to convert loop_extent and loop_step to int64_t
    if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
      if (const int64_t *step_ptr = as_const_int(loop_step)) {
        // Calculate ceil(extent / step)
        int64_t extent = *extent_ptr;
        int64_t step = *step_ptr;
        if (step > 0) {
          // ceil(extent / step) = (extent + step - 1) / step
          return (extent + step - 1) / step;
        } else {
          // Invalid step, use extent as fallback
          return extent;
        }
      } else {
        // Step is not constant, use 100 as default
        return 100;
      }
    } else {
      // Extent is not constant, use 100 as default
      return 100;
    }
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
  int64_t ii_per_iter_{0};

public:
  std::map<Buffer, int64_t> buffer_num_versions; // Map from buffer to number of
                                                 // versions for multi-buffering
};

// Wrapper node: contains a Wrapper statement with variable, value, and child
// IRStructure
class WrapperNode : public IRStructure {
public:
  Stmt wrapper;
  std::shared_ptr<IRStructure> child;
  std::shared_ptr<TaskNode> task;

  Kind GetKind() const override { return Kind::kWrapper; }

  // Resource pipe mask (aggregate from child)
  uint16_t GetPipeMask() const override {
    return child ? child->GetPipeMask() : 0;
  }
  uint16_t GetHbmMask() const override {
    return child ? child->GetHbmMask() : 0;
  }

  // Memory access regions (aggregate from child & task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetReadRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions =
        child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  // Variable access (aggregate from child & task)
  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars = child ? child->GetReadVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars = child ? child->GetWriteVars() : std::vector<Var>{};
    if (task) {
      auto task_vars = task->GetWriteVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (wrapper.defined()) {
      wrapper = Substitute(wrapper, {{old_var, new_var}});
    }
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    if (task) {
      task->SubstituteVar(old_var, new_var);
    }
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to child if exists)
  void SetPipeMask(uint16_t mask) override {
    if (child)
      child->SetPipeMask(mask);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetReadRegions(regions);
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetWriteRegions(regions);
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child && child->containWarpgroupId(id);
  }

  bool ContainsLoopBreak() const override {
    return (task && task->ContainsLoopBreak()) ||
           (child && child->ContainsLoopBreak());
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

// If node: represents an IfThenElse conditional with independently schedulable
// branches. Acts as an atomic unit for outer scheduling but allows recursive
// scheduling within each branch.
class IfNode : public IRStructure {
public:
  PrimExpr condition;
  std::shared_ptr<IRStructure> then_child;
  std::shared_ptr<IRStructure> else_child; // optional
  std::shared_ptr<TaskNode> task; // resource analysis for the condition expr

  Kind GetKind() const override { return Kind::kIf; }

  // Resource pipe mask (union of both branches)
  uint16_t GetPipeMask() const override {
    uint16_t result = 0;
    if (then_child)
      result |= then_child->GetPipeMask();
    if (else_child)
      result |= else_child->GetPipeMask();
    return result;
  }
  uint16_t GetHbmMask() const override {
    uint16_t result = 0;
    if (then_child)
      result |= then_child->GetHbmMask();
    if (else_child)
      result |= else_child->GetHbmMask();
    return result;
  }

  // Memory access regions (union of both branches + task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions;
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    if (then_child) {
      auto r = then_child->GetReadRegions();
      regions.insert(regions.end(), r.begin(), r.end());
    }
    if (else_child) {
      auto r = else_child->GetReadRegions();
      regions.insert(regions.end(), r.begin(), r.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions;
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    if (then_child) {
      auto r = then_child->GetWriteRegions();
      regions.insert(regions.end(), r.begin(), r.end());
    }
    if (else_child) {
      auto r = else_child->GetWriteRegions();
      regions.insert(regions.end(), r.begin(), r.end());
    }
    return regions;
  }

  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars;
    if (task) {
      auto v = task->GetReadVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    if (then_child) {
      auto v = then_child->GetReadVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    if (else_child) {
      auto v = else_child->GetReadVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars;
    if (task) {
      auto v = task->GetWriteVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    if (then_child) {
      auto v = then_child->GetWriteVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    if (else_child) {
      auto v = else_child->GetWriteVars();
      vars.insert(vars.end(), v.begin(), v.end());
    }
    return vars;
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    condition = Substitute(condition, {{old_var, new_var}});
    if (then_child)
      then_child->SubstituteVar(old_var, new_var);
    if (else_child)
      else_child->SubstituteVar(old_var, new_var);
    if (task)
      task->SubstituteVar(old_var, new_var);
  }

  // Latency = max of both branches
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to both branches)
  void SetPipeMask(uint16_t mask) override {
    if (then_child)
      then_child->SetPipeMask(mask);
    if (else_child)
      else_child->SetPipeMask(mask);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {}
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {}
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return (then_child && then_child->containWarpgroupId(id)) ||
           (else_child && else_child->containWarpgroupId(id));
  }

  bool ContainsLoopBreak() const override {
    return (task && task->ContainsLoopBreak()) ||
           (then_child && then_child->ContainsLoopBreak()) ||
           (else_child && else_child->ContainsLoopBreak());
  }

private:
  int64_t latency_{0};
  int64_t ii_{0};
};

class ScheduleUnit : public IRStructure {
public:
  std::map<int, std::vector<Stmt>> before, after;
  std::shared_ptr<IRStructure> child;
  int stage{0}; // SW-pipeline stage assigned by Z3 scheduler

  ScheduleUnit() {}

  Kind GetKind() const override { return Kind::kSchedule; }

  int GetStage() const { return stage; }

  // Resource pipe mask (aggregate from child)
  uint16_t GetPipeMask() const override {
    return child ? child->GetPipeMask() : 0;
  }
  uint16_t GetHbmMask() const override {
    return child ? child->GetHbmMask() : 0;
  }

  // Memory access regions (aggregate from child)
  std::vector<BufferRegion> GetReadRegions() const override {
    return child ? child->GetReadRegions() : std::vector<BufferRegion>{};
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return child ? child->GetWriteRegions() : std::vector<BufferRegion>{};
  }

  std::vector<Var> GetReadVars() const override {
    return child ? child->GetReadVars() : std::vector<Var>{};
  }
  std::vector<Var> GetWriteVars() const override {
    return child ? child->GetWriteVars() : std::vector<Var>{};
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    if (child) {
      child->SubstituteVar(old_var, new_var);
    }
    for (auto &[_, stmts] : before) {
      for (auto &stmt : stmts) {
        stmt = Substitute(stmt, {{old_var, new_var}});
      }
    }
    for (auto &[_, stmts] : after) {
      for (auto &stmt : stmts) {
        stmt = Substitute(stmt, {{old_var, new_var}});
      }
    }
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters (delegate to child if exists)
  void SetPipeMask(uint16_t mask) override {
    if (child)
      child->SetPipeMask(mask);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetReadRegions(regions);
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override {
    if (child)
      child->SetWriteRegions(regions);
  }
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  bool isInnerTask() const { return child->IsTask(); }
  int GetWarpgroupId() const override {
    ICHECK(isInnerTask());
    const TaskNode *task = static_cast<const TaskNode *>(child.get());
    return task->GetWarpgroupId();
  }
  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    return child && child->containWarpgroupId(id);
  }

  bool ContainsLoopBreak() const override {
    return child && child->ContainsLoopBreak();
  }

  // Start time assigned by the Z3 scheduler. Only meaningful on ScheduleUnits
  // produced by ScheduleUnitBuilder — i.e. the children of a scheduled
  // SequenceNode. Expressed as cycle offset relative to that parent sequence's
  // start.
  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
  int64_t start_time_{0};
};

// Sequence node: contains a vector of child IRStructures
class SequenceNode : public IRStructure {
public:
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kSequence; }

  // Resource pipe mask (union of all children)
  uint16_t GetPipeMask() const override;
  uint16_t GetHbmMask() const override;

  // Memory access regions (aggregate from all children)
  std::vector<BufferRegion> GetReadRegions() const override;
  std::vector<BufferRegion> GetWriteRegions() const override;

  std::vector<Var> GetReadVars() const override;
  std::vector<Var> GetWriteVars() const override;

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    for (auto &child : children) {
      child->SubstituteVar(old_var, new_var);
    }
  }

  // Latency estimation (aggregate from all children)
  int64_t GetLatency() const override;
  int64_t GetII() const override;

  // Setters (delegate to all children)
  void SetPipeMask(uint16_t mask) override;
  void SetReadRegions(const std::vector<BufferRegion> &regions) override;
  void SetWriteRegions(const std::vector<BufferRegion> &regions) override;
  void SetLatency(int64_t latency) override;
  void SetII(int64_t ii) override;

  bool CollectFirstAccessPoints(const TaskAccessPredicate &pred,
                                std::set<BufferAccessPoint> &result,
                                std::vector<const IRStructure *> &path,
                                int64_t time_offset = 0) const override;
  bool CollectLastAccessPoints(const TaskAccessPredicate &pred,
                               std::set<BufferAccessPoint> &result,
                               std::vector<const IRStructure *> &path,
                               int64_t time_offset = 0) const override;

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    for (auto &child : children) {
      if (child->containWarpgroupId(id)) {
        return true;
      }
    }
    return false;
  }

  bool ContainsLoopBreak() const override {
    for (const auto &child : children) {
      if (child->ContainsLoopBreak()) {
        return true;
      }
    }
    return false;
  }

private:
  // Latency estimation
  int64_t latency_{0}; // Estimated latency in cycles
  int64_t ii_{0};      // Initiation interval in cycles
};

// Structure to store TaskNode with its loop context information
struct TaskNodeWithContext {
  TaskNode *task;
  ControlNode *control_node{nullptr}; // nullptr if not inside a loop
  int64_t tripcount{1}; // tripcount multiplier for weighted latency
};

// Memory type classification
enum class MemoryType {
  kGlobal,   // Global memory (DRAM)
  kShared,   // Shared memory (L1/shared)
  kRegister, // Register/local memory
  kTmem,     // Tensor memory (shared.tmem, Blackwell/sm_100 only)
  kUnknown   // Unknown memory type
};

// Helper function to determine memory type from buffer scope
inline MemoryType GetMemoryTypeFromScope(const String &scope) {
  if (scope == "global") {
    return MemoryType::kGlobal;
  } else if (scope == "shared.tmem") {
    return MemoryType::kTmem;
  } else if (scope == "shared" || scope == "shared.dyn") {
    return MemoryType::kShared;
  } else if (scope == "local" || scope == "local.var" ||
             scope == "local.fragment") {
    return MemoryType::kRegister;
  }
  return MemoryType::kUnknown;
}

// Helper function to check if a buffer region is in register memory
inline bool IsRegisterRegion(const BufferRegion &region) {
  const Buffer &buffer = region->buffer;
  String scope = buffer.scope();
  MemoryType mem_type = GetMemoryTypeFromScope(scope);
  return mem_type == MemoryType::kRegister;
}

// Helper function to collect all register regions from an IRStructure node
inline std::vector<BufferRegion>
CollectRegisterRegions(const IRStructure *node) {
  std::vector<BufferRegion> reg_regions;
  // Check read regions
  for (const auto &region : node->GetReadRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  // Check write regions
  for (const auto &region : node->GetWriteRegions()) {
    if (IsRegisterRegion(region)) {
      reg_regions.push_back(region);
    }
  }
  return reg_regions;
}

// Helper function to check if two IRStructure nodes use the same register
// region
inline bool UseSameRegisterRegion(const IRStructure *a, const IRStructure *b) {
  if (!a || !b)
    return false;

  // Check if either task contains loop_break
  // Tasks with loop_break should not share register regions with other tasks
  if (a->IsTask() && b->IsTask()) {
    const TaskNode *task_a = static_cast<const TaskNode *>(a);
    const TaskNode *task_b = static_cast<const TaskNode *>(b);
    if (task_a->ContainsLoopBreak() || task_b->ContainsLoopBreak()) {
      return false;
    }
  }

  auto reg_regions_a = CollectRegisterRegions(a);
  auto reg_regions_b = CollectRegisterRegions(b);

  // For each pair of register regions, check if they refer to the same buffer
  // If buffer is the same, consider it as the same region (simplified)
  for (const auto &region_a : reg_regions_a) {
    for (const auto &region_b : reg_regions_b) {
      // Check if same buffer
      if (region_a->buffer.same_as(region_b->buffer)) {
        return true;
      }
    }
  }
  return false;
}

// Helper function to count register regions in an IRStructure node
inline int CountRegisterRegions(const IRStructure *node) {
  auto reg_regions = CollectRegisterRegions(node);
  return static_cast<int>(reg_regions.size());
}

// Collect all ControlNodes at the top level of the IR tree
// (does not recurse into ControlNodes themselves)
inline void
CollectTopLevelControlNodes(IRStructure *node,
                            std::vector<ControlNode *> &control_nodes) {
  if (!node)
    return;
  if (node->IsControl()) {
    control_nodes.push_back(static_cast<ControlNode *>(node));
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      CollectTopLevelControlNodes(child.get(), control_nodes);
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    CollectTopLevelControlNodes(wrapper->child.get(), control_nodes);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectTopLevelControlNodes(unit->child.get(), control_nodes);
  }
  // IfNode is atomic — don't recurse into it
}

// Collect all TaskNodes at the top level (not inside any ControlNode)
inline void CollectTopLevelTaskNodes(IRStructure *node,
                                     std::vector<TaskNode *> &task_nodes) {
  if (!node)
    return;
  if (node->IsTask()) {
    task_nodes.push_back(static_cast<TaskNode *>(node));
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      CollectTopLevelTaskNodes(child.get(), task_nodes);
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    CollectTopLevelTaskNodes(wrapper->child.get(), task_nodes);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectTopLevelTaskNodes(unit->child.get(), task_nodes);
  }
  // IfNode is atomic — don't recurse into it
}

// Collect all top-level leaf items (TaskNodes and ControlNodes) in order.
// Flattens through SequenceNode/WrapperNode/ScheduleUnit but stops at
// TaskNode and ControlNode.
inline void CollectTopLevelItems(IRStructure *node,
                                 std::vector<IRStructure *> &items) {
  if (!node)
    return;
  if (node->IsTask() || node->IsControl() || node->IsIf()) {
    items.push_back(node);
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      CollectTopLevelItems(child.get(), items);
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    CollectTopLevelItems(wrapper->child.get(), items);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectTopLevelItems(unit->child.get(), items);
  }
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(
    IRStructure *node, std::vector<TaskNodeWithContext> &all_tasks,
    ControlNode *current_control_node = nullptr);

// Structure to represent a data dependency between a producer and consumer unit
struct DepInfo {
  ScheduleUnit *prod_unit;
  ScheduleUnit *cons_unit;
  Buffer buffer;
  std::vector<BufferAccessPoint> producer_pts; // prod's last accesses
  std::vector<BufferAccessPoint> consumer_pts; // cons's first accesses
  bool cross_iter;
  int64_t latency;
};

// Analyze data dependencies among `units` (each a ScheduleUnit wrapping a
// scheduling-level child). `current_loop` is present iff this level is a
// ControlNode body, enabling cross-iteration (loop back-edge) dependencies and
// driving the same-iter vs cross-iter overlap test (see RegionsMayConflict in
// the .cc). The analysis is warpgroup-agnostic (broadcast access-point
// queries); the barrier pass expands per-warpgroup from the returned points.
std::vector<DepInfo>
AnalyzeDependencies(const std::vector<ScheduleUnit *> &units,
                    std::optional<For> current_loop);

// Helper function to print BufferRegion details
inline void PrintBufferRegion(const BufferRegion &region,
                              const std::string &indent) {
  const Buffer &buffer = region->buffer;
  const Region &ranges = region->region;

  std::string buffer_name = buffer->name;
  if (buffer_name.empty())
    buffer_name = "unnamed_buffer";

  LOG(INFO) << indent << "Buffer: " << buffer_name;

  // Get scope information
  String scope_str = buffer.scope();
  LOG(INFO) << indent << "  Scope: " << scope_str;

  // Build shape string
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i > 0)
      shape_ss << ", ";
    shape_ss << buffer->shape[i];
  }
  shape_ss << "]";
  LOG(INFO) << indent << "  Shape: " << shape_ss.str();

  LOG(INFO) << indent << "  Region:";
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range &range = ranges[i];
    std::ostringstream range_ss;
    range_ss << "[" << range->min << ", " << range->min + range->extent
             << ") (extent=" << range->extent << ")";
    LOG(INFO) << indent << "    dim " << i << ": " << range_ss.str();
  }
}

// Helper function to print all stmts in IRStructure
inline void PrintAllStmts(const IRStructure *node, int indent = 0) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode with " << task->stmts.size()
              << " statements:";
    for (size_t i = 0; i < task->stmts.size(); i++) {
      LOG(INFO) << indent_str << "  Statement " << i << ":";
      LOG(INFO) << indent_str + "    " << task->stmts[i];
    }
    LOG(INFO) << indent_str << "  Resource pipe mask: 0x" << std::hex
              << task->GetPipeMask() << std::dec;
    LOG(INFO) << indent_str << "  Latency: " << task->GetLatency()
              << " cycles, II: " << task->GetII()
              << " cycles, warpgroup_id: " << task->GetWarpgroupId();
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    // Print the For statement itself
    LOG(INFO) << indent_str << "  For statement:";
    LOG(INFO) << indent_str + "    " << control->control;

    // Recursively print child statements
    if (control->child) {
      LOG(INFO) << indent_str << "  Loop body:";
      PrintAllStmts(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode *seq = static_cast<const SequenceNode *>(node);
    LOG(INFO) << indent_str << "SequenceNode with " << seq->children.size()
              << " children:";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintAllStmts(seq->children[i].get(), indent + 2);
    }
  } else if (node->IsWrapper()) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    // Recursively print child statements
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Wrapper body:";
      PrintAllStmts(wrapper->child.get(), indent + 2);
    }
  } else if (node->IsScheduleUnit()) {
    const ScheduleUnit *promote = static_cast<const ScheduleUnit *>(node);
    LOG(INFO) << indent_str << "ScheduleUnit:";
    for (const auto &[idx, stmts] : promote->before) {
      for (const auto &stmt : stmts) {
        LOG(INFO) << indent_str << "  Before " << idx << " : " << stmt;
      }
    }
    for (const auto &[idx, stmts] : promote->after) {
      for (const auto &stmt : stmts) {
        LOG(INFO) << indent_str << "  After " << idx << " : " << stmt;
      }
    }
    // Recursively print child statements
    if (promote->child) {
      LOG(INFO) << indent_str << "  Promote body:";
      PrintAllStmts(promote->child.get(), indent + 2);
    }
  } else if (node->IsIf()) {
    const IfNode *if_node = static_cast<const IfNode *>(node);
    LOG(INFO) << indent_str << "IfNode:";
    LOG(INFO) << indent_str << "  Condition: " << if_node->condition;
    if (if_node->then_child) {
      LOG(INFO) << indent_str << "  Then:";
      PrintAllStmts(if_node->then_child.get(), indent + 2);
    }
    if (if_node->else_child) {
      LOG(INFO) << indent_str << "  Else:";
      PrintAllStmts(if_node->else_child.get(), indent + 2);
    }
  }
}

inline Stmt GetLetDecl(const BindNode *bind_node) {
  if (bind_node == nullptr) {
    return Stmt();
  }
  return Bind(bind_node->var, bind_node->value);
}

inline Stmt GetAttrDecl(const AttrStmtNode *attr_node) {
  if (attr_node == nullptr) {
    return Stmt();
  }
  return AttrStmt(attr_node->node, attr_node->attr_key, attr_node->value,
                  Evaluate(0));
}

// Original helper function to print IRStructure (kept for backward
// compatibility)
inline void PrintIRStructure(const IRStructure *node, int indent = 0) {
  if (!node)
    return;

  std::string indent_str(indent * 2, ' ');

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmts: " << task->stmts.size()
              << " statements";
    for (auto &stmt : task->stmts) {
      LOG(INFO) << indent_str << "  stmt: " << stmt;
    }
    for (auto &region : task->GetReadRegions()) {
      LOG(INFO) << indent_str << "  Read Region: " << region;
    }
    for (auto &region : task->GetWriteRegions()) {
      LOG(INFO) << indent_str << "  Write Region: " << region;
    }
    for (auto &var : task->GetReadVars()) {
      LOG(INFO) << indent_str << "  Read Var: " << var;
    }
    for (auto &var : task->GetWriteVars()) {
      LOG(INFO) << indent_str << "  Write Var: " << var;
    }
    LOG(INFO) << indent_str << "  pipe_mask: 0x" << std::hex
              << task->GetPipeMask() << std::dec;
    LOG(INFO) << indent_str << "  latency: " << task->GetLatency() << " cycles";
    LOG(INFO) << indent_str << "  II: " << task->GetII() << " cycles";
    LOG(INFO) << indent_str << "  warpgroup_id: " << task->GetWarpgroupId();
  } else if (node->IsControl()) {
    const ControlNode *control = static_cast<const ControlNode *>(node);
    LOG(INFO) << indent_str << "ControlNode (For loop):";
    if (control->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(control->child.get(), indent + 2);
    }
  } else if (node->IsSequence()) {
    const SequenceNode *seq = static_cast<const SequenceNode *>(node);
    LOG(INFO) << indent_str << "SequenceNode: " << seq->children.size()
              << " children";
    for (size_t i = 0; i < seq->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(seq->children[i].get(), indent + 2);
    }
  } else if (node->IsWrapper()) {
    const WrapperNode *wrapper = static_cast<const WrapperNode *>(node);
    LOG(INFO) << indent_str << "WrapperNode:";
    LOG(INFO) << indent_str << "  Wrapper: " << wrapper->wrapper;
    LOG(INFO) << indent_str << "  Task:";
    PrintIRStructure(wrapper->task.get(), indent + 4);
    if (wrapper->child) {
      LOG(INFO) << indent_str << "  Child:";
      PrintIRStructure(wrapper->child.get(), indent + 2);
    }
  } else if (node->IsScheduleUnit()) {
    const ScheduleUnit *promote = static_cast<const ScheduleUnit *>(node);
    LOG(INFO) << indent_str << "ScheduleUnit:";
    for (const auto &[idx, stmts] : promote->before) {
      for (const auto &stmt : stmts) {
        LOG(INFO) << indent_str << "  Before " << idx << " : " << stmt;
      }
    }
    for (const auto &[idx, stmts] : promote->after) {
      for (const auto &stmt : stmts) {
        LOG(INFO) << indent_str << "  After " << idx << " : " << stmt;
      }
    }
    if (promote->child) {
      LOG(INFO) << indent_str << "  Promote body:";
      PrintIRStructure(promote->child.get(), indent + 2);
    }
  } else if (node->IsIf()) {
    const IfNode *if_node = static_cast<const IfNode *>(node);
    LOG(INFO) << indent_str << "IfNode:";
    LOG(INFO) << indent_str << "  Condition: " << if_node->condition;
    if (if_node->task) {
      LOG(INFO) << indent_str << "  Task:";
      PrintIRStructure(if_node->task.get(), indent + 4);
    }
    if (if_node->then_child) {
      LOG(INFO) << indent_str << "  Then:";
      PrintIRStructure(if_node->then_child.get(), indent + 2);
    }
    if (if_node->else_child) {
      LOG(INFO) << indent_str << "  Else:";
      PrintIRStructure(if_node->else_child.get(), indent + 2);
    }
  }
}

} // namespace tl
} // namespace tvm
