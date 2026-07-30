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
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::String;

// Forward declarations
class IRStructure;
class TaskNode;
class ControlNode;

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

// Special scalar "registers" written by config/control intrinsics and read by
// their consumers. Pipe-order registers model a scalar synchronization call
// that fences the hardware pipe named by its argument without making the call
// itself consume that pipe.
enum class SpecialRegister : uint16_t {
  kNone = 0,
  kHf32Mode = 1 << 0,    // set_hf32_mode      <- read by Cube (MAD) ops
  kPadValue = 1 << 1,    // set_copy_pad_value <- read by padded MTE2 copies
  kLoopControl = 1 << 2, // loop_break         <- read by every sibling (fence)
  kAtomicMode = 1 << 3,  // set_atomic         <- read by GM-store copies
  kPipeMTE1 = 1 << 4,
  kPipeMTE2 = 1 << 5,
  kPipeMTE3 = 1 << 6,
  kPipeCube = 1 << 7,
  kPipeVector = 1 << 8,
  kPipeFixpipe = 1 << 9,
  kPipeScalar = 1 << 10,
};

inline uint16_t PipeSpecialRegisterMask(uint16_t pipe_mask) {
  uint16_t mask = 0;
  auto add_pipe = [&](ResourcePipe pipe, SpecialRegister reg) {
    if (pipe_mask & static_cast<uint16_t>(pipe))
      mask |= static_cast<uint16_t>(reg);
  };
  add_pipe(ResourcePipe::kMTE1, SpecialRegister::kPipeMTE1);
  add_pipe(ResourcePipe::kMTE2, SpecialRegister::kPipeMTE2);
  add_pipe(ResourcePipe::kMTE3, SpecialRegister::kPipeMTE3);
  add_pipe(ResourcePipe::kCube, SpecialRegister::kPipeCube);
  add_pipe(ResourcePipe::kVector, SpecialRegister::kPipeVector);
  add_pipe(ResourcePipe::kFixpipe, SpecialRegister::kPipeFixpipe);
  add_pipe(ResourcePipe::kScalar, SpecialRegister::kPipeScalar);
  return mask;
}

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
// access. The caller bakes in any buffer/storage, is_write, and warpgroup
// filters.
class TaskNode;
using TaskAccessPredicate = std::function<bool(const TaskNode *task)>;

PrimExpr CalculateIterationCount(const std::vector<For> &loops);

// Infer the specialized Ascend issue pipe(s) used by a statement. Statements
// with no specialized operation use the scalar pipe. This is shared by
// IRStructure construction and final PerCoreTask synchronization placement so
// the two phases classify candidate statements identically.
uint16_t GetAscendStmtPipeMask(const Stmt &stmt);
uint16_t GetAscendCallPipeMask(const Call &call);
uint16_t GetAscendCopyPipeMask(const Buffer &src, const Buffer &dst);

// A wrapper introduced at a node relative to its nearest surviving structural
// ancestor. Guards are scheduling metadata rather than schedulable tasks.
class Guard {
public:
  enum class Kind { kCondition, kAttribute };

  virtual ~Guard() = default;
  virtual Kind GetKind() const = 0;
  virtual std::unique_ptr<Guard> Clone() const = 0;
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;
  virtual Stmt Wrap(Stmt body) const = 0;

  bool IsCondition() const { return GetKind() == Kind::kCondition; }
  bool IsAttribute() const { return GetKind() == Kind::kAttribute; }
};

class ConditionGuard final : public Guard {
public:
  explicit ConditionGuard(PrimExpr condition)
      : condition(std::move(condition)) {}

  Kind GetKind() const final { return Kind::kCondition; }
  std::unique_ptr<Guard> Clone() const final {
    return std::make_unique<ConditionGuard>(condition);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    condition = Substitute(condition, {{old_var, new_var}});
  }
  Stmt Wrap(Stmt body) const final {
    return IfThenElse(condition, std::move(body));
  }

  PrimExpr condition;
};

class AttributeGuard final : public Guard {
public:
  AttributeGuard(Any node, String key, PrimExpr value, Span span)
      : node(std::move(node)), key(std::move(key)), value(std::move(value)),
        span(std::move(span)) {}

  Kind GetKind() const final { return Kind::kAttribute; }
  std::unique_ptr<Guard> Clone() const final {
    return std::make_unique<AttributeGuard>(node, key, value, span);
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    if (node.as<PrimExprNode>()) {
      node = Substitute(Downcast<PrimExpr>(node), {{old_var, new_var}});
    }
    if (value.defined()) {
      value = Substitute(value, {{old_var, new_var}});
    }
  }
  Stmt Wrap(Stmt body) const final {
    return AttrStmt(node, key, value, std::move(body), span);
  }

  Any node;
  String key;
  PrimExpr value;
  Span span;
};

using GuardList = std::vector<std::unique_ptr<Guard>>;

// Base class for all IR nodes in scheduling
class IRStructure {
public:
  enum class Kind { kTask, kControl };

  virtual ~IRStructure() = default;
  virtual Kind GetKind() const = 0;
  virtual std::shared_ptr<IRStructure> Clone() const = 0;

  // Scheduling metadata assigned after Z3 scheduling. These used to live on a
  // separate wrapper; scheduling now writes them directly onto the TaskNode or
  // ControlNode participating in the scheduled sequence.
  std::map<int, std::vector<Stmt>> before, after;

  // Guard deltas introduced at this node relative to its nearest
  // surviving structural ancestor.
  const GuardList &GetGuards() const { return guards_; }
  bool HasGuards() const { return !guards_.empty(); }
  // True iff the node carries a real *conditional* guard
  bool HasConditions() const {
    for (const auto &guard : guards_) {
      if (guard->IsCondition())
        return true;
    }
    return false;
  }
  void SetGuards(GuardList guards) { guards_ = std::move(guards); }
  // Prepend a conditional guard (the delta stack is built outermost-last, so
  // new outer guards go to the front).
  void PrependCondition(const PrimExpr &cond) {
    guards_.insert(guards_.begin(), std::make_unique<ConditionGuard>(cond));
  }
  // Prepend an AttrStmt wrapper supported by AutoSchedule.
  void PrependAttribute(const Any &node, const String &key,
                        const PrimExpr &value, const Span &span) {
    guards_.insert(guards_.begin(),
                   std::make_unique<AttributeGuard>(node, key, value, span));
  }

  // Pipeline stage assigned by the Z3 scheduler: producer = 0, consumers
  // NEGATIVE (absolute value = latency in stages). See warpgroup_partition.cc.
  int GetStage() const { return stage_; }
  void SetStage(int stage) { stage_ = stage; }

  // Position in the scheduled sibling list. Together with stage this determines
  // the emitted software-pipeline order inside the current parent.
  size_t GetIndex() const { return index_; }
  void SetIndex(size_t index) { index_ = index; }

  // Helper methods for safe casting
  bool IsTask() const { return GetKind() == Kind::kTask; }
  bool IsControl() const { return GetKind() == Kind::kControl; }

  // Substitute a var inside this node's own guard deltas. Call from each
  // node's SubstituteVar override so warpgroup renaming reaches both If
  // conditions and propagated AttrStmt fields.
  void SubstituteVarInGuards(const Var &old_var, const Var &new_var) {
    for (auto &guard : guards_) {
      guard->SubstituteVar(old_var, new_var);
    }
  }

  void SubstituteVarInSchedulingMetadata(const Var &old_var,
                                         const Var &new_var) {
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

  void CopyBaseMetadataFrom(const IRStructure &other) {
    guards_.clear();
    guards_.reserve(other.guards_.size());
    for (const auto &guard : other.guards_) {
      guards_.push_back(guard->Clone());
    }
    before = other.before;
    after = other.after;
    stage_ = other.stage_;
    index_ = other.index_;
    start_time_ = other.start_time_;
  }

  // Start time assigned by the Z3 scheduler. Only meaningful on scheduled
  // TaskNode/ControlNode children of an ordered list (a ControlNode body or
  // the root list). Expressed as cycle offset relative to that list's start.
  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }

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

  // Indirect read access: read regions/vars plus anything reached transitively
  // through scalar Let bindings (see PropagateReadRegions). Falls back to the
  // direct sets for nodes that don't track an indirect set. Aggregate across
  // children like GetReadRegions()/GetReadVars().
  virtual std::vector<BufferRegion> GetIndirectReadRegions() const {
    return GetReadRegions();
  }
  virtual std::vector<Var> GetIndirectReadVars() const { return GetReadVars(); }

  // Latency estimation
  virtual int64_t GetLatency() const = 0; // Estimated latency in cycles
  virtual int64_t GetII() const = 0;      // Initiation interval in cycles

  // Setters (for analysis passes to update these values)
  virtual void SetLatency(int64_t latency) = 0;
  virtual void SetII(int64_t ii) = 0;

  // Substitute a variable throughout this IR node
  virtual void SubstituteVar(const Var &old_var, const Var &new_var) = 0;

  // Get warpgroup id for this node (kWarpgroupUnassigned if not applicable)
  virtual int GetWarpgroupId() const { return kWarpgroupUnassigned; }

  virtual bool containWarpgroupId(int id) const = 0;

  // Special scalar registers this node *writes* (bitmask of SpecialRegister).
  // Only leaf config/control intrinsics write; everything else returns 0.
  virtual uint16_t GetSpecialWriteMask() const { return 0; }

  // Special scalar registers this node *reads*:
  //   Cube (MAD)     reads the HF32-mode register (derived from the Cube pipe)
  //   padded GM->UB copies read the pad-value register (only copies that carry
  //     data_select / pad_value)
  //   loop_break is a fence: every node implicitly reads the loop-control
  //     register, so any loop_break write orders against all siblings.
  uint16_t GetSpecialReadMask() const {
    uint16_t mask = static_cast<uint16_t>(SpecialRegister::kLoopControl);
    if (UsesPipe(ResourcePipe::kCube))
      mask |= static_cast<uint16_t>(SpecialRegister::kHf32Mode);
    if (ReadsPadValueRegister())
      mask |= static_cast<uint16_t>(SpecialRegister::kPadValue);
    if (ReadsAtomicRegister())
      mask |= static_cast<uint16_t>(SpecialRegister::kAtomicMode);
    mask |= PipeSpecialRegisterMask(GetPipeMask());
    return mask;
  }

  // Whether this node contains a padded GM->UB copy.
  virtual bool ReadsPadValueRegister() const { return false; }

  // Whether this node contains a copy that stores to GM.
  virtual bool ReadsAtomicRegister() const { return false; }

  bool WritesSpecialRegister(SpecialRegister reg) const {
    return (GetSpecialWriteMask() & static_cast<uint16_t>(reg)) != 0;
  }

  // Whether this node is the loop_break marker (writes the loop-control
  // register). Used by warpgroup broadcast + sequence scans.
  bool ContainsLoopBreak() const {
    return WritesSpecialRegister(SpecialRegister::kLoopControl);
  }

private:
  GuardList guards_;
  int stage_{0};
  size_t index_{0};
  int64_t start_time_{0};
};

// Task node: contains a single statement
class TaskNode : public IRStructure {
public:
  Stmt stmt;

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

  // Special scalar registers written by this leaf (config/control intrinsics).
  uint16_t GetSpecialWriteMask() const override { return special_write_mask_; }

  // A padded GM->UB copy (data_select / pad_value) in this leaf reads the
  // pad-value register.
  bool ReadsPadValueRegister() const override { return reads_pad_value_; }

  // A copy that stores to GM in this leaf reads the atomic-mode register.
  bool ReadsAtomicRegister() const override { return reads_atomic_; }

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> GetReadRegions() const override {
    return read_regions_;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    return write_regions_;
  }
  std::vector<Var> GetReadVars() const override { return read_vars_; }
  std::vector<Var> GetWriteVars() const override { return write_vars_; }

  // Indirect read access: the direct read regions/vars plus anything reached
  // transitively through scalar Let bindings (e.g. `a = buf[0]; copy(a)` makes
  // the copy task indirectly read `buf`). Filled by PropagateReadRegions; the
  // direct read_regions_/read_vars_ are never mutated. Falls back to the direct
  // set when propagation has not run, so callers always get a valid answer.
  std::vector<BufferRegion> GetIndirectReadRegions() const override {
    return indirect_computed_ ? indirect_read_regions_ : read_regions_;
  }
  std::vector<Var> GetIndirectReadVars() const override {
    return indirect_computed_ ? indirect_read_vars_ : read_vars_;
  }

  // Latency estimation
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters
  void SetPipeMask(uint16_t mask) { pipe_mask_ = mask; }
  void SetHbmMask(uint16_t mask) { hbm_mask_ = mask; }
  void SetSpecialWriteMask(uint16_t mask) { special_write_mask_ = mask; }
  void SetReadsPadValue(bool v) { reads_pad_value_ = v; }
  void SetReadsAtomic(bool v) { reads_atomic_ = v; }
  void SetPerCoreTaskPipeMask(uint16_t mask) {
    per_core_task_pipe_mask_ = mask;
  }
  bool IsPerCoreTask() const { return per_core_task_pipe_mask_ != 0; }
  uint16_t GetPerCoreTaskPipeMask() const { return per_core_task_pipe_mask_; }
  void AddSpecialWrite(SpecialRegister reg) {
    special_write_mask_ |= static_cast<uint16_t>(reg);
  }
  void SetReadRegions(const std::vector<BufferRegion> &regions) {
    read_regions_ = regions;
  }
  void SetWriteRegions(const std::vector<BufferRegion> &regions) {
    write_regions_ = regions;
  }
  void SetReadVars(const std::vector<Var> &vars) { read_vars_ = vars; }
  void SetWriteVars(const std::vector<Var> &vars) { write_vars_ = vars; }
  void SetIndirectReads(const std::vector<BufferRegion> &regions,
                        const std::vector<Var> &vars, bool computed) {
    indirect_read_regions_ = regions;
    indirect_read_vars_ = vars;
    indirect_computed_ = computed;
  }
  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    Map<Var, PrimExpr> substitution{{old_var, new_var}};
    auto substitute_vars = [&](std::vector<Var> &vars) {
      for (auto &var : vars) {
        if (var.same_as(old_var))
          var = new_var;
      }
    };
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        buffer_remap;
    auto substitute_regions = [&](std::vector<BufferRegion> &regions) {
      for (auto &region : regions) {
        Buffer buffer = region->buffer;
        auto buffer_it = buffer_remap.find(buffer);
        if (buffer_it != buffer_remap.end()) {
          buffer = buffer_it->second;
        } else if (buffer->data.same_as(old_var)) {
          Buffer remapped = buffer;
          remapped.CopyOnWrite()->data = new_var;
          buffer_remap.emplace(buffer, remapped);
          buffer = std::move(remapped);
        }
        region = BufferRegion(std::move(buffer),
                              Substitute(region->region, substitution));
      }
    };
    substitute_vars(read_vars_);
    substitute_vars(write_vars_);
    substitute_vars(indirect_read_vars_);
    substitute_regions(read_regions_);
    substitute_regions(write_regions_);
    substitute_regions(indirect_read_regions_);
    outer_ctx = outer_ctx.Substitute(substitution);
    stmt = Substitute(stmt, substitution);
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInSchedulingMetadata(old_var, new_var);
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

  // Indirect read accumulation (used by PropagateReadRegions). Seed once from
  // the direct sets, then add transitively-reached regions/vars.
  void SeedIndirectReads() {
    indirect_read_regions_ = read_regions_;
    indirect_read_vars_ = read_vars_;
    indirect_computed_ = true;
  }
  void AddIndirectReadRegion(const BufferRegion &region) {
    for (const auto &existing : indirect_read_regions_) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        return;
      }
    }
    indirect_read_regions_.push_back(region);
  }
  void AddIndirectReadVar(const Var &var) {
    for (const auto &existing : indirect_read_vars_) {
      if (existing.same_as(var))
        return;
    }
    indirect_read_vars_.push_back(var);
  }

  bool containWarpgroupId(int id) const override {
    return IsWarpgroupBroadcast(warpgroup_id_) || warpgroup_id_ == id;
  }

private:
  // Resource pipe mask (bitmask of ResourcePipe values)
  uint16_t pipe_mask_{0};
  uint16_t hbm_mask_{0};
  // Special scalar registers written by this leaf (bitmask of SpecialRegister).
  uint16_t special_write_mask_{0};
  // Whether a padded GM->UB copy in this leaf reads the pad-value register.
  bool reads_pad_value_{false};
  bool reads_atomic_{false};
  // Non-zero for a T.PerCoreTask task. The stmt is scheduled as one logical
  // group. During final Stmt emission, synchronization attached to this task is
  // moved beside each enclosed statement that issues on this pipe.
  uint16_t per_core_task_pipe_mask_{0};

  // Memory access regions (collected during analysis)
  std::vector<BufferRegion> read_regions_;
  std::vector<BufferRegion> write_regions_;

  std::vector<Var> read_vars_;
  std::vector<Var> write_vars_;

  // Indirect read sets: direct reads plus reads reached transitively through
  // scalar Let bindings. Populated by PropagateReadRegions;
  // `indirect_computed_` gates whether the Get*Indirect* accessors return these
  // or fall back to the direct sets. Never affects barrier/dependency analysis,
  // which uses the direct read_regions_/read_vars_.
  std::vector<BufferRegion> indirect_read_regions_;
  std::vector<Var> indirect_read_vars_;
  bool indirect_computed_{false};

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
};

// Control node: contains a For operation and an ordered list of child
// IRStructures (the loop body). The SequenceNode level was removed: the loop
// body's ordered children live directly here.
class ControlNode : public IRStructure {
public:
  For control; // The For operation
  std::shared_ptr<TaskNode> task;
  std::vector<std::shared_ptr<IRStructure>> children;

  Kind GetKind() const override { return Kind::kControl; }

  bool BodyHasLoopBreak() const {
    for (const auto &child : children) {
      if (child && child->ContainsLoopBreak())
        return true;
    }
    return false;
  }

  // Resource pipe mask (aggregate from children)
  uint16_t GetPipeMask() const override {
    uint16_t mask = 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetPipeMask();
    }
    return mask;
  }
  uint16_t GetHbmMask() const override {
    uint16_t mask = 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetHbmMask();
    }
    return mask;
  }

  // Special-register writes survive the control boundary and must order
  // sibling nodes outside the loop. loop_break is the sole exception: it only
  // controls the loop that directly contains it and must not fence outer
  // siblings.
  uint16_t GetSpecialWriteMask() const override {
    uint16_t mask = task ? task->GetSpecialWriteMask() : 0;
    for (const auto &child : children) {
      if (child)
        mask |= child->GetSpecialWriteMask();
    }
    return mask & static_cast<uint16_t>(
                      ~static_cast<uint16_t>(SpecialRegister::kLoopControl));
  }

  // A loop reads the pad-value register if any child (or its own task) contains
  // a padded GM->UB copy.
  bool ReadsPadValueRegister() const override {
    for (const auto &child : children) {
      if (child && child->ReadsPadValueRegister())
        return true;
    }
    return task && task->ReadsPadValueRegister();
  }

  // A loop reads the atomic register if any child (or its own task) stores to
  // GM.
  bool ReadsAtomicRegister() const override {
    for (const auto &child : children) {
      if (child && child->ReadsAtomicRegister())
        return true;
    }
    return task && task->ReadsAtomicRegister();
  }

  // Memory access regions (aggregate from children & task)
  std::vector<BufferRegion> GetReadRegions() const override {
    std::vector<BufferRegion> regions;
    for (const auto &child : children) {
      if (child) {
        auto child_regions = child->GetReadRegions();
        regions.insert(regions.end(), child_regions.begin(),
                       child_regions.end());
      }
    }
    if (task) {
      auto task_regions = task->GetReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }
  std::vector<BufferRegion> GetWriteRegions() const override {
    std::vector<BufferRegion> regions;
    for (const auto &child : children) {
      if (child) {
        auto child_regions = child->GetWriteRegions();
        regions.insert(regions.end(), child_regions.begin(),
                       child_regions.end());
      }
    }
    if (task) {
      auto task_regions = task->GetWriteRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  // Variable access (aggregate from children & task)
  std::vector<Var> GetReadVars() const override {
    std::vector<Var> vars;
    for (const auto &child : children) {
      if (child) {
        auto child_vars = child->GetReadVars();
        vars.insert(vars.end(), child_vars.begin(), child_vars.end());
      }
    }
    if (task) {
      auto task_vars = task->GetReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<Var> GetWriteVars() const override {
    std::vector<Var> vars;
    for (const auto &child : children) {
      if (child) {
        auto child_vars = child->GetWriteVars();
        vars.insert(vars.end(), child_vars.begin(), child_vars.end());
      }
    }
    if (task) {
      auto task_vars = task->GetWriteVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }

  // Indirect read access (aggregate from children & task)
  std::vector<Var> GetIndirectReadVars() const override {
    std::vector<Var> vars;
    for (const auto &child : children) {
      if (child) {
        auto child_vars = child->GetIndirectReadVars();
        vars.insert(vars.end(), child_vars.begin(), child_vars.end());
      }
    }
    if (task) {
      auto task_vars = task->GetIndirectReadVars();
      vars.insert(vars.end(), task_vars.begin(), task_vars.end());
    }
    return vars;
  }
  std::vector<BufferRegion> GetIndirectReadRegions() const override {
    std::vector<BufferRegion> regions;
    for (const auto &child : children) {
      if (child) {
        auto child_regions = child->GetIndirectReadRegions();
        regions.insert(regions.end(), child_regions.begin(),
                       child_regions.end());
      }
    }
    if (task) {
      auto task_regions = task->GetIndirectReadRegions();
      regions.insert(regions.end(), task_regions.begin(), task_regions.end());
    }
    return regions;
  }

  void SubstituteVar(const Var &old_var, const Var &new_var) override {
    for (auto &child : children) {
      if (child)
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
    SubstituteVarInGuards(old_var, new_var);
    SubstituteVarInSchedulingMetadata(old_var, new_var);
  }

  // Latency estimation (aggregate from child)
  int64_t GetLatency() const override { return latency_; }
  int64_t GetII() const override { return ii_; }

  // Setters
  void SetLatency(int64_t latency) override { latency_ = latency; }
  void SetII(int64_t ii) override { ii_ = ii; }

  void SetIIperIter(int64_t ii) { ii_per_iter_ = ii; }

  int64_t GetIIperIter() const { return ii_per_iter_; }

  // Clone method
  std::shared_ptr<IRStructure> Clone() const override;

  bool containWarpgroupId(int id) const override {
    for (const auto &child : children) {
      if (child && child->containWarpgroupId(id))
        return true;
    }
    return false;
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

// Structure to store TaskNode with its loop context information
struct TaskNodeWithContext {
  TaskNode *task;
  ControlNode *control_node{nullptr}; // nullptr if not inside a loop
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

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(
    IRStructure *node, std::vector<TaskNodeWithContext> &all_tasks,
    ControlNode *current_control_node = nullptr);

// List overload: collect across an ordered list of top-level nodes.
inline void CollectAllTaskNodesWithContext(
    const std::vector<std::shared_ptr<IRStructure>> &nodes,
    std::vector<TaskNodeWithContext> &all_tasks) {
  for (const auto &node : nodes) {
    CollectAllTaskNodesWithContext(node.get(), all_tasks, nullptr);
  }
}

// Structure to represent a data dependency between a producer and consumer node
struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  Buffer buffer;
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

// Table of user-declared manual multi-buffer buffers, keyed by the buffer's
// data Var, value = number of versions N (the user allocated the buffer with an
// explicit leading dim of size N and writes the version index into dim 0
// themselves). Built per kernel from the `tl.manual_multi_buffer` annotation
// and threaded through dependency analysis (to solve the real cross-iteration
// distance) and the barrier pass.
using ManualMultiBufferTable = std::unordered_map<const VarNode *, int64_t>;

// Analyze data dependencies among scheduled TaskNode/ControlNode nodes.
// `loop_stack` is the full enclosing serial loop stack at this level
// (outermost-first). Empty means root/list level. Non-empty means the last loop
// is the current loop body, enabling cross-iteration dependencies.
// `manual` maps user-declared manual multi-buffer data Vars to their version
// count; empty means no manual buffers.
// An `i == j` control-node self-dependency whose producer→consumer pairs are
// already established anywhere inside that node's own subtree is dropped.
std::vector<DepInfo>
AnalyzeDependencies(const std::vector<IRStructure *> &nodes,
                    const std::vector<For> &loop_stack,
                    const ManualMultiBufferTable &manual = {});

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
    LOG(INFO) << indent_str << "TaskNode statement:";
    LOG(INFO) << indent_str + "    " << task->stmt;
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
    LOG(INFO) << indent_str << "  Loop body:";
    for (size_t i = 0; i < control->children.size(); i++) {
      PrintAllStmts(control->children[i].get(), indent + 2);
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

  for (const auto &guard : node->GetGuards()) {
    if (guard->IsAttribute()) {
      const auto *attribute_guard =
          static_cast<const AttributeGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [attr " << attribute_guard->key << "] "
                << attribute_guard->node;
    } else {
      const auto *condition_guard =
          static_cast<const ConditionGuard *>(guard.get());
      LOG(INFO) << indent_str << "  [guard] " << condition_guard->condition;
    }
  }

  if (node->IsTask()) {
    const TaskNode *task = static_cast<const TaskNode *>(node);
    LOG(INFO) << indent_str << "TaskNode:";
    LOG(INFO) << indent_str << "  stmt: " << task->stmt;
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
    LOG(INFO) << indent_str << "  Loop body: " << control->children.size()
              << " children";
    for (size_t i = 0; i < control->children.size(); i++) {
      LOG(INFO) << indent_str << "  Child " << i << ":";
      PrintIRStructure(control->children[i].get(), indent + 2);
    }
  }
}

} // namespace tl
} // namespace tvm
