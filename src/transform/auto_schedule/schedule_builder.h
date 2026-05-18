#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../op/utils.h"
#include "./barrier.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;

class TaskUnionFind;
struct ComponentInfo;

// Warp specialization architecture enum
enum class WarpSpecializeArch : uint8_t {
  kHopper,
  kBlackwell,
  kAscend,
  kUnsupported,
};

// Configuration for warp specialization
struct WarpSpecializeConfig {
  WarpSpecializeArch arch = WarpSpecializeArch::kUnsupported;
};

// Extract all sequential task nodes from the IR structure tree
void GatherTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                     std::vector<std::shared_ptr<IRStructure>> &task_nodes);
void GatherTaskNodesSingle(
    const std::shared_ptr<IRStructure> &node,
    std::vector<std::shared_ptr<IRStructure>> &task_nodes);

// Check if two regions refer to the same buffer
bool SameBuffer(const BufferRegion &a, const BufferRegion &b);

// Check if two variables are the same
bool SameVar(const Var &a, const Var &b);

bool HasDependency(const IRStructure *a, const IRStructure *b);

// Check if two IRStructures have data dependency (excluding read-after-read)
bool HasRegisterDependency(const IRStructure *a, const IRStructure *b);

// Get shared buffers two IRStructures both access (excluding read-after-read)
std::set<Buffer> GetSharedDependencies(const IRStructure *a,
                                       const IRStructure *b);

// Check if an IRStructure has any register region
bool HasRegisterRegion(const IRStructure *node);

// Check if two IRStructures have resource dependency (use same hardware
// resource)
bool HasResourceDependency(const IRStructure *a, const IRStructure *b);

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  void Build(std::shared_ptr<IRStructure> &root) {
    ScheduleRecursive(root, {});
  }

  // Recursive scheduling function
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(std::shared_ptr<IRStructure> &node,
                         const std::set<Buffer> &used_buffers);

  // Z3-based scheduler that calls Python implementation via FFI
  std::vector<IRStructure *>
  Z3SchedulePython(const std::vector<IRStructure *> &nodes) {
    size_t n = nodes.size();
    if (n <= 1) {
      if (n == 1) {
        nodes[0]->SetStartTime(0);
      }
      return nodes;
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<ffi::Function> z3_schedule_func =
          ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                        "topological sort";
        return nodes;
      }

      // Prepare input data
      std::vector<int64_t> latencies;
      std::vector<int64_t> iis;
      std::vector<int64_t> resource_flags;
      std::vector<std::pair<int64_t, int64_t>> data_deps;
      std::vector<std::pair<int64_t, int64_t>> resource_deps;

      latencies.reserve(n);
      iis.reserve(n);
      resource_flags.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        const IRStructure *node = nodes[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Pass pipe mask directly as resource flags
        resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
      }

      // Collect data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasDependency(nodes[i], nodes[j])) {
            data_deps.emplace_back(i, j);
          }
        }
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(nodes[i], nodes[j])) {
            resource_deps.emplace_back(i, j);
          }
        }
      }

      // Convert vectors to TVM containers
      ffi::Array<int64_t> tvm_latencies;
      ffi::Array<int64_t> tvm_iis;
      ffi::Array<int64_t> tvm_resource_flags;
      ffi::Array<ffi::Array<int64_t>> tvm_data_deps;
      ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;

      for (auto val : latencies) {
        tvm_latencies.push_back(val);
      }
      for (auto val : iis) {
        tvm_iis.push_back(val);
      }
      for (auto val : resource_flags) {
        tvm_resource_flags.push_back(val);
      }
      for (const auto &dep : data_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_data_deps.push_back(pair);
      }
      for (const auto &dep : resource_deps) {
        ffi::Array<int64_t> pair;
        pair.push_back(dep.first);
        pair.push_back(dep.second);
        tvm_resource_deps.push_back(pair);
      }

      // Extract results
      // Python function returns only start_times, C++ side will sort by
      // start_time
      auto start_times =
          z3_schedule_func
              .value()(tvm_latencies, tvm_iis, tvm_resource_flags,
                       tvm_data_deps, tvm_resource_deps)
              .cast<ffi::Array<int64_t>>();

      if (start_times.size() != n) {
        LOG(WARNING) << "Python Z3 scheduler returned invalid results (size "
                        "mismatch), falling back to topological sort";
        return nodes;
      }

      // Apply start times to nodes
      for (size_t i = 0; i < n; ++i) {
        nodes[i]->SetStartTime(start_times[i]);
      }

      // Create sorted task list based on start_time (and original index as
      // tie-breaker)
      std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
      start_time_with_idx.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        start_time_with_idx.emplace_back(start_times[i], i);
      }

      // Sort by start_time, then by original index
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t> &a,
                   const std::pair<int64_t, size_t> &b) {
                  if (a.first != b.first)
                    return a.first < b.first;
                  return a.second < b.second;
                });

      // Create sorted node list
      std::vector<IRStructure *> sorted_nodes;
      sorted_nodes.reserve(n);
      for (const auto &p : start_time_with_idx) {
        sorted_nodes.push_back(nodes[p.second]);
      }

      return sorted_nodes;

    } catch (const std::exception &e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return nodes;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                      "falling back to topological sort";
      return nodes;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  void Z3SchedulePythonLoop(ControlNode *ctrl,
                            const std::set<Buffer> &used_buffers) {
    if (ctrl->child == nullptr) {
      LOG(WARNING)
          << "Z3SchedulePythonLoop called on a control node without child";
      return;
    }
    std::vector<std::shared_ptr<IRStructure>> flat_children;
    if (!ctrl->child->IsSequence()) {
      GatherTaskNodesSingle(ctrl->child, flat_children);
    } else {
      auto seq_node = static_cast<SequenceNode *>(ctrl->child.get());
      GatherTaskNodes(seq_node->children, flat_children);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = flat_children;
    ctrl->child = std::move(seq_node);

    auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
    std::vector<IRStructure *> nodes;
    nodes.reserve(seq_body->children.size());
    for (const auto &child : seq_body->children) {
      nodes.push_back(child.get());
    }

    size_t n = nodes.size();
    int64_t ii = 1; // default II

    auto num_stages = 1;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages = num_stages_val.value().cast<IntImm>()->value;
    }

    static std::optional<ffi::Function> z3_schedule_loop_func =
        ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
    if (!z3_schedule_loop_func.has_value()) {
      LOG(FATAL) << "Python Z3 loop scheduler not registered, falling back "
                    "to topological sort";
    }

    // Prepare input data
    std::vector<int64_t> latencies;
    std::vector<int64_t> iis;
    std::vector<int64_t> resource_flags;
    std::vector<std::tuple<int64_t, int64_t, int64_t>>
        data_deps; // (i, j, distance)
    std::vector<std::pair<int64_t, int64_t>> resource_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = nodes[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());

      // Pass pipe mask directly as resource flags
      resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
    }

    // Helper function to check if a buffer is written before being read.
    // Recurses in program order through the child subtrees and (at TaskNode
    // leaves) into the TIR Stmt itself, so a child that internally produces
    // before consuming counts as write-first. Conservative on if/while:
    // conditional writes and loops that may not execute do NOT count as
    // guaranteed writes.
    auto check_buffer_write_first = [&nodes](const Buffer &buffer) {
      for (const auto &node : nodes) {
        auto order = node->ClassifyBufferAccess(buffer);
        if (order == BufferAccessOrder::kUntouched)
          continue;
        return order == BufferAccessOrder::kWriteFirst;
      }
      return false;
    };

    // Collect all shared buffers and detect multi-buffer candidates.
    // Then we need to add variables and constraints in Z3 scheduler for these
    // buffers.
    std::vector<int64_t> buffer_sizes;
    std::set<Buffer> visited_buffers;
    std::map<Buffer, size_t> buffer_var_id;
    // Build memory groups from memory_limits_ map.
    // std::map iteration is ordered by key: "shared" then "shared.l1".
    std::map<std::string, int> scope_to_group;
    std::vector<int64_t> group_capacities;
    std::vector<std::vector<int64_t>> group_indices;
    for (const auto &entry : memory_limits_) {
      if (entry.second > 0) {
        scope_to_group[entry.first] = (int)group_capacities.size();
        group_capacities.push_back(entry.second);
        group_indices.push_back({});
      }
    }
    std::map<Buffer, int64_t> annotated_buffer_versions;
    for (const auto &buffer_access : ctrl->GetBufferAccessInfo()) {
      const auto &buffer = buffer_access.buffer;
      if (!IsSharedBuffer(buffer) && !IsL1Buffer(buffer) &&
          !IsL0CBuffer(buffer)) {
        continue; // Only consider on-chip buffers for multi-buffer
      }
      if (visited_buffers.count(buffer)) {
        continue;
      }
      visited_buffers.insert(buffer);
      std::string scope = IsL1Buffer(buffer)
                              ? "shared.l1"
                              : (IsL0CBuffer(buffer) ? "shared.l0c" : "shared");
      int group_id = scope_to_group[scope];
      // If the buffer is used outside the loop or is read before being written,
      // we cannot multi-buffer it
      if (used_buffers.count(buffer) || !check_buffer_write_first(buffer)) {
        // These buffers cannot be multi-buffered, and cannot share memory with
        // other buffers either, so we directly reduce the capacity of the
        // memory group by their size
        group_capacities[group_id] -= GetBufferSize(buffer);
      } else {
        // Check if this buffer has a manually annotated version count
        auto override_it =
            buffer_versions_overrides_.find(buffer->data->name_hint);
        if (override_it != buffer_versions_overrides_.end()) {
          int64_t override_val = override_it->second;
          annotated_buffer_versions[buffer] = override_val;
          group_capacities[group_id] -= GetBufferSize(buffer) * override_val;
        } else {
          buffer_var_id[buffer] = buffer_sizes.size();
          group_indices[group_id].push_back((int64_t)buffer_sizes.size());
          buffer_sizes.push_back(GetBufferSize(buffer));
        }
      }
    }

    // Collect data dependencies with distance
    // distance = 0 if i < j (same iteration), distance = 1 if i > j (next
    // iteration)
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (HasDependency(nodes[i], nodes[j])) {
          if (i < j) {
            data_deps.emplace_back(i, j, 0);
          } else {
            if (HasRegisterDependency(nodes[i], nodes[j])) {
              data_deps.emplace_back(i, j, 1);
            } else {
              auto deps = GetSharedDependencies(nodes[i], nodes[j]);
              for (const auto &buffer : deps) {
                auto it = buffer_var_id.find(buffer);
                int64_t distance = 1;
                if (it != buffer_var_id.end()) {
                  // Encode the buffer id in the negative distance for the Z3
                  // scheduler to recognize
                  distance = -(int64_t)it->second - 1;
                } else {
                  // Check if this buffer has a manually annotated version count
                  auto ann_it = annotated_buffer_versions.find(buffer);
                  if (ann_it != annotated_buffer_versions.end()) {
                    distance = ann_it->second;
                  }
                }
                data_deps.emplace_back(i, j, distance);
              }
            }
          }
        }
      }
    }

    // Collect resource dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasResourceDependency(nodes[i], nodes[j])) {
          resource_deps.emplace_back(i, j);
        }
      }
    }

    // Convert vectors to TVM containers
    ffi::Array<int64_t> tvm_latencies;
    ffi::Array<int64_t> tvm_iis;
    ffi::Array<int64_t> tvm_resource_flags;
    ffi::Array<ffi::Array<int64_t>>
        tvm_data_deps; // each element is [i, j, distance]
    ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;
    ffi::Array<int64_t> tvm_buffer_sizes;

    for (auto val : latencies) {
      tvm_latencies.push_back(val);
    }
    for (auto val : iis) {
      tvm_iis.push_back(val);
    }
    for (auto val : resource_flags) {
      tvm_resource_flags.push_back(val);
    }
    for (const auto &dep : data_deps) {
      ffi::Array<int64_t> triple;
      triple.push_back(std::get<0>(dep));
      triple.push_back(std::get<1>(dep));
      triple.push_back(std::get<2>(dep));
      tvm_data_deps.push_back(triple);
    }
    for (const auto &dep : resource_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_resource_deps.push_back(pair);
    }
    for (auto val : buffer_sizes) {
      tvm_buffer_sizes.push_back(val);
    }

    // Build memory groups for FFI: each inner array is [capacity, idx0, idx1,
    // ...]
    ffi::Array<ffi::Array<int64_t>> tvm_memory_groups;
    for (size_t g = 0; g < group_capacities.size(); ++g) {
      ffi::Array<int64_t> group;
      group.push_back(group_capacities[g]);
      for (auto idx : group_indices[g]) {
        group.push_back(idx);
      }
      tvm_memory_groups.push_back(group);
    }

    // Extract results
    // Python function returns (start_times, buffer_versions, best_ii) as a
    // tuple
    auto return_val =
        z3_schedule_loop_func
            .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                     tvm_data_deps, tvm_resource_deps, tvm_buffer_sizes,
                     tvm_memory_groups)
            .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>,
                             int64_t>>();

    ffi::Array<int64_t> start_times = return_val.get<0>();
    ffi::Array<int64_t> buffer_versions = return_val.get<1>();
    ii = return_val.get<2>();

    // Apply start times and promote flags to nodes
    for (size_t i = 0; i < n; ++i) {
      nodes[i]->SetStartTime(start_times[i]);
    }
    std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
    start_time_with_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      start_time_with_idx.emplace_back(start_times[i], i);
    }

    // Sort by start_time, then by original index
    std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
              [](const std::pair<int64_t, size_t> &a,
                 const std::pair<int64_t, size_t> &b) {
                if (a.first != b.first)
                  return a.first < b.first;
                return a.second < b.second;
              });

    // Create sorted node list
    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes.reserve(n);
    for (const auto &p : start_time_with_idx) {
      sorted_nodes.push_back(nodes[p.second]);
    }

    // Reorder children if order changed
    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < nodes.size(); ++i) {
        node_to_index[nodes[i]] = i;
      }

      std::vector<std::shared_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());
      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        reordered_children.push_back(std::move(seq_body->children[it->second]));
      }
      seq_body->children = std::move(reordered_children);
    }

    // Reorder & Copy Let-defined variables
    auto IsVarDecl = [&](IRStructure *node) -> bool {
      if (!node)
        return false;

      if (node->IsTask()) {
        auto task = static_cast<TaskNode *>(node);
        if (task->stmts.size() == 1) {
          return task->stmts[0].as<LetStmtNode>() != nullptr;
        }
      }
      return false;
    };

    for (auto &node : seq_body->children) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->child = std::move(node);
      node = std::move(unit);
    }

    // Estimate overall latency: II * tripcount
    // Get tripcount from For loop extent and step
    int64_t tripcount = 100; // default if not constant
    const ForNode *for_node = ctrl->control.get();
    PrimExpr loop_extent = for_node->extent;
    PrimExpr loop_step = for_node->step.has_value()
                             ? for_node->step.value()
                             : IntImm(DataType::Int(32), 1);

    if (const auto *extent_int = loop_extent.as<IntImmNode>()) {
      if (const auto *step_int = loop_step.as<IntImmNode>()) {
        // Calculate ceil(extent / step)
        int64_t extent = extent_int->value;
        int64_t step = step_int->value;
        if (step > 0) {
          // ceil(extent / step) = (extent + step - 1) / step
          tripcount = (extent + step - 1) / step;
        } else {
          // Invalid step, use extent as fallback
          tripcount = extent;
        }
      } else {
        // Step is not constant, use 100 as default
        tripcount = 100;
      }
    } else {
      // Extent is not constant, use 100 as default
      tripcount = 100;
    }
    int64_t overall_latency = ii * tripcount;

    // Set II and latency on the ControlNode (which delegates to child)
    ctrl->SetII(overall_latency);
    ctrl->SetLatency(overall_latency);
    ctrl->SetIIperIter(ii);
    ctrl->buffer_num_versions.clear();
    for (auto &[buffer, versions] : annotated_buffer_versions) {
      ctrl->buffer_num_versions[buffer] = versions;
    }
    for (auto &[buffer, id] : buffer_var_id) {
      ctrl->buffer_num_versions[buffer] = buffer_versions[id];
    }
  }

  // Set memory limit for a given buffer scope (e.g. "shared", "shared.l1")
  void SetMemoryLimit(const std::string &scope, int64_t bytes) {
    memory_limits_[scope] = bytes;
  }

  // Set manual buffer version overrides (from annotate_buffer_versions).
  // Keys are buffer data variable names, values are fixed num_versions.
  void
  SetBufferVersionsOverride(const std::map<std::string, int64_t> &overrides) {
    buffer_versions_overrides_ = overrides;
  }

private:
  std::map<std::string, int64_t> memory_limits_ = {{"shared", 48 * 1024}};
  std::map<std::string, int64_t> buffer_versions_overrides_;
};

} // namespace tl
} // namespace tvm
