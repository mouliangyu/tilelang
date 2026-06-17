#pragma once
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./barrier.h"
#include "./ir_structure.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

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

// Check if two variables are the same
bool SameVar(const Var &a, const Var &b);

// Check if there is a variable dependency between two IRStructures
bool HasVarDependency(const IRStructure *a, const IRStructure *b);

// Check if an IRStructure has any register region
bool HasRegisterRegion(const IRStructure *node);

// Check if two IRStructures have resource dependency (use same hardware
// resource)
bool HasResourceDependency(const IRStructure *a, const IRStructure *b);

// Builder that collects ScheduleUnits from IRStructure
class ScheduleUnitBuilder {
public:
  void Build(std::shared_ptr<IRStructure> &root) { ScheduleRecursive(root); }

  // Recursive scheduling function
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(std::shared_ptr<IRStructure> &node);

  // Z3-based scheduler that calls Python implementation via FFI.
  // Wraps each scheduled child in a ScheduleUnit carrying its start_time and
  // returns the units (as shared_ptr<IRStructure>) in start_time order (ties
  // broken by original index). Callers can assign the result directly to a
  // SequenceNode's children.
  std::vector<std::shared_ptr<IRStructure>>
  Z3SchedulePython(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
    size_t n = nodes.size();
    std::vector<std::shared_ptr<IRStructure>> wrapped;
    wrapped.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->child = nodes[i];
      unit->SetStartTime(0);
      unit->SetLatency(unit->child->GetLatency());
      unit->SetII(unit->child->GetII());
      wrapped.push_back(std::move(unit));
    }

    if (n <= 1) {
      return wrapped;
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<ffi::Function> z3_schedule_func =
          ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                        "topological sort";
        return wrapped;
      }

      std::vector<ScheduleUnit *> units;
      units.reserve(n);
      for (const auto &u : wrapped) {
        units.push_back(static_cast<ScheduleUnit *>(u.get()));
      }

      // Prepare input data
      std::vector<int64_t> latencies;
      std::vector<int64_t> iis;
      std::vector<int64_t> resource_flags;
      std::vector<std::tuple<int64_t, int64_t, int64_t>> data_deps;
      std::vector<std::pair<int64_t, int64_t>> resource_deps;

      latencies.reserve(n);
      iis.reserve(n);
      resource_flags.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        const IRStructure *node = units[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Pass pipe mask directly as resource flags
        resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
      }

      // Variable (Let-var write -> read) data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasVarDependency(units[i], units[j])) {
            data_deps.emplace_back(i, j, latencies[i]);
          }
        }
      }

      // Loop-break control dependencies
      for (size_t i = 0; i < n; ++i) {
        if (!units[i]->ContainsLoopBreak())
          continue;
        for (size_t j = 0; j < n; ++j) {
          if (i < j) {
            data_deps.emplace_back(i, j, latencies[i]);
          } else if (i > j) {
            data_deps.emplace_back(j, i, 0);
          }
        }
      }

      // Buffer-region data dependencies
      auto deps = AnalyzeDependencies(units, /*current_loop=*/std::nullopt);
      std::unordered_map<ScheduleUnit *, size_t> unit_idx;
      for (size_t i = 0; i < n; ++i) {
        unit_idx[units[i]] = i;
      }
      for (const auto &dep : deps) {
        size_t i = unit_idx[dep.prod_unit];
        size_t j = unit_idx[dep.cons_unit];
        data_deps.emplace_back(i, j, dep.latency);
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(units[i], units[j])) {
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
        pair.push_back(std::get<0>(dep));
        pair.push_back(std::get<1>(dep));
        pair.push_back(std::get<2>(dep));
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
        return wrapped;
      }

      // Write start_times onto the already-wrapped units, then sort them in
      // place by start_time (stable: ties keep original program order).
      for (size_t i = 0; i < n; ++i) {
        units[i]->SetStartTime(start_times[i]);
      }

      std::stable_sort(
          wrapped.begin(), wrapped.end(),
          [](const std::shared_ptr<IRStructure> &a,
             const std::shared_ptr<IRStructure> &b) {
            return static_cast<ScheduleUnit *>(a.get())->GetStartTime() <
                   static_cast<ScheduleUnit *>(b.get())->GetStartTime();
          });
      return wrapped;

    } catch (const std::exception &e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return wrapped;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                      "falling back to topological sort";
      return wrapped;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  void Z3SchedulePythonLoop(ControlNode *ctrl) {
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

    // Wrap each loop-body child in a ScheduleUnit
    for (auto &child : seq_body->children) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->child = std::move(child);
      unit->SetLatency(unit->child->GetLatency());
      unit->SetII(unit->child->GetII());
      child = std::move(unit);
    }

    std::vector<ScheduleUnit *> units;
    units.reserve(seq_body->children.size());
    for (const auto &child : seq_body->children) {
      units.push_back(static_cast<ScheduleUnit *>(child.get()));
    }

    size_t n = units.size();

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
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        data_deps; // (i, j, distance, latency)
    std::vector<std::pair<int64_t, int64_t>> resource_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = units[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());
      resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
    }

    // Multi-buffer eligible set from AnnotateMultiBufferEligible pass.
    // The pass runs before IfConditionExtract and records on the For loop
    // which on-chip buffers are safe to multi-buffer (written before any
    // read inside the loop, not referenced outside).
    std::set<Buffer> eligible_set;
    {
      auto v = ctrl->control->annotations.Get("multi_buffer_eligible");
      if (v.has_value()) {
        for (const auto &b : v.value().cast<Array<Buffer>>()) {
          eligible_set.insert(b);
        }
      }
    }

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
    std::vector<std::string> group_scopes;
    std::vector<std::vector<int64_t>> group_indices;
    for (const auto &entry : memory_limits_) {
      if (entry.second > 0) {
        scope_to_group[entry.first] = (int)group_capacities.size();
        group_capacities.push_back(entry.second);
        group_scopes.push_back(entry.first);
        group_indices.push_back({});
      }
    }
    std::map<Buffer, int64_t> annotated_buffer_versions;
    auto enumerate_buffers = [&](auto &&visit) {
      for (const auto &r : ctrl->GetReadRegions())
        visit(r->buffer);
      for (const auto &r : ctrl->GetWriteRegions())
        visit(r->buffer);
    };
    enumerate_buffers([&](const Buffer &buffer) {
      if (!IsSharedBuffer(buffer) && !IsL1Buffer(buffer) &&
          !IsL0CBuffer(buffer) && !IsL0ABuffer(buffer) &&
          !IsL0BBuffer(buffer)) {
        return; // Only consider on-chip buffers for multi-buffer
      }
      if (visited_buffers.count(buffer)) {
        return;
      }
      visited_buffers.insert(buffer);
      std::string scope;
      if (IsL1Buffer(buffer)) {
        scope = "shared.l1";
      } else if (IsL0CBuffer(buffer)) {
        scope = "shared.l0c";
      } else if (IsL0ABuffer(buffer)) {
        scope = "shared.l0a";
      } else if (IsL0BBuffer(buffer)) {
        scope = "shared.l0b";
      } else {
        scope = "shared";
      }
      int group_id = scope_to_group[scope];
      if (!eligible_set.count(buffer)) {
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
    });

    // Validate buffer capacities upfront: if the minimum requirement (one
    // version of each eligible buffer) already exceeds the remaining capacity,
    // Z3 will never find a feasible solution.
    for (size_t g = 0; g < group_capacities.size(); ++g) {
      int64_t capacity = group_capacities[g];
      const auto &idxs = group_indices[g];
      if (!idxs.empty()) {
        if (capacity < 0) {
          LOG(FATAL)
              << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
              << "\" capacity is negative (" << capacity
              << "). Non-eligible and fixed-version buffers already exceed"
              << " the memory limit. Try using T.annotate_unlimit_memory(\""
              << group_scopes[g] << "\") or reduce buffer sizes.";
        }
        int64_t min_usage = 0;
        for (int64_t idx : idxs) {
          min_usage += buffer_sizes[idx];
        }
        if (min_usage > capacity) {
          LOG(FATAL) << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
                     << "\" capacity " << capacity
                     << " is insufficient for minimum buffer usage "
                     << min_usage << ". Try using T.annotate_unlimit_memory(\""
                     << group_scopes[g] << "\") or reduce buffer sizes.";
        }
      }
    }

    // Variable (Let-var write -> read) data dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasVarDependency(units[i], units[j])) {
          data_deps.emplace_back(i, j, 0, latencies[i]);
        }
      }
    }

    // Loop-break control dependencies
    for (size_t i = 0; i < n; ++i) {
      if (!units[i]->ContainsLoopBreak())
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (i < j) {
          data_deps.emplace_back(i, j, 0, latencies[i]);
          data_deps.emplace_back(j, i, 1, 0);
        } else if (i > j) {
          data_deps.emplace_back(i, j, 1, latencies[i]);
          data_deps.emplace_back(j, i, 0, 0);
        }
      }
    }

    // Buffer-region data dependencies
    auto deps = AnalyzeDependencies(units, /*current_loop=*/ctrl->control);
    std::unordered_map<ScheduleUnit *, size_t> unit_idx;
    for (size_t i = 0; i < n; ++i) {
      unit_idx[units[i]] = i;
    }
    for (const auto &dep : deps) {
      size_t i = unit_idx[dep.prod_unit];
      size_t j = unit_idx[dep.cons_unit];
      if (!dep.cross_iter) {
        data_deps.emplace_back(i, j, 0, dep.latency);
      } else {
        const Buffer &buffer = dep.buffer;
        auto it = buffer_var_id.find(buffer);
        int64_t distance = 1;
        if (it != buffer_var_id.end()) {
          // Encode the buffer id in the negative distance for the Z3
          // scheduler to recognize.
          distance = -(int64_t)it->second - 1;
        } else {
          // Check if this buffer has a manually annotated version count.
          auto ann_it = annotated_buffer_versions.find(buffer);
          if (ann_it != annotated_buffer_versions.end()) {
            distance = ann_it->second;
          }
        }
        data_deps.emplace_back(i, j, distance, dep.latency);
      }
    }

    // Collect resource dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasResourceDependency(units[i], units[j])) {
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
      triple.push_back(std::get<3>(dep));
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
    // 3-tuple. Stages are computed in C++ from start_times and II — see below.
    auto return_val =
        z3_schedule_loop_func
            .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                     tvm_data_deps, tvm_resource_deps, tvm_buffer_sizes,
                     tvm_memory_groups)
            .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int64_t>,
                             int64_t>>();

    ffi::Array<int64_t> start_times = return_val.get<0>();
    ffi::Array<int64_t> buffer_versions = return_val.get<1>();
    int64_t ii = return_val.get<2>();

    // Derive a per-task stage from (start_time / ii). Z3 schedules
    // `start = k * ii + r` so `start_time / ii` recovers k. Producers (low k,
    // earliest in z3's absolute time) get stage 0; later tasks get NEGATIVE
    // stages.
    std::vector<int> raw_k(n, 0);
    int min_k_raw = std::numeric_limits<int>::max();
    for (size_t i = 0; i < n; ++i) {
      units[i]->SetStartTime(start_times[i]);
      int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
      raw_k[i] = k;
      min_k_raw = std::min(min_k_raw, k);
    }
    for (size_t i = 0; i < n; ++i) {
      units[i]->stage = min_k_raw - raw_k[i];
    }
    auto phys_time = [&](const std::shared_ptr<IRStructure> &c) -> int64_t {
      auto *u = static_cast<ScheduleUnit *>(c.get());
      return u->GetStartTime() + u->stage * ii;
    };
    std::stable_sort(seq_body->children.begin(), seq_body->children.end(),
                     [&](const std::shared_ptr<IRStructure> &a,
                         const std::shared_ptr<IRStructure> &b) {
                       return phys_time(a) < phys_time(b);
                     });

    // Reorder & Copy Let-defined variables. Children are ScheduleUnit
    // wrappers; unwrap to reach the underlying TaskNode.
    auto is_var_decl = [&](const ScheduleUnit *unit) -> bool {
      if (!unit || !unit->child)
        return false;
      IRStructure *node = unit->child.get();
      if (node->IsTask()) {
        auto task = static_cast<TaskNode *>(node);
        if (task->stmts.size() == 1) {
          return task->stmts[0].as<BindNode>() != nullptr;
        }
      }
      return false;
    };
    auto solve_conflict_var = [&]() -> bool {
      for (size_t i = 0; i < n; ++i) {
        auto unit_i = static_cast<ScheduleUnit *>(seq_body->children[i].get());
        if (is_var_decl(unit_i)) {
          for (size_t j = 0; j < n; ++j) {
            if (i == j)
              continue;
            auto unit_j =
                static_cast<ScheduleUnit *>(seq_body->children[j].get());
            if (!HasVarDependency(unit_i, unit_j))
              continue;
            if (unit_j->stage == unit_i->stage)
              continue;
            auto node_i_task = static_cast<TaskNode *>(unit_i->child.get());
            auto node_i_bind = node_i_task->stmts[0].as<BindNode>();

            // Clone the producer task, then rename its Let var so the
            // conflicting consumer stage gets its own copy.
            auto cloned_bind = Bind(node_i_bind->var.copy_with_suffix(""),
                                    node_i_bind->value, node_i_bind->span);
            auto cloned_task =
                std::static_pointer_cast<TaskNode>(node_i_task->Clone());
            cloned_task->stmts = {cloned_bind};
            cloned_task->SetReadVars(node_i_task->GetReadVars());
            {
              auto write_vars = node_i_task->GetWriteVars();
              for (auto &v : write_vars) {
                if (v.same_as(node_i_bind->var)) {
                  v = cloned_bind->var;
                }
              }
              cloned_task->SetWriteVars(write_vars);
            }
            // Wrap the clone in a ScheduleUnit at the consumer stage, sharing
            // the producer's start_time.
            auto cloned_unit = std::make_shared<ScheduleUnit>();
            cloned_unit->child = cloned_task;
            cloned_unit->stage = unit_j->stage;
            cloned_unit->SetStartTime(unit_j->GetStartTime());
            cloned_unit->SetLatency(cloned_task->GetLatency());
            cloned_unit->SetII(cloned_task->GetII());

            for (size_t k = j; k < n; ++k) {
              auto unit_k =
                  static_cast<ScheduleUnit *>(seq_body->children[k].get());
              if (unit_j->stage != unit_k->stage)
                continue;
              if (HasVarDependency(unit_i, unit_k)) {
                unit_k->SubstituteVar(node_i_bind->var, cloned_bind->var);
                unit_k->stage = unit_j->stage;
              }
            }

            seq_body->children.insert(seq_body->children.begin() + j,
                                      std::move(cloned_unit));
            n += 1;
            return true; // Conflict resolved, restart the loop
          }
        }
      }
      return false;
    };
    // Resolve conflicts until no more conflicts exist or max iterations
    // reached (to avoid infinite loop)
    int conflict_count = 0;
    while (solve_conflict_var() && ++conflict_count < 100)
      ;

    // Estimate overall latency: II * tripcount
    int64_t overall_latency = 0;
    for (const auto &child : seq_body->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      overall_latency =
          std::max(overall_latency, unit->GetStartTime() + unit->GetLatency());
    }
    overall_latency += ii * (ctrl->GetTripCount() - 1);

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
