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

// Check if there is a normal dependency (loop break or let vars) between two
// IRStructures
bool HasNormalDependency(const IRStructure *a, const IRStructure *b);

// Get buffer dependencies between two IRStructures (excluding read-after-read)
std::set<Buffer> GetBufferDependencies(const IRStructure *a,
                                       const IRStructure *b);

// Get region-group dependencies between two IRStructures using a
// pre-computed RegionPartition. Excludes read-after-read.
std::set<int> GetRegionGroupDependencies(const IRStructure *a,
                                         const IRStructure *b,
                                         const RegionPartition &partition);

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
  // SequenceNode's children. If `target_seq` is provided, the resulting
  // RegionPartition (keyed on the wrapped ScheduleUnits) is stashed onto it
  // for later consumption by the barrier pass.
  std::vector<std::shared_ptr<IRStructure>>
  Z3SchedulePython(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                   SequenceNode *target_seq = nullptr) {
    size_t n = nodes.size();

    auto fallback = [&]() {
      std::vector<std::shared_ptr<IRStructure>> wrapped;
      wrapped.reserve(nodes.size());
      for (size_t i = 0; i < nodes.size(); ++i) {
        auto unit = std::make_shared<ScheduleUnit>();
        unit->child = nodes[i];
        unit->SetStartTime(0);
        unit->SetLatency(unit->child->GetLatency());
        unit->SetII(unit->child->GetII());
        wrapped.push_back(std::move(unit));
      }
      if (target_seq != nullptr) {
        std::vector<IRStructure *> units_for_partition;
        units_for_partition.reserve(wrapped.size());
        for (const auto &u : wrapped)
          units_for_partition.push_back(u.get());
        auto p = RegionPartition::Build(units_for_partition,
                                        /*current_loop=*/std::nullopt);
        target_seq->region_partition =
            std::make_unique<RegionPartition>(std::move(p));
      }
      return wrapped;
    };

    if (n <= 1) {
      return fallback();
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<ffi::Function> z3_schedule_func =
          ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                        "topological sort";
        return fallback();
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
        const IRStructure *node = nodes[i].get();
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Pass pipe mask directly as resource flags
        resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
      }

      // Build region partition once for this sequence level (no enclosing
      // loop). Drives the dependency edges below.
      std::vector<IRStructure *> units_for_partition;
      units_for_partition.reserve(n);
      for (const auto &child : nodes) {
        units_for_partition.push_back(child.get());
      }
      RegionPartition partition = RegionPartition::Build(
          units_for_partition, /*current_loop=*/std::nullopt);

      // Collect data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasNormalDependency(nodes[i].get(), nodes[j].get())) {
            data_deps.emplace_back(i, j, latencies[i]);
            continue;
          }
          auto &&rgs = GetRegionGroupDependencies(nodes[i].get(),
                                                  nodes[j].get(), partition);
          int64_t latency = -1;
          for (int rg : rgs) {
            auto ra = nodes[i]->GetLastAccessPoints(rg, partition, false);
            auto wa = nodes[i]->GetLastAccessPoints(rg, partition, true);
            auto rb = nodes[j]->GetFirstAccessPoints(rg, partition, false);
            auto wb = nodes[j]->GetFirstAccessPoints(rg, partition, true);
            auto update = [&](const BufferAccessPoint &a,
                              const BufferAccessPoint &b) {
              int64_t l = a.timestamp + a.task()->GetLatency() - b.timestamp;
              ICHECK(l <= nodes[i]->GetLatency());
              if (l < 0)
                l = 0;
              if (l > latency)
                latency = l;
            };
            for (const auto &a : wa) {
              for (const auto &b : rb)
                update(a, b);
              for (const auto &b : wb)
                update(a, b);
            }
            for (const auto &a : ra) {
              for (const auto &b : wb)
                update(a, b);
            }
          }
          if (latency >= 0) {
            data_deps.emplace_back(i, j, latency);
          }
        }
      }

      // Collect resource dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasResourceDependency(nodes[i].get(), nodes[j].get())) {
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
        return fallback();
      }

      // Sort by start_time, then by original index
      std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
      start_time_with_idx.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        start_time_with_idx.emplace_back(start_times[i], i);
      }
      std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
                [](const std::pair<int64_t, size_t> &a,
                   const std::pair<int64_t, size_t> &b) {
                  if (a.first != b.first)
                    return a.first < b.first;
                  return a.second < b.second;
                });

      // Wrap each child in a ScheduleUnit with its start_time, in sorted order.
      std::vector<std::shared_ptr<IRStructure>> sorted_units;
      sorted_units.reserve(n);
      std::unordered_map<IRStructure *, IRStructure *> remap;
      remap.reserve(n);
      for (const auto &p : start_time_with_idx) {
        auto unit = std::make_shared<ScheduleUnit>();
        unit->child = nodes[p.second];
        unit->SetStartTime(p.first);
        unit->SetLatency(unit->child->GetLatency());
        unit->SetII(unit->child->GetII());
        remap[nodes[p.second].get()] = unit.get();
        sorted_units.push_back(std::move(unit));
      }
      if (target_seq != nullptr) {
        partition.RemapUnits(remap);
        target_seq->region_partition =
            std::make_unique<RegionPartition>(std::move(partition));
      }
      return sorted_units;

    } catch (const std::exception &e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return fallback();
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                      "falling back to topological sort";
      return fallback();
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
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        data_deps; // (i, j, distance, latency)
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
    std::vector<std::vector<int64_t>> group_indices;
    for (const auto &entry : memory_limits_) {
      if (entry.second > 0) {
        scope_to_group[entry.first] = (int)group_capacities.size();
        group_capacities.push_back(entry.second);
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
          !IsL0CBuffer(buffer)) {
        return; // Only consider on-chip buffers for multi-buffer
      }
      if (visited_buffers.count(buffer)) {
        return;
      }
      visited_buffers.insert(buffer);
      std::string scope = IsL1Buffer(buffer)
                              ? "shared.l1"
                              : (IsL0CBuffer(buffer) ? "shared.l0c" : "shared");
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

    // Build region partition once for this loop body. `MayOverlap`
    // substitutes the loop var with two fresh Vars (each Bound to the loop
    // range), giving a single test that subsumes same-iter and any
    // cross-iter distance.
    std::vector<IRStructure *> units_for_partition;
    units_for_partition.reserve(n);
    for (auto *node : nodes) {
      units_for_partition.push_back(node);
    }
    RegionPartition partition =
        RegionPartition::Build(units_for_partition,
                               /*current_loop=*/ctrl->control);

    // Collect data dependencies with distance
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i < j && HasNormalDependency(nodes[i], nodes[j])) {
          data_deps.emplace_back(i, j, 0, latencies[i]);
          continue;
        }
        auto &&rgs = GetRegionGroupDependencies(nodes[i], nodes[j], partition);
        for (int rg : rgs) {
          Buffer buffer = partition.GetRepresentativeBuffer(rg);
          auto ra = nodes[i]->GetLastAccessPoints(rg, partition, false);
          auto wa = nodes[i]->GetLastAccessPoints(rg, partition, true);
          auto rb = nodes[j]->GetFirstAccessPoints(rg, partition, false);
          auto wb = nodes[j]->GetFirstAccessPoints(rg, partition, true);
          int64_t latency = -1;
          auto update = [&](const BufferAccessPoint &a,
                            const BufferAccessPoint &b) {
            int64_t l = a.timestamp + a.task()->GetLatency() - b.timestamp;
            ICHECK(l <= nodes[i]->GetLatency());
            if (l < 0)
              l = 0;
            if (l > latency)
              latency = l;
          };
          for (const auto &a : wa) {
            for (const auto &b : rb)
              update(a, b);
            for (const auto &b : wb)
              update(a, b);
          }
          for (const auto &a : ra) {
            for (const auto &b : wb)
              update(a, b);
          }
          if (latency >= 0) {
            if (i < j) {
              data_deps.emplace_back(i, j, 0, latency);
            } else {
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
              data_deps.emplace_back(i, j, distance, latency);
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
    ii = return_val.get<2>();

    // Sort children by start_time (with original index as tie-breaker).
    // Derive a per-task stage from (start_time / ii). Z3 schedules
    // `start = k * ii + r` so `start_time / ii` recovers k. Producers (low k,
    // earliest in z3's absolute time) get stage 0; later tasks get NEGATIVE
    // stages.
    //
    // Downstream (warpgroup_partition.cc) emits the body with
    // `shift = (max_stage - stage) * step`. With max_stage == 0:
    //   - producer (stage 0)         → shift 0   → processes iter k at
    //                                              physical iter k (runs
    //                                              first in the pipeline)
    //   - consumer (stage -D)        → shift D   → processes iter k-D at
    //                                              physical iter k (runs D
    //                                              iters delayed)
    // which matches SW-pipeline semantics: producer at the "early" stage
    // writes for the current iter, consumer at the "late" stage reads
    // producer's output D iters later.
    std::map<IRStructure *, int> stage_map;
    std::map<IRStructure *, int64_t> start_time_map;
    std::vector<int> raw_k(n, 0);
    int min_k_raw = std::numeric_limits<int>::max();
    for (size_t i = 0; i < n; ++i) {
      start_time_map[nodes[i]] = start_times[i];
      // ii is the inner-loop II just returned by z3. Guard the num_stages==1
      // path where ii may not represent a modulo II — every task then lands
      // on stage 0 and we fall back to the non-pipelined emitter.
      int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
      raw_k[i] = k;
      min_k_raw = std::min(min_k_raw, k);
    }
    if (min_k_raw == std::numeric_limits<int>::max())
      min_k_raw = 0;
    for (size_t i = 0; i < n; ++i) {
      // Producer (smallest k) maps to stage 0; later tasks get negative
      // stages whose magnitude is their delay relative to the producer.
      stage_map[nodes[i]] = min_k_raw - raw_k[i];
    }
    std::vector<std::pair<int64_t, size_t>> start_time_with_idx;
    start_time_with_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      int stage = stage_map[nodes[i]];
      start_time_with_idx.emplace_back(start_times[i] + stage * ii, i);
    }

    // Sort by start_time, then by original index
    std::sort(start_time_with_idx.begin(), start_time_with_idx.end(),
              [](const std::pair<int64_t, size_t> &a,
                 const std::pair<int64_t, size_t> &b) {
                if (a.first != b.first)
                  return a.first < b.first;
                return a.second < b.second;
              });

    // Create sorted node list and a parallel vector of start_times in
    // post-sort order so we can write start_time onto each ScheduleUnit
    // after wrapping below.
    std::vector<IRStructure *> sorted_nodes;
    std::vector<int64_t> sorted_start_times;
    sorted_nodes.reserve(n);
    sorted_start_times.reserve(n);
    for (const auto &p : start_time_with_idx) {
      sorted_nodes.push_back(nodes[p.second]);
      sorted_start_times.push_back(p.first);
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
    auto SolveConflictVar = [&]() -> bool {
      auto HasVarRawDep = [](const IRStructure *producer,
                             const IRStructure *consumer) -> bool {
        for (const auto &w : producer->GetWriteVars()) {
          for (const auto &r : consumer->GetReadVars()) {
            if (SameVar(w, r))
              return true;
          }
        }
        return false;
      };
      for (size_t i = 0; i < n; ++i)
        if (IsVarDecl(seq_body->children[i].get())) {
          for (size_t j = 0; j < n; ++j) {
            if (i == j)
              continue;

            auto node_i = seq_body->children[i].get();
            auto node_j = seq_body->children[j].get();
            int rem_stage_j = stage_map[node_j];

            if (!HasVarRawDep(node_i, node_j))
              continue;

            if (stage_map[node_j] == stage_map[node_i])
              continue;

            auto node_i_task = static_cast<TaskNode *>(node_i);
            auto node_i_let_stmt = node_i_task->stmts[0].as<LetStmtNode>();

            auto cloned_value = node_i_let_stmt->value;
            auto cloned_let_stmt =
                LetStmt(node_i_let_stmt->var.copy_with_suffix(""), cloned_value,
                        Evaluate(0));
            auto cloned_task = std::make_shared<TaskNode>();
            cloned_task->stmts.push_back(cloned_let_stmt);
            cloned_task->SetReadRegions(node_i_task->GetReadRegions());
            cloned_task->SetWriteRegions(node_i_task->GetWriteRegions());
            cloned_task->SetReadVars(node_i_task->GetReadVars());
            {
              auto write_vars = node_i_task->GetWriteVars();
              for (auto &v : write_vars) {
                if (v.same_as(node_i_let_stmt->var)) {
                  v = cloned_let_stmt->var;
                }
              }
              cloned_task->SetWriteVars(write_vars);
            }
            cloned_task->SetLatency(node_i_task->GetLatency());
            cloned_task->SetII(node_i_task->GetII());
            cloned_task->SetPipeMask(node_i_task->GetPipeMask());
            stage_map[cloned_task.get()] = rem_stage_j;

            for (size_t k = j; k < n; ++k) {
              auto node_k = seq_body->children[k].get();
              if (rem_stage_j != stage_map[node_k])
                continue;
              if (HasVarRawDep(node_i, node_k)) {
                node_k->SubstituteVar(node_i_let_stmt->var,
                                      cloned_let_stmt->var);
                stage_map[node_k] = rem_stage_j;
              }
            }

            seq_body->children.insert(seq_body->children.begin() + j,
                                      std::move(cloned_task));
            n += 1;
            return true; // Conflict resolved, restart the loop
          }
        }
      return false;
    };
    // Resolve conflicts until no more conflicts exist or max iterations
    // reached (to avoid infinite loop)
    int conflict_count = 0;
    while (SolveConflictVar() && ++conflict_count < 100)
      ;

    std::unordered_map<IRStructure *, IRStructure *> remap;
    remap.reserve(seq_body->children.size());
    for (size_t k = 0; k < seq_body->children.size(); ++k) {
      auto unit = std::make_shared<ScheduleUnit>();
      auto *child_ptr = seq_body->children[k].get();
      unit->stage = stage_map[child_ptr];
      auto it = start_time_map.find(child_ptr);
      unit->SetStartTime(it != start_time_map.end() ? it->second : 0);
      unit->child = std::move(seq_body->children[k]);
      unit->SetLatency(unit->child->GetLatency());
      unit->SetII(unit->child->GetII());
      remap[child_ptr] = unit.get();
      seq_body->children[k] = std::move(unit);
    }
    partition.RemapUnits(remap);
    seq_body->region_partition =
        std::make_unique<RegionPartition>(std::move(partition));

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
