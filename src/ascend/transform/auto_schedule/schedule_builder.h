#pragma once
#include <tvm/arith/analyzer.h>
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

// Check if two variables are the same
bool SameVar(const Var &a, const Var &b);

// Check if there is a variable dependency between two IRStructures
bool HasVarDependency(const IRStructure *a, const IRStructure *b);

// Check if two IRStructures have resource dependency (use same hardware
// resource)
bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask = UINT16_MAX);

// Builder that schedules TaskNode/ControlNode children in IRStructure
class ScheduleBuilder {
public:
  // Recursive scheduling function
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(std::shared_ptr<IRStructure> &node,
                         std::vector<For> &loop_stack);

  // Schedule an ordered child list in place (root list or a ControlNode body):
  // gather + recurse into each child, then Z3-schedule and write the reordered
  // list back.
  void ScheduleList(std::vector<std::shared_ptr<IRStructure>> &children);

  // Z3-based scheduler that calls Python implementation via FFI.
  // Writes start_time onto each scheduled child and returns the children in
  // start_time order (ties broken by original index). Callers can assign the
  // result directly to an ordered child list (root list or ControlNode body).
  std::vector<std::shared_ptr<IRStructure>>
  Z3SchedulePython(const std::vector<std::shared_ptr<IRStructure>> &children) {
    size_t n = children.size();
    std::vector<std::shared_ptr<IRStructure>> scheduled = children;
    for (const auto &node : scheduled) {
      node->SetStartTime(0);
    }

    if (n <= 1) {
      return scheduled;
    }

    try {
      // Get the Python-registered function using ffi::Function::GetGlobal
      static std::optional<ffi::Function> z3_schedule_func =
          ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
      if (!z3_schedule_func.has_value()) {
        LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                        "topological sort";
        return scheduled;
      }

      std::vector<IRStructure *> nodes;
      nodes.reserve(n);
      for (const auto &u : scheduled) {
        nodes.push_back(u.get());
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
        const IRStructure *node = nodes[i];
        latencies.push_back(node->GetLatency());
        iis.push_back(node->GetII());

        // Pass pipe mask directly as resource flags
        resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
      }

      // Variable (Let-var write -> read) data dependencies
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          if (HasVarDependency(nodes[i], nodes[j])) {
            data_deps.emplace_back(i, j, latencies[i]);
          }
        }
      }

      // Special-register ordering: a node writing register R must be ordered
      // against every node reading R (RAW forward, WAR backward), preserving
      // program order. loop_break writes kLoopControl, which every node reads,
      // so it fences against all siblings; hf32/pad match only their consumers.
      for (size_t i = 0; i < n; ++i) {
        uint8_t wmask = nodes[i]->GetSpecialWriteMask();
        if (wmask == 0)
          continue;
        for (size_t j = 0; j < n; ++j) {
          if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
            continue;
          if (i < j) {
            data_deps.emplace_back(i, j, latencies[i]);
          } else {
            data_deps.emplace_back(j, i, 0);
          }
        }
      }

      // Buffer-region data dependencies
      auto deps =
          AnalyzeDependencies(nodes, /*loop_stack=*/{}, manual_multi_buffer_);
      std::unordered_map<IRStructure *, size_t> node_idx;
      for (size_t i = 0; i < n; ++i) {
        node_idx[nodes[i]] = i;
      }
      for (const auto &dep : deps) {
        size_t i = node_idx[dep.prod_node];
        size_t j = node_idx[dep.cons_node];
        data_deps.emplace_back(i, j, dep.latency);
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
        return scheduled;
      }

      // Write start_times onto the scheduled nodes, then sort them in place by
      // start_time (stable: ties keep original program order).
      for (size_t i = 0; i < n; ++i) {
        nodes[i]->SetStartTime(start_times[i]);
      }

      std::stable_sort(scheduled.begin(), scheduled.end(),
                       [](const std::shared_ptr<IRStructure> &a,
                          const std::shared_ptr<IRStructure> &b) {
                         return a->GetStartTime() < b->GetStartTime();
                       });
      return scheduled;

    } catch (const std::exception &e) {
      LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                   << ", falling back to topological sort";
      return scheduled;
    } catch (...) {
      LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                      "falling back to topological sort";
      return scheduled;
    }
  }

  // Z3-based scheduler for loops that calls Python implementation via FFI
  // with distance-aware dependencies
  void Z3SchedulePythonLoop(ControlNode *ctrl,
                            const std::vector<For> &loop_stack) {
    if (ctrl->children.empty()) {
      LOG(WARNING)
          << "Z3SchedulePythonLoop called on a control node without children";
      return;
    }

    std::vector<IRStructure *> nodes;
    nodes.reserve(ctrl->children.size());
    for (const auto &child : ctrl->children) {
      nodes.push_back(child.get());
    }

    size_t n = nodes.size();

    auto num_stages = 1;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages = num_stages_val.value().cast<IntImm>()->value;
    }

    // When false, the Z3 loop scheduler keeps each resource's tasks within a
    // single II window (max(start) - min(start) < II). Sourced from the loop's
    // "enable_offset" annotation; defaults to false (constraint applied).
    bool enable_offset = false;
    auto enable_offset_val =
        ctrl->control.get()->annotations.Get("enable_offset");
    if (enable_offset_val.has_value()) {
      enable_offset = enable_offset_val.value().cast<bool>();
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
    std::vector<std::pair<int64_t, int64_t>> stage_order_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = nodes[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());
      resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
    }

    // Multi-buffer eligible set from AnnotateMultiBufferEligible pass.
    // The pass runs before AutoSchedule and records on the For loop
    // which on-chip buffers are safe to multi-buffer (written before any
    // read inside the loop, not referenced outside).
    std::set<const VarNode *> eligible_storage_set;
    {
      auto v = ctrl->control->annotations.Get("multi_buffer_eligible");
      if (v.has_value()) {
        for (const auto &b : v.value().cast<Array<Buffer>>()) {
          eligible_storage_set.insert(b->data.get());
        }
      }
    }

    // Collect all shared buffers and detect multi-buffer candidates.
    // Then we need to add variables and constraints in Z3 scheduler for these
    // buffers.
    std::vector<int64_t> buffer_sizes;
    std::map<const VarNode *, std::vector<Buffer>> storage_to_buffers;
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
    auto get_buffer_size = [](const Buffer &buffer) -> int64_t {
      // Bits-based: dtype.bits()*lanes()/8 truncates sub-byte dtypes to 0
      // (fp4 -> 0). Accumulate elements, then ceil(elements*bits/8).
      int64_t elem_bits = buffer->dtype.bits() * buffer->dtype.lanes();
      PrimExpr elems = IntImm(DataType::Int(64), 1);
      for (const auto &dim : buffer->shape) {
        elems *= dim;
      }
      PrimExpr size = FloorDiv(elems * IntImm(DataType::Int(64), elem_bits) +
                                   IntImm(DataType::Int(64), 7),
                               IntImm(DataType::Int(64), 8));
      arith::Analyzer analyzer;
      auto size_val = analyzer.Simplify(size);
      if (const auto *int_imm = size_val.as<IntImmNode>()) {
        return int_imm->value;
      }
      return 0;
    };
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
      auto &aliases = storage_to_buffers[buffer->data.get()];
      bool seen_alias = false;
      for (const auto &alias : aliases) {
        if (alias.same_as(buffer)) {
          seen_alias = true;
          break;
        }
      }
      if (!seen_alias)
        aliases.push_back(buffer);
    });

    for (const auto &[storage_key, buffers] : storage_to_buffers) {
      ICHECK(!buffers.empty());
      const Buffer &buffer = buffers.front();
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
      if (!eligible_storage_set.count(storage_key)) {
        // These buffers cannot be multi-buffered, and cannot share memory with
        // other buffers either, so we directly reduce the capacity of the
        // memory group by their size
        group_capacities[group_id] -= get_buffer_size(buffer);
      } else {
        // Alias buffers created by T.view/T.reshape share one storage Var. They
        // must share one version variable so dependency distances and barrier
        // ring sizes stay consistent no matter which logical Buffer appears at
        // an access point.
        // Check if this buffer has a manually annotated version count
        auto override_it =
            buffer_versions_overrides_.find(storage_key->name_hint);
        if (override_it != buffer_versions_overrides_.end()) {
          int64_t override_val = override_it->second;
          for (const auto &alias : buffers) {
            annotated_buffer_versions[alias] = override_val;
          }
          group_capacities[group_id] -= get_buffer_size(buffer) * override_val;
        } else {
          size_t id = buffer_sizes.size();
          for (const auto &alias : buffers) {
            buffer_var_id[alias] = id;
          }
          group_indices[group_id].push_back((int64_t)id);
          buffer_sizes.push_back(get_buffer_size(buffer));
        }
      }
    }

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
        if (HasVarDependency(nodes[i], nodes[j])) {
          data_deps.emplace_back(i, j, 0, latencies[i]);
        }
      }
    }

    // Special-register ordering: a node writing register R must be ordered
    // against every node reading R (RAW forward, WAR backward), preserving
    // program order. loop_break writes kLoopControl, which every node reads,
    // so it fences against all siblings; hf32/pad match only their consumers.
    // The distance bit encodes intra-iter (0) vs next-iter (1) direction.
    for (size_t i = 0; i < n; ++i) {
      uint8_t wmask = nodes[i]->GetSpecialWriteMask();
      if (wmask == 0)
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
          continue;
        if (i < j) {
          data_deps.emplace_back(i, j, 0, latencies[i]);
          data_deps.emplace_back(j, i, 1, 0);
        } else {
          data_deps.emplace_back(i, j, 1, latencies[i]);
          data_deps.emplace_back(j, i, 0, 0);
        }
      }
    }

    // Buffer-region data dependencies
    std::vector<For> current_loop_stack = loop_stack;
    current_loop_stack.push_back(ctrl->control);
    auto deps =
        AnalyzeDependencies(nodes, current_loop_stack, manual_multi_buffer_);
    std::unordered_map<IRStructure *, size_t> node_idx;
    for (size_t i = 0; i < n; ++i) {
      node_idx[nodes[i]] = i;
    }
    for (const auto &dep : deps) {
      size_t i = node_idx[dep.prod_node];
      size_t j = node_idx[dep.cons_node];
      if (dep.distance >= 0) {
        data_deps.emplace_back(i, j, dep.distance, dep.latency);
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
        if (HasResourceDependency(nodes[i], nodes[j],
                                  ~((uint16_t)ResourcePipe::kScalar))) {
          resource_deps.emplace_back(i, j);
        }
      }
    }

    // Stage-order constraints for copied Let variables.
    //
    // A Let task `a = buf[...]` only READS buffers (it writes a scalar var).
    // When `a` is consumed in a different pipeline stage, solve_conflict_var
    // clones the Let into that consumer's stage and prepends it to the loop
    // body. The clone re-reads `buf` at the consumer's iteration. If `buf` is
    // also WRITTEN inside the loop, that re-read must observe the value from
    // before the write — otherwise the clone reads a stale/overwritten buffer.
    //
    // AnalyzeDependencies already feeds the Let's own WAR edge to Z3, but only
    // constrains the Let's original stage, not the stages of nodes that will
    // receive a copy. Here we add: for every node U that (transitively) reads a
    // Let-defined var, and every WAR consumer W of that Let (a node that writes
    // the buffer the Let reads, same-iter or cross-iter), require k_U <= k_W in
    // the Z3 (k, r) model. Equality is allowed because the clone is placed at
    // the front of the loop body, so it reads before any write in that
    // iteration.
    if (enable_offset) {

      auto bind_of = [&](size_t idx) -> const BindNode * {
        IRStructure *node = nodes[idx];
        if (node && node->IsTask()) {
          auto task = static_cast<TaskNode *>(node);
          return task->stmt.as<BindNode>();
        }
        return nullptr;
      };
      std::unordered_map<const VarNode *, size_t> let_var_to_node;
      for (size_t i = 0; i < n; ++i) {
        if (auto *bind = bind_of(i)) {
          let_var_to_node[bind->var.get()] = i;
        }
      }
      std::vector<std::set<size_t>> unit_let_deps(n);
      for (size_t i = 0; i < n; ++i) {
        for (const auto &rv : nodes[i]->GetIndirectReadVars()) {
          auto it = let_var_to_node.find(rv.get());
          if (it != let_var_to_node.end())
            unit_let_deps[i].insert(it->second);
        }
      }
      // Collect WAR consumers per Let node
      std::vector<std::set<size_t>> let_war_consumers(n);
      for (const auto &dep : deps) {
        size_t p = node_idx[dep.prod_node];
        if (bind_of(p) != nullptr) {
          let_war_consumers[p].insert(node_idx[dep.cons_node]);
        }
      }
      for (size_t u = 0; u < n; ++u) {
        for (size_t l : unit_let_deps[u]) {
          for (size_t w : let_war_consumers[l]) {
            if (u == w)
              continue;
            stage_order_deps.emplace_back((int64_t)u, (int64_t)w);
          }
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
    ffi::Array<ffi::Array<int64_t>> tvm_stage_order_deps;
    for (const auto &dep : stage_order_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_stage_order_deps.push_back(pair);
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
                     tvm_memory_groups, tvm_stage_order_deps, enable_offset)
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
    int64_t min_start_time = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < n; ++i) {
      int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
      raw_k[i] = k;
      min_k_raw = std::min(min_k_raw, k);
      min_start_time = std::min(min_start_time, start_times[i]);
    }
    for (size_t i = 0; i < n; ++i) {
      // When offset is disabled, collapse every task into a single pipeline
      // stage so warpgroup_partition emits a flat (non-pipelined) loop body.
      nodes[i]->SetStage(enable_offset ? (min_k_raw - raw_k[i]) : 0);
      // Adjust start_time to be relative to the earliest start_time, so that
      // the first task starts at 0.
      nodes[i]->SetStartTime(start_times[i] - min_start_time);
    }
    auto phys_time = [&](const std::shared_ptr<IRStructure> &c) -> int64_t {
      return c->GetStartTime() + c->GetStage() * ii;
    };
    std::stable_sort(ctrl->children.begin(), ctrl->children.end(),
                     [&](const std::shared_ptr<IRStructure> &a,
                         const std::shared_ptr<IRStructure> &b) {
                       return phys_time(a) < phys_time(b);
                     });

    // Reorder & Copy Let-defined variables.
    auto is_var_decl = [&](const IRStructure *node) -> bool {
      if (!node)
        return false;
      if (node->IsTask()) {
        auto task = static_cast<const TaskNode *>(node);
        return task->stmt.as<BindNode>() != nullptr;
      }
      return false;
    };
    // Map each Let-defined var to the node that declares it.
    std::unordered_map<const VarNode *, size_t> let_var_to_node;
    for (size_t i = 0; i < n; ++i) {
      if (is_var_decl(ctrl->children[i].get())) {
        auto *bind = static_cast<TaskNode *>(ctrl->children[i].get())
                         ->stmt.as<BindNode>();
        let_var_to_node[bind->var.get()] = i;
      }
    }

    // Resolve Let-var stage conflicts
    std::map<int, std::set<size_t>>
        stage_to_decls;                      // stage -> decl node indices
    std::map<int, int64_t> stage_base_start; // stage -> floored start time
    for (size_t u = 0; u < n; ++u) {
      int s = ctrl->children[u]->GetStage();
      for (const auto &rv : ctrl->children[u]->GetIndirectReadVars()) {
        auto it = let_var_to_node.find(rv.get());
        if (it == let_var_to_node.end())
          continue;
        size_t d = it->second;
        if (ctrl->children[d]->GetStage() == s)
          continue; // decl already in the consumer's stage: no clone needed
        stage_to_decls[s].insert(d);
        int64_t st = ctrl->children[u]->GetStartTime();
        auto bit = stage_base_start.find(s);
        if (bit == stage_base_start.end())
          stage_base_start[s] = st;
        else
          bit->second = std::min(bit->second, st);
      }
    }

    std::vector<std::shared_ptr<IRStructure>> new_clones;
    for (auto &[s, decls] : stage_to_decls) {
      int64_t base_start = (ii > 0) ? stage_base_start[s] / ii * ii : 0;
      std::vector<std::pair<Var, Var>> renames; // old var -> fresh clone var
      std::vector<std::shared_ptr<TaskNode>> stage_clones;
      for (size_t d : decls) {
        auto *decl_task = static_cast<TaskNode *>(ctrl->children[d].get());
        auto *decl_bind = decl_task->stmt.as<BindNode>();
        auto new_var = decl_bind->var.copy_with_suffix("");
        auto cloned_task =
            std::static_pointer_cast<TaskNode>(decl_task->Clone());
        cloned_task->stmt = Bind(new_var, decl_bind->value, decl_bind->span);
        cloned_task->SetReadVars(decl_task->GetReadVars());
        auto write_vars = decl_task->GetWriteVars();
        for (auto &v : write_vars) {
          if (v.same_as(decl_bind->var))
            v = new_var;
        }
        cloned_task->SetWriteVars(write_vars);
        cloned_task->SetStage(s);
        cloned_task->SetStartTime(base_start);
        renames.emplace_back(decl_bind->var, new_var);
        stage_clones.push_back(cloned_task);
      }
      // Rewire chained references among the clones
      for (auto &clone : stage_clones)
        for (auto &[old_var, new_var] : renames)
          clone->SubstituteVar(old_var, new_var);
      for (size_t u = 0; u < n; ++u) {
        if (ctrl->children[u]->GetStage() != s)
          continue;
        for (auto &[old_var, new_var] : renames)
          ctrl->children[u]->SubstituteVar(old_var, new_var);
      }
      // Emit the clones defs-before-uses
      std::set<const VarNode *> clone_writes;
      for (auto &c : stage_clones)
        for (auto &w : c->GetWriteVars())
          clone_writes.insert(w.get());
      std::vector<bool> emitted(stage_clones.size(), false);
      std::set<const VarNode *> available;
      size_t done = 0;
      while (done < stage_clones.size()) {
        bool progress = false;
        for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
          if (emitted[ci])
            continue;
          bool ready = true;
          for (auto &rv : stage_clones[ci]->GetReadVars()) {
            if (clone_writes.count(rv.get()) && !available.count(rv.get())) {
              ready = false;
              break;
            }
          }
          if (!ready)
            continue;
          for (auto &w : stage_clones[ci]->GetWriteVars())
            available.insert(w.get());
          new_clones.push_back(stage_clones[ci]);
          emitted[ci] = true;
          ++done;
          progress = true;
        }
        if (!progress) { // defensive: emit any remainder (no cycle expected)
          for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
            if (!emitted[ci]) {
              new_clones.push_back(stage_clones[ci]);
              emitted[ci] = true;
              ++done;
            }
          }
        }
      }
    }
    ctrl->children.insert(ctrl->children.begin(), new_clones.begin(),
                          new_clones.end());

    // Estimate overall latency across loop iterations.
    int64_t overall_latency = 0;
    for (const auto &child : ctrl->children) {
      overall_latency = std::max(overall_latency,
                                 child->GetStartTime() + child->GetLatency());
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

  // Set the user-declared manual multi-buffer table (from
  // annotate_manual_multi_buffer), threaded into dependency analysis so
  // cross-iteration distances are solved per access pair.
  void SetManualMultiBuffer(const ManualMultiBufferTable &manual) {
    manual_multi_buffer_ = manual;
  }

private:
  std::map<std::string, int64_t> memory_limits_ = {{"shared", 48 * 1024}};
  std::map<std::string, int64_t> buffer_versions_overrides_;
  ManualMultiBufferTable manual_multi_buffer_;
};

} // namespace tl
} // namespace tvm
