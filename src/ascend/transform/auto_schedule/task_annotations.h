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

#pragma once

#include <tvm/ffi/any.h>
#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <optional>

namespace tvm {
namespace tl {

using namespace tirx;

// Compiler-owned fields carried by tl.ascend_task after AutoSchedule. These
// form the short-lived task metadata contract consumed by CorePartition.
constexpr const char *kAscendScheduledCoreMask = "core_mask";
constexpr const char *kAscendScheduledStage = "stage";

struct TaskCost {
  int64_t latency{0};
  int64_t ii{0};
};

struct TaskAnnotations {
  std::optional<int64_t> latency;
  std::optional<int64_t> ii;
  std::optional<uint16_t> pipe;
};

// Parse compiler and user annotations from the structured node payload of
// tl.ascend_task or tl.ascend_per_core_task. Legacy scalar payloads carry no
// annotations.
TaskAnnotations ParseTaskAnnotations(const ffi::Any &node);

// Merge compiler-produced fields into the structured node payload while
// preserving annotations owned by other consumers.
ffi::Any MakeTaskPipeAnnotation(uint16_t pipe, const ffi::Any &existing);
ffi::Any MakeTaskAnnotations(const TaskCost &cost, const ffi::Any &existing);

// Apply the fields explicitly supplied by T.Task to an estimated cost.
TaskCost ResolveTaskCost(const TaskCost &estimated,
                         const TaskAnnotations &annotations);

// Read the complete fields required by AutoSchedule after LatencyEstimator.
TaskCost RequireTaskCost(const TaskAnnotations &annotations,
                         const char *context);
uint16_t RequireTaskPipe(const TaskAnnotations &annotations,
                         const char *context);

} // namespace tl
} // namespace tvm
