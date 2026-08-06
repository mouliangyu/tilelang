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

#include "./task_annotations.h"

#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>

#include <limits>

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

ffi::Map<ffi::String, ffi::Any> CopyTaskAnnotations(const ffi::Any &existing) {
  ffi::Map<ffi::String, ffi::Any> annotations;
  if (const auto *map = existing.as<ffi::MapObj>()) {
    for (const auto &[key, value] : *map)
      annotations.Set(key.cast<ffi::String>(), value);
  }
  return annotations;
}

} // namespace

TaskAnnotations ParseTaskAnnotations(const ffi::Any &node) {
  TaskAnnotations result;
  const auto *map = node.as<ffi::MapObj>();
  if (map == nullptr)
    return result;
  for (const auto &[key_any, value_any] : *map) {
    ffi::String key = key_any.cast<ffi::String>();
    if (key != "latency" && key != "ii" && key != "pipe")
      continue;
    PrimExpr value_expr = value_any.cast<PrimExpr>();
    const auto *value = value_expr.as<IntImmNode>();
    TVM_FFI_CHECK(value != nullptr, ValueError)
        << "T.Task " << key << " must be a constant integer, got "
        << value_expr;
    if (key == "latency") {
      TVM_FFI_CHECK_GE(value->value, 0, ValueError)
          << "T.Task latency must be non-negative";
      result.latency = value->value;
    } else if (key == "ii") {
      TVM_FFI_CHECK_GT(value->value, 0, ValueError)
          << "T.Task ii must be positive";
      result.ii = value->value;
    } else {
      TVM_FFI_CHECK_GT(value->value, 0, ValueError)
          << "T.Task pipe must be a non-zero hardware pipe mask";
      TVM_FFI_CHECK_LE(
          value->value,
          static_cast<int64_t>(std::numeric_limits<uint16_t>::max()),
          ValueError)
          << "T.Task pipe exceeds the supported hardware pipe mask range";
      result.pipe = static_cast<uint16_t>(value->value);
    }
  }
  return result;
}

ffi::Any MakeTaskPipeAnnotation(uint16_t pipe, const ffi::Any &existing) {
  ICHECK_NE(pipe, 0);
  ffi::Map<ffi::String, ffi::Any> annotations = CopyTaskAnnotations(existing);
  annotations.Set("pipe", IntImm(DataType::Int(64), pipe));
  return annotations;
}

ffi::Any MakeTaskAnnotations(const TaskCost &cost, const ffi::Any &existing) {
  ICHECK_GE(cost.latency, 0);
  ICHECK_GT(cost.ii, 0);
  ffi::Map<ffi::String, ffi::Any> annotations = CopyTaskAnnotations(existing);
  annotations.Set("latency", IntImm(DataType::Int(64), cost.latency));
  annotations.Set("ii", IntImm(DataType::Int(64), cost.ii));
  return annotations;
}

TaskCost ResolveTaskCost(const TaskCost &estimated,
                         const TaskAnnotations &annotations) {
  TaskCost result = estimated;
  if (annotations.latency.has_value())
    result.latency = annotations.latency.value();
  if (annotations.ii.has_value())
    result.ii = annotations.ii.value();
  ICHECK_GE(result.latency, 0);
  ICHECK_GT(result.ii, 0);
  return result;
}

TaskCost RequireTaskCost(const TaskAnnotations &annotations,
                         const char *context) {
  TVM_FFI_CHECK(annotations.latency.has_value() && annotations.ii.has_value(),
                ValueError)
      << context << " is missing latency/II annotations. Run "
      << "tl.transform.LatencyEstimator immediately before AutoSchedule.";
  return {annotations.latency.value(), annotations.ii.value()};
}

uint16_t RequireTaskPipe(const TaskAnnotations &annotations,
                         const char *context) {
  TVM_FFI_CHECK(annotations.pipe.has_value(), ValueError)
      << context << " is missing a pipe annotation. Run "
      << "tl.transform.NormalizeTask before LatencyEstimator and AutoSchedule.";
  return annotations.pipe.value();
}

} // namespace tl
} // namespace tvm
