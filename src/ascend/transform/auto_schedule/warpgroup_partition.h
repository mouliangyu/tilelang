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

#include <tvm/ffi/optional.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <optional>

namespace tvm {
namespace tl {

// Partition scheduled TIR directly into its AIV/AIC bodies and lower software
// pipeline stages. No parallel syntax tree is reconstructed by this consumer.
tirx::Stmt
PartitionScheduledTIR(const tirx::Stmt &scheduled_body,
                      ffi::Optional<tirx::Var> external_sid = std::nullopt);

} // namespace tl
} // namespace tvm
