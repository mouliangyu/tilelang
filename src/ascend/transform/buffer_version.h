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

/*!
 * \file tl/ascend/transform/buffer_version.h
 * \brief Internal annotation contract for physical buffer versions.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_BUFFER_VERSION_H_
#define TVM_TL_ASCEND_TRANSFORM_BUFFER_VERSION_H_

#include <tvm/ffi/container/map.h>
#include <tvm/tirx/expr.h>

#include "transform/common/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

/*! \brief Frontend staging annotation for manually versioned buffers. */
constexpr const char *kManualMultiBuffer = "tl.manual_multi_buffer";

/*! \brief Storage data Var -> number of equal-sized physical versions. */
using BufferVersionMap = ffi::Map<tirx::Var, PrimExpr>;

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_BUFFER_VERSION_H_
