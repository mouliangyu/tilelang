/*!
 * \file tl/backend/common/target_utils.h
 * \brief Common entry points for target attribute helpers.
 *
 */

#ifndef TVM_TL_BACKEND_COMMON_TARGET_UTILS_H_
#define TVM_TL_BACKEND_COMMON_TARGET_UTILS_H_

#include <tvm/target/target.h>

#include "cpu/target_utils.h"
#include "cuda/target_utils.h"
#include "metal/target_utils.h"
#include "rocm/target_utils.h"

namespace tvm {
namespace tl {

bool TargetHasAsyncCopy(Target target);

// Ascend-specific helpers (the Ascend backend has no per-backend target_utils;
// these live in the common layer).
bool TargetIsAscend(Target target);
bool IsAscendVectorizableFP8(DataType dtype);
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_TARGET_UTILS_H_
