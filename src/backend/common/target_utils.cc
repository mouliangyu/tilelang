/*!
 * \file tl/backend/common/target_utils.cc
 * \brief Common target helper dispatch.
 */

#include "backend/common/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

bool TargetHasAsyncCopy(Target target) {
  if (TargetIsCuda(target)) {
    return TargetCudaHasAsyncCopy(target);
  }
  if (TargetIsRocm(target)) {
    return TargetRocmHasAsyncCopy(target);
  }
  return false;
}

bool TargetIsAscend(Target target) {
  if (!target.defined()) {
    return false;
  }
  if (target->kind->name == "ascend") {
    return true;
  }
  for (const ffi::String &key : target->keys) {
    if (key == "ascend") {
      return true;
    }
  }
  return false;
}

bool IsAscendVectorizableFP8(DataType dtype) {
  // NOTE: E8M0 is a special type of FP8 which is not handled here.
  // We only handle FP8 types which can be represented with
  // __asc_fp8_interpretation_t here.
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetHasAsyncCopy",
           [](Target target) { return TargetHasAsyncCopy(target); })
      .def("tl.TargetIsAscend",
           [](Target target) { return TargetIsAscend(target); });
}

} // namespace tl
} // namespace tvm
