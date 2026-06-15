/*!
 * \file tl/ascend/op/fill.cc
 * \brief Ascend implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchAscendFillTarget(Target target) { return TargetIsAscend(target); }

bool RegisterAscendFill() {
  RegisterFillImpl(FillImpl{
      "ascend.Fill",
      MatchAscendFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool ascend_fill_registered = RegisterAscendFill();

} // namespace

} // namespace tl
} // namespace tvm
