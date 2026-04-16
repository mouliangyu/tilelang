/*!
 * \file intrin_rule_ascend.cc
 * \brief Ascend intrinsic rules for math functions.
 *
 * The default FloatSuffix lowering uses exact dtype matching
 * (t == DataType::Float(32)) which fails for vectorized types.
 * AscendMath uses t.is_float() like CUDAMath to support any lane count.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tir::FLowerIntrinsic;

struct AscendMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      default:
        return "";
      }
    }
    return "";
  }
};

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("ascend.FLowerIntrinsic",
                               DispatchPureExtern<AscendMath>);

TVM_REGISTER_OP("tir.rsqrt")
    .set_attr<FLowerIntrinsic>("ascend.FLowerIntrinsic",
                               DispatchPureExtern<AscendMath>);

} // namespace intrin
} // namespace codegen
} // namespace tvm
