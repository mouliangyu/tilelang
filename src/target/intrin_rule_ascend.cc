/*!
 * \file intrin_rule_ascend.cc
 * \brief Ascend intrinsic rules for math functions.
 *
 * The default FloatSuffix lowering uses exact dtype matching
 * (t == DataType::Float(32)) which fails for vectorized types.
 * AscendMath uses t.is_float() like CUDAMath to support any lane count.
 *
 * Registered with "c.FLowerIntrinsic" because the Ascend target uses
 * kind.name = "c" (normalized from "ascend" in normalize_ascend_target).
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
      case 16:
        return "h" + name;
      default:
        return "";
      }
    }
    return "";
  }
};

#define REGISTER_ASCEND_INTRIN(OpName)                                         \
  TVM_REGISTER_OP("tir." OpName)                                               \
      .set_attr<FLowerIntrinsic>("c.FLowerIntrinsic",                          \
                                 DispatchPureExtern<AscendMath>)

REGISTER_ASCEND_INTRIN("sqrt");
REGISTER_ASCEND_INTRIN("rsqrt");
REGISTER_ASCEND_INTRIN("exp");
REGISTER_ASCEND_INTRIN("exp2");
REGISTER_ASCEND_INTRIN("exp10");
REGISTER_ASCEND_INTRIN("log");
REGISTER_ASCEND_INTRIN("log2");
REGISTER_ASCEND_INTRIN("log10");
REGISTER_ASCEND_INTRIN("log1p");
REGISTER_ASCEND_INTRIN("tanh");
REGISTER_ASCEND_INTRIN("tan");
REGISTER_ASCEND_INTRIN("sin");
REGISTER_ASCEND_INTRIN("cos");
REGISTER_ASCEND_INTRIN("atan");
REGISTER_ASCEND_INTRIN("atan2");
REGISTER_ASCEND_INTRIN("erf");
REGISTER_ASCEND_INTRIN("floor");
REGISTER_ASCEND_INTRIN("ceil");
REGISTER_ASCEND_INTRIN("fabs");
REGISTER_ASCEND_INTRIN("round");
REGISTER_ASCEND_INTRIN("trunc");

#undef REGISTER_ASCEND_INTRIN

} // namespace intrin
} // namespace codegen
} // namespace tvm
