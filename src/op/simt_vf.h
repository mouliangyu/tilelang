/*!
 * \file tl/op/simt_vf.h
 * \brief SimtVF control TileOperator.
 */

#ifndef TVM_TL_OP_SIMT_VF_H_
#define TVM_TL_OP_SIMT_VF_H_

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "./operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class SimtVFOpNode : public TileOperatorNode {
public:
  For root_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.SimtVFOp", SimtVFOpNode,
                                    TileOperatorNode);

  static void RegisterReflection();

  explicit SimtVFOpNode(For root) : root_(std::move(root)) {}

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;

  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const override;

private:
  Array<Var> CollectCaptures(const LowerArgs &T) const;
  static bool IsSupportedCaptureVar(const Var &var);
};

class SimtVFOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SimtVFOp, TileOperator,
                                             SimtVFOpNode);

  explicit SimtVFOp(const For &root) {
    auto op = tvm::ffi::make_object<SimtVFOpNode>(root);
    data_ = std::move(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_SIMT_VF_H_
