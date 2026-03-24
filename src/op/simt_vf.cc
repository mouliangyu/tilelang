/*!
 * \file tl/op/simt_vf.cc
 * \brief SimtVF control TileOperator.
 */

#include "simt_vf.h"

#include <unordered_set>

#include <tvm/tir/analysis.h>

#include "../transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

void SimtVFOpNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SimtVFOpNode>().def_ro("root", &SimtVFOpNode::root_);
}

TileOperator SimtVFOpNode::Clone() const { return SimtVFOp(root_); }

bool SimtVFOpNode::IsSupportedCaptureVar(const Var &var) {
  const DataType dtype = var->dtype;
  if (dtype.lanes() != 1) {
    return false;
  }
  return dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
         dtype.is_handle() || dtype.is_bool();
}

Array<Var> SimtVFOpNode::CollectCaptures(const LowerArgs &T) const {
  Array<Var> predefined;
  if (T.thread_var.defined()) {
    predefined.push_back(T.thread_var);
  }

  Array<Var> undefined = UndefinedVars(root_->body, predefined);

  Array<Var> captures;
  std::unordered_set<const VarNode *> seen;
  for (const Var &var : undefined) {
    if (seen.count(var.get())) {
      continue;
    }
    seen.insert(var.get());
    ICHECK(IsSupportedCaptureVar(var))
        << "SimtVF capture variable `" << var << "` has unsupported dtype `"
        << var->dtype
        << "`. Only scalar int/uint/float/bool/handle are currently supported.";
    captures.push_back(var);
  }
  return captures;
}

Stmt SimtVFOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  Block lowered = root_;
  auto captures = CollectCaptures(T);

  auto ptr = lowered.CopyOnWrite();
  ptr->annotations.Set(tilelang_simt_vf_captures, captures);
  return lowered;
}

LayoutMap SimtVFOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TVM_FFI_STATIC_INIT_BLOCK() { SimtVFOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
