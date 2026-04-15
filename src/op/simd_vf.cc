/*!
 * \file tl/op/simd_vf.cc
 * \brief SimdVF control TileOperator.
 */

#include "simd_vf.h"

#include <unordered_set>

#include <tvm/tir/analysis.h>

#include "../transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

void SimdVFOpNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SimdVFOpNode>().def_ro("root", &SimdVFOpNode::root_);
}

TileOperator SimdVFOpNode::Clone() const { return SimdVFOp(root_); }

bool SimdVFOpNode::IsSupportedCaptureVar(const Var &var) {
  const DataType dtype = var->dtype;
  if (dtype.lanes() != 1) {
    return false;
  }
  return dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
         dtype.is_handle() || dtype.is_bool();
}

Array<Var> SimdVFOpNode::CollectCaptures(const LowerArgs &T) const {
  Array<Var> predefined;
  // SimdVF has no thread variables — do NOT add T.thread_var to predefined.

  // Exclude Block-owned alloc_buffers from capture analysis.
  for (const tir::Buffer &buf : root_->alloc_buffers) {
    predefined.push_back(buf->data);
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
        << "SimdVF capture variable `" << var << "` has unsupported dtype `"
        << var->dtype
        << "`. Only scalar int/uint/float/bool/handle are currently supported.";
    captures.push_back(var);
  }
  return captures;
}

Stmt SimdVFOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  Block lowered = root_;
  auto captures = CollectCaptures(T);

  auto ptr = lowered.CopyOnWrite();
  ptr->annotations.Set(tilelang_simd_vf_captures, captures);
  return lowered;
}

LayoutMap SimdVFOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TVM_FFI_STATIC_INIT_BLOCK() { SimdVFOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
