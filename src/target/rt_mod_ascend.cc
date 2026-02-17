#include "codegen_ascend.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    // Ascend text-only path may receive PrimFuncs without explicit calling_conv.
    if (calling_conv.defined()) {
      ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    }
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "asc", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_ascend_without_compile",
                        BuildTileLangAscendWithoutCompile);
}

} // namespace codegen
} // namespace tvm
