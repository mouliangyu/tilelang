#include "codegen_ascend.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

namespace {

std::string AscendCodeGen(IRModule mod) {
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    auto is_global_func =
        f->GetAttr<Bool>(tvm::tir::attr::kIsGlobalFunc).value_or(Bool(false));
    // For Ascend, we accept functions that are either:
    // 1. Device kernel launches (calling_conv == kDeviceKernelLaunch), or
    // 2. Global functions called via call_extern (is_global_func == true)
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch || is_global_func)
        << "CodeGenTileLangAscend: Function must have "
           "calling_conv=kDeviceKernelLaunch "
        << "or be marked as a global function (tir.is_global_func=True)";
    cg.AddFunction(f);
  }

  return cg.Finish();
}

} // namespace

ffi::Module BuildTileLangAscend(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_compile")) {
    (*f)(code, target);
  }

  // Return a CSourceModule so that inspect_source() still works.
  // The actual binary compilation for execution is handled by the
  // cython adapter (LibraryGenerator) or the callback above.
  return CSourceModuleCreate(code, "asc", ffi::Array<ffi::String>());
}

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return CSourceModuleCreate(code, "asc", ffi::Array<ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend_without_compile",
           BuildTileLangAscendWithoutCompile);
}

} // namespace codegen
} // namespace tvm
