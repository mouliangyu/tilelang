#include "codegen_ascend.h"

#include <string>

#include "../transform/common/attr.h"
#include "../../3rdparty/tvm/src/tir/transforms/ir_utils.h"

namespace tvm {
namespace codegen {

void CodeGenTileLangAscend::Init(bool output_ssa) {
  CodeGenTileLangCPP::Init(output_ssa, /*emit_asserts=*/false,
                           /*emit_fwd_func_decl=*/false, "ascend", {});
}

void CodeGenTileLangAscend::AddFunction(const PrimFunc &f) {
  this->InitFuncState(f);
  ReserveKeywordsAsUnique();

  // Register function parameters with their storage scope for codegen
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      auto scope = GetPtrStorageScope(param);
      alloc_storage_scope_[param.get()] = scope;
    }
  }

  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenTileLangAscend: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  this->stream << "__global__ __vector__ ";
  CodeGenC::PrintType(f->ret_type, stream);
  CodeGenC::PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) {
      stream << ", ";
    }
    if (v.dtype().is_handle()) {
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          stream << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangAscend::VisitStmt_(const ForNode *op) {
  auto opt_threads = op->annotations.Get(tvm::tl::tilelang_simt_vf_threads);
  if (!opt_threads) {
    CodeGenC::VisitStmt_(op);
    return;
  }

  int64_t threads = -1;
  PrimExpr threads_expr = Downcast<PrimExpr>(opt_threads.value());
  if (const auto *imm = threads_expr.as<IntImmNode>()) {
    threads = imm->value;
  }

  ffi::Array<Var> captures;
  if (auto opt_captures = op->annotations.Get(tvm::tl::tilelang_simt_vf_captures)) {
    if (auto arr = opt_captures.value().as<ffi::Array<Var>>()) {
      captures = arr.value();
    }
  }
  std::string helper_name = "simt_vf_" + std::to_string(simtvf_helper_counter_++);
  auto resolve_var_name = [this](const Var &v) -> std::string {
    auto it = var_idmap_.find(v.get());
    if (it != var_idmap_.end()) {
      return it->second;
    }
    return v->name_hint;
  };

  // Capture helper body by rendering the region body to a temporary stream.
  // Register captures with their storage scope so GetBufferRef adds correct qualifier
  for (const auto &v : captures) {
    auto scope = GetPtrStorageScope(v);
    alloc_storage_scope_[v.get()] = scope;
  }
  std::ostringstream saved_main_stream;
  stream.swap(saved_main_stream);
  CodeGenC::VisitStmt(op->body);
  std::string helper_body = stream.str();
  stream.str("");
  stream.clear();
  stream.swap(saved_main_stream);

  decl_stream << "__simt_vf__ ";
  if (threads >= 0) {
    decl_stream << "__launch_bounds__(" << threads << ") ";
  }
  decl_stream << "inline void " << helper_name << "(";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0) {
      decl_stream << ", ";
    }
    const Var &v = captures[i];
    std::string vname = resolve_var_name(v);
    auto it = handle_data_type_.find(v.get());
    if (it == handle_data_type_.end()) {
      for (const auto &kv : handle_data_type_) {
        if (kv.first->name_hint == v->name_hint) {
          it = handle_data_type_.find(kv.first);
          break;
        }
      }
    }
    // simt_vf helper functions need memory qualifier based on storage scope
    auto scope = GetPtrStorageScope(v);
    PrintStorageScope(scope, decl_stream);
    if (it != handle_data_type_.end()) {
      PrintType(it->second, decl_stream);
      decl_stream << "* " << vname;
    } else {
      decl_stream << "void* " << vname;
    }
  }
  decl_stream << ") {\n";
  decl_stream << helper_body;
  decl_stream << "}\n\n";

  PrintIndent();
  stream << "// __simt_vf__ begin";
  if (threads >= 0) {
    stream << " threads=" << threads;
  }
  stream << "\n";
  PrintIndent();
  stream << "pipe_barrier(pipe_t::PIPE_ALL);\n";
  PrintIndent();
  stream << "asc_vf_call<" << helper_name << ">(cce::dim3(";
  if (threads >= 0) {
    stream << threads;
  } else {
    stream << "1";
  }
  stream << ")";
  for (size_t i = 0; i < captures.size(); ++i) {
    stream << ", " << resolve_var_name(captures[i]);
  }
  stream << ");\n";

  PrintIndent();
  stream << "pipe_barrier(pipe_t::PIPE_ALL);\n";
  PrintIndent();
  stream << "// __simt_vf__ end\n";
}

void CodeGenTileLangAscend::VisitStmt_(const AttrStmtNode *op) {
  // Ascend text-only backend currently does not lower thread binding attrs into
  // executable launch semantics. Keep generating the enclosed body instead of
  // failing in the generic C backend.
  if (op->attr_key == tir::attr::thread_extent ||
      op->attr_key == tir::attr::virtual_thread) {
    if (const auto *iv = op->node.as<IterVarNode>()) {
      // Materialize a placeholder symbol so later uses (e.g. simtvf_tx) are
      // printable in expression codegen.
      std::string var_name = AllocVarID(iv->var.get());
      bool is_simtvf_thread_var = var_name.rfind("simtvf_", 0) == 0;
      if ((iv->thread_tag == "threadIdx.x" || iv->thread_tag == "threadIdx.y" ||
           iv->thread_tag == "threadIdx.z") &&
          !is_simtvf_thread_var) {
        this->VisitStmt(op->body);
        return;
      }
      PrintIndent();
      PrintType(iv->var.dtype(), stream);
      stream << " " << var_name << " = ";
      if (iv->thread_tag == "threadIdx.x") {
        stream << "threadIdx.x";
      } else if (iv->thread_tag == "threadIdx.y") {
        stream << "threadIdx.y";
      } else if (iv->thread_tag == "threadIdx.z") {
        stream << "threadIdx.z";
      } else if (iv->thread_tag == "blockIdx.x") {
        stream << "blockIdx.x";
      } else if (iv->thread_tag == "blockIdx.y") {
        stream << "blockIdx.y";
      } else if (iv->thread_tag == "blockIdx.z") {
        stream << "blockIdx.z";
      } else {
        stream << "0";
      }
      stream << ";\n";
    }
    this->VisitStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::PrintStorageSync(const CallNode *op) {
  ICHECK_GE(op->args.size(), 1U);
  PrintIndent();
  stream << "pipe_barrier(pipe_t::PIPE_ALL);\n";
}

void CodeGenTileLangAscend::PrintStorageScope(const std::string& scope, std::ostream& os) {
  if (scope == "global") {
    os << "__gm__ ";
  } else if (scope == "shared") {
    os << "__ubuf__ ";
  } else if (!scope.empty()) {
    LOG(FATAL) << "CodeGenTileLangAscend: Unsupported storage scope: " << scope;
  }
  // Empty scope: output nothing
}

bool CodeGenTileLangAscend::IsScopePartOfType() const {
  // Ascend requires memory scope qualifiers as part of the pointer type
  return true;
}

} // namespace codegen
} // namespace tvm
