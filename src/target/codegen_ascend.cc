#include "codegen_ascend.h"

#include "../op/builtin.h"
#include <sstream>
#include <string>

#include "../../3rdparty/tvm/src/tir/transforms/ir_utils.h"
#include "../transform/common/attr.h"
#include <tvm/tir/stmt_functor.h>

static std::string FixIndent(const std::string &body) {
  size_t min_indent = std::string::npos;
  std::istringstream iss(body);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty())
      continue;
    size_t pos = line.find_first_not_of(' ');
    if (pos != std::string::npos && pos < min_indent) {
      min_indent = pos;
    }
  }
  if (min_indent == std::string::npos || min_indent <= 2) {
    return body;
  }
  size_t strip = min_indent - 2;
  iss.clear();
  iss.seekg(0);
  std::ostringstream oss;
  while (std::getline(iss, line)) {
    if (line.size() >= strip &&
        line.find_first_not_of(' ') != std::string::npos) {
      oss << line.substr(strip) << "\n";
    } else {
      oss << line << "\n";
    }
  }
  return oss.str();
}

namespace tvm {
namespace codegen {

void CodeGenTileLangAscend::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\"\n";
}

void CodeGenTileLangAscend::PrintType(DataType t, std::ostream &os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types for handles";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    if (lanes != 1) {
      LOG(FATAL) << "Ascend does not support vector bool type: " << t;
    }
    os << "bool";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      if (lanes == 1) {
        os << "half";
      } else if (lanes == 2) {
        os << "float16x2_t";
      } else if (lanes == 4) {
        os << "float16x4_t";
      } else if (lanes == 8) {
        os << "float16x8_t";
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes == 1) {
        os << "float";
      } else if (lanes == 2) {
        os << "float2";
      } else if (lanes == 4) {
        os << "float4";
      } else {
        fail = true;
      }
      break;
    case 64:
      if (lanes == 1) {
        os << "double";
      } else {
        fail = true; // double vector not supported
      }
      break;
    default:
      fail = true;
    }
  } else if (t.is_bfloat16()) {
    if (lanes == 1) {
      os << "bfloat16_t";
    } else if (lanes == 2) {
      os << "bfloat16x2_t";
    } else {
      fail = true;
    }
  } else if (t.is_int() || t.is_uint()) {
    switch (t.bits()) {
    case 8:
      if (lanes == 1) {
        os << (t.is_int() ? "int8_t" : "uint8_t");
      } else {
        fail = true; // 8-bit vector not yet supported
      }
      break;
    case 16:
      if (lanes == 1) {
        os << (t.is_int() ? "int16_t" : "uint16_t");
      } else {
        fail = true; // 16-bit vector not yet supported
      }
      break;
    case 32:
      if (lanes == 1) {
        os << (t.is_int() ? "int32_t" : "uint32_t");
      } else if (lanes == 2) {
        os << (t.is_int() ? "int2" : "uint2");
      } else if (lanes == 4) {
        os << (t.is_int() ? "int4" : "uint4");
      } else {
        fail = true;
      }
      break;
    case 64:
      if (lanes == 1) {
        os << (t.is_int() ? "int64_t" : "uint64_t");
      } else {
        fail = true; // 64-bit vector not supported
      }
      break;
    default:
      fail = true;
    }
  } else {
    fail = true;
  }

  if (fail) {
    LOG(FATAL) << "Cannot convert type " << t << " (lanes=" << lanes
               << ") to Ascend type";
  }
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

  // Pre-scan body to determine kernel mode (Mix / Vector / Cube)
  bool has_cube = false, has_vector = false, has_simtvf = false,
       has_simdvf = false;
  int vector_count = 2;
  tir::PostOrderVisit(f->body, [&](const ObjectRef &n) {
    if (const auto *block = n.as<tir::BlockNode>()) {
      if (block->name_hint == "CUBE")
        has_cube = true;
      else if (block->name_hint == "VECTOR") {
        has_vector = true;
        if (auto opt = block->annotations.Get("vector_count")) {
          if (const auto *imm = opt.value().as<IntImmNode>()) {
            vector_count = imm->value;
          }
        }
      } else if (block->name_hint == "SIMT_VF")
        has_simtvf = true;
      else if (block->name_hint == "SIMD_VF")
        has_simdvf = true;
    }
    if (const auto *call = n.as<tir::CallNode>()) {
      if (call->op.same_as(tl::ascend_gemm_l1())) {
        has_gemm_l1_ = true;
      }
      if (call->op.same_as(tl::ascend_mad())) {
        has_gemm_l0_ = true;
      }
      if (call->op.same_as(tl::ascend_nd2nz_scatter()) ||
          call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_nd2nz_copy_ = true;
      }
    }
  });

  // Pre-scan for bf16 buffers needing cast intrinsics
  tir::PostOrderVisit(f->body, [&](const ObjectRef &n) {
    if (const auto *alloc = n.as<tir::AllocateNode>()) {
      if (alloc->dtype.is_bfloat16()) {
        has_bf16_cast_ = true;
      }
    }
    if (const auto *store = n.as<tir::BufferStoreNode>()) {
      if (store->buffer->dtype.is_bfloat16()) {
        has_bf16_cast_ = true;
      }
    }
    if (const auto *load = n.as<tir::BufferLoadNode>()) {
      if (load->buffer->dtype.is_bfloat16()) {
        has_bf16_cast_ = true;
      }
    }
    if (const auto *cast = n.as<tir::CastNode>()) {
      if (cast->dtype.is_bfloat16() || cast->value->dtype.is_bfloat16()) {
        has_bf16_cast_ = true;
      }
    }
  });
  if (!has_bf16_cast_) {
    for (const auto &kv : f->buffer_map) {
      if (kv.second->dtype.is_bfloat16()) {
        has_bf16_cast_ = true;
        break;
      }
    }
  }

  // Conditionally include gemm.h when ascend_gemm_l1 is used
  if (has_gemm_l1_ && !has_gemm_l1_included_) {
    decl_stream << "#include <tl_templates/ascend/gemm.h>\n";
    has_gemm_l1_included_ = true;
  }

  // Conditionally include <limits> for std::numeric_limits<T>::infinity()
  if (!has_limits_included_) {
    decl_stream << "#include <limits>\n";
    has_limits_included_ = true;
  }

  // Include asc_bf16.h when bf16 buffers exist (needed for vector casts)
  if (has_bf16_cast_ && !has_bf16_cast_included_) {
    decl_stream << "#include <simt_api/asc_bf16.h>\n";
    has_bf16_cast_included_ = true;
  }

  // Conditionally include nd2nz_copy.h when ascend_nd2nz_copy is used
  if (has_nd2nz_copy_ && !has_nd2nz_copy_included_) {
    decl_stream << "#include <tl_templates/ascend/nd2nz_copy.h>\n";
    has_nd2nz_copy_included_ = true;
  }

  // Conditionally include micro_api header when SimdVF or nd2nz_copy is used
  if ((has_simdvf || has_nd2nz_copy_) && !has_simdvf_included_) {
    decl_stream << "#include <micro_api/kernel_micro_intf.h>\n";
    has_simdvf_included_ = true;
  }

  if (has_cube && has_vector) {
    kernel_mode_ = AscendKernelMode::kMix;
    mix_aiv_count_ = vector_count;
  } else if (has_cube || has_vector) {
    LOG(FATAL) << "T.Cube() and T.Vector() must be used together for Mix mode";
  } else if (has_gemm_l0_ || has_gemm_l1_ || has_nd2nz_copy_) {
    kernel_mode_ = AscendKernelMode::kCube;
  } else {
    kernel_mode_ = AscendKernelMode::kVector;
  }

  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol) << "CodeGenTileLangAscend: Expect PrimFunc to have the "
                           "global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  switch (kernel_mode_) {
  case AscendKernelMode::kMix:
    this->stream << "__global__ __mix__(1, " << mix_aiv_count_ << ") ";
    break;
  case AscendKernelMode::kVector:
    this->stream << "__global__ __vector__ ";
    break;
  case AscendKernelMode::kCube:
    this->stream << "__global__ __cube__ ";
    break;
  }
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

void CodeGenTileLangAscend::VisitStmt_(const tir::BlockNode *op) {
  if (op->name_hint == "CUBE") {
    if (kernel_mode_ == AscendKernelMode::kMix) {
      PrintIndent();
      stream << "if ASCEND_IS_AIC {\n";
      int scope = BeginScope();
      PrintStmt(op->body);
      EndScope(scope);
      PrintIndent();
      stream << "}\n";
    } else {
      PrintStmt(op->body);
    }
    return;
  }
  if (op->name_hint == "VECTOR") {
    if (kernel_mode_ == AscendKernelMode::kMix) {
      PrintIndent();
      stream << "if ASCEND_IS_AIV {\n";
      int scope = BeginScope();
      PrintStmt(op->body);
      EndScope(scope);
      PrintIndent();
      stream << "}\n";
    } else {
      PrintStmt(op->body);
    }
    return;
  }
  if (op->name_hint == "SIMD_VF") {
    ffi::Array<Var> captures;
    if (auto opt_captures =
            op->annotations.Get(tvm::tl::tilelang_simd_vf_captures)) {
      if (auto arr = opt_captures.value().as<ffi::Array<Var>>()) {
        captures = arr.value();
      }
    }
    std::string helper_name =
        "simd_vf_" + std::to_string(simdvf_helper_counter_++);
    auto resolve_var_name = [this](const Var &v) -> std::string {
      auto it = var_idmap_.find(v.get());
      if (it != var_idmap_.end()) {
        return it->second;
      }
      return v->name_hint;
    };

    // Register captures with their storage scope
    for (const auto &v : captures) {
      if (v->type_annotation.as<PointerTypeNode>()) {
        auto scope = GetPtrStorageScope(v);
        alloc_storage_scope_[v.get()] = scope;
      }
    }
    // Reset variable naming state so the helper function body gets fresh names
    std::unordered_map<const tir::VarNode *, std::string> saved_var_idmap;
    saved_var_idmap.swap(var_idmap_);
    NameSupply saved_name_supply = name_supply_;
    name_supply_ = NameSupply();
    // Carry over all outer variable names so they remain accessible inside the
    // helper body. Don't reserve them in name_supply_ so inner variables get
    // clean unsuffixed names.
    for (const auto &kv : saved_var_idmap) {
      var_idmap_[kv.first] = kv.second;
    }
    // Render body to temporary stream
    std::ostringstream saved_main_stream;
    stream.swap(saved_main_stream);
    bool old_inside_simdvf_body = inside_simdvf_body_;
    inside_simdvf_body_ = true;
    CodeGenC::VisitStmt(op->body);
    inside_simdvf_body_ = old_inside_simdvf_body;
    std::string helper_body = FixIndent(stream.str());
    stream.str("");
    stream.clear();
    stream.swap(saved_main_stream);
    // Restore naming state
    var_idmap_.swap(saved_var_idmap);
    name_supply_ = saved_name_supply;

    // Emit __simd_vf__ helper function to decl_stream
    decl_stream << "__simd_vf__ inline void " << helper_name << "(";
    for (size_t i = 0; i < captures.size(); ++i) {
      if (i != 0) {
        decl_stream << ", ";
      }
      const Var &v = captures[i];
      std::string vname = resolve_var_name(v);
      if (!v->type_annotation.as<PointerTypeNode>()) {
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vname;
      } else {
        auto it = handle_data_type_.find(v.get());
        if (it == handle_data_type_.end()) {
          for (const auto &kv : handle_data_type_) {
            if (kv.first->name_hint == v->name_hint) {
              it = handle_data_type_.find(kv.first);
              break;
            }
          }
        }
        auto scope = GetPtrStorageScope(v);
        PrintStorageScope(scope, decl_stream);
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
          decl_stream << "* " << vname;
        } else {
          decl_stream << "void* " << vname;
        }
      }
    }
    decl_stream << ") {\n";
    decl_stream << helper_body;
    decl_stream << "}\n\n";

    // Emit direct inline call (no asc_vf_call)
    PrintIndent();
    stream << helper_name << "(";
    for (size_t i = 0; i < captures.size(); ++i) {
      if (i != 0) {
        stream << ", ";
      }
      stream << resolve_var_name(captures[i]);
    }
    stream << ");\n";
    return;
  }
  if (op->name_hint != "SIMT_VF") {
    PrintStmt(op->body);
    return;
  }

  int64_t thread_x = 1, thread_y = 1, thread_z = 1;
  {
    const tir::Stmt *cur = &op->body;
    while (const auto *attr = (*cur).as<AttrStmtNode>()) {
      if (attr->attr_key != tir::attr::thread_extent)
        break;
      const auto *iv = attr->node.as<IterVarNode>();
      if (!iv)
        break;
      if (const auto *imm = attr->value.as<IntImmNode>()) {
        if (iv->thread_tag == "threadIdx.x")
          thread_x = imm->value;
        else if (iv->thread_tag == "threadIdx.y")
          thread_y = imm->value;
        else if (iv->thread_tag == "threadIdx.z")
          thread_z = imm->value;
      }
      cur = &attr->body;
    }
  }

  // Check that no shared buffer allocations exist inside SimtVF region
  tir::PostOrderVisit(op->body, [](const ObjectRef &n) {
    if (const auto *alloc = n.as<AllocateNode>()) {
      auto alloc_scope = GetPtrStorageScope(alloc->buffer_var);
      ICHECK(alloc_scope != "shared" && alloc_scope != "shared.dyn")
          << "Shared buffer allocation (T.alloc_shared) must be placed OUTSIDE "
             "SimtVF region. "
          << "Found buffer '" << alloc->buffer_var->name_hint
          << "' with scope '" << alloc_scope << "' inside SimtVF body.";
    }
  });

  ffi::Array<Var> captures;
  if (auto opt_captures =
          op->annotations.Get(tvm::tl::tilelang_simt_vf_captures)) {
    if (auto arr = opt_captures.value().as<ffi::Array<Var>>()) {
      captures = arr.value();
    }
  }
  std::string helper_name =
      "simt_vf_" + std::to_string(simtvf_helper_counter_++);
  auto resolve_var_name = [this](const Var &v) -> std::string {
    auto it = var_idmap_.find(v.get());
    if (it != var_idmap_.end()) {
      return it->second;
    }
    return v->name_hint;
  };

  // Capture helper body by rendering the region body to a temporary stream.
  // Register captures with their storage scope so GetBufferRef adds correct
  // qualifier
  for (const auto &v : captures) {
    if (v->type_annotation.as<PointerTypeNode>()) {
      auto scope = GetPtrStorageScope(v);
      alloc_storage_scope_[v.get()] = scope;
    }
  }
  // Reset variable naming state so the helper function body gets fresh names
  std::unordered_map<const tir::VarNode *, std::string> saved_var_idmap;
  saved_var_idmap.swap(var_idmap_);
  NameSupply saved_name_supply = name_supply_;
  name_supply_ = NameSupply();
  // Carry over all outer variable names so they remain accessible inside the
  // helper body. Don't reserve them in name_supply_ so inner variables get
  // clean unsuffixed names.
  for (const auto &kv : saved_var_idmap) {
    var_idmap_[kv.first] = kv.second;
  }
  std::ostringstream saved_main_stream;
  stream.swap(saved_main_stream);
  bool old_inside_simtvf_body = inside_simtvf_body_;
  inside_simtvf_body_ = true;
  CodeGenC::VisitStmt(op->body);
  inside_simtvf_body_ = old_inside_simtvf_body;
  std::string helper_body = FixIndent(stream.str());
  stream.str("");
  stream.clear();
  stream.swap(saved_main_stream);
  // Restore naming state
  var_idmap_.swap(saved_var_idmap);
  name_supply_ = saved_name_supply;

  int64_t total_threads = thread_x * thread_y * thread_z;
  decl_stream << "__simt_vf__ ";
  if (total_threads > 0) {
    decl_stream << "__launch_bounds__(" << total_threads << ") ";
  }
  decl_stream << "inline void " << helper_name << "(";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0) {
      decl_stream << ", ";
    }
    const Var &v = captures[i];
    std::string vname = resolve_var_name(v);
    if (!v->type_annotation.as<PointerTypeNode>()) {
      PrintType(v.dtype(), decl_stream);
      decl_stream << " " << vname;
    } else {
      auto it = handle_data_type_.find(v.get());
      if (it == handle_data_type_.end()) {
        for (const auto &kv : handle_data_type_) {
          if (kv.first->name_hint == v->name_hint) {
            it = handle_data_type_.find(kv.first);
            break;
          }
        }
      }
      auto scope = GetPtrStorageScope(v);
      PrintStorageScope(scope, decl_stream);
      if (it != handle_data_type_.end()) {
        PrintType(it->second, decl_stream);
        decl_stream << "* " << vname;
      } else {
        decl_stream << "void* " << vname;
      }
    }
  }
  decl_stream << ") {\n";
  decl_stream << helper_body;
  decl_stream << "}\n\n";

  PrintIndent();
  stream << "asc_vf_call<" << helper_name << ">(cce::dim3(";
  stream << thread_x;
  if (thread_y > 1 || thread_z > 1) {
    stream << ", " << thread_y;
    if (thread_z > 1) {
      stream << ", " << thread_z;
    }
  }
  stream << ")";
  for (size_t i = 0; i < captures.size(); ++i) {
    stream << ", " << resolve_var_name(captures[i]);
  }
  stream << ");\n";
}

void CodeGenTileLangAscend::VisitExpr_(const BroadcastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8) {
    const int64_t *p = as_const_int(op->value);
    if (p) {
      if (lanes == 4) {
        // make_int8x4
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
        return;
      } else if (lanes == 32) {
        // make_int8x32
        const int64_t *p = as_const_int(op->value);
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (op->dtype.is_uint()) {
          os << "make_ulonglong4(" << v << ", " << v << ", " << v << ", " << v
             << ")";
        } else {
          os << "make_longlong4(" << v << ", " << v << ", " << v << ", " << v
             << ")";
        }
        return;
      }
    }
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_float16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_bfloat16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_float() && op->dtype.bits() == 32 &&
      op->dtype.lanes() == 8) {
    std::string v = PrintExpr(op->value);
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "*(unsigned long long*)&make_float2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t *p = as_const_int(op->value);
    ICHECK(p) << "BroadcastNode " << op << " value: " << op->value
              << " is not a constant";
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenTileLangAscend::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    ICHECK_EQ(op->args.size(), 1)
        << "tl.ascend_pipe_barrier expects exactly 1 argument (pipe_t string)";
    std::string pipe_t_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "AscendC::PipeBarrier<pipe_t::" << pipe_t_str << ">();\n";
  } else if (op->op.same_as(tl::ascend_set_flag())) {
    ICHECK_EQ(op->args.size(), 2) << "tl.ascend_set_flag expects exactly 2 "
                                     "arguments (hard_event string, event_id)";
    std::string hard_event_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "AscendC::SetFlag<AscendC::HardEvent::" << hard_event_str
                 << ">(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_wait_flag())) {
    ICHECK_EQ(op->args.size(), 2) << "tl.ascend_wait_flag expects exactly 2 "
                                     "arguments (hard_event string, event_id)";
    std::string hard_event_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "AscendC::WaitFlag<AscendC::HardEvent::" << hard_event_str
                 << ">(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_threadfence())) {
    ICHECK_EQ(op->args.size(), 0)
        << "tl.ascend_threadfence expects 0 arguments";
    this->PrintIndent();
    this->stream << "asc_threadfence();\n";
  } else if (op->op.same_as(tl::ascend_get_buf())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_get_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str = Downcast<StringImm>(op->args[0])->value;
    int64_t mode = Downcast<IntImm>(op->args[2])->value;
    this->PrintIndent();
    this->stream << "get_buf(" << pipe_str << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", " << (mode ? "true" : "false") << ");\n";
  } else if (op->op.same_as(tl::ascend_rls_buf())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_rls_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str = Downcast<StringImm>(op->args[0])->value;
    int64_t mode = Downcast<IntImm>(op->args[2])->value;
    this->PrintIndent();
    this->stream << "rls_buf(" << pipe_str << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", " << (mode ? "true" : "false") << ");\n";
  } else if (op->op.same_as(tl::ascend_reduce_add())) {
    os << "asc_reduce_add(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ascend_reduce_max())) {
    os << "asc_reduce_max(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ascend_reduce_min())) {
    os << "asc_reduce_min(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ascend_cross_core_set_flag())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_cross_core_set_flag expects exactly 3 arguments "
           "(mode_id int, pipe string, flag_id)";
    int64_t mode_id = Downcast<IntImm>(op->args[0])->value;
    std::string pipe_str = Downcast<StringImm>(op->args[1])->value;
    this->PrintIndent();
    this->stream << "AscendC::CrossCoreSetFlag<" << mode_id << ", " << pipe_str
                 << ">(";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_cross_core_wait_flag())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_cross_core_wait_flag expects exactly 3 arguments "
           "(mode_id int, pipe string, flag_id)";
    int64_t mode_id = Downcast<IntImm>(op->args[0])->value;
    std::string pipe_str = Downcast<StringImm>(op->args[1])->value;
    this->PrintIndent();
    this->stream << "AscendC::CrossCoreWaitFlag<" << mode_id << ", " << pipe_str
                 << ">(";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
    ICHECK_EQ(op->args.size(), 7)
        << "tl.ascend_copy_gm_to_ubuf expects exactly 7 arguments";
    this->PrintIndent();
    this->stream << "copy_gm_to_ubuf((__ubuf__ void*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__gm__ void*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "), ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[3], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[4], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[5], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[6], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
    ICHECK_EQ(op->args.size(), 8)
        << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";
    this->PrintIndent();
    this->stream << "bisheng::cce::copy_ubuf_to_gm_align_v2((__gm__ void*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__ubuf__ void*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "), ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[3], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[4], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[5], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[6], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[7], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_gm_to_cbuf())) {
    ICHECK_EQ(op->args.size(), 10)
        << "tl.ascend_copy_gm_to_cbuf expects exactly 10 arguments";
    // arg[9] is the transpose flag: 0 → nd2nz, nonzero → dn2nz
    const IntImmNode *transpose_node = op->args[9].as<IntImmNode>();
    bool use_dn2nz = transpose_node && transpose_node->value != 0;
    // Emit set_mte2_nz_para: configure MTE2 NZ layout for GM→L1 copy
    // mte2NzPara encoding: [15:0]=ndNum, [31:16]=dstNzNStride,
    // [47:32]=dstNzC0Stride SDK convention (from data_copy_wrapper_nd.h):
    //   dstNzNStride = 1 (always)
    //   dstNzC0Stride = ceil(height/16)*16  where height = n_rows for nd2nz,
    //   inner_elems for dn2nz
    {
      const IntImmNode *rows_node = op->args[5].as<IntImmNode>();
      const IntImmNode *inner_node = op->args[6].as<IntImmNode>();
      int n_rows_val = rows_node ? rows_node->value : 16;
      int inner_val = inner_node ? inner_node->value : 16;
      // For nd2nz: height=n_rows, width=inner_elems; dstNzC0Stride =
      // ceil(height/16)*16 For dn2nz: height=inner_elems(K), width=n_rows(N);
      // dstNzC0Stride = ceil(width/16)*16 nd2nz [n_rows, d]: dstNzC0Stride =
      // ceil(n_rows/16)*16  (C0 tiles the row dimension) dn2nz [n_rows, d]:
      // dstNzC0Stride = ceil(d/16)*16      (C0 tiles the col=N dimension after
      // transpose)
      //   SDK: dn2nz dValue=height(=n_rows/K), dstNzC0Stride=Ceil(width=d/N,
      //   16)*16
      // dstNzC0Stride = Ceil(nValue, 16) * 16
      // After copy.cc swaps n_value/d_value for dn2nz, args[5] is always
      // nValue (= height for nd2nz, = width for dn2nz). Both cases:
      // dstNzC0Stride = Ceil(nValue, 16) * 16.
      int nz_c0_stride = ((n_rows_val + 15) / 16) * 16;
      this->PrintIndent();
      this->stream << "bisheng::cce::set_mte2_nz_para("
                   << "uint64_t(1) | (uint64_t(1) << 16) | (uint64_t("
                   << nz_c0_stride << ") << 32));\n";
    }
    this->PrintIndent();
    this->stream << (use_dn2nz ? "bisheng::cce::copy_gm_to_cbuf_multi_dn2nz("
                               : "bisheng::cce::copy_gm_to_cbuf_multi_nd2nz(");
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_load_cbuf_to_ca())) {
    ICHECK_EQ(op->args.size(), 9)
        << "tl.ascend_load_cbuf_to_ca expects exactly 9 arguments";
    this->PrintIndent();
    this->stream << "bisheng::cce::load_cbuf_to_ca(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_load_cbuf_to_cb())) {
    ICHECK_EQ(op->args.size(), 9)
        << "tl.ascend_load_cbuf_to_cb expects exactly 9 arguments";
    this->PrintIndent();
    this->stream << "bisheng::cce::load_cbuf_to_cb(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_matrix_cc_to_ub())) {
    ICHECK_EQ(op->args.size(), 26)
        << "tl.ascend_copy_matrix_cc_to_ub expects exactly 26 arguments";
    this->PrintIndent();
    this->stream << "copy_matrix_cc_to_ub(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 26; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    ICHECK_EQ(op->args.size(), 25)
        << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
    // Emit set_loop3_para: configure LOOP3_PARA for L0C→GM NZ-to-ND conversion
    // ndNum=1, srcNdStride=0, dstNdStride=0 (single ND block)
    this->PrintIndent();
    this->stream << "bisheng::cce::set_loop3_para(uint64_t(1));\n";
    this->PrintIndent();
    this->stream << "bisheng::cce::copy_matrix_cc_to_gm(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 25; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_ubuf_to_cbuf())) {
    ICHECK_EQ(op->args.size(), 7)
        << "tl.ascend_copy_ubuf_to_cbuf expects exactly 7 arguments";
    this->PrintIndent();
    this->stream << "copy_ubuf_to_cbuf((__cbuf__ void*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__ubuf__ void*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "), ";
    for (int i = 2; i < 7; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
    // ascend_nd2nz_scatter(src_ub, tmp_ub, rows, cols, fuse_cast,
    //                      dst_dtype_str, src_dtype_str)
    // Emits the SimdVF scatter from an ND tile in `src_ub` into the NZ-laid
    // tmp buffer `tmp_ub`. The post-scatter UB→L1 raw DMA is emitted as a
    // separate ascend_nd2nz_post_copy stmt.
    ICHECK_EQ(op->args.size(), 7)
        << "tl.ascend_nd2nz_scatter expects exactly 7 arguments";
    int rows = op->args[2].as<IntImmNode>()->value;
    int cols = op->args[3].as<IntImmNode>()->value;
    int fuse_cast = op->args[4].as<IntImmNode>()->value;
    std::string dst_dtype = op->args[5].as<StringImmNode>()->value;
    std::string src_dtype = op->args[6].as<StringImmNode>()->value;
    this->PrintIndent();
    this->stream << "ascend_nd2nz_scatter<" << rows << ", " << cols << ", "
                 << rows << ", " << (fuse_cast ? "true" : "false") << ", "
                 << dst_dtype << ", " << src_dtype << ">((__ubuf__ "
                 << src_dtype << "*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__ubuf__ " << dst_dtype << "*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "));\n";
    this->PrintIndent();
    // this->stream << "pipe_barrier(PIPE_ALL);\n";
  } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
    // ascend_nd2nz_post_copy(dst_l1, src_ub, rows, cols, full_rows,
    //                        dst_dtype_str)
    // For multi-fractal: src_ub is the NZ tmp buffer produced by the scatter
    // — emit per-D-group copy_ubuf_to_cbuf with sid correction.
    // For single-fractal (rows<=16 && cols<=16): src_ub is the raw ND source —
    // emit one direct copy (ND == NZ in this case).
    ICHECK_EQ(op->args.size(), 6)
        << "tl.ascend_nd2nz_post_copy expects exactly 6 arguments";
    int rows = op->args[2].as<IntImmNode>()->value;
    int cols = op->args[3].as<IntImmNode>()->value;
    int full_rows = op->args[4].as<IntImmNode>()->value;
    std::string dst_dtype = op->args[5].as<StringImmNode>()->value;

    int dst_elem_bytes = (dst_dtype == "float") ? 4 : 2;
    int full_nBlk = full_rows / 16;
    int dBlk = cols / 16;

    if (rows <= 16 && cols <= 16) {
      // Single fractal: ND == NZ, emit one raw copy.
      int total_bytes = rows * cols * dst_elem_bytes;
      this->PrintIndent();
      this->stream << "copy_ubuf_to_cbuf((__cbuf__ void*)(";
      this->PrintExpr(op->args[0], this->stream);
      this->stream << "), (__ubuf__ void*)(";
      this->PrintExpr(op->args[1], this->stream);
      this->stream << "), 0, " << total_bytes / 32 << ", 1, 0, 0);\n";
    } else {
      // Multi-fractal: per-D-group strided copy from NZ tmp to L1.
      //
      // Sub-tile NZ:  nBlk_sub  = rows / 16,         dBlk = cols / 16
      // Full NZ:      nBlk_full = full_rows / 16
      // Sub-tile fractal (n_sub, d) sits in tmp at offset
      //     (d * nBlk_sub + n_sub) * 256 elements.
      // Full fractal in L1 lives at
      //     p_l1_base + (d * nBlk_full + n_start + n_sub) * 256
      // where n_start = sid * nBlk_sub depends on the AIV id.
      //
      // The dst_ptr we receive from the lowering already includes an
      // ND-style sid offset: dst_ptr = p_l1_base + sid * rows * cols. We
      // need the NZ-style base p_l1_base + sid * nBlk_sub * 256 instead.
      // We can't recover p_l1_base + sid at codegen time, so we emit the
      // correction inline against the supplied dst_ptr:
      //     target = dst_ptr
      //            + (d * nBlk_full * 256 + sid * nBlk_sub * 256
      //               - sid * rows * cols)
      //            = dst_ptr + d * nBlk_full * 256
      //                       - sid * (rows * cols - nBlk_sub * 256)
      //                      [= rows * (cols - 16), per-sid correction]
      // sid is unknown at compile time, so it stays in the emitted C++.
      int nBlk_sub = rows / 16;
      int frac_elems = 256;
      int frac_bytes = frac_elems * dst_elem_bytes;
      int nd_vs_nz_correction = rows * cols - nBlk_sub * frac_elems;
      this->PrintIndent();
      this->stream << "{\n";
      this->PrintIndent();
      this->stream << "  __ubuf__ " << dst_dtype << "* _nd2nz_tmp = (__ubuf__ "
                   << dst_dtype << "*)(";
      this->PrintExpr(op->args[1], this->stream);
      this->stream << ");\n";
      // For each D-group, copy nBlk_sub fractals from tmp to the right NZ
      // slot in L1. d_group_offset = d * nBlk_full * 256 (full-NZ stride);
      // the runtime `- sid * nd_vs_nz_correction` undoes the ND sid offset
      // baked into dst_ptr.
      for (int d = 0; d < dBlk; ++d) {
        int tmp_offset_elems = d * nBlk_sub * frac_elems;
        int copy_bytes = nBlk_sub * frac_bytes;
        int d_group_offset = d * full_nBlk * frac_elems;
        this->PrintIndent();
        this->stream << "  copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ "
                     << dst_dtype << "*)(";
        this->PrintExpr(op->args[0], this->stream);
        this->stream << ") + " << d_group_offset << " - sid * "
                     << nd_vs_nz_correction
                     << "), (__ubuf__ void*)(_nd2nz_tmp + " << tmp_offset_elems
                     << "), 0, " << copy_bytes / 32 << ", 1, 0, 0);\n";
      }
      this->PrintIndent();
      this->stream << "}\n";
    }
  } else if (op->op.same_as(tl::ascend_mad())) {
    ICHECK_EQ(op->args.size(), 10)
        << "tl.ascend_mad expects exactly 10 arguments";
    this->PrintIndent();
    this->stream << "bisheng::cce::mad(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ", ";
    for (int i = 3; i < 10; ++i) {
      if (i > 3)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_gemm_l1())) {
    // ascend_gemm_l1(cc_ptr, cbuf_a_ptr, cbuf_b_ptr, M, K, N, tile_k_sub,
    //                trans_b, clear_accum, dtype_str, buf_offset)
    ICHECK_EQ(op->args.size(), 11)
        << "tl.ascend_gemm_l1 expects exactly 11 arguments";
    int M = op->args[3].as<IntImmNode>()->value;
    int K = op->args[4].as<IntImmNode>()->value;
    int N = op->args[5].as<IntImmNode>()->value;
    int tile_k_sub = op->args[6].as<IntImmNode>()->value;
    int trans_b = op->args[7].as<IntImmNode>()->value;
    std::string dtype_str = op->args[9].as<StringImmNode>()->value;

    this->PrintIndent();
    this->stream << "ascend_gemm_l1<" << M << ", " << K << ", " << N << ", "
                 << tile_k_sub << ", " << (trans_b ? "true" : "false") << ", "
                 << dtype_str << ">(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[8], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[10], this->stream);
    this->stream << ");\n";
  } else if (op->op.same_as(tir::builtin::address_of())) {
    const auto *load = op->args[0].as<tir::BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    const auto *buf_var = load->buffer->data.get();
    std::string scope;
    if (alloc_storage_scope_.count(buf_var)) {
      scope = alloc_storage_scope_.at(buf_var);
    }
    bool needs_ptr_arith = (scope.find("l0") != std::string::npos ||
                            scope.find("l1") != std::string::npos);
    if (needs_ptr_arith) {
      os << "(";
      PrintStorageScope(scope, os);
      PrintType(load->dtype, os);
      os << "*)(";
      os << GetVarID(buf_var);
      if (load->indices.size() == 1) {
        PrimExpr idx = load->indices[0];
        if (!is_zero(idx)) {
          os << " + ";
          PrintExpr(idx, os);
        }
      }
      os << ")";
    } else {
      CodeGenC::VisitExpr_(op, os);
    }
  }
  // --- SimdVF binary ops: (dst, src0, src1, mask) ---
  else if (op->op.same_as(tl::ascend_simd_add()) ||
           op->op.same_as(tl::ascend_simd_sub()) ||
           op->op.same_as(tl::ascend_simd_mul()) ||
           op->op.same_as(tl::ascend_simd_div())) {
    ICHECK_EQ(op->args.size(), 4);
    const char *name = nullptr;
    if (op->op.same_as(tl::ascend_simd_add()))
      name = "Add";
    else if (op->op.same_as(tl::ascend_simd_sub()))
      name = "Sub";
    else if (op->op.same_as(tl::ascend_simd_mul()))
      name = "Mul";
    else
      name = "Div";
    this->PrintIndent();
    this->stream << "AscendC::MicroAPI::" << name << "(";
    this->PrintExpr(op->args[0], this->stream);
    for (int i = 1; i < 4; ++i) {
      this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  }
  // --- SimdVF unary ops: (dst, src, mask) ---
  else if (op->op.same_as(tl::ascend_simd_exp()) ||
           op->op.same_as(tl::ascend_simd_ln()) ||
           op->op.same_as(tl::ascend_simd_sqrt()) ||
           op->op.same_as(tl::ascend_simd_rsqrt()) ||
           op->op.same_as(tl::ascend_simd_abs()) ||
           op->op.same_as(tl::ascend_simd_neg()) ||
           op->op.same_as(tl::ascend_simd_duplicate())) {
    ICHECK_EQ(op->args.size(), 3);
    const char *name = nullptr;
    if (op->op.same_as(tl::ascend_simd_exp()))
      name = "Exp";
    else if (op->op.same_as(tl::ascend_simd_ln()))
      name = "Ln";
    else if (op->op.same_as(tl::ascend_simd_sqrt()))
      name = "Sqrt";
    else if (op->op.same_as(tl::ascend_simd_rsqrt()))
      name = "Rsqrt";
    else if (op->op.same_as(tl::ascend_simd_abs()))
      name = "Abs";
    else if (op->op.same_as(tl::ascend_simd_duplicate()))
      name = "Duplicate";
    else
      name = "Neg";
    this->PrintIndent();
    this->stream << "AscendC::MicroAPI::" << name << "(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ");\n";
  }
  // --- SimdVF load: 3 args (simple) or 4 args (with AddrReg) ---
  else if (op->op.same_as(tl::ascend_simd_load())) {
    ICHECK(op->args.size() == 3 || op->args.size() == 4);
    std::string dtype_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "AscendC::MicroAPI::LoadAlign<" << dtype_str << ">(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[2], this->stream);
    if (op->args.size() == 4) {
      this->stream << ", ";
      this->PrintExpr(op->args[3], this->stream);
    }
    this->stream << ");\n";
  }
  // --- SimdVF load_single: (dtype_str, dst, src) ---
  else if (op->op.same_as(tl::ascend_simd_load_single())) {
    ICHECK_EQ(op->args.size(), 3);
    std::string dtype_str = Downcast<StringImm>(op->args[0])->value;
    // Infer LoadDist bit width from dtype:
    //   float, int32/uint32 → 4 bytes → DIST_BRC_B32
    //   half_t, bfloat16_t, int16/uint16 → 2 bytes → DIST_BRC_B16
    //   int8/uint8 → 1 byte → DIST_BRC_B8
    int bit_width = 32;
    if (dtype_str == "float") {
      bit_width = 32;
    } else if (dtype_str == "half_t" || dtype_str == "bfloat16_t") {
      bit_width = 16;
    } else {
      // Parse "intN_t" / "uintN_t" → extract N
      auto pos = dtype_str.find_first_of("0123456789");
      if (pos != std::string::npos)
        bit_width = std::stoi(dtype_str.substr(pos));
    }
    int dist_val = bit_width / 8; // 4, 2, or 1
    const char *dist_str = nullptr;
    switch (dist_val) {
    case 1:
      dist_str = "MicroAPI::LoadDist::DIST_BRC_B8";
      break;
    case 2:
      dist_str = "MicroAPI::LoadDist::DIST_BRC_B16";
      break;
    case 4:
      dist_str = "MicroAPI::LoadDist::DIST_BRC_B32";
      break;
    default:
      LOG(FATAL) << "Unsupported bit width for load_single: " << bit_width;
    }
    this->PrintIndent();
    this->stream << "AscendC::MicroAPI::LoadAlign<" << dtype_str << ", "
                 << dist_str << ">(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[2], this->stream);
    this->stream << ");\n";
  }
  // --- SimdVF store: 4 args (simple) or 5 args (with AddrReg) ---
  else if (op->op.same_as(tl::ascend_simd_store())) {
    ICHECK(op->args.size() == 4 || op->args.size() == 5);
    std::string dtype_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "AscendC::MicroAPI::StoreAlign<" << dtype_str << ">(";
    this->PrintExpr(op->args[1], this->stream);
    for (size_t i = 2; i < op->args.size(); ++i) {
      this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  }
  // --- SimdVF create_addr_reg: (dtype_str, idx0, stride0, ...) → expression
  // ---
  else if (op->op.same_as(tl::ascend_simd_create_addr_reg())) {
    ICHECK_GE(op->args.size(), 3);
    std::string dtype_str = Downcast<StringImm>(op->args[0])->value;
    os << "AscendC::MicroAPI::CreateAddrReg<" << dtype_str << ">(";
    for (size_t i = 1; i < op->args.size(); ++i) {
      if (i > 1)
        os << ", ";
      this->PrintExpr(op->args[i], os);
    }
    os << ")";
  }
  // --- SimdVF update_mask: (dtype_str, count) → expression ---
  else if (op->op.same_as(tl::ascend_simd_update_mask())) {
    ICHECK_EQ(op->args.size(), 2);
    std::string dtype_str = Downcast<StringImm>(op->args[0])->value;
    os << "AscendC::MicroAPI::UpdateMask<" << dtype_str << ">((uint32_t)";
    this->PrintExpr(op->args[1], os);
    os << ")";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitStmt_(const ForNode *op) {
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
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
      if ((iv->thread_tag == "threadIdx.x" || iv->thread_tag == "threadIdx.y" ||
           iv->thread_tag == "threadIdx.z") &&
          !inside_simtvf_body_ && !inside_simdvf_body_) {
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
      } else if (iv->thread_tag == "cthread") {
        stream << "get_subblockid()";
      } else {
        stream << "0";
      }
      stream << ";\n";
    }
    this->VisitStmt(op->body);
    return;
  }
  // SimdVF scope marker: transparent in codegen, just visit body
  if (op->attr_key == "tl.simdvf_scope") {
    this->VisitStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

static std::string GetAscendScopeQualifier(const std::string &scope) {
  if (scope == "shared" || scope == "shared.dyn")
    return "__ubuf__";
  if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
    return "__ca__";
  if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return "__cb__";
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return "__cc__";
  if (scope == "shared.l1" || scope == "shared.l1.dyn")
    return "__cbuf__";
  return "";
}

void CodeGenTileLangAscend::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);

  if (scope == "shared") {
    LOG(FATAL)
        << "Static shared memory (scope='shared') is not supported on Ascend. "
        << "Use dynamic shared memory (scope='shared.dyn') with "
           "T.alloc_shared.";
  }

  // SimdVF: emit MicroAPI::RegTensor<T> for fragment buffers
  if (inside_simdvf_body_ && scope == "local.fragment") {
    std::string vid = AllocVarID(op->buffer_var.get());
    this->PrintIndent();
    stream << "AscendC::MicroAPI::RegTensor<";
    PrintType(op->dtype, stream);
    stream << "> " << vid << ";\n";
    alloc_storage_scope_[op->buffer_var.get()] = scope;
    RegisterHandleType(op->buffer_var.get(), op->dtype);
    this->PrintStmt(op->body);
    return;
  }

  std::string qualifier = GetAscendScopeQualifier(scope);
  if (!qualifier.empty()) {
    // Generate: __qualifier__ <type> *buf = (__qualifier__ <type> *)0;
    std::string vid = AllocVarID(op->buffer_var.get());
    DataType dtype = op->dtype;
    this->PrintIndent();
    stream << qualifier << " ";
    PrintType(dtype, stream);
    stream << " *" << vid << " = (" << qualifier << " ";
    PrintType(dtype, stream);
    stream << " *)0;\n";
    alloc_storage_scope_[op->buffer_var.get()] = scope;
    RegisterHandleType(op->buffer_var.get(), dtype);
    this->PrintStmt(op->body);
    return;
  }

  // For other scopes, use parent class implementation
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const tir::LetStmtNode *op) {
  // Inside SimdVF body, handle-typed let-bindings (e.g. MaskReg from
  // UpdateMask) should use 'auto' instead of 'void*'.
  if (inside_simdvf_body_ && op->var.dtype().is_handle()) {
    std::string vid = AllocVarID(op->var.get());
    if (const auto *call = op->value.as<CallNode>()) {
      // ascend_init_mask(count) → uint32_t vid = count;
      if (call->op.same_as(tl::ascend_init_mask())) {
        PrintIndent();
        stream << "uint32_t " << vid << " = ";
        PrintExpr(call->args[0], stream);
        stream << ";\n";
        PrintStmt(op->body);
        return;
      }
      // ascend_simd_update_mask(dtype, count) → auto vid =
      // UpdateMask<dtype>(count); When count is a Var (from ascend_init_mask),
      // use it directly as lvalue.
      if (call->op.same_as(tl::ascend_simd_update_mask())) {
        std::string dtype_str = Downcast<StringImm>(call->args[0])->value;
        if (call->args[1].as<VarNode>()) {
          PrintIndent();
          stream << "auto " << vid << " = AscendC::MicroAPI::UpdateMask<"
                 << dtype_str << ">(";
          PrintExpr(call->args[1], stream);
          stream << ");\n";
        } else {
          PrintIndent();
          stream << "uint32_t " << vid << "_cnt = ";
          PrintExpr(call->args[1], stream);
          stream << ";\n";
          PrintIndent();
          stream << "auto " << vid << " = AscendC::MicroAPI::UpdateMask<"
                 << dtype_str << ">(" << vid << "_cnt);\n";
        }
        PrintStmt(op->body);
        return;
      }
    }
    PrintIndent();
    stream << "auto " << vid << " = ";
    PrintExpr(op->value, stream);
    stream << ";\n";
    PrintStmt(op->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (op->dtype.is_bfloat16() || op->dtype.is_float16() ||
      op->dtype.is_float()) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "std::numeric_limits<";
      p->PrintType(op->dtype, temp);
      temp << ">::infinity()";
    } else if (std::isnan(op->value)) {
      temp << "std::numeric_limits<";
      p->PrintType(op->dtype, temp);
      temp << ">::quiet_NaN()";
    } else {
      p->PrintType(op->dtype, temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (op->dtype.bits()) {
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangAscend::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::VisitExpr_(const CastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  DataType from = op->value->dtype;
  DataType to = op->dtype;
  if (from.is_bfloat16() && from.lanes() == 2 && to.is_float() &&
      to.bits() == 32 && to.lanes() == 2) {
    has_bf16_cast_ = true;
    os << "__bfloat1622float2(";
    PrintExpr(op->value, os);
    os << ")";
    return;
  }
  if (from.is_float() && from.bits() == 32 && from.lanes() == 2 &&
      to.is_bfloat16() && to.lanes() == 2) {
    has_bf16_cast_ = true;
    os << "__float22bfloat162_rn(";
    PrintExpr(op->value, os);
    os << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangAscend::PrintStorageSync(const CallNode *op) {
  ICHECK_GE(op->args.size(), 1U);
  PrintIndent();
  stream << "asc_syncthreads();\n";
}

void CodeGenTileLangAscend::PrintStorageScope(const std::string &scope,
                                              std::ostream &os) {
  if (scope == "global") {
    os << "__gm__ ";
  } else {
    std::string qualifier = GetAscendScopeQualifier(scope);
    if (!qualifier.empty()) {
      os << qualifier << " ";
    }
  }
}

bool CodeGenTileLangAscend::IsScopePartOfType() const {
  // Ascend requires memory scope qualifiers as part of the pointer type
  return true;
}

void CodeGenTileLangAscend::PrintVecElemLoad(const std::string &vec, DataType t,
                                             int i, std::ostream &os) {
  if (t.is_scalar()) {
    os << vec;
    return;
  }
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 4) << "Ascend vector lanes max 4: i=" << i;
  if (t.lanes() <= 4) {
    os << vec << "." << access[i];
  } else {
    os << "((" << vec << ")." << access[i / 2] << ")";
  }
}

void CodeGenTileLangAscend::PrintVecElemStore(const std::string &vec,
                                              DataType t, int i,
                                              const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 4) << "Ascend vector lanes max 4: i=" << i;
  if (t.lanes() <= 4) {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  } else {
    stream << "((" << vec << ")." << access[i / 2] << ") = " << value << ";\n";
  }
}

void CodeGenTileLangAscend::PrintVecBinaryOp(const std::string &op, DataType t,
                                             PrimExpr lhs, PrimExpr rhs,
                                             std::ostream &os) {
  // Only scalarize float4 division; other vector binary ops fall through
  // to the base CodeGenC implementation.
  if (op != "/" || t.lanes() <= 1) {
    CodeGenC::PrintVecBinaryOp(op, t, lhs, rhs, os);
    return;
  }

  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      value_temp << "(";
      PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
      value_temp << " / ";
      PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
      value_temp << ")";
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  os << sret;
}

void CodeGenTileLangAscend::PrintCallExtern(Type ret_type,
                                            ffi::String global_symbol,
                                            const ffi::Array<PrimExpr> &args,
                                            bool skip_first_arg,
                                            std::ostream &os) {
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
    // Scalarize: for each lane, extract scalar args, call the scalar function,
    // store result into the result vector lane.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
    return;
  }

  // Intercept debug_print_* call_extern functions for Ascend.
  // blockIdx is an ASC builtin available only inside kernel functions, so we
  // emit AscendC::printf inline instead of calling through template helpers
  // (which live at file scope and can't see the builtins).
  {
    size_t arg_start = static_cast<size_t>(skip_first_arg);

    auto emit_val = [&](size_t idx) {
      DataType dtype = args[idx].dtype();
      if (dtype.is_bool()) {
        PrintExpr(args[idx], os);
        os << " ? \"true\" : \"false\"";
      } else if ((dtype.is_float() || dtype.is_bfloat16()) &&
                 dtype.bits() < 32) {
        os << "(float)(";
        PrintExpr(args[idx], os);
        os << ")";
      } else if (!dtype.is_float() && !dtype.is_bfloat16()) {
        if (dtype.bits() < 32) {
          os << "(int)(";
          PrintExpr(args[idx], os);
          os << ")";
        } else {
          PrintExpr(args[idx], os);
        }
      } else {
        PrintExpr(args[idx], os);
      }
    };

    auto get_fmt = [](DataType dtype) -> std::string {
      if (dtype.is_bool())
        return "%s";
      if (dtype.is_float() || dtype.is_bfloat16())
        return "%f";
      return "%d";
    };

    const char *pfx = inside_simtvf_body_ ? "printf" : "AscendC::printf";
    const char *ctx_var = inside_simtvf_body_ ? "threadIdx.x" : "blockIdx.x";
    const char *ctx_fmt = inside_simtvf_body_ ? "threadIdx=%d" : "blockIdx=%d";

    if (global_symbol == "debug_print_msg") {
      // args: [msg_string]
      std::string msg = Downcast<StringImm>(args[arg_start])->value;
      os << pfx << "(\"[" << ctx_fmt << "] " << msg << "\\n\", " << ctx_var
         << ")";
      return;
    }

    if (global_symbol == "debug_print_var") {
      // args: [msg_string, val]
      std::string msg = Downcast<StringImm>(args[arg_start])->value;
      std::string fmt = get_fmt(args[arg_start + 1].dtype());
      os << pfx << "(\"[" << ctx_fmt << "] " << msg << " = " << fmt << "\\n\", "
         << ctx_var << ", ";
      emit_val(arg_start + 1);
      os << ")";
      return;
    }

    if (global_symbol == "debug_print_buffer_value") {
      // args: [msg_string, buf_name, index, val]
      std::string msg = Downcast<StringImm>(args[arg_start])->value;
      std::string buf_name = Downcast<StringImm>(args[arg_start + 1])->value;
      std::string fmt = get_fmt(args[arg_start + 3].dtype());
      os << pfx << "(\"[" << ctx_fmt << "] " << msg << " " << buf_name
         << "[%d] = " << fmt << "\\n\", " << ctx_var << ", ";
      PrintExpr(args[arg_start + 2], os);
      os << ", ";
      emit_val(arg_start + 3);
      os << ")";
      return;
    }

    if (global_symbol == "debug_print_buffer_dump") {
      // args: [msg, buf_name, ptr, size, type_str]
      uint64_t size_val = Downcast<IntImm>(args[arg_start + 3])->value;
      std::string dtype_str = Downcast<StringImm>(args[arg_start + 4])->value;
      std::string position = "VECIN";
      auto scope = args[arg_start + 2]
                       .as<CallNode>()
                       ->args[0]
                       .as<BufferLoadNode>()
                       ->buffer.scope();
      if (scope == "shared.l0c") {
        position = "CO1";
      }

      os << "AscendC::DumpTensor(AscendC::LocalTensor<" << dtype_str
         << ">(AscendC::TPosition::" << position << ", (uint32_t)(uintptr_t)";
      PrintExpr(args[arg_start + 2], os);
      os << ", " << size_val << "), 0, " << size_val << ")";
      return;
    }
    if (global_symbol == "debug_print_global_buffer_dump") {
      // args: [msg, buf_name, ptr, size, type_str]
      uint64_t size_val = Downcast<IntImm>(args[arg_start + 3])->value;
      std::string dtype_str = Downcast<StringImm>(args[arg_start + 4])->value;
      this->PrintIndent();
      this->stream << "{ AscendC::GlobalTensor<" << dtype_str
                   << "> __dump_gt; __dump_gt.SetGlobalBuffer(";
      // args[arg_start+2] via access_ptr → resolve to the raw buffer Var
      if (auto *call = args[arg_start + 2].as<tir::CallNode>()) {
        // tvm_access_ptr(base, ...) → extract base Var
        if (call->args.size() >= 2) {
          PrintExpr(call->args[1], this->stream);
        } else {
          PrintExpr(args[arg_start + 2], this->stream);
        }
      } else {
        PrintExpr(args[arg_start + 2], this->stream);
      }
      this->stream << ", " << size_val
                   << "); AscendC::DumpTensor(__dump_gt, 0, " << size_val
                   << "); }\n";
      return;
    }
  }
  // debug_print_global_buffer_dump handled in VisitStmt_(EvaluateNode)

  CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
}

} // namespace codegen
} // namespace tvm
