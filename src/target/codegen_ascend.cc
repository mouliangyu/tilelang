#include "codegen_ascend.h"

#include "../op/builtin.h"
#include "./utils.h"
#include <sstream>
#include <string>
#include <unordered_set>

#include "../../3rdparty/tvm/src/tir/transforms/ir_utils.h"
#include <tvm/tir/analysis.h>
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

// Map a TVM DataType to the Ascend CCE vector type suffix (e.g. Float32 ->
// "f32").
static std::string CCEVectorSuffix(tvm::DataType dtype) {
  if (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn())
    return "f8e4m3";
  if (dtype.is_float8_e5m2())
    return "f8e5m2";
  if (dtype.is_float8_e8m0fnu())
    return "f8e8m0";
  if (dtype.is_float())
    return (dtype.bits() == 16) ? "f16" : "f32";
  if (dtype.is_bfloat16())
    return "bf16";
  if (dtype.is_int())
    return "s" + std::to_string(dtype.bits());
  if (dtype.is_uint())
    return "u" + std::to_string(dtype.bits());
  return "f32";
}

// Get element width in bits for choosing b8/b16/b32 predicate/load/store
// variants.
static int CCEElemWidthBits(tvm::DataType dtype) { return dtype.bits(); }

// Map DataType to the C++ type name used in __ubuf__ pointer casts.
static std::string CCEUBufType(tvm::DataType dtype) {
  if (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn())
    return "float8_e4m3_t";
  if (dtype.is_float8_e5m2())
    return "float8_e5m2_t";
  if (dtype.is_float8_e8m0fnu())
    return "float8_e8m0_t";
  if (dtype.is_float())
    return (dtype.bits() == 16) ? "half" : "float";
  if (dtype.is_bfloat16())
    return "bfloat16_t";
  if (dtype.is_int())
    return "int" + std::to_string(dtype.bits()) + "_t";
  if (dtype.is_uint())
    return "uint" + std::to_string(dtype.bits()) + "_t";
  return "float";
}

namespace tvm {
namespace codegen {

void CodeGenTileLangAscend::PrintCCEVectorSuffix(DataType t, std::ostream &os) {
  os << CCEVectorSuffix(t);
}

void CodeGenTileLangAscend::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  decl_stream << "#include <tl_templates/ascend/common.h>\n";
  decl_stream << "\n";
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\"\n";
}

static std::string GetAscendFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8) for "
                  "Ascend fp8";
  }
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2()) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.is_float8_e8m0fnu()) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported Ascend fp8 type " << type;
  }
  return stream.str();
}

static std::string GetAscendFP8ScalarValueType(DataType type) {
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    return "tl::float_e4m3_t";
  }
  if (type.is_float8_e5m2()) {
    return "tl::float_e5m2_t";
  }
  LOG(FATAL) << "Unsupported Ascend scalar fp8 conversion type " << type;
  return "";
}

void CodeGenTileLangAscend::PrintType(DataType t, std::ostream &os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types for handles";
    os << "void*";
    return;
  }
  // CCE predicate register: boolx256 = vector_bool
  if (t.is_bool() && t.lanes() == 256) {
    os << "vector_bool";
    return;
  }
  // CCE vector register: high lane counts map to vector_<T>
  if (lanes >= 32) {
    DataType elem = t.element_of();
    os << "vector_" << CCEVectorSuffix(elem);
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
      } else if (lanes <= 8) {
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
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
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for bfloat16 type";
      os << "uint" << lanes / 2;
    } else {
      fail = true;
    }
  } else if (t.is_float8()) {
    os << GetAscendFP8Type(t);
    return;
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
  bool has_cube = false, has_vector = false;
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
      }
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

  // Conditionally include nd2nz_copy.h when ascend_nd2nz_copy is used
  if (has_nd2nz_copy_ && !has_nd2nz_copy_included_) {
    decl_stream << "#include <tl_templates/ascend/nd2nz_copy.h>\n";
    has_nd2nz_copy_included_ = true;
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

Array<Var>
CodeGenTileLangAscend::CollectVFCaptures(const tir::BlockNode *op) const {
  Array<Var> predefined;
  for (const tir::Buffer &buf : op->alloc_buffers) {
    predefined.push_back(buf->data);
  }
  Array<Var> undefined = UndefinedVars(op->body, predefined);
  Array<Var> captures;
  std::unordered_set<const VarNode *> seen;
  for (const Var &var : undefined) {
    if (seen.count(var.get()))
      continue;
    seen.insert(var.get());
    const DataType dtype = var->dtype;
    ICHECK(dtype.lanes() == 1 &&
           (dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
            dtype.is_handle() || dtype.is_bool()))
        << op->name_hint << " capture variable `" << var
        << "` has unsupported dtype `" << dtype
        << "`. Only scalar int/uint/float/bool/handle are supported.";
    captures.push_back(var);
  }
  return captures;
}

void CodeGenTileLangAscend::EmitVFFunction(const tir::BlockNode *op,
                                           const ffi::Array<tir::Var> &captures,
                                           const std::string &helper_name,
                                           const std::string &func_attrs,
                                           bool &inside_vf_body) {
  // Register captures with their storage scope
  for (const auto &v : captures) {
    if (v->type_annotation.as<PointerTypeNode>()) {
      alloc_storage_scope_[v.get()] = GetPtrStorageScope(v);
    }
  }
  // Save and reset variable naming state
  std::unordered_map<const tir::VarNode *, std::string> saved_var_idmap;
  saved_var_idmap.swap(var_idmap_);
  NameSupply saved_name_supply = name_supply_;
  name_supply_ = NameSupply();
  for (const auto &kv : saved_var_idmap) {
    var_idmap_[kv.first] = kv.second;
  }
  // Reserve all capture parameter names in the fresh NameSupply so that
  // newly generated variables inside the VF body do not shadow them.
  for (const auto &v : captures) {
    auto it = var_idmap_.find(v.get());
    std::string cap_name =
        (it != var_idmap_.end()) ? it->second : std::string(v->name_hint);
    name_supply_->ReserveName(cap_name, false);
  }
  // Render body to temporary stream
  std::ostringstream saved_main_stream;
  stream.swap(saved_main_stream);
  bool old_inside = inside_vf_body;
  inside_vf_body = true;
  CodeGenC::VisitStmt(op->body);
  inside_vf_body = old_inside;
  std::string helper_body = FixIndent(stream.str());
  stream.str("");
  stream.clear();
  stream.swap(saved_main_stream);
  // Restore naming state
  var_idmap_.swap(saved_var_idmap);
  name_supply_ = saved_name_supply;

  auto resolve_var_name = [this](const Var &v) -> std::string {
    auto it = var_idmap_.find(v.get());
    if (it != var_idmap_.end())
      return it->second;
    return v->name_hint;
  };

  // Emit helper function to decl_stream
  decl_stream << func_attrs << " " << helper_name << "(";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0)
      decl_stream << ", ";
    const Var &v = captures[i];
    std::string vname = resolve_var_name(v);
    if (!v->type_annotation.as<PointerTypeNode>()) {
      PrintType(v.dtype(), decl_stream);
      decl_stream << " " << vname;
    } else {
      auto scope_it = alloc_storage_scope_.find(v.get());
      if (scope_it != alloc_storage_scope_.end() &&
          scope_it->second == "local.var") {
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
        } else {
          PrintType(v.dtype(), decl_stream);
        }
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
        PrintStorageScope(GetPtrStorageScope(v), decl_stream);
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
          decl_stream << "* " << vname;
        } else {
          decl_stream << "void* " << vname;
        }
      }
    }
  }
  decl_stream << ") {\n" << helper_body << "}\n\n";
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
    if (!has_simd_inst_included_) {
      decl_stream << "#include <tl_templates/ascend/simd_inst.h>\n";
      has_simd_inst_included_ = true;
    }
    auto captures = CollectVFCaptures(op);
    std::string helper_name =
        "simd_vf_" + std::to_string(simdvf_helper_counter_++);
    EmitVFFunction(op, captures, helper_name, "__simd_vf__ inline void",
                   inside_simdvf_body_);

    PrintIndent();
    stream << helper_name << "(";
    for (size_t i = 0; i < captures.size(); ++i) {
      if (i != 0)
        stream << ", ";
      auto it = var_idmap_.find(captures[i].get());
      stream << (it != var_idmap_.end() ? it->second
                                        : std::string(captures[i]->name_hint));
    }
    stream << ");\n";
    return;
  }
  if (op->name_hint == "SIMT_VF") {
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

    auto captures = CollectVFCaptures(op);
    std::string helper_name =
        "simt_vf_" + std::to_string(simtvf_helper_counter_++);

    int64_t total_threads = thread_x * thread_y * thread_z;
    std::string func_attrs = "__simt_vf__ ";
    if (total_threads > 0) {
      func_attrs += "__launch_bounds__(" + std::to_string(total_threads) + ") ";
    }
    func_attrs += "inline void";
    EmitVFFunction(op, captures, helper_name, func_attrs, inside_simtvf_body_);

    PrintIndent();
    stream << "asc_vf_call<" << helper_name << ">(cce::dim3(" << thread_x;
    if (thread_y > 1 || thread_z > 1) {
      stream << ", " << thread_y;
      if (thread_z > 1)
        stream << ", " << thread_z;
    }
    stream << ")";
    for (size_t i = 0; i < captures.size(); ++i) {
      auto it = var_idmap_.find(captures[i].get());
      stream << ", "
             << (it != var_idmap_.end() ? it->second
                                        : std::string(captures[i]->name_hint));
    }
    stream << ");\n";
    return;
  }
  PrintStmt(op->body);
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
    ICHECK_EQ(op->args.size(), 1) << "tl.ascend_pipe_barrier expects exactly "
                                     "1 argument (pipe_t string)";
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
  } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
    ICHECK_EQ(op->args.size(), 1)
        << "tl.ascend_set_hf32_mode expects exactly 1 argument (mode_int)";
    int mode = Downcast<IntImm>(op->args[0])->value;
    this->PrintIndent();
    if (mode == 0) {
      this->stream << "AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);\n";
    } else {
      this->stream << "AscendC::SetHF32Mode(AscendC::HF32Mode::ENABLE);\n";
      this->PrintIndent();
      this->stream << "AscendC::SetHF32TransMode(AscendC::"
                   << (mode == 1 ? "HF32TransMode::NEAREST_ZERO"
                                 : "HF32TransMode::NEAREST_EVEN")
                   << ");\n";
    }
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
    ICHECK_EQ(op->args.size(), 11)
        << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";
    this->PrintIndent();
    this->stream << "copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__gm__ uint8_t*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ")";
    for (size_t i = 2; i < op->args.size(); ++i) {
      this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
    ICHECK_EQ(op->args.size(), 8)
        << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";
    this->PrintIndent();
    this->stream << "copy_ubuf_to_gm_align_v2((__gm__ void*)(";
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
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_gm_to_cbuf())) {
    ICHECK_EQ(op->args.size(), 11)
        << "tl.ascend_copy_gm_to_cbuf expects exactly 11 arguments";
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
      const IntImmNode *rows_node = op->args[10].as<IntImmNode>();
      const IntImmNode *inner_node = op->args[6].as<IntImmNode>();
      int n_rows_val = rows_node ? rows_node->value : 16;
      int inner_val = inner_node ? inner_node->value : 16;
      // For nd2nz: height=n_rows, width=inner_elems; dstNzC0Stride =
      // ceil(height/16)*16 For dn2nz: height=inner_elems(K), width=n_rows(N);
      // dstNzC0Stride = ceil(width/16)*16 nd2nz [n_rows, d]: dstNzC0Stride =
      // ceil(n_rows/16)*16  (C0 tiles the row dimension) dn2nz [n_rows, d]:
      // dstNzC0Stride = ceil(d/16)*16      (C0 tiles the col=N dimension
      // after transpose)
      //   SDK: dn2nz dValue=height(=n_rows/K), dstNzC0Stride=Ceil(width=d/N,
      //   16)*16
      // dstNzC0Stride = Ceil(nValue, 16) * 16
      // After copy.cc swaps n_value/d_value for dn2nz, args[5] is always
      // nValue (= height for nd2nz, = width for dn2nz). Both cases:
      // dstNzC0Stride = Ceil(nValue, 16) * 16.
      int nz_c0_stride = ((n_rows_val + 15) / 16) * 16;
      this->PrintIndent();
      this->stream << "set_mte2_nz_para("
                   << "uint64_t(1) | (uint64_t(1) << 16) | (uint64_t("
                   << nz_c0_stride << ") << 32));\n";
    }
    this->PrintIndent();
    this->stream << (use_dn2nz ? "copy_gm_to_cbuf_multi_dn2nz("
                               : "copy_gm_to_cbuf_multi_nd2nz(");
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_load_cbuf_to_ca())) {
    ICHECK_EQ(op->args.size(), 9)
        << "tl.ascend_load_cbuf_to_ca expects exactly 9 arguments";
    this->PrintIndent();
    this->stream << "load_cbuf_to_ca(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_load_cbuf_to_cb())) {
    ICHECK_EQ(op->args.size(), 9)
        << "tl.ascend_load_cbuf_to_cb expects exactly 9 arguments";
    this->PrintIndent();
    this->stream << "load_cbuf_to_cb(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 9; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
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
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    ICHECK_EQ(op->args.size(), 25)
        << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
    // Emit set_loop3_para: configure LOOP3_PARA for L0C→GM NZ-to-ND
    // conversion ndNum=1, srcNdStride=0, dstNdStride=0 (single ND block)
    this->PrintIndent();
    this->stream << "set_loop3_para(uint64_t(1));\n";
    this->PrintIndent();
    this->stream << "copy_matrix_cc_to_gm(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ", ";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << ", ";
    for (int i = 2; i < 25; ++i) {
      if (i > 2)
        this->stream << ", ";
      this->PrintExpr(op->args[i], this->stream);
    }
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
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
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
    // ascend_nd2nz_scatter(src_ub, tmp_ub, rows, cols,
    //                      dst_dtype_str, src_dtype_str)
    // Emits the SimdVF scatter from an ND tile in `src_ub` into the NZ-laid
    // tmp buffer `tmp_ub`. The post-scatter UB→L1 raw DMA is emitted as a
    // separate ascend_nd2nz_post_copy stmt.
    // Template: <ROWS, COLS, SrcT, DstT> — the (SrcT, DstT) pair selects
    // the specialization (same-type or fused-cast).
    ICHECK_EQ(op->args.size(), 6)
        << "tl.ascend_nd2nz_scatter expects exactly 6 arguments";
    int rows = op->args[2].as<IntImmNode>()->value;
    int cols = op->args[3].as<IntImmNode>()->value;
    std::string dst_dtype = op->args[4].as<StringImmNode>()->value;
    std::string src_dtype = op->args[5].as<StringImmNode>()->value;
    this->PrintIndent();
    this->stream << "ascend_nd2nz_scatter<" << rows << ", " << cols << ", "
                 << src_dtype << ", " << dst_dtype << ">((__ubuf__ "
                 << src_dtype << "*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << "), (__ubuf__ " << dst_dtype << "*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "));\n";
    this->PrintIndent();
    // this->stream << "pipe_barrier(PIPE_ALL);\n";
  } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
    // ascend_nd2nz_post_copy(dst_l1, src_nz_ub, rows, cols, full_rows,
    //                        dst_dtype_str)
    // 2D copy_ubuf_to_cbuf from NZ tmp to L1.
    //
    // Sid correction: warpgroup_partition gives dst_ptr an ND offset
    //   (sid * rows * cols elements).  The true NZ offset is
    //   sid * rows * elems_perC0 (elems_perC0 = 32 / sizeof(elem)).
    // We subtract the difference inline — sid is a runtime variable.
    //
    // 2D copy model: each D-group = one "row".
    //   burst_num = cols / elems_perC0    (D-groups)
    //   burst_len = rows                  (32B blocks per D-group: one per row)
    //   src_gap   = 1                     (pad block in tmp between D-groups)
    //   dst_gap   = full_rows - rows      (other AIV rows in L1 per D-group)
    ICHECK_EQ(op->args.size(), 6)
        << "tl.ascend_nd2nz_post_copy expects exactly 6 arguments";
    int rows = op->args[2].as<IntImmNode>()->value;
    int cols = op->args[3].as<IntImmNode>()->value;
    int full_rows = op->args[4].as<IntImmNode>()->value;
    std::string dst_dtype = op->args[5].as<StringImmNode>()->value;

    int elem_bytes = (dst_dtype == "float") ? 4 : 2;
    int elems_perC0 = 32 / elem_bytes; // 16 (half) or 8 (float)

    int burst_num = cols / elems_perC0;           // D-groups
    int burst_len = rows;                         // 32B blocks per D-group
    int src_gap = 1;                              // pad in tmp between D-groups
    int dst_gap = full_rows - rows;               // other AIV rows in L1
    int correction = rows * (cols - elems_perC0); // ND→NZ ptr delta

    this->PrintIndent();
    this->stream << "copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ " << dst_dtype
                 << "*)(";
    this->PrintExpr(op->args[0], this->stream);
    this->stream << ") - sid * " << correction << "), (__ubuf__ void*)(";
    this->PrintExpr(op->args[1], this->stream);
    this->stream << "), 0, " << burst_num << ", " << burst_len << ", "
                 << src_gap << ", " << dst_gap << ");\n";
  } else if (op->op.same_as(tl::ascend_mad())) {
    ICHECK_EQ(op->args.size(), 10)
        << "tl.ascend_mad expects exactly 10 arguments";
    this->PrintIndent();
    this->stream << "mad(";
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
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
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
    if (op->args.size() >= 4 && op->args[3].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[3])->value;
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
  // --- SimdVF binary ops: (src0, src1, mask, mode) ---
  else if (op->op.same_as(tl::simd_vadd()) || op->op.same_as(tl::simd_vsub()) ||
           op->op.same_as(tl::simd_vmul()) || op->op.same_as(tl::simd_vdiv()) ||
           op->op.same_as(tl::simd_vmax()) || op->op.same_as(tl::simd_vmin())) {
    ICHECK_EQ(op->args.size(), 4);
    const char *name = nullptr;
    if (op->op.same_as(tl::simd_vadd()))
      name = "vadd";
    else if (op->op.same_as(tl::simd_vsub()))
      name = "vsub";
    else if (op->op.same_as(tl::simd_vmul()))
      name = "vmul";
    else if (op->op.same_as(tl::simd_vmax()))
      name = "vmax";
    else if (op->op.same_as(tl::simd_vmin()))
      name = "vmin";
    else
      name = "vdiv";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ", ";
    PrintExpr(op->args[2], os);                            // mask
    os << ", " << Downcast<StringImm>(op->args[3])->value; // mode
    os << ")";
  }
  // --- SimdVF bitwise: (src0, src1, mask, mode) ---
  else if (op->op.same_as(tl::simd_vand()) || op->op.same_as(tl::simd_vor())) {
    ICHECK_EQ(op->args.size(), 4);
    const char *name = op->op.same_as(tl::simd_vand()) ? "vand" : "vor";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ", ";
    PrintExpr(op->args[2], os);                            // mask
    os << ", " << Downcast<StringImm>(op->args[3])->value; // mode
    os << ")";
  }
  // --- SimdVF unary ops: (src, mask, mode) ---
  else if (op->op.same_as(tl::simd_vexp()) || op->op.same_as(tl::simd_vln()) ||
           op->op.same_as(tl::simd_vsqrt()) ||
           op->op.same_as(tl::simd_vrsqrt()) ||
           op->op.same_as(tl::simd_vabs()) || op->op.same_as(tl::simd_vneg()) ||
           op->op.same_as(tl::simd_vrelu()) ||
           op->op.same_as(tl::simd_vrec())) {
    ICHECK_EQ(op->args.size(), 3);
    const char *name = nullptr;
    if (op->op.same_as(tl::simd_vexp()))
      name = "vexp";
    else if (op->op.same_as(tl::simd_vln()))
      name = "vln";
    else if (op->op.same_as(tl::simd_vsqrt()))
      name = "vsqrt";
    else if (op->op.same_as(tl::simd_vrsqrt()))
      name = "vrsqrt";
    else if (op->op.same_as(tl::simd_vabs()))
      name = "vabs";
    else if (op->op.same_as(tl::simd_vrelu()))
      name = "vrelu";
    else if (op->op.same_as(tl::simd_vrec()))
      name = "vrec";
    else
      name = "vneg";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os);                            // mask
    os << ", " << Downcast<StringImm>(op->args[2])->value; // mode
    os << ")";
  }
  // --- SimdVF cross-lane reduce with vcgadd addition ---
  else if (op->op.same_as(tl::simd_vcmax()) ||
           op->op.same_as(tl::simd_vcadd()) ||
           op->op.same_as(tl::simd_vcgadd())) {
    ICHECK_EQ(op->args.size(), 3);
    const char *name =
        op->op.same_as(tl::simd_vcmax())
            ? "vcmax"
            : (op->op.same_as(tl::simd_vcgadd()) ? "vcgadd" : "vcadd");
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os);                            // mask
    os << ", " << Downcast<StringImm>(op->args[2])->value; // mode
    os << ")";
  }
  // --- SimdVF register permutation: (src0, src1) → pair ---
  else if (op->op.same_as(tl::simd_vintlv()) ||
           op->op.same_as(tl::simd_vdintlv())) {
    ICHECK_EQ(op->args.size(), 2);
    const char *name = op->op.same_as(tl::simd_vintlv()) ? "vintlv" : "vdintlv";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ")";
  }
  // --- SimdVF pair element access: pair_get(pair, index) → pair.vN ---
  else if (op->op.same_as(tl::simd_pair_get())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintExpr(op->args[0], os); // pair
    int idx = (int)Downcast<IntImm>(op->args[1])->value;
    os << (idx == 0 ? ".v0" : ".v1");
  }
  // --- SimdVF register pack: (src, part) ---
  else if (op->op.same_as(tl::simd_vpack())) {
    ICHECK_EQ(op->args.size(), 2);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vpack<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", " << Downcast<StringImm>(op->args[1])->value << ")";
  }
  // --- SimdVF vector broadcast vdupv: (src, mask, pos, mode) ---
  else if (op->op.same_as(tl::simd_vdupv())) {
    ICHECK_EQ(op->args.size(), 4);
    os << "simd_inst::vdupv(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os);                                   // mask
    os << ", " << Downcast<StringImm>(op->args[2])->value;        // pos
    os << ", " << Downcast<StringImm>(op->args[3])->value << ")"; // mode
  }
  // --- SimdVF gather 32B blocks: (base, index [, mask]) ---
  else if (op->op.same_as(tl::simd_vgatherb())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vgatherb<";
    PrintType(elem, os);
    os << ">((" << "__ubuf__" << " ";
    PrintType(elem, os);
    os << "*)";
    PrintExpr(op->args[0], os); // base
    os << ", ";
    PrintExpr(op->args[1], os); // index
    if (op->args.size() == 3) {
      os << ", ";
      PrintExpr(op->args[2], os); // mask
    }
    os << ")";
  }
  // --- SimdVF load: (addr, dist [, offset]) ---
  else if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() >= 2 && op->args.size() <= 3);
    DataType elem = op->dtype.element_of();
    std::string dist = Downcast<StringImm>(op->args[1])->value;
    bool has_off = (op->args.size() == 3);
    os << "simd_inst::vld<";
    PrintType(elem, os);
    os << ">((" << "__ubuf__" << " ";
    PrintType(elem, os);
    os << "*)";
    PrintExpr(op->args[0], os); // addr
    os << ", ";
    if (has_off) {
      os << "vag_b" << CCEElemWidthBits(elem) << "(";
      PrintExpr(op->args[2], os);
      os << ")";
    } else {
      os << dist;
    }
    os << ")";
  }
  // --- SimdVF store: (addr, src, mask, dist [, offset]) ---
  else if (op->op.same_as(tl::simd_vsts())) {
    ICHECK(op->args.size() >= 3 && op->args.size() <= 5);
    DataType dtype = op->args[1].dtype().element_of();
    int n = op->args.size();
    std::string dist = "NORM_B32";
    int off_idx = -1;
    if (n >= 4 && op->args[3].as<StringImmNode>()) {
      dist = Downcast<StringImm>(op->args[3])->value;
      if (n == 5)
        off_idx = 4;
    } else if (n == 4 && !op->args[3].as<StringImmNode>()) {
      off_idx = 3;
    }
    this->PrintIndent();
    this->stream << "simd_inst::vsts(";
    PrintExpr(op->args[1], this->stream); // src
    this->stream << ", (__ubuf__ " << CCEUBufType(dtype) << "*)";
    PrintExpr(op->args[0], this->stream); // addr
    if (off_idx >= 0) {
      this->stream << ", ";
      PrintExpr(op->args[off_idx], this->stream);
    } else {
      this->stream << ", 0";
    }
    this->stream << ", " << dist << ", ";
    PrintExpr(op->args[2], this->stream); // mask
    if (op->args.size() >= 5 && op->args[4].as<StringImmNode>())
      this->stream << ", " << Downcast<StringImm>(op->args[4])->value;
    this->stream << ");\n";
  }

  // --- SimdVF scatter-store blocks: (src, base, stride, mask) ---
  else if (op->op.same_as(tl::simd_vsstb())) {
    ICHECK_EQ(op->args.size(), 4);
    DataType dtype = op->args[0].dtype().element_of();
    this->PrintIndent();
    this->stream << "simd_inst::vsstb(";
    PrintExpr(op->args[0], this->stream); // src
    this->stream << ", (__ubuf__ " << CCEUBufType(dtype) << "*)";
    PrintExpr(op->args[1], this->stream); // base
    this->stream << ", ";
    PrintExpr(op->args[2], this->stream); // stride
    this->stream << ", ";
    PrintExpr(op->args[3], this->stream); // mask
    this->stream << ");\n";
  }
  // --- SimdVF scalar-vector ops: (src, scalar, mask, mode) ---
  else if (op->op.same_as(tl::simd_vadds()) ||
           op->op.same_as(tl::simd_vmuls())) {
    ICHECK_EQ(op->args.size(), 4);
    const char *name = op->op.same_as(tl::simd_vadds()) ? "vadds" : "vmuls";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os); // scalar
    os << ", ";
    PrintExpr(op->args[2], os);                                   // mask
    os << ", " << Downcast<StringImm>(op->args[3])->value << ")"; // mode
  }
  // --- SimdVF dup: scalar broadcast ---
  else if (op->op.same_as(tl::simd_vdup())) {
    ICHECK_EQ(op->args.size(), 3);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vdup<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // scalar
    os << ", ";
    PrintExpr(op->args[1], os);                                   // mask
    os << ", " << Downcast<StringImm>(op->args[2])->value << ")"; // mode
  }
  // --- SimdVF cast: (src, mask, ...extra) ---
  else if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4 && op->args.size() <= 6);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vcvt<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os); // mask
    for (int i = 2; i < static_cast<int>(op->args.size()); ++i) {
      os << ", " << Downcast<StringImm>(op->args[i])->value;
    }
    os << ")";
  }
  // --- SimdVF vexpdif: (src0, src1, mask, part) ---
  else if (op->op.same_as(tl::simd_vexpdif())) {
    ICHECK_EQ(op->args.size(), 4);
    int part = (int)Downcast<IntImm>(op->args[3])->value;
    os << "simd_inst::vexpdif(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ", ";
    PrintExpr(op->args[2], os); // mask
    os << ", " << (part == 0 ? "PART_EVEN" : "PART_ODD") << ")";
  }
  // --- SimdVF sel: (src0, src1, mask, mode) ---
  else if (op->op.same_as(tl::simd_vsel())) {
    ICHECK_EQ(op->args.size(), 4);
    os << "simd_inst::vsel(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ", ";
    PrintExpr(op->args[2], os);                                   // mask
    os << ", " << Downcast<StringImm>(op->args[3])->value << ")"; // mode
  } else if (op->op.same_as(tl::simd_mem_bar())) {
    std::string mem_type_str = Downcast<StringImm>(op->args[0])->value;
    this->PrintIndent();
    this->stream << "simd_inst::mem_bar(" << mem_type_str << ");\n";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitExpr_(const BufferLoadNode *op,
                                       std::ostream &os) {
  DataType value_dtype = op->dtype;
  const VarNode *buf_var = op->buffer->data.get();

  // For Ascend: scalar FP8 loads from shared/UBUF memory need conversion
  // through raw bytes, because the scalar FP8 type (tl::float_e4m3_t) is a
  // struct whose operator= does not support __ubuf__ to local assignment.
  if (value_dtype.is_scalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        PrimExpr index = op->indices[0];
        std::string buf_name = GetVarID(buf_var);

        // tl::float_e4m3_t::from_bits(*(__ubuf__ uint8_t*)(buf + off))
        os << GetAscendFP8ScalarValueType(value_dtype) << "::from_bits("
           << "*(";
        PrintStorageScope(scope, os);
        os << "uint8_t*)(" << buf_name;
        if (!is_zero(index)) {
          os << " + ";
          PrintExpr(index, os);
        }
        os << "))";
        return;
      }
    }
  }

  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangAscend::VisitStmt_(const BufferStoreNode *op) {
  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  const VarNode *buf_var = op->buffer->data.get();
  PrimExpr index_expr = op->indices[0];

  // For Ascend: scalar FP8 stores to shared/UBUF memory need to write the raw
  // .data byte, because tl::float_e4m3_t::operator= does not accept a local
  // struct on a __ubuf__ lvalue target.
  if (value_dtype.is_scalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        std::string buf_name = GetVarID(buf_var);
        std::string value = PrintExpr(op->value);

        // ((__ubuf__ uint8_t*)buf)[off] = value.data;
        PrintIndent();
        stream << "((";
        PrintStorageScope(scope, stream);
        stream << "uint8_t*)" << buf_name << ")[" << PrintExpr(index_expr)
               << "] = " << value << ".data;\n";
        return;
      }
    }
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const ForNode *op) {
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const AttrStmtNode *op) {
  // Ascend text-only backend currently does not lower thread binding attrs
  // into executable launch semantics. Keep generating the enclosed body
  // instead of failing in the generic C backend.
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
    LOG(FATAL) << "Static shared memory (scope='shared') is not supported on "
                  "Ascend. "
               << "Use dynamic shared memory (scope='shared.dyn') with "
                  "T.alloc_shared.";
  }

  // SimdVF: emit vector_<T> for fragment buffers
  if (inside_simdvf_body_ && scope == "local.fragment") {
    std::string vid = AllocVarID(op->buffer_var.get());
    this->PrintIndent();
    stream << "vector_";
    PrintCCEVectorSuffix(op->dtype, stream);
    stream << " " << vid << ";\n";
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

  if (scope == "local.var") {
    ICHECK_EQ(op->ConstantAllocationSize(), 1U) << "local.var should be scalar";
    std::string vid = AllocVarID(op->buffer_var.get());
    PrimExpr init = tir::make_const(op->dtype, 0);
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
        user_init = tir::Cast(op->dtype, user_init);
      }
      init = user_init;
    }
    PrintIndent();
    PrintType(op->dtype, stream);
    stream << " " << vid << " = ";
    PrintExpr(init, stream);
    stream << ";\n";
    alloc_storage_scope_[op->buffer_var.get()] = scope;
    RegisterHandleType(op->buffer_var.get(), op->dtype);
    this->PrintStmt(op->body);
    return;
  }

  // For other scopes, use parent class implementation
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const tir::LetStmtNode *op) {
  // Inside SimdVF body: special handling for simd_pset.
  if (inside_simdvf_body_) {
    std::string vid = AllocVarID(op->var.get());
    if (const auto *call = op->value.as<CallNode>()) {
      // simd_pset(elem_width_bits, dist) → vector_bool vid = pset_bXX(dist);
      if (call->op.same_as(tl::simd_pset())) {
        int elem_bits = (int)Downcast<IntImm>(call->args[0])->value;
        std::string dist =
            (call->args.size() >= 2 && call->args[1].as<StringImmNode>())
                ? Downcast<StringImm>(call->args[1])->value
                : "PAT_ALL";
        PrintIndent();
        stream << "vector_bool " << vid << " = pset_b" << elem_bits << "("
               << dist << ");\n";
        PrintStmt(op->body);
        return;
      }
      // pair_get(pair, idx) → inline as pair.vN, no auto decl
      if (call->op.same_as(tl::simd_pair_get())) {
        int idx = (int)Downcast<IntImm>(call->args[1])->value;
        std::ostringstream tmp;
        PrintExpr(call->args[0], tmp);
        tmp << (idx == 0 ? ".v0" : ".v1");
        var_idmap_[op->var.get()] = tmp.str();
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
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.is_scalar()) {
    if (from_ty.is_float() && from_ty.bits() == 32 &&
        tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (from_ty.is_float16() && tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (from_ty.is_bfloat16() && tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(__bfloat162float(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float() &&
        target_ty.bits() == 32) {
      os << "static_cast<float>(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float16()) {
      os << "static_cast<half>(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_bfloat16()) {
      os << "__float2bfloat16_rn(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    return CodeGenC::VisitExpr_(op, os);
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  std::string src = SSAGetID(PrintExpr(op->value), from_ty);

  int lanes = from_ty.lanes();

  auto PrintVectorizedCast =
      [&](const std::string &cast_func, const std::string &src_type,
          const std::string &dst_type, const std::string &extra_args = "",
          bool src_needs_reinterpret = false,
          bool dst_needs_reinterpret = false) {
        int num_chunks = lanes / 2;
        std::string src_cast = src_needs_reinterpret
                                   ? "reinterpret_cast<" + src_type + "*>"
                                   : "(" + src_type + "*)";
        std::string dst_cast = dst_needs_reinterpret
                                   ? "reinterpret_cast<" + dst_type + "*>"
                                   : "(" + dst_type + "*)";

        for (int i = 0; i < num_chunks; i++) {
          PrintIndent();
          stream << "(" << dst_cast << "(&" << sret << "))[" << i
                 << "] = " << cast_func << "((" << src_cast << "(&" << src
                 << "))[" << i << "]" << extra_args << ");\n";
        }
        os << sret;
      };

  // Handle conversion from float16 to float32.
  if (from_ty.is_float16() && target_ty.is_float() && target_ty.bits() == 32) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__half22float2", "half2", "float2");
      return;
    }
  }

  // Handle conversion from float32 to float16.
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_float16()) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22half2_rn", "float2", "half2");
      return;
    }
  }

  // Handle conversion from bfloat16 to float32
  if (from_ty.is_bfloat16() && from_ty.lanes() == 2 && target_ty.is_float() &&
      target_ty.bits() == 32 && target_ty.lanes() == 2) {
    os << "__bfloat1622float2(";
    PrintExpr(op->value, os);
    os << ")";
    return;
  }

  // Handle conversion from float32 to bfloat16
  if (from_ty.is_float() && from_ty.bits() == 32 && from_ty.lanes() == 2 &&
      target_ty.is_bfloat16() && target_ty.lanes() == 2) {
    os << "__float22bfloat162_rn(";
    PrintExpr(op->value, os);
    os << ")";
    return;
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.is_float() && from_ty.bits() == 32 &&
      tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string type_suffix = target_type_is_e4m3 ? "__ASC_E4M3" : "__ASC_E5M2";

    // Use __asc_cvt_float2_to_fp8x2 for vectorized conversion (float2 -> fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      std::string extra_args = std::string(", __ASC_SATFINITE, ") + type_suffix;
      PrintVectorizedCast("__asc_cvt_float2_to_fp8x2", "float2", "uint16_t",
                          extra_args, false, true);
      return;
    }
  }

  // Handle conversion from float16 to float8 (E4M3/E5M2)
  if (from_ty.is_float16() && tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_half2_to_fp8_e4m3x2"
                               : "tl::detail::cast_half2_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "half2", "uint16_t", "", false, true);
      return;
    }
  }

  // Handle conversion from bfloat16 to float8 (E4M3/E5M2)
  if (from_ty.is_bfloat16() && tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_bfloat162_to_fp8_e4m3x2"
                               : "tl::detail::cast_bfloat162_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "bfloat16x2_t", "uint16_t", "", true, true);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float32
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float() &&
      target_ty.bits() == 32) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func =
        from_type_is_e4m3 ? "__e4m3x22float2" : "__e5m2x22float2";

    // Use Ascend x2 intrinsics for vectorized conversion (fp8x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "float2", "", true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float16
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float16()) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func = from_type_is_e4m3
                               ? "tl::detail::cast_fp8_e4m3x2_to_half2"
                               : "tl::detail::cast_fp8_e5m2x2_to_half2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "half2", "", true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to bfloat16
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_bfloat16()) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func = from_type_is_e4m3
                               ? "tl::detail::cast_fp8_e4m3x2_to_bfloat162"
                               : "tl::detail::cast_fp8_e5m2x2_to_bfloat162";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "bfloat16x2_t", "", true,
                          true);
      return;
    }
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
  ICHECK(i >= 0 && i < t.lanes() && i < 64 / t.bits())
      << "Ascend " << t << " vector lanes max " << (64 / t.bits())
      << ": i=" << i << " t.bits()=" << t.bits() << " t.lanes()=" << t.lanes();
  if (t.is_float16()) {
    os << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]." << access[i % 2];
  } else if (t.is_bfloat16()) {
    os << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
       << access[i % 2];
  } else if (t.lanes() <= 4) {
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
  ICHECK(i >= 0 && i < t.lanes() && i < 64 / t.bits())
      << "Ascend " << t << " vector lanes max " << (64 / t.bits())
      << ": i=" << i << " t.bits()=" << t.bits() << " t.lanes()=" << t.lanes();
  if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.is_bfloat16()) {
    stream << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.lanes() <= 4) {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  } else {
    stream << "((" << vec << ")." << access[i / 2] << ") = " << value << ";\n";
  }
}

std::string CodeGenTileLangAscend::GetBufferRef(DataType t,
                                                const BufferNode *buffer,
                                                PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope == "local.var") {
    return GetVarID(buffer_var);
  }
  return CodeGenC::GetBufferRef(t, buffer, index);
}

void CodeGenTileLangAscend::PrintVecBinaryOp(const std::string &op, DataType t,
                                             PrimExpr lhs, PrimExpr rhs,
                                             std::ostream &os) {
  // Only scalarize float4 division; other vector binary ops fall through
  // to the base CodeGenC implementation.
  // TODO add support for fp16x4 since it becomes uint2
  if ((op != "/" || t.lanes() <= 1) && !(t.is_float16() || t.is_bfloat16())) {
    CodeGenC::PrintVecBinaryOp(op, t, lhs, rhs, os);
    return;
  }

  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangAscend::PrintCallExtern(Type ret_type,
                                            ffi::String global_symbol,
                                            const ffi::Array<PrimExpr> &args,
                                            bool skip_first_arg,
                                            std::ostream &os) {
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
    // Scalarize: for each lane, extract scalar args, call the scalar
    // function, store result into the result vector lane.
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
