#include "codegen_ascend.h"

#include "../op/builtin.h"
#include <string>

#include "../../3rdparty/tvm/src/tir/transforms/ir_utils.h"
#include "../transform/common/attr.h"
#include <tvm/tir/stmt_functor.h>

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
        os << "half_t";
      } else {
        fail = true; // half vector not yet supported
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
  bool has_cube = false, has_vector = false, has_simtvf = false;
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
    }
  });

  if (has_cube && has_vector) {
    kernel_mode_ = AscendKernelMode::kMix;
    mix_aiv_count_ = vector_count;
  } else if (has_cube || has_vector) {
    LOG(FATAL) << "T.Cube() and T.Vector() must be used together for Mix mode";
  } else if (has_simtvf) {
    kernel_mode_ = AscendKernelMode::kVector;
  } else {
    kernel_mode_ = AscendKernelMode::kCube;
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
  std::ostringstream saved_main_stream;
  stream.swap(saved_main_stream);
  bool old_inside_simtvf_body = inside_simtvf_body_;
  inside_simtvf_body_ = true;
  CodeGenC::VisitStmt(op->body);
  inside_simtvf_body_ = old_inside_simtvf_body;
  std::string helper_body = stream.str();
  stream.str("");
  stream.clear();
  stream.swap(saved_main_stream);

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
    ICHECK_EQ(op->args.size(), 9)
        << "tl.ascend_copy_gm_to_cbuf expects exactly 9 arguments";
    this->PrintIndent();
    this->stream << "bisheng::cce::copy_gm_to_cbuf_multi_nd2nz(";
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
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitStmt_(const ForNode *op) {
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
          !inside_simtvf_body_) {
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

} // namespace codegen
} // namespace tvm
