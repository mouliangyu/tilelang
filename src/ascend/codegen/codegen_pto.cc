/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"

#include "backend/common/codegen/codegen_utils.h"
#include "op/builtin.h"
#include "support/check.h"
#include "tvm/ir/repr.h"

#include <algorithm>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using namespace tirx;

namespace {

std::string PtoTypeName(DataType t) {
  ICHECK(t.is_scalar()) << "PTO scalar type expected, got " << t;
  if (t.is_float()) {
    if (t.bits() == 32)
      return "pto.f32";
    if (t.bits() == 16)
      return "pto.f16";
  } else if (t.is_float8_e4m3fn()) {
    return "pto.f8e4m3";
  } else if (t.is_bfloat16()) {
    return "pto.bf16";
  } else if (t.is_uint()) {
    if (t.bits() == 64)
      return "pto.ui64";
    if (t.bits() == 32)
      return "pto.ui32";
    if (t.bits() == 16)
      return "pto.ui16";
    if (t.bits() == 8)
      return "pto.ui8";
  } else if (t.is_int()) {
    if (t.bits() == 64)
      return "pto.i64";
    if (t.bits() == 32)
      return "pto.i32";
    if (t.bits() == 16)
      return "pto.i16";
    if (t.bits() == 8)
      return "pto.i8";
    if (t.bits() == 1)
      return "pto.i1";
  }
  LOG(FATAL) << "Unsupported PTO type: " << t;
  return "";
}

std::string PtoSignedIntegerTypeName(DataType t) {
  ICHECK(t.is_scalar() && t.is_int())
      << "PTO signed integer scalar type expected, got " << t;
  switch (t.bits()) {
  case 8:
    return "pto.si8";
  case 16:
    return "pto.si16";
  case 32:
    return "pto.si32";
  case 64:
    return "pto.si64";
  default:
    LOG(FATAL) << "Unsupported PTO signed integer type: " << t;
  }
  return "";
}

std::string StripPipePrefix(const std::string &name) {
  if (name.rfind("PIPE_", 0) == 0) {
    return name.substr(5);
  }
  return name;
}

DataType ParsePTODtype(const std::string &dtype_name) {
  if (dtype_name == "float" || dtype_name == "float32")
    return DataType::Float(32);
  if (dtype_name == "half" || dtype_name == "float16")
    return DataType::Float(16);
  if (dtype_name == "bfloat16" || dtype_name == "bfloat16_t")
    return DataType::BFloat(16);
  if (dtype_name == "float8_e4m3fn" || dtype_name == "float8_e4m3_t")
    return DataType(DataType::kFloat8_e4m3fn, 8, 1);
  if (dtype_name == "int64")
    return DataType::Int(64);
  if (dtype_name == "int32")
    return DataType::Int(32);
  if (dtype_name == "int16")
    return DataType::Int(16);
  if (dtype_name == "uint64")
    return DataType::UInt(64);
  if (dtype_name == "uint32")
    return DataType::UInt(32);
  if (dtype_name == "uint16")
    return DataType::UInt(16);
  if (dtype_name == "int8")
    return DataType::Int(8);
  if (dtype_name == "uint8")
    return DataType::UInt(8);
  LOG(FATAL) << "Unsupported PTO dtype string: " << dtype_name;
  return DataType::Void();
}

bool TryGetConstInt(const PrimExpr &expr, int64_t *value) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    *value = imm->value;
    return true;
  }
  return false;
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.rfind(prefix, 0) == 0;
}

bool IsFloat32(DataType t) {
  return t.is_float() && t.bits() == 32 && t.lanes() == 1;
}

void CheckContiguousRampStride(const PrimExpr &index, const char *access_kind) {
  if (const auto *ramp = index.as<RampNode>()) {
    PrimExpr stride_expr = arith::Analyzer().Simplify(ramp->stride);
    int64_t stride = 0;
    ICHECK(TryGetConstInt(stride_expr, &stride) && stride == 1)
        << "PTO SIMT vector " << access_kind
        << " emits a contiguous vector access, so Ramp stride must be 1, got "
        << ramp->stride;
  }
}

void CheckConstZero(const PrimExpr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value) && value == 0)
      << "PTO codegen currently only supports " << name
      << " == 0 for tl.ascend_copy_gm_to_ubuf, got " << expr;
}

std::string PtoStoreL2CacheToken(int64_t value) {
  switch (value) {
  case 0:
    return "nmfv";
  case 1:
    return "nmlv";
  case 2:
    return "nmprs";
  case 3:
    return "nmred";
  case 4:
    return "naci";
  case 5:
    return "napw";
  case 6:
    return "napi";
  case 7:
    return "nared";
  case 8:
    return "wbhfv";
  case 9:
    return "wbhlv";
  case 10:
    return "wbhprs";
  case 11:
    return "wbhred";
  case 12:
    return "wtsfv";
  case 13:
    return "wtslv";
  case 14:
    return "wtsprs";
  case 15:
    return "wtsred";
  default:
    LOG(FATAL) << "Unsupported PTO store l2 cache control value: " << value;
    return "nmfv";
  }
}

bool IsOpName(const ObjectRef &op, const std::string &name) {
  if (auto opt_call_op = op.as<Op>()) {
    return opt_call_op.value()->name == name;
  }
  return false;
}

int64_t ConstShapeDim(const PrimExpr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value))
      << name << " must be static, got " << expr;
  return value;
}

int64_t ConstArgDim(const CallNode *call, size_t index, const char *name) {
  ICHECK_GT(call->args.size(), index) << name << " argument is missing";
  return ConstShapeDim(call->args[index], name);
}

void CheckPTOLocalVarBuffer(const BufferNode *buffer) {
  ICHECK_EQ(buffer->shape.size(), 1U)
      << "PTO local.var only supports scalar alloc_var buffers, got rank "
      << buffer->shape.size();
  int64_t extent = 0;
  ICHECK(TryGetConstInt(buffer->shape[0], &extent) && extent == 1)
      << "PTO local.var only supports scalar alloc_var buffers with shape "
         "(1,), got "
      << buffer->shape[0];

  DataType dtype = buffer->dtype;
  if (dtype.lanes() > 1) {
    // VMI registers are logical vectors; integer, floating-point and
    // low-precision element types are all valid here.
    return;
  }
  ICHECK(dtype.is_int() || dtype.is_uint())
      << "PTO local.var only supports integer scalar values, got " << dtype;
}

void CheckPTOKernel(const PrimFunc &func) {
  bool has_cube_block = false;
  bool has_gemm_l1 = false;
  bool has_unsupported_cube_op = false;

  PostOrderVisit(func->body, [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      if (block->name_hint == "CUBE") {
        has_cube_block = true;
      }
    }

    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::ascend_gemm_l1())) {
        has_gemm_l1 = true;
      } else if (call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
                 call->op.same_as(tl::ascend_mad()) ||
                 call->op.same_as(tl::ascend_nd2nz_scatter()) ||
                 call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_unsupported_cube_op = true;
      }
    }
  });

  ICHECK(!has_cube_block || has_gemm_l1)
      << "PTO codegen only supports cube kernels through tl.ascend_gemm_l1";
  ICHECK(!has_unsupported_cube_op)
      << "PTO codegen does not support this Ascend cube op yet.";
}

std::string AccStoreUnitFlagArg(int64_t unit_flag_ctrl) {
  if (unit_flag_ctrl == 0)
    return "None";
  if (unit_flag_ctrl == 2)
    return "pto.AccStoreUnitFlagCtrl.CHECK_ONLY";
  if (unit_flag_ctrl == 3)
    return "pto.AccStoreUnitFlagCtrl.CHECK_AND_CLEAR";
  LOG(FATAL) << "PTO GEMM unsupported L0C store unit_flag_ctrl="
             << unit_flag_ctrl;
  return "None";
}

bool IsSupportedPTOGemmInputDtype(DataType dtype) {
  return dtype.is_bfloat16() || dtype.is_float8_e4m3fn();
}

int64_t PTOGemmInputC0(DataType dtype) {
  ICHECK(IsSupportedPTOGemmInputDtype(dtype))
      << "PTO GEMM L1 helper currently only supports bfloat16 and "
         "float8_e4m3fn inputs, got "
      << dtype;
  int64_t elem_bytes = dtype.bytes();
  ICHECK_GT(elem_bytes, 0) << "Invalid PTO GEMM input dtype size: " << dtype;
  ICHECK_EQ(32 % elem_bytes, 0)
      << "PTO GEMM input dtype byte size must divide 32, got " << dtype;
  return 32 / elem_bytes;
}

DataType GetAnnotatedPointerDtype(const PrimExpr &expr,
                                  DataType fallback_dtype) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return fallback_dtype;
  }

  if (call->op.same_as(builtin::tvm_access_ptr()) && !call->args.empty()) {
    if (const auto *type_call = call->args[0].as<CallNode>()) {
      if (!type_call->args.empty()) {
        if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
          return ParsePTODtype(dtype_name->value);
        }
      }
    }
    return fallback_dtype;
  }

  if (call->op.same_as(builtin::address_of()) && call->args.size() == 1U) {
    if (const auto *load = call->args[0].as<BufferLoadNode>()) {
      return load->buffer->dtype;
    }
  }

  return fallback_dtype;
}

bool GetAddressOfIndex(const PrimExpr &expr, PrimExpr *index,
                       const VarNode **buffer_var) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  if (call->op.same_as(builtin::address_of())) {
    if (call->args.size() != 1U) {
      return false;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return false;
    }
    *index = load->indices[0];
    *buffer_var = load->buffer->data.get();
    return true;
  }
  if (call->op.same_as(builtin::tvm_access_ptr())) {
    if (call->args.size() < 3U) {
      return false;
    }
    const auto *var = call->args[1].as<VarNode>();
    if (var == nullptr) {
      return false;
    }
    *index = call->args[2];
    *buffer_var = var;
    return true;
  }
  return false;
}

} // namespace

void CodeGenTileLangPTO::AddFunction(const GlobalVar &gvar,
                                     const PrimFunc &func) {
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  CheckPTOKernel(func);
  fragment_info_.clear();
  local_var_buffers_.clear();
  gemm_emit_ctx_ = PTOGemmEmitContext();
  current_function_has_gemm_ = HasAscendGemmL1(func);
  has_gemm_l1_ = has_gemm_l1_ || current_function_has_gemm_;

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(current_function_name_, func, stream);
  stream << ":\n";
  int func_scope = BeginScope();
  if (current_function_has_gemm_) {
    const CallNode *first_gemm = nullptr;
    tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (first_gemm == nullptr && call->op.same_as(tl::ascend_gemm_l1())) {
          first_gemm = call;
        }
      }
    });
    ICHECK(first_gemm != nullptr);
    EnsurePTOGemmHelper(first_gemm);
  }
  PrintStmt_(func->body);
  EndScope(func_scope);
  stream << "\n";
}

std::string CodeGenTileLangPTO::Finish() {
  std::ostringstream code;
  code << "from ptodsl import pto, scalar\n";
  code << "from ptodsl._ops import _coerce_i64 as _tl_coerce_i64\n";
  code << "from ptodsl._surface_values import wrap_surface_value as "
          "_tl_wrap_surface_value\n";
  if (has_gemm_l1_) {
    code << "from tilelang.contrib.ptodsl.gemm import PTOGemmL1Template\n";
  }
  code << "\n";
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangPTO::PrintFuncDecorator_(std::ostream &os) { // NOLINT(*)
  os << "@pto.jit(name=\"" << current_function_name_ << "\", kernel_kind=\""
     << (current_function_has_gemm_ ? "cube" : "vector")
     << "\", target=\"a5\", mode=\"explicit\"";
  if (current_function_has_gemm_) {
    os << ", insert_sync=False";
  }
  os << ")\n";
}

void CodeGenTileLangPTO::PrintFunctionSignature_(
    const ffi::String &function_name, const PrimFunc &func,
    std::ostream &os) { // NOLINT(*)
  os << "def " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
    if (func->buffer_map.count(v)) {
      Buffer buffer = func->buffer_map[v];
      os << ": " << PtoPtrType(buffer->dtype, "gm");
    } else if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope =
            ptr->storage_scope.empty() ? "gm" : ptr->storage_scope;
        if (scope == "global") {
          scope = "gm";
        }
        os << ": " << PtoPtrType(prim->dtype, scope);
      } else {
        os << ": " << PtoScalarType(v->dtype);
      }
    } else {
      os << ": " << PtoScalarType(v->dtype);
    }
  }
  os << ")";

  for (const auto &param : func->params) {
    if (auto *ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType_(param.get(), prim->dtype);
      }
    }
  }
}

std::string CodeGenTileLangPTO::PtoScalarType(DataType t) const {
  return PtoTypeName(t);
}

std::string CodeGenTileLangPTO::PtoPtrType(DataType t,
                                           const std::string &space) const {
  std::ostringstream os;
  os << "pto.ptr(" << PtoTypeName(t) << ", \"" << space << "\")";
  return os.str();
}

std::pair<std::string, std::string>
CodeGenTileLangPTO::ParseHardEventPair(const std::string &hard_event) const {
  auto pos = hard_event.find('_');
  if (pos == std::string::npos) {
    return {hard_event, hard_event};
  }
  return {hard_event.substr(0, pos), hard_event.substr(pos + 1)};
}

bool CodeGenTileLangPTO::HasAscendGemmL1(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1());
    }
  });
  return found;
}

std::string CodeGenTileLangPTO::ResolveVarName(const Var &v) const {
  auto it = var_idmap_.find(v.get());
  if (it != var_idmap_.end()) {
    return it->second;
  }
  return v->name_hint;
}

std::string CodeGenTileLangPTO::ScopeOfBuffer(const BufferNode *buffer) const {
  std::string scope;
  auto it = alloc_storage_scope_.find(buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }
  return scope;
}

ffi::Array<Var>
CodeGenTileLangPTO::CollectVFCaptures(const SBlockNode *op) const {
  ffi::Array<Var> predefined;
  for (const Buffer &buf : op->alloc_buffers) {
    predefined.push_back(buf->data);
  }

  ffi::Array<Var> undefined = UndefinedVars(op->body, predefined);
  ffi::Array<Var> captures;
  std::unordered_set<const VarNode *> seen;
  for (const Var &var : undefined) {
    if (seen.count(var.get())) {
      continue;
    }
    // Kernel blockIdx.x is mapped directly to the PTODSL builtin in
    // VisitStmt_(AttrStmtNode), so SIMT helpers can reference it without a
    // scalar capture parameter.
    auto it = var_idmap_.find(var.get());
    if (it != var_idmap_.end() && it->second == "pto.get_block_idx()") {
      continue;
    }
    seen.insert(var.get());

    const DataType dtype = var->dtype;
    ICHECK(dtype.lanes() == 1 &&
           (dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
            dtype.is_handle() || dtype.is_bool()))
        << op->name_hint << " capture variable `" << var
        << "` has unsupported dtype `" << dtype
        << "`. Only scalar int/uint/float/bool/handle captures are supported.";
    captures.push_back(var);
  }
  return captures;
}

void CodeGenTileLangPTO::ExtractSimtThreadExtents(const SBlockNode *op,
                                                  int64_t *thread_x,
                                                  int64_t *thread_y,
                                                  int64_t *thread_z) const {
  *thread_x = 1;
  *thread_y = 1;
  *thread_z = 1;

  tirx::PostOrderVisit(op->body, [&](const ffi::ObjectRef &node) {
    const auto *attr = node.as<AttrStmtNode>();
    if (attr == nullptr || attr->attr_key != tirx::attr::thread_extent) {
      return;
    }
    const auto *iv = attr->node.as<IterVarNode>();
    if (!iv) {
      return;
    }
    int64_t value = 1;
    if (TryGetConstInt(attr->value, &value)) {
      if (iv->thread_tag == "threadIdx.x") {
        *thread_x = value;
      } else if (iv->thread_tag == "threadIdx.y") {
        *thread_y = value;
      } else if (iv->thread_tag == "threadIdx.z") {
        *thread_z = value;
      }
    }
  });
}

void CodeGenTileLangPTO::EmitSimtVFFunction(const SBlockNode *op,
                                            const ffi::Array<Var> &captures,
                                            const std::string &helper_name,
                                            int64_t thread_x, int64_t thread_y,
                                            int64_t thread_z) {
  std::unordered_map<const VarNode *, std::string> capture_scope;
  for (const Var &v : captures) {
    if (v->type_annotation.as<PointerTypeNode>()) {
      alloc_storage_scope_[v.get()] = GetPtrStorageScope(v);
    }
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      capture_scope[v.get()] = it->second;
    }
  }

  std::unordered_map<const VarNode *, std::string> saved_var_idmap;
  saved_var_idmap.swap(var_idmap_);
  NameSupply saved_name_supply = name_supply_;
  name_supply_ = NameSupply();
  ReserveKeywordsAsUnique_();
  for (const auto &kv : saved_var_idmap) {
    var_idmap_[kv.first] = kv.second;
  }
  for (const Var &v : captures) {
    if (!var_idmap_.count(v.get())) {
      AllocVarID(v.get());
    }
    name_supply_->ReserveName(ResolveVarName(v), false);
  }

  std::vector<std::string> capture_names;
  capture_names.reserve(captures.size());
  for (const Var &v : captures) {
    capture_names.push_back(ResolveVarName(v));
  }

  CodeGenTileLangPTO body_codegen;
  body_codegen.var_idmap_ = var_idmap_;
  body_codegen.name_supply_ = name_supply_;
  body_codegen.alloc_storage_scope_ = alloc_storage_scope_;
  body_codegen.inside_simtvf_body_ = true;
  for (const Var &v : captures) {
    if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        body_codegen.RegisterHandleType_(v.get(), prim->dtype);
      }
    }
  }
  int helper_scope = body_codegen.BeginScope();
  for (const Buffer &buf : op->alloc_buffers) {
    body_codegen.EmitPtoBufferAllocation(buf);
  }
  body_codegen.VisitStmt(op->body);
  body_codegen.EndScope(helper_scope);
  std::string helper_body = body_codegen.stream.str();

  var_idmap_.swap(saved_var_idmap);
  name_supply_ = saved_name_supply;

  const int64_t total_threads = thread_x * thread_y * thread_z;
  decl_stream << "@pto.simt(name=\"" << helper_name << "\"";
  if (total_threads > 0) {
    decl_stream << ", max_threads=" << total_threads;
  }
  decl_stream << ")\n";
  decl_stream << "def " << helper_name << "(";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0) {
      decl_stream << ", ";
    }
    const Var &v = captures[i];
    std::string vname = capture_names[i];
    decl_stream << vname << ": ";
    if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope =
            ptr->storage_scope.empty() ? "gm" : ptr->storage_scope;
        auto scope_it = capture_scope.find(v.get());
        if (scope_it != capture_scope.end() &&
            (scope_it->second == "shared" ||
             scope_it->second == "shared.dyn")) {
          decl_stream << PtoPtrType(prim->dtype, "ub");
          RegisterHandleType_(v.get(), prim->dtype);
          continue;
        }
        if (scope == "global") {
          scope = "gm";
        } else if (scope == "shared" || scope == "shared.dyn") {
          scope = "ub";
        }
        decl_stream << PtoPtrType(prim->dtype, scope);
        RegisterHandleType_(v.get(), prim->dtype);
      } else {
        decl_stream << PtoScalarType(v->dtype);
      }
    } else {
      decl_stream << PtoScalarType(v->dtype);
    }
  }
  decl_stream << "):\n";
  if (helper_body.empty()) {
    decl_stream << "  return\n";
  } else {
    decl_stream << helper_body;
  }
  decl_stream << "\n";
}

void CodeGenTileLangPTO::EmitSimtVFLaunch(const ffi::Array<Var> &captures,
                                          const std::string &helper_name,
                                          int64_t thread_x, int64_t thread_y,
                                          int64_t thread_z) {
  PrintIndent();
  stream << helper_name << "[" << thread_x << ", " << thread_y << ", "
         << thread_z << "](";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    const Var &v = captures[i];
    std::string arg = ResolveVarName(v);
    stream << arg;
  }
  stream << ")\n";
}

std::string CodeGenTileLangPTO::GetPtoPointerExpr(const VarNode *buffer_var,
                                                  DataType elem_dtype,
                                                  const PrimExpr &index) {
  std::string scope = "global";
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }

  std::string base = GetVarID(buffer_var);
  if (scope == "shared" || scope == "shared.dyn") {
    if (HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
        !HandleTypeMatch_(buffer_var, elem_dtype)) {
      base = "pto.castptr(" + base + ", " + PtoPtrType(elem_dtype, "ub") + ")";
    }
  }

  if (is_zero(index)) {
    return base;
  }

  std::string index_str;
  int64_t const_index = 0;
  if (TryGetConstInt(index, &const_index)) {
    index_str = std::to_string(const_index);
  } else {
    index_str = RemoveOutermostParentheses(PrintExpr_(index));
  }

  if (scope == "global" || scope.empty() || scope == "shared" ||
      scope == "shared.dyn") {
    return "pto.addptr(" + base + ", " + index_str + ")";
  }

  LOG(FATAL) << "Unsupported storage scope in PTO pointer emission: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::GetPtoPointerExpr(const BufferNode *buffer,
                                                  const PrimExpr &index) {
  return GetPtoPointerExpr(buffer->data.get(), buffer->dtype, index);
}

std::string CodeGenTileLangPTO::PtoScalarLoad(const BufferNode *buffer,
                                              const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  if (scope == "local.var") {
    return GetVarID(buffer->data.get());
  }

  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
  if (inside_simtvf_body_ && (scope == "local.fragment" || scope == "local")) {
    ICHECK(IsFloat32(buffer->dtype))
        << "PTO SIMT local scalar load currently supports float32 only, got "
        << buffer->dtype;
    return "scalar.load(" + GetVarID(buffer->data.get()) + ", " + index_str +
           ")";
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    const VarNode *buffer_var = buffer->data.get();
    std::string base = GetVarID(buffer_var);
    std::string pto_space =
        (scope == "shared" || scope == "shared.dyn") ? "ub" : "gm";
    if (scope == "shared" || scope == "shared.dyn") {
      const bool need_cast = HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
                             !HandleTypeMatch_(buffer_var, buffer->dtype);
      if (need_cast) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(buffer->dtype, pto_space) + ")";
      }
    }
    return "scalar.load(" + base + ", " + index_str + ")";
  }

  if (scope == "local.fragment" || scope == "local") {
    return GetVarID(buffer->data.get()) + "[" + index_str + "]";
  }

  LOG(FATAL) << "Unsupported PTO scalar load scope: " << scope;
  return "";
}

void CodeGenTileLangPTO::EmitPtoScalarStore(const BufferNode *buffer,
                                            const std::string &value,
                                            const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));

  if (scope == "local.var") {
    stream << GetVarID(buffer->data.get()) << " = " << value << "\n";
    return;
  }

  if (scope == "local.fragment" || scope == "local") {
    if (inside_simtvf_body_) {
      ICHECK(IsFloat32(buffer->dtype))
          << "PTO SIMT local scalar store currently supports float32 only, got "
          << buffer->dtype;
      stream << "scalar.store(" << value << ", " << GetVarID(buffer->data.get())
             << ", " << index_str << ")\n";
      return;
    }
    stream << GetVarID(buffer->data.get()) << "[" << index_str
           << "] = " << value << "\n";
    return;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    const VarNode *buffer_var = buffer->data.get();
    std::string base = GetVarID(buffer_var);
    std::string pto_space =
        (scope == "shared" || scope == "shared.dyn") ? "ub" : "gm";
    if (scope == "shared" || scope == "shared.dyn") {
      const bool need_cast = HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
                             !HandleTypeMatch_(buffer_var, buffer->dtype);
      if (need_cast) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(buffer->dtype, pto_space) + ")";
      }
    }
    stream << "scalar.store(" << value << ", " << base << ", " << index_str
           << ")\n";
    return;
  }

  LOG(FATAL) << "Unsupported PTO scalar store scope: " << scope;
}

void CodeGenTileLangPTO::EmitPtoBufferAllocation(const Buffer &buffer) {
  std::string scope = GetPtrStorageScope(buffer->data);
  alloc_storage_scope_[buffer->data.get()] = scope;

  if (scope == "shared" || scope == "shared.dyn") {
    PrintIndent();
    std::string vid = AllocVarID(buffer->data.get());
    auto alloc = AllocBuffer(buffer);
    auto opt_size = alloc.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO shared allocation currently requires constant allocation size";
    stream << vid << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PtoPtrType(buffer->dtype, "ub") << ")\n";
    RegisterHandleType_(buffer->data.get(), buffer->dtype);
    return;
  } else if (scope == "local.fragment" || scope == "local") {
    std::string vid = AllocVarID(buffer->data.get());
    auto alloc = AllocBuffer(buffer);
    auto opt_size = alloc.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO local.fragment currently requires constant allocation size";
    PrintIndent();
    if (inside_simtvf_body_) {
      ICHECK(IsFloat32(buffer->dtype))
          << "PTO SIMT local allocation currently supports float32 only, got "
          << buffer->dtype;
      stream << vid << " = pto.alloc_buffer((" << opt_size.value()
             << ",), pto.f32)\n";
    } else {
      stream << vid << " = [None] * " << opt_size.value() << "\n";
    }
  } else if (scope == "local.var") {
    PrintIndent();
    stream << AllocVarID(buffer->data.get()) << " = 0\n";
  }

  RegisterHandleType_(buffer->data.get(), buffer->dtype);
}

std::string CodeGenTileLangPTO::GetAddressOfExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1U);
  const auto *load = op->args[0].as<BufferLoadNode>();
  ICHECK(load) << "address_of expects BufferLoad";
  ICHECK_EQ(load->indices.size(), 1U)
      << "CodeGenTileLangPTO only supports flat memory";
  return GetPtoPointerExpr(load->buffer.get(), load->indices[0]);
}

std::string CodeGenTileLangPTO::GetAccessPtrExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U);
  auto buffer_var = Downcast<Var>(op->args[1]);
  DataType elem_dtype = DataType::Float(32);

  if (const auto *type_call = op->args[0].as<CallNode>()) {
    if (!type_call->args.empty()) {
      if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
        elem_dtype = ParsePTODtype(dtype_name->value);
      }
    }
  } else if (HandleTypeMatch_(buffer_var.get(), DataType::Float(32))) {
    elem_dtype = DataType::Float(32);
  }

  return GetPtoPointerExpr(buffer_var.get(), elem_dtype, op->args[2]);
}

std::string CodeGenTileLangPTO::GetAscendCopyGmUbExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 11U)
      << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";
  CheckConstZero(op->args[2], "sid");
  CheckConstZero(op->args[5], "leftPadding");
  CheckConstZero(op->args[6], "rightPadding");
  // TODO: Support dataSelect=true by finding the pad value and emitting
  // pto.mte_gm_ub(..., pad=(pad_value, leftPadding, rightPadding)).
  CheckConstZero(op->args[7], "dataSelect");

  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string l2_cache_ctl =
      RemoveOutermostParentheses(PrintExpr_(op->args[8]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[9]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  std::ostringstream os;
  os << "pto.mte_gm_ub(" << src << ", " << dst << ", " << l2_cache_ctl << ", "
     << burst_len << ", nburst=(" << burst_num << ", " << src_stride << ", "
     << dst_stride << "))";
  return os.str();
}

std::string CodeGenTileLangPTO::GetAscendCopyUbGmExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 8U);
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  int64_t l2_cache_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[5], &l2_cache_ctrl))
      << "PTO UB-to-GM copy expects constant l2_cache_ctrl";
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  std::ostringstream os;
  os << "pto.mte_ub_gm(" << src << ", " << dst << ", " << burst_len
     << ", nburst=(" << burst_num << ", " << src_stride << ", " << dst_stride
     << "), l2_cache=\"" << PtoStoreL2CacheToken(l2_cache_ctrl) << "\")";
  return os.str();
}

std::string CodeGenTileLangPTO::GetPtoLocalByteAddrExpr(
    const PrimExpr &index, DataType elem_dtype, const std::string &context) {
  int64_t const_index = 0;
  int64_t elem_bytes = elem_dtype.bytes();
  if (TryGetConstInt(index, &const_index)) {
    return "pto.const(" + std::to_string(const_index * elem_bytes) +
           ", dtype=pto.int64)";
  }

  std::string index_expr = RemoveOutermostParentheses(PrintExpr_(index));
  std::string coerced =
      "_tl_coerce_i64(" + index_expr + ", context=\"" + context + "\")";
  if (elem_bytes == 1) {
    return coerced;
  }
  return "scalar.muli(" + coerced + ", pto.const(" +
         std::to_string(elem_bytes) + ", dtype=pto.int64))";
}

std::string CodeGenTileLangPTO::GetPtoLocalPtrExpr(const PrimExpr &expr,
                                                   const std::string &space,
                                                   DataType fallback_dtype) {
  PrimExpr index;
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO local pointer expects address_of/tvm_access_ptr, got " << expr;

  DataType elem_dtype = fallback_dtype;
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && !call->args.empty()) {
      if (auto *type_call = call->args[0].as<CallNode>()) {
        if (!type_call->args.empty()) {
          if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
            elem_dtype = ParsePTODtype(dtype_name->value);
          }
        }
      }
    } else if (call->op.same_as(builtin::address_of())) {
      if (const auto *load = call->args[0].as<BufferLoadNode>()) {
        elem_dtype = load->buffer->dtype;
      }
    }
  }

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  ICHECK(scope == "shared" || scope == "shared.dyn" || scope == "shared.l1" ||
         scope == "shared.l1.dyn" || scope == "shared.l0c" ||
         scope == "local.fragment" || scope.empty())
      << "PTO local pointer expected shared/local.fragment storage, got "
      << scope;

  std::string byte_addr =
      GetPtoLocalByteAddrExpr(index, elem_dtype, "PTO local pointer offset");
  return "pto.castptr(" + byte_addr + ", " + PtoPtrType(elem_dtype, space) +
         ")";
}

std::string CodeGenTileLangPTO::GetPtoAccPtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "acc", dtype);
}

std::string CodeGenTileLangPTO::LocalVarID(const VarNode *var) {
  return GetVarID(var);
}

bool CodeGenTileLangPTO::IsLocalVarBuffer(const VarNode *var) const {
  return local_var_buffers_.count(var) != 0;
}

bool CodeGenTileLangPTO::IsVmiLocalRegisterBuffer(
    const BufferNode *buffer) const {
  return !inside_simtvf_body_ && ScopeOfBuffer(buffer) == "local" &&
         buffer->dtype.lanes() > 1;
}

void CodeGenTileLangPTO::CheckVmiLocalRegisterIndex(
    const BufferNode *buffer, const PrimExpr &index) const {
  bool is_constant = true;
  tirx::PostOrderVisit(index, [&](const ObjectRef &node) {
    if (node.as<VarNode>()) {
      is_constant = false;
    }
  });
  ICHECK(is_constant)
      << "PTO VMI local register buffer `" << buffer->name
      << "` requires a compile-time constant index. Register arrays cannot "
         "be accessed inside T.unroll(..., explicit=False) or other runtime "
         "loops; use T.unroll(..., explicit=True), got "
      << index;
}

void CodeGenTileLangPTO::EnsurePTOGemmHelper(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_gemm_l1 expects exactly 12 arguments";
  int64_t tile_m = ConstArgDim(op, 3, "tl.ascend_gemm_l1 M");
  int64_t tile_k = ConstArgDim(op, 4, "tl.ascend_gemm_l1 K");
  int64_t tile_n = ConstArgDim(op, 5, "tl.ascend_gemm_l1 N");
  int64_t base_k = ConstArgDim(op, 6, "tl.ascend_gemm_l1 tile_k_sub");
  int64_t trans_b = ConstArgDim(op, 7, "tl.ascend_gemm_l1 trans_b");
  ICHECK_EQ(trans_b, 1) << "PTO GEMM L1 helper currently requires trans_b=1";
  ICHECK_EQ(tile_k % base_k, 0);

  const auto *dtype_name = op->args[9].as<StringImmNode>();
  ICHECK(dtype_name) << "PTO GEMM L1 helper requires a constant input dtype "
                        "string at tl.ascend_gemm_l1 arg 9";
  DataType input_dtype = ParsePTODtype(dtype_name->value);
  ICHECK(IsSupportedPTOGemmInputDtype(input_dtype))
      << "PTO GEMM L1 helper currently only supports bfloat16 and "
         "float8_e4m3fn inputs, got "
      << input_dtype;

  DataType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  DataType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(a_dtype == input_dtype)
      << "PTO GEMM L1 A pointer dtype must match input dtype " << input_dtype
      << ", got " << a_dtype;
  ICHECK(b_dtype == input_dtype)
      << "PTO GEMM L1 B pointer dtype must match input dtype " << input_dtype
      << ", got " << b_dtype;

  DataType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(accum_dtype.is_float() && accum_dtype.bits() == 32)
      << "PTO GEMM L1 helper currently only supports float32 accum/output, got "
      << accum_dtype;

  if (gemm_emit_ctx_.initialized) {
    ICHECK_EQ(gemm_emit_ctx_.tile_m, tile_m);
    ICHECK_EQ(gemm_emit_ctx_.tile_n, tile_n);
    ICHECK_EQ(gemm_emit_ctx_.tile_k, tile_k);
    ICHECK_EQ(gemm_emit_ctx_.base_k, base_k);
    ICHECK(gemm_emit_ctx_.input_dtype == input_dtype)
        << "PTO GEMM L1 helper saw inconsistent input dtype: expected "
        << gemm_emit_ctx_.input_dtype << ", got " << input_dtype;
    ICHECK(gemm_emit_ctx_.accum_dtype == accum_dtype)
        << "PTO GEMM L1 helper saw inconsistent accum/output dtype: expected "
        << gemm_emit_ctx_.accum_dtype << ", got " << accum_dtype;
    return;
  }

  gemm_emit_ctx_.initialized = true;
  gemm_emit_ctx_.tile_m = tile_m;
  gemm_emit_ctx_.tile_n = tile_n;
  gemm_emit_ctx_.tile_k = tile_k;
  gemm_emit_ctx_.base_k = base_k;
  gemm_emit_ctx_.input_dtype = input_dtype;
  gemm_emit_ctx_.accum_dtype = accum_dtype;
  gemm_emit_ctx_.helper_name = "_tl_gemm_l1";
  gemm_emit_ctx_.a_l0_name = "a_l0_0";
  gemm_emit_ctx_.b_l0_name = "b_l0_0";

  int64_t input_c0 = PTOGemmInputC0(input_dtype);
  ICHECK_EQ(base_k % input_c0, 0)
      << "PTO GEMM base_k must be divisible by input C0=" << input_c0
      << " for dtype " << input_dtype;
  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / input_c0;
  int64_t a_l0_stage_elems = tile_m * base_k;
  int64_t b_l0_stage_elems = base_k * tile_n;

  PrintIndent();
  stream << "zero_addr = pto.const(0, dtype=pto.int64)\n";
  PrintIndent();
  stream << gemm_emit_ctx_.a_l0_name << " = pto.castptr(zero_addr, "
         << PtoPtrType(input_dtype, "left") << ")\n";
  PrintIndent();
  stream << gemm_emit_ctx_.b_l0_name << " = pto.castptr(zero_addr, "
         << PtoPtrType(input_dtype, "right") << ")\n";
  PrintIndent();
  stream << gemm_emit_ctx_.helper_name << " = PTOGemmL1Template(" << tile_m
         << ", " << tile_n << ", " << tile_k << ", " << base_k << ", "
         << sub_k_tiles << ", " << input_c0 << ", " << sub_k_c0_blocks << ", "
         << a_l0_stage_elems << ", " << b_l0_stage_elems << ")\n";
}

void CodeGenTileLangPTO::EmitAscendCopyGmToCbuf(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 12 arguments";
  CheckConstZero(op->args[2], "sid");
  CheckConstZero(op->args[7], "loop4_src_stride");

  // arg[11] physical_dtype is consumed by the AscendC codegen for packed-SF
  // pointer casts. Reject that variant because PTO cannot emit it yet.
  ICHECK(Downcast<StringImm>(op->args[11])->value.empty())
      << "PTO GM-to-L1 copy does not support packed scale-factor layouts";

  int64_t transpose = 0;
  ICHECK(TryGetConstInt(op->args[9], &transpose))
      << "PTO GM-to-L1 fractal copy expects constant transpose flag";

  std::string dst =
      GetPtoLocalPtrExpr(op->args[0], "mat", DataType::BFloat(16));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string l2_cache_ctrl =
      RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string n_value = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string d_value = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string smallc0_en = RemoveOutermostParentheses(PrintExpr_(op->args[8]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string dst_n_value =
      RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  PrintIndent();
  stream << "pto.mte_gm_l1_frac(" << src << ", " << dst << ", "
         << (transpose == 0 ? "pto.FractalMode.ND2NZ" : "pto.FractalMode.DN2NZ")
         << ", shape=(" << n_value << ", " << d_value << "), src_layout=("
         << src_stride << ",), dst_group=(1, 1, " << dst_n_value
         << ", 0), ctrl=(" << l2_cache_ctrl << ", "
         << (smallc0_en == "0" ? "False" : smallc0_en) << "))\n";
}

void CodeGenTileLangPTO::EmitPTOGemmRun(const std::string &a_mat,
                                        const std::string &b_mat,
                                        const std::string &acc,
                                        const std::string &clear_accum,
                                        const std::string &unit_flag_ctrl) {
  PrintIndent();
  stream << gemm_emit_ctx_.helper_name << ".run_l1_tile(" << a_mat << ", "
         << b_mat << ", " << gemm_emit_ctx_.a_l0_name << ", "
         << gemm_emit_ctx_.b_l0_name << ", " << acc
         << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl << ")\n";
}

void CodeGenTileLangPTO::EmitAscendGemmL1(const CallNode *op) {
  EnsurePTOGemmHelper(op);
  std::string acc = GetPtoAccPtrExpr(op->args[0], DataType::Float(32));
  std::string a_mat =
      GetPtoLocalPtrExpr(op->args[1], "mat", gemm_emit_ctx_.input_dtype);
  std::string b_mat =
      GetPtoLocalPtrExpr(op->args[2], "mat", gemm_emit_ctx_.input_dtype);

  EmitPTOGemmRun(a_mat, b_mat, acc,
                 RemoveOutermostParentheses(PrintExpr_(op->args[8])),
                 RemoveOutermostParentheses(PrintExpr_(op->args[11])));
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToGm(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 25U)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = GetPtoAccPtrExpr(op->args[1], DataType::Float(32));
  std::string sid = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string n_size = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string m_size = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string l2_cache_ctrl =
      RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  int64_t unit_flag_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[9], &unit_flag_ctrl))
      << "PTO L0C-to-GM expects constant unit_flag_ctrl";

  PrintIndent();
  stream << "pto.mte_l0c_gm(" << src << ", " << dst << ", " << m_size << ", "
         << n_size << ", " << src_stride << ", " << dst_stride << ", " << sid
         << ", " << l2_cache_ctrl;
  if (unit_flag_ctrl != 0) {
    stream << ", unit_flag=" << AccStoreUnitFlagArg(unit_flag_ctrl);
  }
  stream << ", layout=\"nz2nd\")\n";
}

std::string
CodeGenTileLangPTO::EmitPTOAllReduceExpr_(const std::string &func_name,
                                          const CallNode *op) {
  ICHECK_GE(op->args.size(), 2U)
      << "tl::AscendAllReduce call expects a value argument";

  const size_t begin = func_name.find("tl::AscendAllReduce");
  ICHECK_NE(begin, std::string::npos)
      << "Cannot parse AscendAllReduce template arguments from: " << func_name;
  ICHECK_NE(func_name.find("tl::AscendAllReduce<tl::SumOp", begin),
            std::string::npos)
      << "PTO codegen currently maps only sum reductions to "
         "pto.simt_allreduce_sum";

  long long parsed_threads = 0;       // NOLINT(runtime/int)
  long long parsed_scale = 1;         // NOLINT(runtime/int)
  long long parsed_thread_offset = 0; // NOLINT(runtime/int)
  const char *pattern = "tl::AscendAllReduce<tl::SumOp, %lld, %lld, %lld";
  const int parsed =
      std::sscanf(func_name.c_str() + begin, pattern, &parsed_threads,
                  &parsed_scale, &parsed_thread_offset);
  ICHECK_GE(parsed, 1)
      << "AscendAllReduce expects at least a threads template parameter: "
      << func_name;

  const int64_t threads = parsed_threads;
  const int64_t scale = parsed >= 2 ? parsed_scale : 1;
  const int64_t thread_offset = parsed >= 3 ? parsed_thread_offset : 0;

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string scratch = "None";
  if (op->args.size() >= 3U) {
    scratch = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  } else {
    ICHECK_LE(threads, scale)
        << "AscendAllReduce with threads > scale requires a scratch pointer";
  }

  std::ostringstream os;
  os << "pto.simt_allreduce_sum(" << value << ", threads=" << threads
     << ", scale=" << scale << ", thread_offset=" << thread_offset
     << ", scratch=" << scratch << ")";
  return os.str();
}

std::string
CodeGenTileLangPTO::PrintVmiAnnotationValue(const std::string &key,
                                            const ObjectRef &value) {
  if (key == "to_dtype") {
    if (const auto *dtype_name = value.as<StringImmNode>()) {
      return PtoScalarType(ParsePTODtype(dtype_name->value));
    }
  }

  if (const auto *expr = value.as<PrimExprNode>()) {
    return PrintExpr_(GetRef<PrimExpr>(expr));
  }

  std::ostringstream os;
  os << value;
  return os.str();
}

void CodeGenTileLangPTO::PrintPtoVmiCall_(const CallNode *op,
                                          std::ostream &os) {
  auto opt_call_op = op->op.as<Op>();
  ICHECK(opt_call_op.has_value());
  std::string op_name = opt_call_op.value()->name;
  ICHECK(StartsWith(op_name, "tl.vmi."))
      << "Expected a tl.vmi.* call, got " << op_name;

  std::vector<std::pair<std::string, ObjectRef>> kwargs;
  kwargs.reserve(op->annotations.size());
  for (const auto &[key, value] : op->annotations) {
    std::string key_str = key;
    if (key_str == "loc" || key_str == "ip") {
      continue;
    }
    kwargs.emplace_back(std::move(key_str), value);
  }
  std::sort(kwargs.begin(), kwargs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  os << "pto." << op_name.substr(3) << "(";
  bool needs_comma = false;

  auto print_scalar_literal_value = [&](const PrimExpr &arg) {
    if (const auto *imm = arg.as<IntImmNode>()) {
      if (imm->dtype == DataType::Bool()) {
        os << (imm->value ? "True" : "False");
      } else {
        os << imm->value;
      }
      return;
    }
    if (const auto *imm = arg.as<FloatImmNode>()) {
      os << "float.fromhex('" << FlexibleHexFormat(imm->value) << "')";
      return;
    }
    PrintExpr_(arg, os);
  };

  // PTODSL vgather/vgatherb/vscatter take a single pointer operand. TileLang
  // lowers buffer addresses to (ptr, elem_offset); fold them with addptr here.
  auto print_ptr_with_offset = [&](const PrimExpr &ptr,
                                   const PrimExpr &offset) {
    if (is_zero(offset)) {
      os << PrintExpr_(ptr);
      return;
    }
    os << "pto.addptr(" << PrintExpr_(ptr) << ", "
       << RemoveOutermostParentheses(PrintExpr_(offset)) << ")";
  };

  if (op_name == "tl.vmi.vgather" || op_name == "tl.vmi.vgatherb") {
    ICHECK_EQ(op->args.size(), 4U)
        << op_name << " expects (ptr, offset, offsets, mask)";
    print_ptr_with_offset(op->args[0], op->args[1]);
    os << ", " << PrintExpr_(op->args[2]) << ", " << PrintExpr_(op->args[3]);
    needs_comma = true;
  } else if (op_name == "tl.vmi.vscatter") {
    ICHECK_EQ(op->args.size(), 5U)
        << op_name << " expects (value, ptr, offset, offsets, mask)";
    os << PrintExpr_(op->args[0]) << ", ";
    print_ptr_with_offset(op->args[1], op->args[2]);
    os << ", " << PrintExpr_(op->args[3]) << ", " << PrintExpr_(op->args[4]);
    needs_comma = true;
  } else if (op_name == "tl.vmi.vstore") {
    auto dist_mode_it = op->annotations.find("dist_mode");
    ObjectRef dist_mode = dist_mode_it != op->annotations.end()
                              ? (*dist_mode_it).second
                              : ObjectRef();
    const bool is_dintlv_store =
        dist_mode.defined() && dist_mode.as<StringImmNode>() != nullptr &&
        Downcast<StringImm>(dist_mode)->value == "dintlv";
    if (is_dintlv_store) {
      ICHECK_GE(op->args.size(), 4U)
          << op_name
          << " with dist_mode=dintlv expects (even, odd, ptr, offset[, mask])";
      os << "(" << PrintExpr_(op->args[0]) << ", " << PrintExpr_(op->args[1])
         << ")";
      for (size_t i = 2; i < op->args.size(); ++i) {
        os << ", " << PrintExpr_(op->args[i]);
      }
      needs_comma = true;
    } else {
      for (size_t i = 0; i < op->args.size(); ++i) {
        const PrimExpr &arg = op->args[i];
        if (needs_comma) {
          os << ", ";
        }
        os << PrintExpr_(arg);
        needs_comma = true;
      }
    }
  } else {
    for (size_t i = 0; i < op->args.size(); ++i) {
      const PrimExpr &arg = op->args[i];
      if (needs_comma) {
        os << ", ";
      }
      const bool is_scalar_literal =
          arg.as<FloatImmNode>() != nullptr || arg.as<IntImmNode>() != nullptr;
      const bool should_wrap_typed_literal =
          (op_name == "tl.vmi.vbrc" || op_name == "tl.vmi.vci") && i == 0 &&
          is_scalar_literal;
      if (should_wrap_typed_literal) {
        // PTODSL needs typed literal scalars for these VMI sources; preserve
        // the literal dtype instead of assuming every source is f32.
        os << PtoScalarType(arg.dtype()) << "(";
        print_scalar_literal_value(arg);
        os << ")";
      } else {
        os << PrintExpr_(arg);
      }
      needs_comma = true;
    }
  }

  for (const auto &[key, value] : kwargs) {
    if (needs_comma) {
      os << ", ";
    }
    os << key << "=";
    if (op_name == "tl.vmi.vcvt" && key == "to_dtype" && op->dtype.is_int()) {
      os << PtoSignedIntegerTypeName(op->dtype.element_of());
    } else {
      os << PrintVmiAnnotationValue(key, value);
    }
    needs_comma = true;
  }
  os << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const CallNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->op.same_as(builtin::bitwise_and())) {
    PrintBinaryExpr_("&", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::bitwise_or())) {
    PrintBinaryExpr_("|", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::bitwise_xor())) {
    PrintBinaryExpr_("^", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::shift_left())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_left";
    PrintBinaryExpr_("*", op->dtype, op->args[0],
                     IntImm(op->args[0].dtype(), 1LL << shift), os);
    return;
  }

  if (op->op.same_as(builtin::shift_right())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_right";
    PrintBinaryExpr_("//", op->dtype, op->args[0],
                     IntImm(op->args[0].dtype(), 1LL << shift), os);
    return;
  }

  if (op->op.same_as(builtin_call_extern_) ||
      op->op.same_as(builtin_call_pure_extern_)) {
    ICHECK_GE(op->args.size(), 1U);
    std::string func_name = Downcast<StringImm>(op->args[0])->value;
    if ((func_name == "sqrt" || func_name == "sqrtf") &&
        op->args.size() == 2U) {
      std::string value = PrintExpr_(op->args[1]);
      os << (inside_simtvf_body_ ? "pto.sqrt(" : "scalar.sqrt(") << value
         << ")";
      return;
    }
    if ((func_name == "rsqrt" || func_name == "rsqrtf") &&
        op->args.size() == 2U) {
      std::string value = PrintExpr_(op->args[1]);
      os << "(1.0 / " << (inside_simtvf_body_ ? "pto.sqrt(" : "scalar.sqrt(")
         << value << "))";
      return;
    }
    if (func_name.find("tl::AscendAllReduce") != std::string::npos) {
      os << EmitPTOAllReduceExpr_(func_name, op);
      return;
    }
  }

  if (op->op.same_as(builtin::tvm_storage_sync())) {
    ICHECK_GE(op->args.size(), 1U)
        << "tvm_storage_sync expects at least the storage scope argument";
    std::string sync_scope = Downcast<StringImm>(op->args[0])->value;
    if (sync_scope == "warp") {
      return;
    }
    if (sync_scope == "shared" || sync_scope == "shared.dyn") {
      PrintIndent();
      stream << (inside_simtvf_body_ ? "pto.syncthreads()\n"
                                     : "pto.pipe_barrier(pto.Pipe.ALL)\n");
      return;
    }
    LOG(FATAL) << "Unsupported PTO storage sync scope: " << sync_scope;
  }

  if (op->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
    PrintIndent();
    stream << GetAscendCopyGmUbExpr_(op) << "\n";
    return;
  }

  if (op->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
    PrintIndent();
    stream << GetAscendCopyUbGmExpr_(op) << "\n";
    return;
  }

  if (op->op.same_as(tl::ascend_copy_gm_to_cbuf())) {
    EmitAscendCopyGmToCbuf(op);
    return;
  }

  if (op->op.same_as(tl::ascend_gemm_l1())) {
    EmitAscendGemmL1(op);
    return;
  }

  if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    EmitAscendCopyMatrixCcToGm(op);
    return;
  }

  if (op->op.same_as(builtin::address_of())) {
    os << GetAddressOfExpr_(op);
    return;
  }

  if (op->op.same_as(builtin::tvm_access_ptr())) {
    os << GetAccessPtrExpr_(op);
    return;
  }

  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    auto pipe_name = Downcast<StringImm>(op->args[0])->value;
    PrintIndent();
    stream << "pto.pipe_barrier(\"" << StripPipePrefix(pipe_name) << "\")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_flag()) ||
      op->op.same_as(tl::ascend_wait_flag())) {
    auto hard_event = Downcast<StringImm>(op->args[0])->value;
    auto [src_pipe, dst_pipe] = ParseHardEventPair(hard_event);
    std::string eid = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
    PrintIndent();
    stream << (op->op.same_as(tl::ascend_set_flag()) ? "pto.set_flag("
                                                     : "pto.wait_flag(")
           << "\"" << src_pipe << "\", \"" << dst_pipe << "\", event_id=" << eid
           << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_pset())) {
    ICHECK_GE(op->args.size(), 1U)
        << "tl.simd.pset expects at least 1 argument (element width)";
    int64_t elem_width = 0;
    ICHECK(TryGetConstInt(op->args[0], &elem_width))
        << "tl.simd.pset element width must be constant for PTO codegen";
    std::string dist = "PAT_ALL";
    if (op->args.size() >= 2U) {
      dist = Downcast<StringImm>(op->args[1])->value;
    }
    os << "pto.pset_b" << elem_width << "(\"" << dist << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() >= 2U && op->args.size() <= 3U)
        << "tl.simd.vld expects 2 or 3 arguments (addr, dist[, offset])";
    DataType elem_dtype = op->dtype.element_of();
    ICHECK_GT(op->dtype.lanes(), 1)
        << "tl.simd.vld should return a vector type, got " << op->dtype;
    std::string dist = Downcast<StringImm>(op->args[1])->value;
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    os << "pto.vlds(" << PrintExpr_(op->args[0]) << ", " << offset
       << ", pto.vreg_type(" << op->dtype.lanes() << ", "
       << PtoScalarType(elem_dtype) << ")";
    if (!dist.empty() && dist != "NORM") {
      os << ", dist=\"" << dist << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vadd())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vadd expects 4 arguments (src0, src1, mask, mode)";
    std::string mode = Downcast<StringImm>(op->args[3])->value;
    ICHECK_EQ(mode, "MODE_ZEROING")
        << "PTO codegen currently only supports MODE_ZEROING for tl.simd.vadd";
    os << "pto.vadd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vsts())) {
    ICHECK(op->args.size() >= 3U && op->args.size() <= 5U)
        << "tl.simd.vsts expects 3 to 5 arguments "
           "(addr, src, mask[, dist][, offset])";
    std::string dist;
    int offset_index = -1;
    if (op->args.size() >= 4U && op->args[3].as<StringImmNode>()) {
      dist = Downcast<StringImm>(op->args[3])->value;
      if (op->args.size() == 5U) {
        offset_index = 4;
      }
    } else if (op->args.size() == 4U) {
      offset_index = 3;
    }
    std::string offset =
        offset_index >= 0
            ? RemoveOutermostParentheses(PrintExpr_(op->args[offset_index]))
            : "pto.const(0)";
    PrintIndent();
    stream << "pto.vsts(" << PrintExpr_(op->args[1]) << ", "
           << PrintExpr_(op->args[0]) << ", " << offset << ", "
           << PrintExpr_(op->args[2]);
    if (!dist.empty()) {
      stream << ", dist=\"" << dist << "\"";
    }
    stream << ")\n";
    return;
  }

  if (auto opt_call_op = op->op.as<Op>()) {
    const auto &call_op = opt_call_op.value();
    std::string op_name = call_op->name;
    if ((op_name == "tir.rsqrt" || op_name == "tirx.rsqrt") &&
        op->args.size() == 1U) {
      std::string value = PrintExpr_(op->args[0]);
      os << "(1.0 / " << (inside_simtvf_body_ ? "pto.sqrt(" : "scalar.sqrt(")
         << value << "))";
      return;
    }
    if ((op_name == "tir.sqrt" || op_name == "tirx.sqrt") &&
        op->args.size() == 1U) {
      std::string value = PrintExpr_(op->args[0]);
      os << (inside_simtvf_body_ ? "pto.sqrt(" : "scalar.sqrt(") << value
         << ")";
      return;
    }
    if (StartsWith(op_name, "tl.vmi.")) {
      if (op_name == "tl.vmi.pair_get") {
        ICHECK_EQ(op->args.size(), 2U)
            << "tl.vmi.pair_get expects exactly 2 arguments";
        os << "(" << PrintExpr_(op->args[0]) << ")[" << PrintExpr_(op->args[1])
           << "]";
        return;
      }
      PrintPtoVmiCall_(op, os);
      return;
    }
    if (StartsWith(op_name, "tl.")) {
      LOG(FATAL) << "PTO codegen does not support TileLang op `" << op_name
                 << "`. Please add a handler in CodeGenTileLangPTO or lower "
                    "it before PTO codegen.";
    }
  }

  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::VisitExpr_(const CastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  DataType from = op->value.dtype();
  DataType to = op->dtype;
  bool from_integer = from.is_int() || from.is_uint();
  bool to_integer = to.is_int() || to.is_uint();
  if (from_integer && to_integer && from.lanes() == to.lanes() &&
      from.bits() < to.bits()) {
    // Address-index widening casts should not become Python int(...), because
    // that coerces PTODSL runtime values and breaks tracing.
    PrintExpr_(op->value, os);
    return;
  }

  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::VisitExpr_(const AndNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("&", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::VisitExpr_(const OrNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("|", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::VisitExpr_(const SelectNode *op,
                                    std::ostream &os) { // NOLINT(*)
  auto print_select_value = [&](const PrimExpr &expr) {
    if (const auto *imm = expr.as<IntImmNode>()) {
      os << "pto.const(" << imm->value << ", dtype=" << PtoTypeName(op->dtype)
         << ")";
      return;
    }
    PrintExpr_(expr, os);
  };
  os << "scalar.select(";
  PrintExpr_(op->condition, os);
  os << ", ";
  print_select_value(op->true_value);
  os << ", ";
  print_select_value(op->false_value);
  os << ")";
}

void CodeGenTileLangPTO::PrintBinaryExpr_(const std::string &opstr,
                                          DataType dtype, PrimExpr lhs,
                                          PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
  if (dtype.is_scalar()) {
    CodeGenTileLangPY::PrintBinaryExpr_(opstr, dtype, lhs, rhs, os);
    return;
  }

  ICHECK(inside_simtvf_body_)
      << "PTO vector binary expressions are only supported inside SIMT bodies";
  if (opstr != "+" && opstr != "-" && opstr != "*" && opstr != "/") {
    LOG(FATAL) << "Unsupported PTO SIMT vector binary op: " << opstr;
  }
  os << "(" << PrintExpr_(lhs) << " " << opstr << " " << PrintExpr_(rhs) << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const BroadcastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  DataType elem_dtype = op->value.dtype();
  ICHECK(IsFloat32(elem_dtype))
      << "PTO vector broadcast currently supports float32 only, got "
      << elem_dtype;
  std::string value = PrintExpr_(op->value);
  os << "pto.Vec(pto.f32, " << op->dtype.lanes() << ", init=" << value << ")";
}

void CodeGenTileLangPTO::VisitStmt_(const DeclBufferNode *op) { (void)op; }

void CodeGenTileLangPTO::VisitStmt_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>()) {
    if (IsOpName(call->op, "tl.simd.alloc")) {
      AllocVarID(op->var.get());
      DataType elem_dtype = op->var.dtype().element_of();
      ICHECK_GT(op->var.dtype().lanes(), 1)
          << "tl.simd.alloc should bind a vector-typed Var, got "
          << op->var.dtype();
      FragmentInfo info;
      info.lanes = op->var.dtype().lanes();
      info.dtype = elem_dtype;
      fragment_info_[op->var.get()] = info;
      return;
    }
  }

  PrintSSAAssign(AllocVarID(op->var.get()), PrintExpr_(op->value),
                 op->var.dtype());
}

void CodeGenTileLangPTO::VisitStmt_(const AllocBufferNode *op) {
  // VMI/SIMD mutable registers are local.var buffers even for vector-only
  // kernels (which do not enter the GEMM allocation path).
  if (GetPtrStorageScope(op->buffer->data) == "local.var") {
    const Var &buffer_var = op->buffer->data;
    CheckPTOLocalVarBuffer(op->buffer.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    if (op->buffer->dtype.lanes() > 1) {
      stream << AllocVarID(buffer_var.get()) << " = pto.vmi.vreg("
             << op->buffer->dtype.lanes() << ", "
             << PtoTypeName(op->buffer->dtype.element_of()) << ")\n";
    } else {
      stream << AllocVarID(buffer_var.get())
             << " = pto.const(0, dtype=pto.int64)\n";
    }
    RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
    return;
  }
  if (!current_function_has_gemm_) {
    EmitPtoBufferAllocation(op->buffer);
    return;
  }

  const Var &buffer_var = op->buffer->data;
  std::string scope = GetPtrStorageScope(buffer_var);
  alloc_storage_scope_[buffer_var.get()] = scope;

  if (scope == "shared" || scope == "shared.dyn" || scope == "shared.l1" ||
      scope == "shared.l1.dyn" || scope == "shared.l0c") {
    PrintIndent();
    stream << AllocVarID(buffer_var.get())
           << " = pto.const(0, dtype=pto.int64)\n";
  } else if (scope == "local.fragment") {
    AllocVarID(buffer_var.get());
    auto alloc_ref = GetRef<AllocBuffer>(op);
    auto opt_size = alloc_ref.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO local.fragment allocation expects a constant size";
    FragmentInfo info;
    info.lanes = static_cast<int>(opt_size.value());
    info.dtype = op->buffer->dtype;
    fragment_info_[buffer_var.get()] = info;
  } else if (scope == "local.var") {
    // Scalar alloc_var is used by GEMM tile mapping; vector alloc_var is a
    // mutable VMI/SIMD register and is emitted as a PTODSL Vec value.
    CheckPTOLocalVarBuffer(op->buffer.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    if (op->buffer->dtype.lanes() > 1) {
      stream << AllocVarID(buffer_var.get()) << " = pto.vmi.vreg("
             << op->buffer->dtype.lanes() << ", "
             << PtoTypeName(op->buffer->dtype.element_of()) << ")\n";
    } else {
      stream << AllocVarID(buffer_var.get())
             << " = pto.const(0, dtype=pto.int64)\n";
    }
  }

  RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
}

void CodeGenTileLangPTO::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x") {
      var_idmap_[iv->var.get()] = "pto.get_block_idx()";
      VisitStmt(op->body);
      return;
    }

    std::string vid = AllocVarID(iv->var.get());
    std::string thread_value;
    if (iv->thread_tag == "cthread") {
      thread_value = "pto.get_subblock_idx()";
    } else if (iv->thread_tag == "threadIdx.x") {
      thread_value = "pto.get_tid_x()";
    } else if (iv->thread_tag == "threadIdx.y") {
      thread_value = "pto.get_tid_y()";
    } else if (iv->thread_tag == "threadIdx.z") {
      thread_value = "pto.get_tid_z()";
    } else {
      LOG(FATAL) << "Unsupported PTO thread tag: " << iv->thread_tag;
    }
    PrintIndent();
    stream << vid << " = " << thread_value << "\n";
    VisitStmt(op->body);
    return;
  }

  if (op->attr_key == "tl.simdvf_scope") {
    VisitStmt(op->body);
    return;
  }

  if (op->attr_key == "tl.simtvf_scope") {
    VisitStmt(op->body);
    return;
  }

  VisitStmt(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const ForNode *op) {
  if (!current_function_has_gemm_) {
    PrintIndent();
    std::string vid = AllocVarID(op->loop_var.get());
    std::string begin = RemoveOutermostParentheses(PrintExpr_(op->min));
    PrimExpr upper_bound = arith::Analyzer().Simplify(op->extent + op->min);
    std::string end = RemoveOutermostParentheses(PrintExpr_(upper_bound));
    stream << "for " << vid << " in range(" << begin << ", " << end << "):\n";
    int for_scope = BeginScope();
    PrintStmt_(op->body);
    EndScope(for_scope);
    return;
  }

  PrimExpr start = arith::Analyzer().Simplify(op->min);
  PrimExpr stop = arith::Analyzer().Simplify(op->min + op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "with pto.for_(" << PrintExpr_(start) << ", " << PrintExpr_(stop)
         << ", step=1) as " << vid << ":\n";
  int scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(scope);
}

void CodeGenTileLangPTO::VisitStmt_(const SBlockNode *op) {
  if (op->name_hint == "SIMT_VF") {
    int64_t thread_x = 1;
    int64_t thread_y = 1;
    int64_t thread_z = 1;
    ExtractSimtThreadExtents(op, &thread_x, &thread_y, &thread_z);

    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simtvf_helper_counter_++;
    auto it = op->annotations.find("tl.vf_source_index");
    if (it != op->annotations.end()) {
      if (auto *imm = (*it).second.as<IntImmNode>()) {
        vf_idx = imm->value;
      }
    }
    std::string helper_name = "simt_vf_" + std::to_string(vf_idx);
    EmitSimtVFFunction(op, captures, helper_name, thread_x, thread_y, thread_z);
    EmitSimtVFLaunch(captures, helper_name, thread_x, thread_y, thread_z);
    return;
  }

  if (op->name_hint == "SIMD_VF") {
    for (const Buffer &buf : op->alloc_buffers) {
      EmitPtoBufferAllocation(buf);
    }
    if (op->init.defined()) {
      PrintStmt_(op->init.value());
    }
    PrintStmt_(op->body);
    return;
  }

  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const IfThenElseNode *op) {
  std::string cond = RemoveOutermostParentheses(PrintExpr_(op->condition));
  PrintIndent();
  stream << "if " << cond << ":\n";
  int if_scope = BeginScope();
  PrintStmt_(op->then_case);
  EndScope(if_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangPTO::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  std::string emitted = PrintExpr_(op->value);
  if (!emitted.empty()) {
    PrintIndent();
    stream << emitted << "\n";
  }
}

void CodeGenTileLangPTO::VisitExpr_(const BufferLoadNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Read it back as the scalar surface
    // value used by GEMM tile mapping, not as a general PTO buffer load.
    CheckPTOLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      os << LocalVarID(op->buffer->data.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    os << LocalVarID(op->buffer->data.get());
    return;
  }

  if (IsVmiLocalRegisterBuffer(op->buffer.get())) {
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO VMI local register buffers must be flattened before codegen";
    ICHECK(!op->predicate.defined())
        << "PTO VMI local register buffers do not support predicated loads";
    CheckVmiLocalRegisterIndex(op->buffer.get(), op->indices[0]);
    os << GetVarID(op->buffer->data.get()) << "["
       << RemoveOutermostParentheses(PrintExpr_(op->indices[0])) << "]";
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer loads";
  ICHECK(!op->predicate.defined())
      << "CodeGenTileLangPTO does not support predicated loads yet";

  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  if (value_dtype.lanes() > 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  ICHECK_EQ(value_dtype, element_dtype)
      << "PTO scalar BufferLoad expects value dtype to match buffer element "
         "dtype, got "
      << value_dtype << " vs " << element_dtype;

  os << PtoScalarLoad(op->buffer.get(), op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedLoad(const BufferLoadNode *op,
                                            std::ostream &os) {
  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  ICHECK_EQ(element_dtype.lanes(), 1)
      << "PTO vector BufferLoad scalarization currently expects scalar "
         "buffer elements, got "
      << element_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsFloat32(element_dtype))
        << "PTO SIMT vector BufferLoad currently supports float32 only, got "
        << element_dtype;
    std::string scope = ScopeOfBuffer(op->buffer.get());
    const int lanes = value_dtype.lanes();
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "load");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    if (scope == "local.fragment" || scope == "local") {
      os << "scalar.load(" << GetVarID(op->buffer->data.get()) << ", "
         << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = GetVarID(op->buffer->data.get());
      if (HandleTypeMatch_(op->buffer->data.get(), DataType::Int(8)) ||
          !HandleTypeMatch_(op->buffer->data.get(), element_dtype)) {
        base = "pto.castptr(" + base + ", " + PtoPtrType(element_dtype, "ub") +
               ")";
      }
      os << "scalar.load(" << base << ", " << index_str
         << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      os << "scalar.load(" << GetVarID(op->buffer->data.get()) << ", "
         << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector load scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferLoad is not supported yet";
}

void CodeGenTileLangPTO::VisitStmt_(const BufferStoreNode *op) {
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Store it as a scalar surface value
    // for GEMM tile mapping, not as a general PTO buffer store.
    CheckPTOLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      PrintIndent();
      stream << LocalVarID(op->buffer->data.get()) << " = "
             << RemoveOutermostParentheses(PrintExpr_(op->value)) << "\n";
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var store expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var store expects index 0";
    PrintIndent();
    stream << LocalVarID(op->buffer->data.get()) << " = "
           << "_tl_wrap_surface_value(_tl_coerce_i64("
           << RemoveOutermostParentheses(PrintExpr_(op->value))
           << ", context=\"PTO local.var store\"))\n";
    return;
  }

  if (IsVmiLocalRegisterBuffer(op->buffer.get())) {
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO VMI local register buffers must be flattened before codegen";
    ICHECK(!op->predicate.defined())
        << "PTO VMI local register buffers do not support predicated stores";
    CheckVmiLocalRegisterIndex(op->buffer.get(), op->indices[0]);
    PrintIndent();
    stream << GetVarID(op->buffer->data.get()) << "["
           << RemoveOutermostParentheses(PrintExpr_(op->indices[0]))
           << "] = " << RemoveOutermostParentheses(PrintExpr_(op->value))
           << "\n";
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer stores";
  ICHECK(!op->predicate.defined())
      << "CodeGenTileLangPTO does not support predicated stores yet";

  if (op->value.dtype().lanes() > 1) {
    EmitScalarizedStore(op);
    return;
  }

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
  PrintIndent();
  EmitPtoScalarStore(op->buffer.get(), value, op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedStore(const BufferStoreNode *op) {
  ICHECK_EQ(op->buffer->dtype.lanes(), 1)
      << "PTO vector BufferStore scalarization currently expects scalar "
         "buffer elements, got "
      << op->buffer->dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsFloat32(op->buffer->dtype))
        << "PTO SIMT vector BufferStore currently supports float32 only, got "
        << op->buffer->dtype;
    std::string scope = ScopeOfBuffer(op->buffer.get());
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "store");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    PrintIndent();
    if (scope == "local.fragment" || scope == "local") {
      stream << "scalar.store(" << value << ", "
             << GetVarID(op->buffer->data.get()) << ", " << index_str << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = GetVarID(op->buffer->data.get());
      if (HandleTypeMatch_(op->buffer->data.get(), DataType::Int(8)) ||
          !HandleTypeMatch_(op->buffer->data.get(), op->buffer->dtype)) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(op->buffer->dtype, "ub") + ")";
      }
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      stream << "scalar.store(" << value << ", "
             << GetVarID(op->buffer->data.get()) << ", " << index_str << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector store scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferStore is not supported yet";
}

} // namespace codegen
} // namespace tvm
