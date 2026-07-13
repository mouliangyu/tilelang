/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <sstream>
#include <string>

#include "backend/common/codegen/codegen_utils.h"
#include "op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace codegen {

using namespace tirx;

namespace {

std::string PtoTypeName(DataType t) {
  ICHECK(t.is_scalar()) << "PTO scalar type expected, got " << t;
  if (t.is_float()) {
    if (t.bits() == 32)
      return "pto.float32";
    if (t.bits() == 16)
      return "pto.float16";
  } else if (t.is_bfloat16()) {
    return "pto.bf16";
  } else if (t.is_int() || t.is_uint()) {
    if (t.bits() == 64)
      return "pto.int64";
    if (t.bits() == 32)
      return "pto.int32";
    if (t.bits() == 16)
      return "pto.int16";
    if (t.bits() == 8)
      return "pto.int8";
    if (t.bits() == 1)
      return "pto.int1";
  }
  LOG(FATAL) << "Unsupported PTO type: " << t;
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
  if (dtype_name == "int64")
    return DataType::Int(64);
  if (dtype_name == "int32")
    return DataType::Int(32);
  if (dtype_name == "int16")
    return DataType::Int(16);
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

void CheckConstZero(const PrimExpr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value) && value == 0)
      << "PTO codegen currently only supports " << name
      << " == 0 for tl.ascend_copy_gm_to_ubuf, got " << expr;
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
  ICHECK_EQ(dtype.lanes(), 1)
      << "PTO local.var only supports scalar integer values, got " << dtype;
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
    tirx::Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
    if (func->buffer_map.count(v)) {
      tirx::Buffer buffer = func->buffer_map[v];
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

std::string CodeGenTileLangPTO::GetPtoPointerExpr(const VarNode *buffer_var,
                                                  DataType elem_dtype,
                                                  const PrimExpr &index) {
  std::string scope = "global";
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }

  std::string base = GetVarID(buffer_var);
  if (scope == "shared" || scope == "shared.dyn") {
    base = "pto.castptr(" + base + ", " + PtoPtrType(elem_dtype, "ub") + ")";
  }

  if (is_zero(index)) {
    return base;
  }

  std::string index_str;
  int64_t const_index = 0;
  if (TryGetConstInt(index, &const_index)) {
    index_str = "pto.const(" + std::to_string(const_index) + ")";
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

  if (auto *type_call = op->args[0].as<CallNode>()) {
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
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  std::ostringstream os;
  os << "pto.mte_ub_gm(" << src << ", " << dst << ", " << burst_len
     << ", nburst=(" << burst_num << ", " << src_stride << ", " << dst_stride
     << "))";
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
  ICHECK_EQ(base_k % 16, 0);

  const auto *dtype_name = op->args[9].as<StringImmNode>();
  ICHECK(dtype_name) << "PTO GEMM L1 helper requires a constant input dtype "
                        "string at tl.ascend_gemm_l1 arg 9";
  DataType input_dtype = ParsePTODtype(dtype_name->value);
  ICHECK(input_dtype.is_bfloat16())
      << "PTO GEMM L1 helper currently only supports bfloat16 inputs, got "
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

  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / 16;
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
         << sub_k_tiles << ", " << sub_k_c0_blocks << ", " << a_l0_stage_elems
         << ", " << b_l0_stage_elems << ")\n";
}

void CodeGenTileLangPTO::EmitAscendCopyGmToCbuf(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 11U)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 11 arguments";
  CheckConstZero(op->args[2], "sid");
  CheckConstZero(op->args[7], "loop4_src_stride");

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

  PrintIndent();
  stream << "pto.mte_gm_l1_frac(" << src << ", " << dst << ", "
         << (transpose == 0 ? "pto.FractalMode.ND2NZ" : "pto.FractalMode.DN2NZ")
         << ", shape=(" << n_value << ", " << d_value << "), src_layout=("
         << src_stride << ",), dst_group=(1, 1, " << d_value << ", 0), ctrl=("
         << l2_cache_ctrl << ", " << (smallc0_en == "0" ? "False" : smallc0_en)
         << "))\n";
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
    if (StartsWith(op_name, "tl.")) {
      LOG(FATAL) << "PTO codegen does not support TileLang op `" << op_name
                 << "`. Please add a handler in CodeGenTileLangPTO or "
                    "lower it before PTO codegen.";
    }
  }

  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::VisitExpr_(const BufferLoadNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Read it back as the scalar surface
    // value used by GEMM tile mapping, not as a general PTO buffer load.
    CheckPTOLocalVarBuffer(op->buffer.get());
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    os << LocalVarID(op->buffer->data.get());
    return;
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

void CodeGenTileLangPTO::VisitStmt_(const DeclBufferNode *op) {
  // DeclBuffer is a leaf statement in tirx. The surrounding SeqStmt owns order.
}

void CodeGenTileLangPTO::VisitStmt_(const BufferStoreNode *op) {
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Store it as a scalar surface value
    // for GEMM tile mapping, not as a general PTO buffer store.
    CheckPTOLocalVarBuffer(op->buffer.get());
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

  CodeGenTileLangPY::VisitStmt_(op);
}

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
    // T.alloc_var lowers to a one-element local.var buffer.
    CheckPTOLocalVarBuffer(op->buffer.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    stream << AllocVarID(buffer_var.get())
           << " = pto.const(0, dtype=pto.int64)\n";
  }

  RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
}

void CodeGenTileLangPTO::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    std::string vid = AllocVarID(iv->var.get());
    std::string thread_value;
    if (iv->thread_tag == "blockIdx.x") {
      thread_value = "pto.get_block_idx()";
    } else if (iv->thread_tag == "cthread") {
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

  VisitStmt(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const ForNode *op) {
  if (!current_function_has_gemm_) {
    CodeGenTileLangPY::VisitStmt_(op);
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
  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const IfThenElseNode *op) {
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(PrintExpr_(op->condition))
         << ":\n";
  int if_scope = BeginScope();
  PrintStmt_(op->then_case);
  EndScope(if_scope);

  if (op->else_case.defined()) {
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

} // namespace codegen
} // namespace tvm
