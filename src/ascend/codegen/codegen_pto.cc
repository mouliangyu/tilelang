/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"

#include "backend/common/codegen/codegen_utils.h"
#include "op/builtin.h"
#include "support/check.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <sstream>
#include <string>

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

void CheckConstZero(const PrimExpr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value) && value == 0)
      << "PTO codegen currently only supports " << name
      << " == 0 for tl.ascend_copy_gm_to_ubuf, got " << expr;
}

// TODO: Support PTO CUBE and MIX kernels.
void CheckPTOVectorKernel(const PrimFunc &func) {
  bool has_cube_block = false;
  bool has_vector_block = false;
  bool has_cube_op = false;

  PostOrderVisit(func->body, [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      if (block->name_hint == "CUBE") {
        has_cube_block = true;
      } else if (block->name_hint == "VECTOR") {
        has_vector_block = true;
      }
    }

    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::ascend_gemm_l1()) ||
          call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
          call->op.same_as(tl::ascend_mad()) ||
          call->op.same_as(tl::ascend_nd2nz_scatter()) ||
          call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_cube_op = true;
      }
    }
  });

  ICHECK(!has_cube_block && !has_vector_block)
      << "PTO codegen only supports flat vector kernels in the first porting "
         "phase.";
  ICHECK(!has_cube_op) << "PTO codegen does not support Ascend cube ops yet.";
}

} // namespace

void CodeGenTileLangPTO::AddFunction(const GlobalVar &gvar,
                                     const PrimFunc &func) {
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  CheckPTOVectorKernel(func);

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(current_function_name_, func, stream);
  stream << ":\n";
  int func_scope = BeginScope();
  PrintStmt_(func->body);
  EndScope(func_scope);
  stream << "\n";
}

std::string CodeGenTileLangPTO::Finish() {
  std::ostringstream code;
  code << "from ptodsl import pto\n";
  code << "\n";
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangPTO::PrintFuncDecorator_(std::ostream &os) { // NOLINT(*)
  os << "@pto.jit(name=\"" << current_function_name_
     << "\", kernel_kind=\"vector\", target=\"a5\", mode=\"explicit\")\n";
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
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  std::ostringstream os;
  os << "pto.mte_ub_gm(" << src << ", " << dst << ", " << burst_len
     << ", nburst=(" << burst_num << ", " << src_stride << ", " << dst_stride
     << "))";
  return os.str();
}

void CodeGenTileLangPTO::VisitExpr_(const CallNode *op,
                                    std::ostream &os) { // NOLINT(*)
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
                 << "`. Please add a handler in CodeGenTileLangPTO or lower "
                    "it before PTO codegen.";
    }
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

void CodeGenTileLangPTO::VisitStmt_(const DeclBufferNode *op) { (void)op; }

void CodeGenTileLangPTO::VisitStmt_(const BindNode *op) {
  PrintSSAAssign(AllocVarID(op->var.get()), PrintExpr_(op->value),
                 op->var.dtype());
}

void CodeGenTileLangPTO::VisitStmt_(const AllocBufferNode *op) {
  std::string scope = GetPtrStorageScope(op->buffer->data);
  alloc_storage_scope_[op->buffer->data.get()] = scope;

  if (scope == "shared" || scope == "shared.dyn") {
    PrintIndent();
    stream << AllocVarID(op->buffer->data.get())
           << " = pto.const(0, dtype=pto.int64)\n";
  } else if (scope == "local.fragment") {
    AllocVarID(op->buffer->data.get());
  } else if (scope == "local.var") {
    PrintIndent();
    stream << AllocVarID(op->buffer->data.get()) << " = 0\n";
  }

  RegisterHandleType_(op->buffer->data.get(), op->buffer->dtype);
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

  LOG(FATAL) << "Unsupported PTO attr: " << op->attr_key;
}

void CodeGenTileLangPTO::VisitStmt_(const ForNode *op) {
  CodeGenTileLangPY::VisitStmt_(op);
}

void CodeGenTileLangPTO::VisitStmt_(const SBlockNode *op) {
  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const IfThenElseNode *op) {
  CodeGenTileLangPY::VisitStmt_(op);
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
