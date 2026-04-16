// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_simdvf_lower_parallel.cc
 * \brief Lower T.Parallel loops inside SIMD_VF blocks to MicroAPI
 *        register-level vector instructions (LoadAlign/Add/StoreAlign).
 *
 * VReg size: 8 VLanes × 32 Bytes / sizeof(dtype)
 *   float32 → 64, float16 → 128, int8 → 256
 *
 * For 2D parallel, the dimension with stride=1 (contiguous access) is
 * chosen as the VReg dimension; the other becomes serial.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

class AscendSimdVFLowerParallel : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    AscendSimdVFLowerParallel substituter;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return ffi::GetRef<PrimFunc>(fptr);
  }

private:
  bool inside_simdvf_{false};

  // VReg (inner/contiguous) dimension
  const VarNode *parallel_var_{nullptr};
  int64_t parallel_extent_{0};

  // Outer (serial) dimension for 2D parallel
  const VarNode *outer_parallel_var_{nullptr};
  Var outer_serial_var_;

  int64_t vreg_size_{0};
  Var chunk_var_;
  Var mask_cnt_var_;
  Var mask_var_;

  std::vector<Buffer> new_fragments_;
  int frag_counter_{0};
  std::string dtype_str_;

  //-----------------------------------------------------------------------
  // VReg / type helpers
  //-----------------------------------------------------------------------
  static int64_t ComputeVRegSize(DataType dtype) {
    return 8 * 32 / (dtype.bits() / 8);
  }

  static std::string DTypeToAscendStr(DataType dtype) {
    if (dtype.is_float())
      return (dtype.bits() == 16) ? "half_t" : "float";
    if (dtype.is_int())
      return "int" + std::to_string(dtype.bits()) + "_t";
    if (dtype.is_uint())
      return "uint" + std::to_string(dtype.bits()) + "_t";
    return "float";
  }

  //-----------------------------------------------------------------------
  // Fragment allocation
  //-----------------------------------------------------------------------
  Buffer AllocFragmentReg(DataType dtype) {
    std::string name = "vreg_" + std::to_string(frag_counter_++);
    Var data(name, PointerType(PrimType(dtype), "local.fragment"));
    Array<PrimExpr> shape = {IntImm(DataType::Int(32), vreg_size_)};
    Buffer buf(data, dtype, shape, {}, PrimExpr(0), name, 0, 0, kDefault);
    new_fragments_.push_back(buf);
    return buf;
  }

  //-----------------------------------------------------------------------
  // Stride analysis: coefficient of `var` in linear expression
  // Returns 0 if not found, -1 if non-linear, >0 = stride
  //-----------------------------------------------------------------------
  static int64_t GetVarStride(const PrimExpr &expr, const VarNode *var) {
    if (const auto *v = expr.as<VarNode>())
      return (v == var) ? 1 : 0;
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>())
      return 0;
    if (const auto *add = expr.as<AddNode>()) {
      int64_t sa = GetVarStride(add->a, var);
      int64_t sb = GetVarStride(add->b, var);
      return (sa < 0 || sb < 0) ? -1 : sa + sb;
    }
    if (const auto *sub = expr.as<SubNode>()) {
      int64_t sa = GetVarStride(sub->a, var);
      int64_t sb = GetVarStride(sub->b, var);
      return (sa < 0 || sb < 0) ? -1 : sa - sb;
    }
    if (const auto *mul = expr.as<MulNode>()) {
      if (const auto *imm = mul->b.as<IntImmNode>()) {
        int64_t sa = GetVarStride(mul->a, var);
        return sa < 0 ? -1 : sa * imm->value;
      }
      if (const auto *imm = mul->a.as<IntImmNode>()) {
        int64_t sb = GetVarStride(mul->b, var);
        return sb < 0 ? -1 : sb * imm->value;
      }
      int64_t sa = GetVarStride(mul->a, var);
      int64_t sb = GetVarStride(mul->b, var);
      return (sa != 0 && sb != 0) ? -1 : ((sa == 0) ? sb : sa);
    }
    if (const auto *fd = expr.as<FloorDivNode>()) {
      int64_t sa = GetVarStride(fd->a, var);
      if (sa == 0)
        return 0;
      if (const auto *imm = fd->b.as<IntImmNode>())
        return (sa > 0 && sa % imm->value == 0) ? sa / imm->value : -1;
      return -1;
    }
    if (expr.as<FloorModNode>()) {
      return GetVarStride(expr.as<FloorModNode>()->a, var) == 0 ? 0 : -1;
    }
    if (const auto *cast = expr.as<CastNode>())
      return GetVarStride(cast->value, var);
    return -1;
  }

  //-----------------------------------------------------------------------
  // Strip AttrStmt wrappers to get to the real body
  //-----------------------------------------------------------------------
  static Stmt StripAttr(const Stmt &body) {
    Stmt cur = body;
    while (const auto *attr = cur.as<AttrStmtNode>())
      cur = attr->body;
    return cur;
  }

  //-----------------------------------------------------------------------
  // Get flattened stride of a given var in the first store's indices
  //-----------------------------------------------------------------------
  int64_t GetStrideInStore(const Stmt &body, const VarNode *var) {
    auto get = [&](const BufferStoreNode *store) -> int64_t {
      PrimExpr flat = store->indices[0];
      for (size_t d = 1; d < store->indices.size(); ++d)
        flat = flat * store->buffer->shape[d] + store->indices[d];
      return GetVarStride(flat, var);
    };
    Stmt b = StripAttr(body);
    if (const auto *store = b.as<BufferStoreNode>())
      return get(store);
    if (const auto *seq = b.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (const auto *store = s.as<BufferStoreNode>())
          return get(store);
      }
    }
    return -1;
  }

  //-----------------------------------------------------------------------
  // Index substitution: replace both parallel vars with their new values
  // VReg var → chunk * VL,  outer var → serial loop var
  //-----------------------------------------------------------------------
  PrimExpr SubstituteIndex(const PrimExpr &index) {
    class Replacer : public ExprMutator {
      const VarNode *pv_;
      PrimExpr pr_;
      const VarNode *ov_;
      PrimExpr or_;

    public:
      Replacer(const VarNode *pv, PrimExpr pr, const VarNode *ov, PrimExpr or_)
          : pv_(pv), pr_(pr), ov_(ov), or_(or_) {}
      PrimExpr VisitExpr_(const VarNode *op) override {
        if (pv_ && op == pv_)
          return pr_;
        if (ov_ && op == ov_)
          return or_;
        return ffi::GetRef<PrimExpr>(op);
      }
    };
    PrimExpr par_repl = chunk_var_ * IntImm(DataType::Int(32), vreg_size_);
    PrimExpr outer_repl = outer_parallel_var_
                              ? PrimExpr(outer_serial_var_)
                              : PrimExpr(IntImm(DataType::Int(32), 0));
    return Replacer(parallel_var_, par_repl, outer_parallel_var_,
                    outer_repl)(index);
  }

  PrimExpr MakeAddressOf(const Buffer &buffer,
                         const Array<PrimExpr> &orig_indices) {
    Array<PrimExpr> idx;
    for (const auto &i : orig_indices)
      idx.push_back(SubstituteIndex(i));
    return Call(DataType::Handle(), builtin::address_of(),
                {BufferLoad(buffer, idx)});
  }

  //-----------------------------------------------------------------------
  // Expression decomposition (recursive)
  //-----------------------------------------------------------------------
  Var DecomposeExpr(const PrimExpr &expr, DataType dtype, Array<Stmt> *stmts) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Buffer reg = AllocFragmentReg(dtype);
      stmts->push_back(
          Evaluate(Call(DataType::Void(), ascend_simd_load(),
                        {StringImm(dtype_str_), reg->data,
                         MakeAddressOf(load->buffer, load->indices)})));
      return reg->data;
    }

    auto EmitBinary = [&](const PrimExpr &a, const PrimExpr &b,
                          const Op &op) -> Var {
      Var lhs = DecomposeExpr(a, dtype, stmts);
      Var rhs = DecomposeExpr(b, dtype, stmts);
      Buffer dst = AllocFragmentReg(dtype);
      stmts->push_back(Evaluate(
          Call(DataType::Void(), op, {dst->data, lhs, rhs, mask_var_})));
      return dst->data;
    };
    if (auto *n = expr.as<AddNode>())
      return EmitBinary(n->a, n->b, ascend_simd_add());
    if (auto *n = expr.as<SubNode>())
      return EmitBinary(n->a, n->b, ascend_simd_sub());
    if (auto *n = expr.as<MulNode>())
      return EmitBinary(n->a, n->b, ascend_simd_mul());
    if (auto *n = expr.as<DivNode>())
      return EmitBinary(n->a, n->b, ascend_simd_div());

    auto EmitUnary = [&](const PrimExpr &a, const Op &op) -> Var {
      Var src = DecomposeExpr(a, dtype, stmts);
      Buffer dst = AllocFragmentReg(dtype);
      stmts->push_back(
          Evaluate(Call(DataType::Void(), op, {dst->data, src, mask_var_})));
      return dst->data;
    };
    if (const auto *call = expr.as<CallNode>()) {
      if (const auto *opn = call->op.as<OpNode>()) {
        std::string n = opn->name;
        if (n == "tir.exp" && call->args.size() == 1)
          return EmitUnary(call->args[0], ascend_simd_exp());
        if ((n == "tir.log" || n == "tir.ln") && call->args.size() == 1)
          return EmitUnary(call->args[0], ascend_simd_ln());
        if (n == "tir.sqrt" && call->args.size() == 1)
          return EmitUnary(call->args[0], ascend_simd_sqrt());
        if (n == "tir.rsqrt" && call->args.size() == 1)
          return EmitUnary(call->args[0], ascend_simd_rsqrt());
        if (n == "tir.fabs" && call->args.size() == 1)
          return EmitUnary(call->args[0], ascend_simd_abs());
      }
    }
    if (const auto *sub = expr.as<SubNode>())
      if (is_zero(sub->a))
        return EmitUnary(sub->b, ascend_simd_neg());

    LOG(FATAL) << "AscendSimdVFLowerParallel: unsupported expr: " << expr;
    return Var();
  }

  //-----------------------------------------------------------------------
  // Store lowering
  //-----------------------------------------------------------------------
  Stmt LowerBufferStore(const BufferStoreNode *store, DataType dtype) {
    Array<Stmt> stmts;
    Var result = DecomposeExpr(store->value, dtype, &stmts);
    stmts.push_back(Evaluate(Call(DataType::Void(), ascend_simd_store(),
                                  {StringImm(dtype_str_),
                                   MakeAddressOf(store->buffer, store->indices),
                                   result, mask_var_})));
    return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
  }

  bool CanLowerBody(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (b.as<BufferStoreNode>())
      return true;
    if (auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq)
        if (!st.as<BufferStoreNode>())
          return false;
      return true;
    }
    return false;
  }

  Stmt LowerParallelBody(const Stmt &body, DataType dtype) {
    Stmt b = StripAttr(body);
    if (auto *s = b.as<BufferStoreNode>())
      return LowerBufferStore(s, dtype);
    if (auto *s = b.as<SeqStmtNode>()) {
      Array<Stmt> lowered;
      for (const Stmt &st : s->seq)
        lowered.push_back(LowerBufferStore(st.as<BufferStoreNode>(), dtype));
      return SeqStmt(lowered);
    }
    LOG(FATAL) << "AscendSimdVFLowerParallel: bad parallel body: "
               << b->GetTypeKey();
    return Stmt();
  }

  DataType InferDType(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (auto *s = b.as<BufferStoreNode>())
      return s->buffer->dtype;
    if (auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq)
        if (auto *bs = st.as<BufferStoreNode>())
          return bs->buffer->dtype;
    }
    return DataType::Float(32);
  }

  //-----------------------------------------------------------------------
  // VisitStmt overrides
  //-----------------------------------------------------------------------
  Stmt VisitStmt_(const BlockNode *op) override {
    if (op->name_hint != "SIMD_VF")
      return StmtExprMutator::VisitStmt_(op);

    bool saved = inside_simdvf_;
    inside_simdvf_ = true;
    auto saved_frags = std::move(new_fragments_);
    int saved_cnt = frag_counter_;
    new_fragments_.clear();
    frag_counter_ = 0;

    Stmt new_body = VisitStmt(op->body);

    Array<Buffer> alloc_bufs = op->alloc_buffers;
    for (const Buffer &b : new_fragments_)
      alloc_bufs.push_back(b);

    inside_simdvf_ = saved;
    new_fragments_ = std::move(saved_frags);
    frag_counter_ = saved_cnt;

    if (alloc_bufs.same_as(op->alloc_buffers) && new_body.same_as(op->body))
      return ffi::GetRef<Stmt>(op);

    return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                 op->init, alloc_bufs, op->match_buffers, op->annotations);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (!inside_simdvf_ || op->kind != ForKind::kParallel)
      return StmtExprMutator::VisitStmt_(op);

    // 2D parallel: parallel → [attrs...] → parallel → body
    Stmt stripped = StripAttr(op->body);
    const auto *inner = stripped.as<ForNode>();
    if (inner && inner->kind == ForKind::kParallel) {
      if (auto *t = inner->body.as<ForNode>())
        if (t->kind == ForKind::kParallel)
          LOG(FATAL) << "3D+ parallel not supported";
      return Lower2DParallel(op, inner);
    }

    if (!CanLowerBody(op->body))
      return StmtExprMutator::VisitStmt_(op);
    return Lower1DParallel(op);
  }

  //-----------------------------------------------------------------------
  // 1D Parallel
  //-----------------------------------------------------------------------
  Stmt Lower1DParallel(const ForNode *op) {
    int64_t extent = Downcast<IntImm>(op->extent)->value;
    DataType dtype = InferDType(op->body);
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);
    ICHECK_EQ(extent % vreg_size_, 0)
        << "Extent " << extent << " not divisible by VReg size " << vreg_size_;
    int64_t nchunks = extent / vreg_size_;

    auto saved_pv = parallel_var_, saved_ov = outer_parallel_var_;
    parallel_var_ = op->loop_var.get();
    outer_parallel_var_ = nullptr;

    chunk_var_ = Var("vchunk", DataType::Int(32));
    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    mask_var_ = Var("mask", DataType::Handle());

    PrimExpr im = Call(DataType::Handle(), ascend_init_mask(),
                       {IntImm(DataType::Int(32), extent)});
    PrimExpr um = Call(DataType::Handle(), ascend_simd_update_mask(),
                       {StringImm(dtype_str_), mask_cnt_var_});

    Stmt body = LowerParallelBody(op->body, dtype);
    Stmt ml = LetStmt(mask_var_, um, body);
    Stmt loop = For(chunk_var_, IntImm(DataType::Int(32), 0),
                    IntImm(DataType::Int(32), nchunks), ForKind::kSerial, ml);
    Stmt result = LetStmt(mask_cnt_var_, im, loop);

    parallel_var_ = saved_pv;
    outer_parallel_var_ = saved_ov;
    return result;
  }

  //-----------------------------------------------------------------------
  // 2D Parallel: pick stride=1 var as VReg, other as serial
  //-----------------------------------------------------------------------
  Stmt Lower2DParallel(const ForNode *outer, const ForNode *inner) {
    if (!CanLowerBody(inner->body))
      return StmtExprMutator::VisitStmt_(outer);

    int64_t outer_ext = Downcast<IntImm>(outer->extent)->value;
    int64_t inner_ext = Downcast<IntImm>(inner->extent)->value;

    DataType dtype = InferDType(inner->body);
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);

    // Determine which var has stride=1 → that's the VReg dimension
    int64_t outer_stride = GetStrideInStore(inner->body, outer->loop_var.get());
    int64_t inner_stride = GetStrideInStore(inner->body, inner->loop_var.get());

    const ForNode *vreg_for;
    const ForNode *serial_for;
    int64_t vreg_extent, serial_extent;

    if (inner_stride == 1) {
      vreg_for = inner;
      serial_for = outer;
      vreg_extent = inner_ext;
      serial_extent = outer_ext;
    } else if (outer_stride == 1) {
      vreg_for = outer;
      serial_for = inner;
      vreg_extent = outer_ext;
      serial_extent = inner_ext;
    } else {
      LOG(FATAL) << "2D parallel requires one dimension with stride=1, got "
                 << "outer stride=" << outer_stride
                 << " inner stride=" << inner_stride;
    }

    ICHECK_EQ(vreg_extent % vreg_size_, 0)
        << "VReg extent " << vreg_extent << " not divisible by " << vreg_size_;
    int64_t nchunks = vreg_extent / vreg_size_;

    auto saved_pv = parallel_var_, saved_ov = outer_parallel_var_;
    parallel_var_ = vreg_for->loop_var.get();
    parallel_extent_ = vreg_extent;
    outer_parallel_var_ = serial_for->loop_var.get();
    outer_serial_var_ = Var("vserial", DataType::Int(32));

    chunk_var_ = Var("vchunk", DataType::Int(32));
    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    mask_var_ = Var("mask", DataType::Handle());

    PrimExpr im =
        Call(DataType::Handle(), ascend_init_mask(),
             {IntImm(DataType::Int(32), vreg_extent * serial_extent)});
    PrimExpr um = Call(DataType::Handle(), ascend_simd_update_mask(),
                       {StringImm(dtype_str_), mask_cnt_var_});

    Stmt body = LowerParallelBody(inner->body, dtype);
    Stmt ml = LetStmt(mask_var_, um, body);
    Stmt inner_loop =
        For(chunk_var_, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, ml);
    Stmt outer_loop = For(outer_serial_var_, IntImm(DataType::Int(32), 0),
                          IntImm(DataType::Int(32), serial_extent),
                          ForKind::kSerial, inner_loop);
    Stmt result = LetStmt(mask_cnt_var_, im, outer_loop);

    parallel_var_ = saved_pv;
    outer_parallel_var_ = saved_ov;
    return result;
  }
};

using namespace tir::transform;

tvm::transform::Pass AscendSimdVFLowerParallel() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AscendSimdVFLowerParallel::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendSimdVFLowerParallel", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendSimdVFLowerParallel",
                        AscendSimdVFLowerParallel);
}

} // namespace tl
} // namespace tvm
