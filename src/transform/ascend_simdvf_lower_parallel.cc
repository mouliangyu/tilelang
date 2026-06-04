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
#include "../op/utils.h"

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
  Var mask_var_;

  int frag_counter_{0};
  int64_t elem_bits_{0};
  std::vector<Stmt> scalar_loads_;

  //-----------------------------------------------------------------------
  // VReg / type helpers
  //-----------------------------------------------------------------------
  static int64_t ComputeVRegSize(DataType dtype) {
    return 8 * 32 / (dtype.bits() / 8);
  }

  static int ElemWidthBits(DataType dtype) { return dtype.bits(); }

  //-----------------------------------------------------------------------
  // SIMD register helpers (return-value style)
  //-----------------------------------------------------------------------
  Var MakeReg(DataType dtype) {
    std::string name = "vreg_" + std::to_string(frag_counter_++);
    int vl = ComputeVRegSize(dtype);
    return Var(name, dtype.with_lanes(vl));
  }

  static Stmt FoldLetStmts(const Array<Stmt> &stmts, Stmt body) {
    for (auto it = stmts.rbegin(); it != stmts.rend(); ++it) {
      Stmt s = *it;
      if (const auto *let = s.as<LetStmtNode>()) {
        body = LetStmt(let->var, let->value, body);
      } else {
        body = SeqStmt({s, body});
      }
    }
    return body;
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
  // Expression decomposition (recursive) — return-value style
  //-----------------------------------------------------------------------
  Var DecomposeExpr(const PrimExpr &expr, DataType dtype, Array<Stmt> *stmts) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Var reg = MakeReg(dtype);

      // Check if any index depends on the VReg variable
      bool depends_on_vreg = false;
      if (parallel_var_) {
        for (const auto &idx : load->indices) {
          if (GetVarStride(idx, parallel_var_) != 0) {
            depends_on_vreg = true;
            break;
          }
        }
      }

      if (!depends_on_vreg && outer_parallel_var_) {
        // Scalar load in 2D context: hoist outside chunk loop
        PrimExpr addr = MakeAddressOf(load->buffer, load->indices);
        PrimExpr vld_call =
            Call(reg->dtype, simd_vld(), {addr, StringImm("BRC_B32")});
        scalar_loads_.push_back(
            LetStmt(reg, vld_call, Evaluate(IntImm(DataType::Int(32), 0))));
      } else if (!depends_on_vreg) {
        // Scalar load in 1D context: inline
        PrimExpr addr = MakeAddressOf(load->buffer, load->indices);
        stmts->push_back(LetStmt(
            reg, Call(reg->dtype, simd_vld(), {addr, StringImm("BRC_B32")}),
            Evaluate(IntImm(DataType::Int(32), 0))));
      } else {
        // Vector load
        stmts->push_back(
            LetStmt(reg,
                    Call(reg->dtype, simd_vld(),
                         {MakeAddressOf(load->buffer, load->indices),
                          StringImm("NORM")}),
                    Evaluate(IntImm(DataType::Int(32), 0))));
      }
      return reg;
    }

    auto EmitBinary = [&](const PrimExpr &a, const PrimExpr &b,
                          const Op &op) -> Var {
      Var lhs = DecomposeExpr(a, dtype, stmts);
      Var rhs = DecomposeExpr(b, dtype, stmts);
      Var dst = MakeReg(dtype);
      stmts->push_back(
          LetStmt(dst,
                  Call(dst->dtype, op,
                       {lhs, rhs, mask_var_, StringImm("MODE_ZEROING")}),
                  Evaluate(IntImm(DataType::Int(32), 0))));
      return dst;
    };
    if (auto *n = expr.as<AddNode>())
      return EmitBinary(n->a, n->b, simd_vadd());
    if (auto *n = expr.as<SubNode>())
      return EmitBinary(n->a, n->b, simd_vsub());
    if (auto *n = expr.as<MulNode>())
      return EmitBinary(n->a, n->b, simd_vmul());
    if (auto *n = expr.as<DivNode>())
      return EmitBinary(n->a, n->b, simd_vdiv());

    auto EmitUnary = [&](const PrimExpr &a, const Op &op) -> Var {
      Var src = DecomposeExpr(a, dtype, stmts);
      Var dst = MakeReg(dtype);
      stmts->push_back(LetStmt(
          dst,
          Call(dst->dtype, op, {src, mask_var_, StringImm("MODE_ZEROING")}),
          Evaluate(IntImm(DataType::Int(32), 0))));
      return dst;
    };
    if (const auto *call = expr.as<CallNode>()) {
      if (const auto *opn = call->op.as<OpNode>()) {
        std::string n = opn->name;
        if (n == "tir.exp" && call->args.size() == 1)
          return EmitUnary(call->args[0], simd_vexp());
        if ((n == "tir.log" || n == "tir.ln") && call->args.size() == 1)
          return EmitUnary(call->args[0], simd_vln());
        if (n == "tir.sqrt" && call->args.size() == 1)
          return EmitUnary(call->args[0], simd_vsqrt());
        if (n == "tir.rsqrt" && call->args.size() == 1)
          return EmitUnary(call->args[0], simd_vrsqrt());
        if (n == "tir.fabs" && call->args.size() == 1)
          return EmitUnary(call->args[0], simd_vabs());
      }
    }
    if (const auto *sub = expr.as<SubNode>())
      if (is_zero(sub->a))
        return EmitUnary(sub->b, simd_vneg());

    // Scalar constants: broadcast via vdup
    if (expr.as<FloatImmNode>() || expr.as<IntImmNode>()) {
      Var reg = MakeReg(dtype);
      stmts->push_back(
          LetStmt(reg,
                  Call(reg->dtype, simd_vdup(),
                       {expr, mask_var_, StringImm("MODE_ZEROING")}),
                  Evaluate(IntImm(DataType::Int(32), 0))));
      return reg;
    }

    LOG(FATAL) << "AscendSimdVFLowerParallel: unsupported expr: " << expr;
    return Var();
  }

  //-----------------------------------------------------------------------
  // Store lowering
  //-----------------------------------------------------------------------
  Stmt LowerBufferStore(const BufferStoreNode *store, DataType dtype) {
    Array<Stmt> stmts;
    Var result = DecomposeExpr(store->value, dtype, &stmts);
    Stmt vsts_stmt = Evaluate(Call(
        DataType::Void(), simd_vsts(),
        {MakeAddressOf(store->buffer, store->indices), result, mask_var_}));
    return FoldLetStmts(stmts, vsts_stmt);
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
    int saved_cnt = frag_counter_;
    frag_counter_ = 0;

    Stmt new_body = VisitStmt(op->body);

    inside_simdvf_ = saved;
    frag_counter_ = saved_cnt;

    if (new_body.same_as(op->body))
      return ffi::GetRef<Stmt>(op);

    return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                 op->init, op->alloc_buffers, op->match_buffers,
                 op->annotations);
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

      if (!CanLowerBody(inner->body))
        return StmtExprMutator::VisitStmt_(op);

      // Pick stride=1 dim as VReg, other as serial
      int64_t outer_stride = GetStrideInStore(inner->body, op->loop_var.get());
      int64_t inner_stride =
          GetStrideInStore(inner->body, inner->loop_var.get());

      if (inner_stride == 1)
        return LowerParallel(inner, op);
      if (outer_stride == 1)
        return LowerParallel(op, inner);

      LOG(FATAL) << "2D parallel requires one dimension with stride=1, got "
                 << "outer stride=" << outer_stride
                 << " inner stride=" << inner_stride;
    }

    // 1D parallel
    if (!CanLowerBody(op->body))
      return StmtExprMutator::VisitStmt_(op);
    return LowerParallel(op, nullptr);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (!inside_simdvf_)
      return StmtExprMutator::VisitStmt_(op);

    const auto *call = op->value.as<CallNode>();
    if (!call)
      return StmtExprMutator::VisitStmt_(op);

    // Check if it's a tl.tileop.fill or tl.tileop.copy call
    if (const auto *opnode = call->op.as<OpNode>()) {
      if (opnode->name == "tl.tileop.fill")
        return LowerFill(call);
      if (opnode->name == "tl.tileop.copy")
        return LowerCopy(call);
      if (opnode->name == "tl.tileop.reduce")
        return LowerReduce(call);
      return StmtExprMutator::VisitStmt_(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  //-----------------------------------------------------------------------
  // Fill lowering: T.fill(buf, value) → Duplicate + StoreAlign loop
  //-----------------------------------------------------------------------
  Stmt LowerFill(const CallNode *call) {
    auto saved_frag_counter = frag_counter_;
    frag_counter_ = 0;
    BufferRegion buf_region = NormalizeToBufferRegion(call->args[0]);
    Buffer buf = buf_region->buffer;
    Array<Range> ranges = buf_region->region;
    PrimExpr value = call->args[1];
    DataType dtype = buf->dtype;

    if (value->dtype != dtype)
      value = Cast(dtype, value);

    int ndim = ranges.size();
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();

    bool is_full = true;
    int64_t total = 1;
    for (int i = 0; i < ndim; i++) {
      const auto *min_imm = ranges[i]->min.as<IntImmNode>();
      const auto *ext_imm = ranges[i]->extent.as<IntImmNode>();
      const auto *shape_imm = buf->shape[i].as<IntImmNode>();
      if (!min_imm || !ext_imm || !shape_imm) {
        is_full = false;
        break;
      }
      if (min_imm->value != 0 || ext_imm->value != shape_imm->value) {
        is_full = false;
        break;
      }
      total *= ext_imm->value;
    }

    ICHECK(is_full) << "T.fill inside SimdVF requires full-buffer regions";
    ICHECK_EQ(total % vreg_size_, 0)
        << "T.fill inside SimdVF requires extent divisible by VReg size ("
        << vreg_size_ << "), got " << total;

    int64_t nchunks = total / vreg_size_;

    // One fragment holds the broadcast value
    Var frag = MakeReg(dtype);

    // CreateMask<dtype>()
    Var mask_var("mask", DataType::Bool(256));
    PrimExpr create_mask =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});

    Var chunk_var("vchunk", DataType::Int(32));
    PrimExpr flat_idx = chunk_var * IntImm(DataType::Int(32), vreg_size_);

    Array<PrimExpr> idxs;
    if (ndim == 1) {
      idxs.push_back(flat_idx);
    } else {
      PrimExpr remaining = flat_idx;
      for (int i = ndim - 1; i >= 1; i--) {
        idxs.insert(idxs.begin(), FloorMod(remaining, buf->shape[i]));
        remaining = FloorDiv(remaining, buf->shape[i]);
      }
      idxs.insert(idxs.begin(), remaining);
    }

    PrimExpr addr = Call(DataType::Handle(), builtin::address_of(),
                         {BufferLoad(buf, idxs)});
    Stmt store =
        Evaluate(Call(DataType::Void(), simd_vsts(), {addr, frag, mask_var}));

    Stmt loop =
        For(chunk_var, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, store);

    // Bind frag via vdup in LetStmt
    PrimExpr vdup_call = Call(frag->dtype, simd_vdup(),
                              {value, mask_var, StringImm("MODE_ZEROING")});
    Stmt result =
        LetStmt(mask_var, create_mask, LetStmt(frag, vdup_call, loop));
    frag_counter_ = saved_frag_counter;
    return result;
  }

  //-----------------------------------------------------------------------
  // Copy lowering: T.copy(src, dst) → LoadAlign + StoreAlign loop
  //-----------------------------------------------------------------------
  Stmt LowerCopy(const CallNode *call) {
    auto saved_frag_counter = frag_counter_;
    frag_counter_ = 0;
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src_buf = src_region->buffer;
    Buffer dst_buf = dst_region->buffer;
    Array<Range> src_ranges = src_region->region;
    Array<Range> dst_ranges = dst_region->region;
    DataType src_dtype = src_buf->dtype;
    DataType dst_dtype = dst_buf->dtype;
    bool is_cast = (dst_dtype != src_dtype);

    ICHECK(!is_cast || (src_dtype.is_float() &&
                        (dst_dtype.is_float16() || dst_dtype.is_bfloat16())))
        << "T.copy inside SimdVF: cast copy only supports f32→bf16 or f32→half";

    vreg_size_ = ComputeVRegSize(src_dtype);
    int src_elem_bits = src_dtype.bits();
    int dst_elem_bits = dst_dtype.bits();

    auto is_full = [](const Array<Range> &ranges, const Array<PrimExpr> &shape,
                      int64_t *total) -> bool {
      *total = 1;
      for (size_t i = 0; i < ranges.size(); i++) {
        const auto *mi = ranges[i]->min.as<IntImmNode>();
        const auto *ei = ranges[i]->extent.as<IntImmNode>();
        const auto *si = shape[i].as<IntImmNode>();
        if (!mi || !ei || !si)
          return false;
        if (mi->value != 0 || ei->value != si->value)
          return false;
        *total *= ei->value;
      }
      return true;
    };

    int64_t src_total = 0, dst_total = 0;
    ICHECK(is_full(src_ranges, src_buf->shape, &src_total) &&
           is_full(dst_ranges, dst_buf->shape, &dst_total))
        << "T.copy inside SimdVF requires full-buffer regions";

    ICHECK(src_total == dst_total) << "T.copy inside SimdVF requires src and "
                                      "dst to have same element count";

    int64_t total = src_total;
    ICHECK(total % vreg_size_ == 0) << "T.copy inside SimdVF requires total "
                                       "elements divisible by VReg size ("
                                    << vreg_size_ << "), got " << total;

    int64_t nchunks = total / vreg_size_;

    Var load_frag = MakeReg(src_dtype);
    Var cast_frag;
    if (is_cast)
      cast_frag = MakeReg(dst_dtype);

    Var chunk_var("vchunk", DataType::Int(32));
    mask_var_ = Var("mask", DataType::Bool(256));

    int mask_elem_bits = is_cast ? dst_elem_bits : src_elem_bits;
    PrimExpr mask_val =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), mask_elem_bits), StringImm("PAT_ALL")});

    auto flat_addr = [&](const Buffer &buf, int ndim) -> PrimExpr {
      PrimExpr flat_idx = chunk_var * IntImm(DataType::Int(32), vreg_size_);
      Array<PrimExpr> idxs;
      if (ndim == 1) {
        idxs.push_back(flat_idx);
      } else {
        PrimExpr r = flat_idx;
        for (int i = ndim - 1; i >= 1; i--) {
          idxs.insert(idxs.begin(), FloorMod(r, buf->shape[i]));
          r = FloorDiv(r, buf->shape[i]);
        }
        idxs.insert(idxs.begin(), r);
      }
      return Call(DataType::Handle(), builtin::address_of(),
                  {BufferLoad(buf, idxs)});
    };

    PrimExpr src_addr = flat_addr(src_buf, src_ranges.size());
    PrimExpr dst_addr = flat_addr(dst_buf, dst_ranges.size());

    // Load from source: LetStmt(load_frag, vld(addr), ...)
    PrimExpr vld_call =
        Call(load_frag->dtype, simd_vld(), {src_addr, StringImm("NORM")});

    Stmt store;
    if (is_cast) {
      // Cast + store: LetStmt(cast_frag, vcvt(load_frag, mask, ...), store)
      PrimExpr vcvt_call = Call(cast_frag->dtype, simd_vcvt(),
                                {load_frag, mask_var_, StringImm("ROUND_R"),
                                 StringImm("RS_ENABLE"), StringImm("PART_EVEN"),
                                 StringImm("MODE_ZEROING")});
      store = Evaluate(Call(DataType::Void(), simd_vsts(),
                            {dst_addr, cast_frag, mask_var_}));
      Stmt body = LetStmt(cast_frag, vcvt_call, store);
      body = LetStmt(load_frag, vld_call, body);
      Stmt loop =
          For(chunk_var, IntImm(DataType::Int(32), 0),
              IntImm(DataType::Int(32), nchunks), ForKind::kSerial, body);
      Stmt result = LetStmt(mask_var_, mask_val, loop);
      frag_counter_ = saved_frag_counter;
      return result;
    } else {
      store = Evaluate(Call(DataType::Void(), simd_vsts(),
                            {dst_addr, load_frag, mask_var_}));
      Stmt body = LetStmt(load_frag, vld_call, store);
      Stmt loop =
          For(chunk_var, IntImm(DataType::Int(32), 0),
              IntImm(DataType::Int(32), nchunks), ForKind::kSerial, body);
      Stmt result = LetStmt(mask_var_, mask_val, loop);
      frag_counter_ = saved_frag_counter;
      return result;
    }
  }

  //-----------------------------------------------------------------------
  // Reduce lowering: T.reduce_max/sum(src, dst, dim)
  //
  // dim=0: reduce across rows → element-wise Max loop, no Reduce<MAX>
  // dim=1: reduce across columns → Max per chunk, then Reduce<MAX> final
  //-----------------------------------------------------------------------
  Stmt LowerReduce(const CallNode *call) {
    auto saved_frag_counter = frag_counter_;
    frag_counter_ = 0;
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src = src_region->buffer, dst = dst_region->buffer;
    Array<Range> sr = src_region->region, dr = dst_region->region;

    std::string rtype = call->args[2].as<StringImmNode>()->value;
    int dim = (int)call->args[3].as<IntImmNode>()->value;
    bool clear = call->args[4].as<Bool>().value();

    DataType dtype = src->dtype;
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();

    ICHECK(sr.size() == 2 && dr.size() == 1 && (dim == 0 || dim == 1))
        << "T.reduce inside SimdVF requires 2D src → 1D dst, dim=0 or 1";

    auto is_full = [](const Array<Range> &r, const Array<PrimExpr> &sh,
                      int64_t *t) -> bool {
      *t = 1;
      for (size_t i = 0; i < r.size(); i++) {
        auto *mi = r[i]->min.as<IntImmNode>();
        auto *ei = r[i]->extent.as<IntImmNode>();
        auto *si = sh[i].as<IntImmNode>();
        if (!mi || !ei || !si || mi->value || ei->value != si->value)
          return false;
        *t *= ei->value;
      }
      return true;
    };
    int64_t st, dt;
    ICHECK(is_full(sr, src->shape, &st) && is_full(dr, dst->shape, &dt))
        << "T.reduce inside SimdVF requires full-buffer regions";

    int64_t de = Downcast<IntImm>(sr[dim]->extent)->value;
    int64_t oe = Downcast<IntImm>(sr[1 - dim]->extent)->value;
    ICHECK_EQ(de % vreg_size_, 0)
        << "T.reduce requires dim extent multiple of VReg (" << vreg_size_
        << "), got " << de;
    int64_t nc = de / vreg_size_;

    const Op &eop = (rtype == "sum") ? simd_vadd() : simd_vmax();
    const Op &rop = (rtype == "sum") ? simd_vcadd() : simd_vcmax();
    PrimExpr ival = (rtype == "sum") ? PrimExpr(FloatImm(dtype, 0.0))
                                     : PrimExpr(FloatImm(dtype, -3.402823e+38));

    auto NewA = [&]() { return MakeReg(dtype); };
    auto NewB = [&]() { return MakeReg(dtype); };

    PrimExpr mask_val =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});

    mask_var_ = Var("mask", DataType::Bool(256));

    auto i32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };
    auto a1d = [&](const Buffer &buf, const PrimExpr &i0) {
      return Call(DataType::Handle(), builtin::address_of(),
                  {BufferLoad(buf, {i0})});
    };
    auto a2d = [&](const Buffer &buf, const PrimExpr &i0, const PrimExpr &i1) {
      return Call(DataType::Handle(), builtin::address_of(),
                  {BufferLoad(buf, {i0, i1})});
    };
    Var rep("vrepeat", DataType::Int(32));

    // === dim=1, single VReg: ReduceVF ===
    if (dim == 1 && nc == 1) {
      Var a = NewA();
      Var b = NewB();
      // a = vld(src[rep, 0])
      Stmt L = LetStmt(a,
                       Call(a->dtype, simd_vld(),
                            {a2d(src, rep, i32(0)), StringImm("NORM")}),
                       Evaluate(i32(0)));
      // a2 = rop(a, mask)
      Var a2 = NewA();
      Stmt R = LetStmt(
          a2, Call(a2->dtype, rop, {a, mask_var_, StringImm("MODE_ZEROING")}),
          Evaluate(i32(0)));
      // S = vsts(dst[rep*VL], a2, mask)
      Stmt S = Evaluate(Call(DataType::Void(), simd_vsts(),
                             {a1d(dst, rep * i32(vreg_size_)), a2, mask_var_}));

      Stmt row_body;
      if (!clear) {
        // b = vld(dst[rep*VL], BRC_B32)
        Stmt LP = LetStmt(
            b,
            Call(b->dtype, simd_vld(),
                 {a1d(dst, rep * i32(vreg_size_)), StringImm("BRC_B32")}),
            Evaluate(i32(0)));
        // a3 = eop(a2, b, mask)
        Var a3 = NewA();
        Stmt MP = LetStmt(
            a3,
            Call(a3->dtype, eop, {a2, b, mask_var_, StringImm("MODE_ZEROING")}),
            Evaluate(i32(0)));
        // S3 = vsts(dst[rep*VL], a3, mask)
        Stmt S3 =
            Evaluate(Call(DataType::Void(), simd_vsts(),
                          {a1d(dst, rep * i32(vreg_size_)), a3, mask_var_}));
        row_body = FoldLetStmts({L, R, LP, MP, S3}, S3);
        row_body =
            LetStmt(mask_var_, mask_val, FoldLetStmts({L, R, LP, MP}, S3));
      } else {
        row_body = LetStmt(mask_var_, mask_val, FoldLetStmts({L, R}, S));
      }
      Stmt result = For(rep, i32(0), i32(oe), ForKind::kSerial, row_body);
      frag_counter_ = saved_frag_counter;
      return result;
    }

    // === dim=1, multi-VReg: per-chunk Reduce + accumulate ===
    if (dim == 1) {
      Var a = NewA();
      Var b = NewB();
      // a = vdup(ival, mask)
      Stmt DI = LetStmt(a,
                        Call(a->dtype, simd_vdup(),
                             {ival, mask_var_, StringImm("MODE_ZEROING")}),
                        Evaluate(i32(0)));
      Var ck("vchunk", DataType::Int(32));
      PrimExpr co = ck * i32(vreg_size_);
      // b = vld(src[rep, co])
      Stmt LK = LetStmt(
          b, Call(b->dtype, simd_vld(), {a2d(src, rep, co), StringImm("NORM")}),
          Evaluate(i32(0)));
      // b2 = rop(b, mask)
      Var b2 = NewB();
      Stmt RK = LetStmt(
          b2, Call(b2->dtype, rop, {b, mask_var_, StringImm("MODE_ZEROING")}),
          Evaluate(i32(0)));
      // a2 = eop(a, b2, mask)
      Var a2 = NewA();
      Stmt MA = LetStmt(
          a2,
          Call(a2->dtype, eop, {a, b2, mask_var_, StringImm("MODE_ZEROING")}),
          Evaluate(i32(0)));
      Stmt CB = LetStmt(mask_var_, mask_val,
                        FoldLetStmts({LK, RK, MA}, Evaluate(i32(0))));
      // Rebind a for the outer scope
      a = a2;

      // Final store: vsts(dst[rep*VL], a, mask)
      Stmt S = Evaluate(Call(DataType::Void(), simd_vsts(),
                             {a1d(dst, rep * i32(vreg_size_)), a, mask_var_}));

      PrimExpr mask2_val =
          Call(DataType::Bool(256), simd_pset(),
               {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});
      Var mask2("mask2", DataType::Bool(256));

      Array<Stmt> rb = {DI, For(ck, i32(0), i32(nc), ForKind::kSerial, CB)};
      // The for loop rebinds `a` each iteration; final `a` holds last
      // iteration's result. Need to use the loop-produced `a` for final store.
      // Since LetStmt scoping makes this hard, use AllocateNode for
      // accumulator. For now, handle clear vs non-clear with fresh bindings.

      if (!clear) {
        Var b3 = NewB();
        Stmt LP = LetStmt(
            b3,
            Call(b3->dtype, simd_vld(),
                 {a1d(dst, rep * i32(vreg_size_)), StringImm("BRC_B32")}),
            Evaluate(i32(0)));
        Var a3 = NewA();
        Stmt MP = LetStmt(
            a3,
            Call(a3->dtype, eop, {a2, b3, mask2, StringImm("MODE_ZEROING")}),
            Evaluate(i32(0)));
        Stmt store2 =
            Evaluate(Call(DataType::Void(), simd_vsts(),
                          {a1d(dst, rep * i32(vreg_size_)), a3, mask2}));
        rb.push_back(LetStmt(mask2, mask2_val, FoldLetStmts({LP, MP}, store2)));
      } else {
        Stmt store2 =
            Evaluate(Call(DataType::Void(), simd_vsts(),
                          {a1d(dst, rep * i32(vreg_size_)), a2, mask2}));
        rb.push_back(LetStmt(mask2, mask2_val, store2));
      }
      Stmt result = For(rep, i32(0), i32(oe), ForKind::kSerial,
                        FoldLetStmts(rb, Evaluate(i32(0))));
      frag_counter_ = saved_frag_counter;
      return result;
    }

    // === dim=0: rows NOT contiguous → element-wise Max per column chunk ===
    Var ck("vchunk", DataType::Int(32)), rv("vrow", DataType::Int(32));
    int64_t N = Downcast<IntImm>(src->shape[1])->value;
    PrimExpr co = ck * i32(vreg_size_);
    Var a = NewA();
    Var b = NewB();

    // a = vld(src[0, co])
    Stmt L0 = LetStmt(
        a,
        Call(a->dtype, simd_vld(), {a2d(src, i32(0), co), StringImm("NORM")}),
        Evaluate(i32(0)));
    // b = vld(src[rv, co])
    Stmt LK = LetStmt(
        b, Call(b->dtype, simd_vld(), {a2d(src, rv, co), StringImm("NORM")}),
        Evaluate(i32(0)));
    // a2 = eop(a, b, mask)
    Var a2 = NewA();
    Stmt EA = LetStmt(
        a2, Call(a2->dtype, eop, {a, b, mask_var_, StringImm("MODE_ZEROING")}),
        Evaluate(i32(0)));
    // vsts(dst[co], a2, mask)
    Stmt SK = Evaluate(
        Call(DataType::Void(), simd_vsts(), {a1d(dst, co), a2, mask_var_}));

    Array<Stmt> cb;
    if (oe == 1)
      cb.push_back(L0);
    else {
      cb.push_back(LetStmt(mask_var_, mask_val, L0));
      cb.push_back(
          For(rv, i32(1), i32(oe), ForKind::kSerial,
              LetStmt(mask_var_, mask_val, FoldLetStmts({LK, EA}, SK))));
    }
    cb.push_back(LetStmt(mask_var_, mask_val, SK));
    Stmt result =
        For(ck, i32(0), i32(nc), ForKind::kSerial, FoldLetStmts(cb, SK));
    frag_counter_ = saved_frag_counter;
    return result;
  }

  //-----------------------------------------------------------------------
  // LowerParallel: unified 1D / 2D parallel lowering
  //
  // vreg_for  — the parallel loop whose extent is the VReg dimension
  // serial_for — (nullable) the outer serial loop for 2D parallel
  //-----------------------------------------------------------------------
  Stmt LowerParallel(const ForNode *vreg_for, const ForNode *serial_for) {
    auto saved_frag_counter = frag_counter_;
    frag_counter_ = 0;
    int64_t vreg_ext = Downcast<IntImm>(vreg_for->extent)->value;
    int64_t serial_ext =
        serial_for ? Downcast<IntImm>(serial_for->extent)->value : 1;

    DataType dtype = InferDType(vreg_for->body);
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();
    ICHECK_EQ(vreg_ext % vreg_size_, 0)
        << "T.Parallel inside SimdVF requires VReg extent divisible by VReg "
           "size ("
        << vreg_size_ << "), got " << vreg_ext;
    int64_t nchunks = vreg_ext / vreg_size_;

    auto saved_pv = parallel_var_, saved_ov = outer_parallel_var_;
    parallel_var_ = vreg_for->loop_var.get();
    parallel_extent_ = vreg_ext;
    if (serial_for) {
      outer_parallel_var_ = serial_for->loop_var.get();
      outer_serial_var_ = Var("vserial", DataType::Int(32));
    } else {
      outer_parallel_var_ = nullptr;
    }

    chunk_var_ = Var("vchunk", DataType::Int(32));
    mask_var_ = Var("mask", DataType::Bool(256));
    scalar_loads_.clear();

    PrimExpr mask_val =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});
    const PrimExpr &um = mask_val;

    Stmt body = LowerParallelBody(vreg_for->body, dtype);
    Stmt ml = LetStmt(mask_var_, um, body);
    Stmt inner_loop =
        For(chunk_var_, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, ml);

    // Per-iteration body: scalar loads + VReg chunk loop
    Stmt per_iter_body = inner_loop;
    if (!scalar_loads_.empty()) {
      // Fold scalar loads into LetStmt chain around per_iter_body
      for (auto it = scalar_loads_.rbegin(); it != scalar_loads_.rend(); ++it) {
        if (const auto *let = it->as<LetStmtNode>()) {
          per_iter_body = LetStmt(let->var, let->value, per_iter_body);
        } else {
          per_iter_body = SeqStmt({*it, per_iter_body});
        }
      }
      scalar_loads_.clear();
    }

    Stmt result;
    if (serial_for) {
      result = For(outer_serial_var_, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), serial_ext), ForKind::kSerial,
                   per_iter_body);
    } else {
      result = per_iter_body;
    }

    parallel_var_ = saved_pv;
    outer_parallel_var_ = saved_ov;
    frag_counter_ = saved_frag_counter;
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
