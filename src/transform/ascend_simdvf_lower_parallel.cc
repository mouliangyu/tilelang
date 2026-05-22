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
  Var mask_cnt_var_;
  Var mask_var_;

  std::vector<Buffer> new_fragments_;
  int frag_counter_{0};
  std::string dtype_str_;
  std::vector<Stmt> scalar_loads_;

  //-----------------------------------------------------------------------
  // VReg / type helpers
  //-----------------------------------------------------------------------
  static int64_t ComputeVRegSize(DataType dtype) {
    return 8 * 32 / (dtype.bits() / 8);
  }

  static std::string DTypeToAscendStr(DataType dtype) {
    if (dtype.is_bfloat16())
      return "bfloat16_t";
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
        // Scalar load in 2D context: use load_single to broadcast
        // Put the load in scalar_loads_ to be hoisted outside the VReg chunk
        // loop
        PrimExpr addr = MakeAddressOf(load->buffer, load->indices);
        scalar_loads_.push_back(
            Evaluate(Call(DataType::Void(), ascend_simd_load_single(),
                          {StringImm(dtype_str_), reg->data, addr})));
      } else if (!depends_on_vreg) {
        // Scalar load in 1D context: use load_single in-place
        PrimExpr addr = MakeAddressOf(load->buffer, load->indices);
        stmts->push_back(
            Evaluate(Call(DataType::Void(), ascend_simd_load_single(),
                          {StringImm(dtype_str_), reg->data, addr})));
      } else {
        // Vector load: full LoadAlign
        stmts->push_back(
            Evaluate(Call(DataType::Void(), ascend_simd_load(),
                          {StringImm(dtype_str_), reg->data,
                           MakeAddressOf(load->buffer, load->indices)})));
      }
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

    // Scalar constants: broadcast via Duplicate
    if (expr.as<FloatImmNode>() || expr.as<IntImmNode>()) {
      Buffer reg = AllocFragmentReg(dtype);
      stmts->push_back(Evaluate(
          Call(DataType::Void(), ascend_simd_dup(), {reg->data, expr})));
      return reg->data;
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
    BufferRegion buf_region = NormalizeToBufferRegion(call->args[0]);
    Buffer buf = buf_region->buffer;
    Array<Range> ranges = buf_region->region;
    PrimExpr value = call->args[1];
    DataType dtype = buf->dtype;

    if (value->dtype != dtype)
      value = Cast(dtype, value);

    int ndim = ranges.size();
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);

    // Check if this is a full-buffer fill (all mins=0, all extents=buf shape).
    // For full-buffer fills the elements are always contiguous in row-major.
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
    Buffer frag = AllocFragmentReg(dtype);

    // CreateMask<dtype>()
    Var mask_var("mask", DataType::Handle());
    PrimExpr create_mask = Call(DataType::Handle(), ascend_simd_create_mask(),
                                {StringImm(dtype_str_)});

    // Duplicate(frag, scalar)
    Stmt dup_stmt = Evaluate(
        Call(DataType::Void(), ascend_simd_dup(), {frag->data, value}));

    // StoreAlign<dtype>(buf + i*VL, frag, mask)
    // Use flat 1D addressing: index = chunk * VL + 0 (base=0 for full-buffer
    // fill)
    Var chunk_var("vchunk", DataType::Int(32));
    PrimExpr flat_idx = chunk_var * IntImm(DataType::Int(32), vreg_size_);

    // Build per-dimension indices from flat_index
    // For ND buffer, flat_index = i0*N1*N2*.. + i1*N2*.. + ... + i_{d-1}
    // Since we're using base=0, we can just use flat index for 1D
    // For ND, convert flat → multi-dim indices
    Array<PrimExpr> idxs;
    if (ndim == 1) {
      idxs.push_back(flat_idx);
    } else {
      // Convert flat_index back to multi-dim indices for the BufferLoad
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
        Evaluate(Call(DataType::Void(), ascend_simd_store(),
                      {StringImm(dtype_str_), addr, frag->data, mask_var}));

    Stmt loop =
        For(chunk_var, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, store);

    return LetStmt(mask_var, create_mask, SeqStmt({dup_stmt, loop}));
  }

  //-----------------------------------------------------------------------
  // Copy lowering: T.copy(src, dst) → LoadAlign + StoreAlign loop
  //-----------------------------------------------------------------------
  Stmt LowerCopy(const CallNode *call) {
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src_buf = src_region->buffer;
    Buffer dst_buf = dst_region->buffer;
    Array<Range> src_ranges = src_region->region;
    Array<Range> dst_ranges = dst_region->region;
    DataType src_dtype = src_buf->dtype;
    DataType dst_dtype = dst_buf->dtype;
    bool is_cast = (dst_dtype != src_dtype);

    // For cast copy, support f32→bf16 and f32→half (32→16 bit)
    ICHECK(!is_cast || (src_dtype.is_float() &&
                        (dst_dtype.is_float16() || dst_dtype.is_bfloat16())))
        << "T.copy inside SimdVF: cast copy only supports f32→bf16 or f32→half";

    vreg_size_ = ComputeVRegSize(src_dtype);
    std::string src_dtype_str = DTypeToAscendStr(src_dtype);
    std::string dst_dtype_str = DTypeToAscendStr(dst_dtype);

    // Validate both are full-buffer regions with same element count
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

    // Load fragment (src dtype), cast fragment (dst dtype for cast copies)
    Buffer load_frag = AllocFragmentReg(src_dtype);
    Buffer cast_frag;
    if (is_cast)
      cast_frag = AllocFragmentReg(dst_dtype);

    Var chunk_var("vchunk", DataType::Int(32));
    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    mask_var_ = Var("mask", DataType::Handle());

    PrimExpr im = Call(DataType::Handle(), ascend_init_mask(),
                       {IntImm(DataType::Int(32), total)});
    PrimExpr um = Call(
        DataType::Handle(), ascend_simd_update_mask(),
        {StringImm(is_cast ? dst_dtype_str : src_dtype_str), mask_cnt_var_});

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

    // LoadAlign from source
    Stmt load =
        Evaluate(Call(DataType::Void(), ascend_simd_load(),
                      {StringImm(src_dtype_str), load_frag->data, src_addr}));

    Stmt store;
    Array<Stmt> body_stmts;
    body_stmts.push_back(load);

    if (is_cast) {
      // Cast f32→bf16, then StoreAlign with DIST_PACK_B32
      Stmt cast =
          Evaluate(Call(DataType::Void(), ascend_simd_cast(),
                        {StringImm(dst_dtype_str), StringImm(src_dtype_str),
                         cast_frag->data, load_frag->data, mask_var_}));
      body_stmts.push_back(cast);
      store = Evaluate(Call(
          DataType::Void(), ascend_simd_store_pack(),
          {StringImm(dst_dtype_str), dst_addr, cast_frag->data, mask_var_}));
    } else {
      store = Evaluate(Call(
          DataType::Void(), ascend_simd_store(),
          {StringImm(src_dtype_str), dst_addr, load_frag->data, mask_var_}));
    }
    body_stmts.push_back(store);

    Stmt body = SeqStmt(body_stmts);
    Stmt ml = LetStmt(mask_var_, um, body);
    Stmt loop = For(chunk_var, IntImm(DataType::Int(32), 0),
                    IntImm(DataType::Int(32), nchunks), ForKind::kSerial, ml);

    return LetStmt(mask_cnt_var_, im, loop);
  }

  //-----------------------------------------------------------------------
  // Reduce lowering: T.reduce_max/sum(src, dst, dim)
  //
  // dim=0: reduce across rows → element-wise Max loop, no Reduce<MAX>
  // dim=1: reduce across columns → Max per chunk, then Reduce<MAX> final
  //-----------------------------------------------------------------------
  Stmt LowerReduce(const CallNode *call) {
    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src = src_region->buffer, dst = dst_region->buffer;
    Array<Range> sr = src_region->region, dr = dst_region->region;

    std::string rtype = call->args[2].as<StringImmNode>()->value;
    int dim = (int)call->args[3].as<IntImmNode>()->value;
    bool clear = call->args[4].as<Bool>().value();

    DataType dtype = src->dtype;
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);

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

    const Op &eop = (rtype == "sum") ? ascend_simd_add() : ascend_simd_max();
    const Op &rop =
        (rtype == "sum") ? ascend_simd_reduce_sum() : ascend_simd_reduce_max();
    PrimExpr ival = (rtype == "sum") ? PrimExpr(FloatImm(dtype, 0.0))
                                     : PrimExpr(FloatImm(dtype, -3.402823e+38));

    Buffer a = AllocFragmentReg(dtype), b = AllocFragmentReg(dtype);

    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    PrimExpr mi = Call(DataType::Handle(), ascend_init_mask(),
                       {IntImm(DataType::Int(32), de)});
    PrimExpr mu = Call(DataType::Handle(), ascend_simd_update_mask(),
                       {StringImm(dtype_str_), mask_cnt_var_});
    mask_var_ = Var("mask", DataType::Handle());

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
    // Store at VReg-aligned address (rep * vreg_size), full mask → 32B aligned
    if (dim == 1 && nc == 1) {
      Stmt L = Evaluate(
          Call(DataType::Void(), ascend_simd_load(),
               {StringImm(dtype_str_), a->data, a2d(src, rep, i32(0))}));
      Stmt R =
          Evaluate(Call(DataType::Void(), rop, {a->data, a->data, mask_var_}));
      Stmt S =
          Evaluate(Call(DataType::Void(), ascend_simd_store(),
                        {StringImm(dtype_str_), a1d(dst, rep * i32(vreg_size_)),
                         a->data, mask_var_}));

      PrimExpr dst_off = rep * i32(vreg_size_);
      Stmt row_body;
      if (!clear) {
        Stmt LP =
            Evaluate(Call(DataType::Void(), ascend_simd_load_single(),
                          {StringImm(dtype_str_), b->data, a1d(dst, dst_off)}));
        Stmt MP = Evaluate(Call(DataType::Void(), eop,
                                {a->data, a->data, b->data, mask_var_}));
        row_body = LetStmt(mask_var_, mu, SeqStmt({L, R, LP, MP, S}));
      } else {
        row_body = LetStmt(mask_var_, mu, SeqStmt({L, R, S}));
      }
      return LetStmt(mask_cnt_var_, mi,
                     For(rep, i32(0), i32(oe), ForKind::kSerial, row_body));
    }

    // === dim=1, multi-VReg: per-chunk Reduce<MAX> + Max-accumulate ===
    if (dim == 1) {
      Stmt DI =
          Evaluate(Call(DataType::Void(), ascend_simd_dup(), {a->data, ival}));
      Var ck("vchunk", DataType::Int(32));
      PrimExpr co = ck * i32(vreg_size_);
      Stmt LK =
          Evaluate(Call(DataType::Void(), ascend_simd_load(),
                        {StringImm(dtype_str_), b->data, a2d(src, rep, co)}));
      Stmt RK =
          Evaluate(Call(DataType::Void(), rop, {b->data, b->data, mask_var_}));
      Stmt MA = Evaluate(
          Call(DataType::Void(), eop, {a->data, a->data, b->data, mask_var_}));
      Stmt CB = LetStmt(mask_var_, mu, SeqStmt({LK, RK, MA}));

      PrimExpr dst_off = rep * i32(vreg_size_);
      Stmt S = Evaluate(
          Call(DataType::Void(), ascend_simd_store(),
               {StringImm(dtype_str_), a1d(dst, dst_off), a->data, mask_var_}));

      // Fresh mask for post-loop store (inner loop consumed mask_cnt_var_)
      Var mask_cnt2("mask_cnt2", DataType::Handle());
      PrimExpr mi2 = Call(DataType::Handle(), ascend_init_mask(),
                          {IntImm(DataType::Int(32), vreg_size_)});
      PrimExpr mu2 = Call(DataType::Handle(), ascend_simd_update_mask(),
                          {StringImm(dtype_str_), mask_cnt2});
      Var mask2("mask2", DataType::Handle());

      Array<Stmt> rb = {DI, For(ck, i32(0), i32(nc), ForKind::kSerial, CB)};
      if (!clear) {
        Stmt LP =
            Evaluate(Call(DataType::Void(), ascend_simd_load_single(),
                          {StringImm(dtype_str_), b->data, a1d(dst, dst_off)}));
        Stmt MP = Evaluate(
            Call(DataType::Void(), eop, {a->data, a->data, b->data, mask2}));
        Stmt store2 = Evaluate(
            Call(DataType::Void(), ascend_simd_store(),
                 {StringImm(dtype_str_), a1d(dst, dst_off), a->data, mask2}));
        rb.push_back(LetStmt(mask_cnt2, mi2,
                             LetStmt(mask2, mu2, SeqStmt({LP, MP, store2}))));
      } else {
        Stmt store2 = Evaluate(
            Call(DataType::Void(), ascend_simd_store(),
                 {StringImm(dtype_str_), a1d(dst, dst_off), a->data, mask2}));
        rb.push_back(LetStmt(mask_cnt2, mi2, LetStmt(mask2, mu2, store2)));
      }
      return LetStmt(mask_cnt_var_, mi,
                     For(rep, i32(0), i32(oe), ForKind::kSerial, SeqStmt(rb)));
    }

    // === dim=0: rows NOT contiguous → element-wise Max per column chunk ===
    Var ck("vchunk", DataType::Int(32)), rv("vrow", DataType::Int(32));
    int64_t N = Downcast<IntImm>(src->shape[1])->value;
    PrimExpr co = ck * i32(vreg_size_);

    Stmt L0 =
        Evaluate(Call(DataType::Void(), ascend_simd_load(),
                      {StringImm(dtype_str_), a->data, a2d(src, i32(0), co)}));
    Stmt LK =
        Evaluate(Call(DataType::Void(), ascend_simd_load(),
                      {StringImm(dtype_str_), b->data, a2d(src, rv, co)}));
    Stmt EA = Evaluate(
        Call(DataType::Void(), eop, {a->data, a->data, b->data, mask_var_}));
    Stmt SK = Evaluate(
        Call(DataType::Void(), ascend_simd_store(),
             {StringImm(dtype_str_), a1d(dst, co), a->data, mask_var_}));

    Array<Stmt> cb;
    if (oe == 1)
      cb.push_back(L0);
    else {
      cb.push_back(LetStmt(mask_var_, mu, L0));
      cb.push_back(For(rv, i32(1), i32(oe), ForKind::kSerial,
                       LetStmt(mask_var_, mu, SeqStmt({LK, EA}))));
    }
    cb.push_back(LetStmt(mask_var_, mu, SK));
    return LetStmt(mask_cnt_var_, mi,
                   For(ck, i32(0), i32(nc), ForKind::kSerial, SeqStmt(cb)));
  }

  //-----------------------------------------------------------------------
  // LowerParallel: unified 1D / 2D parallel lowering
  //
  // vreg_for  — the parallel loop whose extent is the VReg dimension
  // serial_for — (nullable) the outer serial loop for 2D parallel
  //-----------------------------------------------------------------------
  Stmt LowerParallel(const ForNode *vreg_for, const ForNode *serial_for) {
    int64_t vreg_ext = Downcast<IntImm>(vreg_for->extent)->value;
    int64_t serial_ext =
        serial_for ? Downcast<IntImm>(serial_for->extent)->value : 1;

    DataType dtype = InferDType(vreg_for->body);
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);
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
    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    mask_var_ = Var("mask", DataType::Handle());
    scalar_loads_.clear();

    PrimExpr im = Call(DataType::Handle(), ascend_init_mask(),
                       {IntImm(DataType::Int(32), vreg_ext)});
    PrimExpr um = Call(DataType::Handle(), ascend_simd_update_mask(),
                       {StringImm(dtype_str_), mask_cnt_var_});

    Stmt body = LowerParallelBody(vreg_for->body, dtype);
    Stmt ml = LetStmt(mask_var_, um, body);
    Stmt inner_loop =
        For(chunk_var_, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, ml);

    // Per-iteration body: scalar loads + mask_cnt init + VReg chunk loop
    Stmt per_iter_body = inner_loop;
    if (!scalar_loads_.empty()) {
      Array<Stmt> seq;
      for (const Stmt &s : scalar_loads_)
        seq.push_back(s);
      seq.push_back(per_iter_body);
      per_iter_body = SeqStmt(seq);
      scalar_loads_.clear();
    }
    per_iter_body = LetStmt(mask_cnt_var_, im, per_iter_body);

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
