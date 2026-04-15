// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_simdvf_lower_parallel.cc
 * \brief Lower T.Parallel loops inside SIMD_VF blocks to MicroAPI
 *        register-level vector instructions (LoadAlign/Add/StoreAlign).
 *
 * VReg size: 8 VLanes × 32 Bytes / sizeof(dtype)
 *   float32 → 64, float16 → 128, int8 → 256
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

/*!
 * \brief Lower parallel loops inside SIMD_VF blocks to MicroAPI vector ops.
 *
 * Transforms:
 *   for i in T.Parallel(N):
 *       C[i] = A[i] + B[i]
 *
 * Into:
 *   r0 = alloc_fragment((VL,), dtype)
 *   r1 = alloc_fragment((VL,), dtype)
 *   rd = alloc_fragment((VL,), dtype)
 *   mask_cnt = ascend_init_mask(N)
 *   for chunk in serial(N / VL):
 *       mask = ascend_simd_update_mask(mask_cnt)
 *       ascend_simd_load(r0, &A[chunk*VL])
 *       ascend_simd_load(r1, &B[chunk*VL])
 *       ascend_simd_add(rd, r0, r1, mask)
 *       ascend_simd_store(&C[chunk*VL], rd, mask)
 */
class AscendSimdVFLowerParallel : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    AscendSimdVFLowerParallel substituter;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return ffi::GetRef<PrimFunc>(fptr);
  }

private:
  // Whether we are inside a SIMD_VF block
  bool inside_simdvf_{false};

  // Parallel loop variable and extent being lowered
  const VarNode *parallel_var_{nullptr};
  int64_t parallel_extent_{0};

  // VReg size (number of elements per register), determined by dtype
  int64_t vreg_size_{0};

  // The chunk loop variable: iterates over VReg-sized chunks
  Var chunk_var_;

  // Mask variables for the current lowering
  Var mask_cnt_var_;
  Var mask_var_;

  // Fragment registers allocated during expression decomposition
  std::vector<Buffer> new_fragments_;
  int frag_counter_{0};

  // The dtype string for MicroAPI template parameters (e.g. "float")
  std::string dtype_str_;

  //-----------------------------------------------------------------------
  // VReg size computation
  //-----------------------------------------------------------------------
  static int64_t ComputeVRegSize(DataType dtype) {
    // 8 VLanes × 32 Bytes per VLane / sizeof(element)
    int64_t elem_bytes = dtype.bits() / 8;
    return 8 * 32 / elem_bytes;
  }

  static std::string DTypeToAscendStr(DataType dtype) {
    if (dtype.is_float()) {
      if (dtype.bits() == 16)
        return "half_t";
      if (dtype.bits() == 32)
        return "float";
    } else if (dtype.is_int()) {
      return "int" + std::to_string(dtype.bits()) + "_t";
    } else if (dtype.is_uint()) {
      return "uint" + std::to_string(dtype.bits()) + "_t";
    }
    return "float";
  }

  //-----------------------------------------------------------------------
  // Fragment register allocation
  //-----------------------------------------------------------------------
  Buffer AllocFragmentReg(DataType dtype) {
    std::string name =
        "vreg_" + std::to_string(frag_counter_++);
    Var data(name, PointerType(PrimType(dtype), "local.fragment"));
    Array<PrimExpr> shape = {IntImm(DataType::Int(32), vreg_size_)};
    Buffer buf(data, dtype, shape,
               /*strides=*/{},
               /*elem_offset=*/PrimExpr(0),
               /*name=*/name,
               /*data_alignment=*/0,
               /*offset_factor=*/0,
               /*buffer_type=*/kDefault);
    new_fragments_.push_back(buf);
    return buf;
  }

  //-----------------------------------------------------------------------
  // Index substitution: replace parallel_var with chunk_var_ * vreg_size_
  //-----------------------------------------------------------------------
  PrimExpr SubstituteIndex(const PrimExpr &index) {
    if (parallel_var_ == nullptr)
      return index;

    class IndexReplacer : public ExprMutator {
    public:
      const VarNode *from_;
      PrimExpr to_;
      IndexReplacer(const VarNode *from, PrimExpr to)
          : from_(from), to_(std::move(to)) {}
      PrimExpr VisitExpr_(const VarNode *op) override {
        if (op == from_)
          return to_;
        return ffi::GetRef<PrimExpr>(op);
      }
    };

    PrimExpr replacement =
        chunk_var_ * IntImm(DataType::Int(32), vreg_size_);
    IndexReplacer replacer(parallel_var_, replacement);
    return replacer(index);
  }

  //-----------------------------------------------------------------------
  // Make address_of(buffer[substituted_indices])
  //-----------------------------------------------------------------------
  PrimExpr MakeAddressOf(const Buffer &buffer,
                         const Array<PrimExpr> &orig_indices) {
    Array<PrimExpr> new_indices;
    for (const auto &idx : orig_indices) {
      new_indices.push_back(SubstituteIndex(idx));
    }
    PrimExpr load = BufferLoad(buffer, new_indices);
    return Call(DataType::Handle(), builtin::address_of(), {load});
  }

  //-----------------------------------------------------------------------
  // Expression decomposition (recursive)
  // Returns the fragment data Var holding the result
  //-----------------------------------------------------------------------
  Var DecomposeExpr(const PrimExpr &expr, DataType dtype,
                    Array<Stmt> *stmts) {
    // Leaf: BufferLoad → alloc fragment + ascend_simd_load
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Buffer reg = AllocFragmentReg(dtype);
      PrimExpr addr = MakeAddressOf(load->buffer, load->indices);
      PrimExpr call = Call(DataType::Void(), ascend_simd_load(),
                           {StringImm(dtype_str_), reg->data, addr});
      stmts->push_back(Evaluate(call));
      return reg->data;
    }

    // Binary: Add
    if (const auto *add = expr.as<AddNode>()) {
      Var lhs = DecomposeExpr(add->a, dtype, stmts);
      Var rhs = DecomposeExpr(add->b, dtype, stmts);
      Buffer dst = AllocFragmentReg(dtype);
      PrimExpr call = Call(DataType::Void(), ascend_simd_add(),
                           {dst->data, lhs, rhs, mask_var_});
      stmts->push_back(Evaluate(call));
      return dst->data;
    }

    // Binary: Sub
    if (const auto *sub = expr.as<SubNode>()) {
      Var lhs = DecomposeExpr(sub->a, dtype, stmts);
      Var rhs = DecomposeExpr(sub->b, dtype, stmts);
      Buffer dst = AllocFragmentReg(dtype);
      PrimExpr call = Call(DataType::Void(), ascend_simd_sub(),
                           {dst->data, lhs, rhs, mask_var_});
      stmts->push_back(Evaluate(call));
      return dst->data;
    }

    // Binary: Mul
    if (const auto *mul = expr.as<MulNode>()) {
      Var lhs = DecomposeExpr(mul->a, dtype, stmts);
      Var rhs = DecomposeExpr(mul->b, dtype, stmts);
      Buffer dst = AllocFragmentReg(dtype);
      PrimExpr call = Call(DataType::Void(), ascend_simd_mul(),
                           {dst->data, lhs, rhs, mask_var_});
      stmts->push_back(Evaluate(call));
      return dst->data;
    }

    LOG(FATAL) << "AscendSimdVFLowerParallel: unsupported expression type: "
               << expr->GetTypeKey();
    return Var();
  }

  //-----------------------------------------------------------------------
  // Lower a single BufferStore inside a parallel loop
  //-----------------------------------------------------------------------
  Stmt LowerBufferStore(const BufferStoreNode *store, DataType dtype) {
    Array<Stmt> stmts;

    // Decompose the value expression into MicroAPI ops
    Var result_reg = DecomposeExpr(store->value, dtype, &stmts);

    // Store: ascend_simd_store(&dst[chunk*VL], result_reg, mask)
    PrimExpr addr = MakeAddressOf(store->buffer, store->indices);
    PrimExpr store_call =
        Call(DataType::Void(), ascend_simd_store(),
             {StringImm(dtype_str_), addr, result_reg, mask_var_});
    stmts.push_back(Evaluate(store_call));

    if (stmts.size() == 1)
      return stmts[0];
    return SeqStmt(stmts);
  }

  //-----------------------------------------------------------------------
  // Lower an entire parallel body (single store or seq of stores)
  //-----------------------------------------------------------------------
  Stmt LowerParallelBody(const Stmt &body, DataType dtype) {
    if (const auto *store = body.as<BufferStoreNode>()) {
      return LowerBufferStore(store, dtype);
    }

    if (const auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> lowered;
      for (const Stmt &s : seq->seq) {
        if (const auto *store = s.as<BufferStoreNode>()) {
          lowered.push_back(LowerBufferStore(store, dtype));
        } else {
          LOG(FATAL) << "AscendSimdVFLowerParallel: expected BufferStore in "
                        "parallel seq, got "
                     << s->GetTypeKey();
        }
      }
      return SeqStmt(lowered);
    }

    LOG(FATAL)
        << "AscendSimdVFLowerParallel: unsupported parallel body type: "
        << body->GetTypeKey();
    return Stmt();
  }

  //-----------------------------------------------------------------------
  // VisitStmt overrides
  //-----------------------------------------------------------------------
  Stmt VisitStmt_(const BlockNode *op) override {
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Enter SIMD_VF scope
    bool saved = inside_simdvf_;
    inside_simdvf_ = true;
    auto saved_fragments = std::move(new_fragments_);
    int saved_counter = frag_counter_;
    new_fragments_.clear();
    frag_counter_ = 0;

    // Visit body — parallel loops will be lowered here
    Stmt new_body = VisitStmt(op->body);

    // Collect new fragment allocations into alloc_buffers
    Array<Buffer> alloc_buffers = op->alloc_buffers;
    for (const Buffer &buf : new_fragments_) {
      alloc_buffers.push_back(buf);
    }

    // Restore state
    inside_simdvf_ = saved;
    new_fragments_ = std::move(saved_fragments);
    frag_counter_ = saved_counter;

    if (alloc_buffers.same_as(op->alloc_buffers) &&
        new_body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }

    return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                 new_body, op->init, alloc_buffers, op->match_buffers,
                 op->annotations);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (!inside_simdvf_ || op->kind != ForKind::kParallel) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Extract extent
    const auto *extent_imm = op->extent.as<IntImmNode>();
    ICHECK(extent_imm)
        << "AscendSimdVFLowerParallel: parallel extent must be constant, got "
        << op->extent;
    int64_t extent = extent_imm->value;

    // Determine dtype from the first BufferStore in the body
    DataType dtype = DataType::Float(32); // default
    if (const auto *store = op->body.as<BufferStoreNode>()) {
      dtype = store->buffer->dtype;
    } else if (const auto *seq = op->body.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (const auto *store = s.as<BufferStoreNode>()) {
          dtype = store->buffer->dtype;
          break;
        }
      }
    }

    // Compute VReg size
    vreg_size_ = ComputeVRegSize(dtype);
    dtype_str_ = DTypeToAscendStr(dtype);

    ICHECK_EQ(extent % vreg_size_, 0)
        << "Parallel extent " << extent << " must be divisible by VReg size "
        << vreg_size_;
    int64_t num_chunks = extent / vreg_size_;

    // Save/set parallel variable
    const VarNode *saved_pvar = parallel_var_;
    int64_t saved_extent = parallel_extent_;
    parallel_var_ = op->loop_var.get();
    parallel_extent_ = extent;

    // Create chunk loop variable
    chunk_var_ = Var("simdvf_chunk", DataType::Int(32));

    // Create mask variables
    // mask_cnt = ascend_init_mask(extent)
    mask_cnt_var_ = Var("mask_cnt", DataType::Handle());
    PrimExpr init_mask_call =
        Call(DataType::Handle(), ascend_init_mask(),
             {IntImm(DataType::Int(32), extent)});

    // mask = ascend_simd_update_mask(mask_cnt)
    mask_var_ = Var("mask", DataType::Handle());
    PrimExpr update_mask_call =
        Call(DataType::Handle(), ascend_simd_update_mask(),
             {StringImm(dtype_str_), mask_cnt_var_});

    // Lower the parallel body
    Stmt chunk_body = LowerParallelBody(op->body, dtype);

    // Wrap: let mask = update_mask(mask_cnt) { chunk_body }
    Stmt mask_let = LetStmt(mask_var_, update_mask_call, chunk_body);

    // Wrap in chunk loop: for chunk_var in serial(num_chunks) { mask_let }
    Stmt chunk_loop =
        For(chunk_var_, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), num_chunks), ForKind::kSerial, mask_let);

    // Wrap: let mask_cnt = init_mask(extent) { chunk_loop }
    Stmt result = LetStmt(mask_cnt_var_, init_mask_call, chunk_loop);

    // Restore state
    parallel_var_ = saved_pvar;
    parallel_extent_ = saved_extent;

    return result;
  }
};

using namespace tir::transform;

tvm::transform::Pass AscendSimdVFLowerParallel() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AscendSimdVFLowerParallel::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.AscendSimdVFLowerParallel", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendSimdVFLowerParallel",
                        AscendSimdVFLowerParallel);
}

} // namespace tl
} // namespace tvm
