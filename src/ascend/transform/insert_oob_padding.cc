/*!
 * \file insert_oob_padding.cc
 * \brief Handle Ascend DMA copy out-of-bounds tails and GM->L1 padding before
 *        AutoSchedule.
 *
 * For every DMA `tl.tileop.copy` (GM<->UB, GM->L1, L0C->GM) this pass clamps
 * the out-of-bounds tail: it rewrites the copy's region args to the in-bounds
 * ranges and wraps the copy in the has_data guard (dropping statically-empty
 * tiles). For a padded GM->L1 copy it also appends the create_cbuf_matrix
 * padding fill(s) as standalone ascend_fill_l1 statements.
 *
 * Doing this before AutoSchedule (rather than inside copy.cc's LowerTileOp
 * lowering, which runs after AutoSchedule) makes the fill visible to
 * pipe/barrier analysis: it is scheduled as its own MTE1 task (see
 * auto_schedule.cc's ResourceAnalyzer) and ordered against the L1->L0 loads
 * that read the padded buffer. copy.cc lowering consumes the already-clamped
 * ranges and no longer performs any OOB/padding handling. Both this pass and
 * (historically) copy.cc share the clamp/fill helpers in oob_padding.h.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/op/oob_padding.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

// Rebuild a tl.region call from a buffer, its clamped per-axis ranges and the
// access mask, matching RegionOp's arg layout:
//   region(BufferLoad(buffer, mins), access_mask, extent0, extent1, ...)
PrimExpr MakeRegionCall(const Buffer &buffer, const Array<Range> &ranges,
                        int access_mask) {
  static const Op &region_op = Op::Get("tl.tileop.region");
  Array<PrimExpr> mins;
  Array<PrimExpr> args;
  for (const auto &r : ranges)
    mins.push_back(r->min);
  args.push_back(BufferLoad(buffer, mins));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto &r : ranges)
    args.push_back(r->extent);
  return Call(DataType::Handle(), region_op, args);
}

// Returns the destination-base access_ptr for the fill, encoded as a
// tl.access_ptr(BufferLoad(dst, dst_range mins)) so LowerAccessPtr computes the
// same flat offset copy.cc used (ComputeFlatOffset over row-major strides).
PrimExpr MakeFillBasePtr(const CopyNode *copy) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  Array<PrimExpr> dst_mins;
  PrimExpr total_elems = make_const(DataType::Int(32), 1);
  for (const auto &r : copy->dst_range) {
    dst_mins.push_back(r->min);
    total_elems = total_elems * r->extent;
  }
  BufferLoad base_load(copy->dst, dst_mins);
  // tl.access_ptr(BufferLoad, extent, rw_mask): rw_mask 2 == write.
  return Call(DataType::Handle(), tl::access_ptr(),
              {base_load, total_elems, I(2)});
}

// Build the standalone padding fill statements for a padded GM->L1 copy, or an
// empty vector when the copy needs no padding. Mirrors copy.cc's dma_path 2
// major-layout branch (valid extents from the clamped source, dst geometry from
// the full destination range).
std::vector<Stmt> MakeL1PaddingStmts(const CopyNode *copy,
                                     const LayoutMap &layout_map,
                                     arith::Analyzer *analyzer) {
  std::vector<Stmt> stmts;
  if (!(IsGlobalBuffer(copy->src) && IsL1Buffer(copy->dst)))
    return stmts;
  if (copy->dst_range.size() < 2 || copy->src_range.size() < 2)
    return stmts;

  Optional<Layout> layout = ascend::FindLayoutForBuffer(layout_map, copy->dst);
  if (!layout.defined())
    return stmts;
  AscendFractalLayoutInfo dst_layout;
  if (!TryExtractAscendFractalLayout(layout.value(), copy->dst, &dst_layout))
    return stmts;
  // Only the major-like (data) NZ fractal needs an explicit padding fill; the
  // scale-factor (kSF) layout does not go through this path.
  if (!IsAscendMajorKind(dst_layout.kind))
    return stmts;

  // Clamp the copy's ranges the same way copy.cc lowering will, to recover the
  // valid (in-bounds) source extents that determine the padding tail size.
  ascend::BoundedDMACopyRanges bounded =
      ascend::ClampDMACopyTail(*copy, analyzer);
  // Only pad the tail that an out-of-bounds clamp actually cut off. A copy
  // whose destination L1 buffer is merely over-allocated (larger than the valid
  // source region, but the source itself is fully in bounds) is left untouched:
  // its extra columns/rows are not clamped, so has_data stays undefined and no
  // fill is emitted. When the source *is* clamped (OOB), we pad every tail
  // dimension (M/N and the K reduction axis) so the gemm never reads
  // uninitialized L1.
  if (!bounded.has_data.defined())
    return stmts;
  const Array<Range> &src_range = bounded.src;
  size_t src_ndim = src_range.size();
  PrimExpr valid_src_rows = src_range[src_ndim - 2]->extent;
  PrimExpr valid_src_cols = src_range[src_ndim - 1]->extent;

  bool needs_transpose = copy->GetTranspose();
  PrimExpr dst_n_value = dst_layout.outer1 * dst_layout.row_frac;
  PrimExpr valid_rows = needs_transpose ? valid_src_cols : valid_src_rows;
  PrimExpr valid_cols = needs_transpose ? valid_src_rows : valid_src_cols;

  PrimExpr base_ptr = MakeFillBasePtr(copy);

  Optional<Stmt> col_padding = ascend::MakeL1ColPadding(
      *copy, dst_layout, analyzer, base_ptr, dst_n_value, valid_cols);
  if (col_padding.defined())
    stmts.push_back(col_padding.value());

  Optional<Stmt> row_padding = ascend::MakeL1RowPadding(
      *copy, dst_layout, analyzer, base_ptr, dst_n_value, valid_rows);
  if (row_padding.defined())
    stmts.push_back(row_padding.value());

  return stmts;
}

class OOBPaddingInserter : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Apply(Stmt stmt, arith::Analyzer *analyzer) {
    OOBPaddingInserter inserter(analyzer);
    return inserter(std::move(stmt));
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  explicit OOBPaddingInserter(arith::Analyzer *analyzer) : Parent(analyzer) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    LayoutMap saved = layout_map_;
    if (auto layout_map_ref = op->annotations.Get(attr::kLayoutMap)) {
      if (auto map = layout_map_ref.value().as<Map<Buffer, Layout>>()) {
        layout_map_ = map.value();
      }
    }
    // Parent binds this block's iter vars into the analyzer before recursing.
    Stmt stmt = Parent::VisitStmt_(op);
    layout_map_ = saved;
    return stmt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(copy_op)) {
      return Parent::VisitStmt_(op);
    }

    Copy copy_obj(call->args, call->annotations);
    const CopyNode *copy = copy_obj.get();

    int dma_path = ascend::GetDMAPath(copy->src, copy->dst);
    // Only DMA paths that copy.cc used to clamp participate in OOB handling.
    bool clampable =
        dma_path == 0 || dma_path == 1 || dma_path == 2 || dma_path == 7;
    if (!clampable) {
      return Parent::VisitStmt_(op);
    }

    // Clamp the copy's tail to the in-bounds region. The traversal analyzer has
    // the enclosing loop / thread / block-iter ranges bound (via
    // IRMutatorWithAnalyzer), matching copy.cc's lowering-time analyzer.
    ascend::BoundedDMACopyRanges bounded =
        ascend::ClampDMACopyTail(*copy, analyzer_);

    // Rebuild the copy call with clamped ranges when clamping shrank them.
    Stmt copy_stmt = GetRef<Stmt>(op);
    if (bounded.has_data.defined()) {
      int src_mask = NormalizeToAccessRegion(call->args[0]).access_mask;
      int dst_mask = NormalizeToAccessRegion(call->args[1]).access_mask;
      Array<PrimExpr> new_args = call->args;
      new_args.Set(0, MakeRegionCall(copy->src, bounded.src, src_mask));
      new_args.Set(1, MakeRegionCall(copy->dst, bounded.dst, dst_mask));
      copy_stmt =
          Evaluate(Call(call->dtype, call->op, new_args, call->annotations));

      PrimExpr has_data = bounded.has_data.value();
      if (is_zero(has_data)) {
        // Fully out of bounds: no copy and no fill of valid data. (A padded
        // buffer whose entire valid region is empty still gets its fill below.)
        copy_stmt = Evaluate(0);
      } else if (!is_one(has_data)) {
        // The enclosing loop/thread/if constraints (e.g. the Persistent wave
        // guard `w*64+bx < ceildiv(num_tokens,32)`) may already guarantee at
        // least one valid element. When provable, keep the clamped copy
        // UNCONDITIONAL: the tail guard is redundant, and an unconditional
        // write keeps the dst buffer write-first so multi-buffer eligibility /
        // annotate_buffer_versions apply.
        if (!analyzer_->CanProve(has_data)) {
          copy_stmt = IfThenElse(has_data, copy_stmt);
        }
      }
    }

    // Emit the standalone padding fill(s) for a padded GM->L1 copy.
    std::vector<Stmt> fills = MakeL1PaddingStmts(copy, layout_map_, analyzer_);

    if (fills.empty()) {
      return copy_stmt;
    }
    Array<Stmt> seq;
    seq.push_back(copy_stmt);
    for (auto &f : fills)
      seq.push_back(f);
    return SeqStmt(seq);
  }

  LayoutMap layout_map_;
};

} // namespace

tvm::transform::Pass AscendInsertOOBPadding() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const PassContext &ctx) {
    arith::Analyzer analyzer;
    PrimFuncNode *write_ptr = func.CopyOnWrite();
    write_ptr->body =
        OOBPaddingInserter::Apply(std::move(write_ptr->body), &analyzer);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendInsertOOBPadding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendInsertOOBPadding",
                        AscendInsertOOBPadding);
}

} // namespace tl
} // namespace tvm
