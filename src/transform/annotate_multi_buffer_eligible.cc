/*!
 * \file annotate_multi_buffer_eligible.cc
 * \brief Pre-AutoSchedule analysis: for each For loop, decide which on-chip
 *        buffers can be multi-buffered and record the set on the For's
 *        annotations under "multi_buffer_eligible".
 *
 *  Runs before IfConditionExtract so that the original if-then-else structure
 *  is still intact — IfConditionExtract splits one if into many small ifs,
 *  which makes "write-first" analysis overly conservative.
 *
 *  A buffer is eligible inside For L iff
 *    1) it lives in on-chip memory (shared / shared.l1 / shared.l0c, including
 *       their .dyn variants),
 *    2) it is not referenced anywhere outside L (siblings or ancestors),
 *    3) inside L's body it is always written before being read.
 *
 *  Frontend may pre-set the annotation; we never overwrite an existing value.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// ---------------------------------------------------------------------------
// WriteFirstClassifier: program-order classification of a single buffer's
// access pattern inside a Stmt.
//
// Four-state lattice:
//   kUntouched         — subtree provably never accesses the buffer
//   kMaybeWriteFirst   — either kUntouched or kWriteFirst (write not certain
//                        but no read-before-write risk)
//   kWriteFirst        — first access is a guaranteed write
//   kReadFirst         — a read may happen before any guaranteed write
//                        (absorbing/unsafe state)
//
// Both kUntouched and kMaybeWriteFirst and kWriteFirst are "safe" for
// multi-buffering (prior content is irrelevant). kReadFirst is not.
// ---------------------------------------------------------------------------
enum class AccessOrder {
  kUntouched,
  kMaybeWriteFirst,
  kWriteFirst,
  kReadFirst,
};

class WriteFirstClassifier : public StmtExprVisitor {
public:
  AccessOrder Classify(const Stmt &s, const Buffer &buf) {
    target_ = buf;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(s);
    return result_;
  }

private:
  Buffer target_;
  AccessOrder result_ = AccessOrder::kUntouched;
  arith::Analyzer analyzer_;

  // Terminal states determine the answer; further sub-events can't change it.
  static bool IsTerminal(AccessOrder o) {
    return o == AccessOrder::kReadFirst || o == AccessOrder::kWriteFirst;
  }

  // Sequential composition: order of two sub-events in program order.
  // (kReadFirst / kWriteFirst absorb; kMaybeWriteFirst+kReadFirst is
  //  conservatively kReadFirst since the prefix's skip case would leak.)
  static AccessOrder SeqCompose(AccessOrder a, AccessOrder b) {
    if (a == AccessOrder::kReadFirst || a == AccessOrder::kWriteFirst)
      return a;
    if (a == AccessOrder::kUntouched)
      return b;
    // a == kMaybeWriteFirst
    if (b == AccessOrder::kUntouched)
      return AccessOrder::kMaybeWriteFirst;
    return b;
  }

  // Dispatch short-circuit: skip further work once a terminal is reached.
  void VisitStmt(const Stmt &s) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitStmt(s);
  }
  void VisitExpr(const PrimExpr &e) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitExpr(e);
  }

  template <typename F> AccessOrder ClassifyLocal(F f) {
    AccessOrder saved = result_;
    result_ = AccessOrder::kUntouched;
    f();
    AccessOrder local = result_;
    result_ = saved;
    return local;
  }

  // Merge of an if/then/else (both branches taken under mutually exclusive
  // conditions). Returns the safest classification consistent with both.
  static AccessOrder MergeIfThenElse(AccessOrder t, AccessOrder e) {
    if (t == AccessOrder::kReadFirst || e == AccessOrder::kReadFirst)
      return AccessOrder::kReadFirst;
    if (t == AccessOrder::kUntouched && e == AccessOrder::kUntouched)
      return AccessOrder::kUntouched;
    if (t == AccessOrder::kWriteFirst && e == AccessOrder::kWriteFirst)
      return AccessOrder::kWriteFirst;
    // Mix among {kUntouched, kWriteFirst, kMaybeWriteFirst}: maybe-write.
    return AccessOrder::kMaybeWriteFirst;
  }
  // Single-branch if (no else); implicit else is kUntouched.
  static AccessOrder MergeIfThenOnly(AccessOrder t) {
    if (t == AccessOrder::kReadFirst)
      return AccessOrder::kReadFirst;
    if (t == AccessOrder::kUntouched)
      return AccessOrder::kUntouched;
    // kWriteFirst or kMaybeWriteFirst: either the branch ran and wrote, or
    // didn't run at all — never a leaked read.
    return AccessOrder::kMaybeWriteFirst;
  }
  // If the surrounding loop / block may not execute, demote write-first to
  // maybe-write-first (write not guaranteed). kReadFirst and kUntouched and
  // kMaybeWriteFirst pass through unchanged.
  static AccessOrder DemoteIfNotMustExecute(AccessOrder r, bool must_execute) {
    if (must_execute)
      return r;
    if (r == AccessOrder::kWriteFirst)
      return AccessOrder::kMaybeWriteFirst;
    return r;
  }

  bool ForMustExecute(const ForNode *op) {
    // step >= 1 (default is 1) and extent > 0.
    if (op->step.has_value()) {
      if (!analyzer_.CanProve(op->step.value() >= 1))
        return false;
    }
    return analyzer_.CanProve(op->extent > 0);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_.same_as(op->buffer))
      result_ = AccessOrder::kReadFirst;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // value first, then indices, then the store itself.
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_.same_as(op->buffer))
      result_ = AccessOrder::kWriteFirst;
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (const auto &s : op->seq) {
      VisitStmt(s);
      if (IsTerminal(result_))
        return;
    }
  }

  void VisitStmt_(const EvaluateNode *op) final { VisitExpr(op->value); }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    if (IsTerminal(result_))
      return;
    VisitExpr(op->extent);
    if (IsTerminal(result_))
      return;
    if (op->step.has_value()) {
      VisitExpr(op->step.value());
      if (IsTerminal(result_))
        return;
    }
    bool must_exec = ForMustExecute(op);
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    // NOTE: technically unsafe heuristic. kMaybeWriteFirst body means each
    // iteration is independently safe (skip or write-first), but says
    // nothing about whether *any* iteration takes the write branch. If every
    // iteration happens to skip, the loop overall leaves the buffer
    // untouched — yet we report kWriteFirst, which can mislead an outer
    // classifier that sees a subsequent read in the same scope as "safe
    // because the For wrote first". We accept this risk because the typical
    // pattern (conditional write inside a hot loop) almost always has the
    // condition true at least once, and being conservative here was
    // empirically too restrictive for multi-buffer eligibility.
    if (must_exec && body == AccessOrder::kMaybeWriteFirst)
      body = AccessOrder::kWriteFirst;
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_exec));
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto t = ClassifyLocal([&] { VisitStmt(op->then_case); });
    AccessOrder branch;
    if (op->else_case) {
      auto e = ClassifyLocal([&] { VisitStmt(op->else_case.value()); });
      branch = MergeIfThenElse(t, e);
    } else {
      branch = MergeIfThenOnly(t);
    }
    result_ = SeqCompose(result_, branch);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    VisitStmt(op->body);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_,
                         DemoteIfNotMustExecute(body, /*must_execute=*/false));
  }

  void VisitStmt_(const BlockNode *op) final {
    // Reduce-init runs only on the first iteration of the surrounding loop;
    // its writes are not guaranteed across iterations -> kMaybeWriteFirst.
    if (op->init.defined()) {
      auto init = ClassifyLocal([&] { VisitStmt(op->init.value()); });
      AccessOrder init_eff = init;
      if (init == AccessOrder::kWriteFirst)
        init_eff = AccessOrder::kMaybeWriteFirst;
      result_ = SeqCompose(result_, init_eff);
      if (IsTerminal(result_))
        return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    VisitExpr(op->predicate);
    if (IsTerminal(result_))
      return;
    bool must_execute = false;
    if (const auto *imm = op->predicate.as<IntImmNode>())
      must_execute = (imm->value != 0);
    auto body = ClassifyLocal([&] { VisitStmt(op->block); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_execute));
  }

  void VisitExpr_(const CallNode *op) final {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto access_ptr_op = Op::Get("tl.access_ptr");

    if (op->op.same_as(region_op)) {
      HandleRegionCall(op);
      return;
    }
    if (op->op.same_as(access_ptr_op)) {
      HandleAccessPtrCall(op);
      return;
    }
    for (const auto &arg : op->args) {
      VisitExpr(arg);
      if (IsTerminal(result_))
        return;
    }
  }

  // tl.tileop.region(BufferLoad(buf, indices...), access_type, extents...)
  //   access_type: bit 0 = read, bit 1 = write.
  void HandleRegionCall(const CallNode *op) {
    if (op->args.size() < 2) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mode = op->args[1].as<IntImmNode>();
    if (!bl || !mode) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      VisitExpr(op->args[i]);
      if (IsTerminal(result_))
        return;
    }
    if (target_.same_as(bl->buffer)) {
      int m = mode->value;
      if (m & 1)
        result_ = AccessOrder::kReadFirst;
      else if (m & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }

  // tl.access_ptr(BufferLoad(buf, indices...), extent, rw_mask)
  //   rw_mask: bit 0 = read, bit 1 = write.
  void HandleAccessPtrCall(const CallNode *op) {
    if (op->args.size() < 3) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mask = op->args[2].as<IntImmNode>();
    if (!bl || !mask) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    VisitExpr(op->args[1]);
    if (IsTerminal(result_))
      return;
    if (target_.same_as(bl->buffer)) {
      int rw = mask->value;
      if (rw & 1)
        result_ = AccessOrder::kReadFirst;
      else if (rw & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }
};

// ---------------------------------------------------------------------------
// BufferAccessCollector: one-shot scan of a Stmt to collect every buffer that
// is read or written anywhere inside it (program order doesn't matter).
// ---------------------------------------------------------------------------
class BufferAccessCollector : public StmtExprVisitor {
public:
  std::set<Buffer> touched;

  void Run(const Stmt &s) { StmtExprVisitor::VisitStmt(s); }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    touched.insert(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    touched.insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) final {
    for (const auto &r : op->reads)
      touched.insert(r->buffer);
    for (const auto &w : op->writes)
      touched.insert(w->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto access_ptr_op = Op::Get("tl.access_ptr");
    if (op->op.same_as(region_op) && op->args.size() >= 1) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        touched.insert(bl->buffer);
    } else if (op->op.same_as(access_ptr_op) && op->args.size() >= 1) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        touched.insert(bl->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

public:
  // (no extra state needed; access_ptr already carries a BufferLoad)
};

// ---------------------------------------------------------------------------
// MultiBufferAnnotator: walk the PrimFunc body, mirroring the recursion that
// the AutoSchedule IR-structure builder will perform. Only loops that the
// builder turns into a ControlNode (i.e. ForKind::kSerial) get annotated; all
// other constructs (parallel/vectorized/unrolled For, While, Block) become
// opaque TaskNodes downstream and are left untouched.
//
// Recursion mirrors auto_schedule.cc's StructureBuilder:
//   - SeqStmt          -> walk children
//   - For (kSerial)    -> ControlNode: annotate + walk body
//   - For (other kind) -> TaskNode: leaf, do not walk
//   - IfThenElse       -> walk both branches
//   - LetStmt/AttrStmt -> WrapperNode: walk body
//   - Block / While    -> TaskNode: leaf
//   - other            -> leaf
// ---------------------------------------------------------------------------
class MultiBufferAnnotator : public StmtMutator {
public:
  Stmt Run(Stmt body) { return VisitStmt(body); }

private:
  std::set<Buffer> used_buffers_;

  static bool IsOnChipMultiBufferScope(const Buffer &b) {
    return IsSharedBuffer(b) || IsL1Buffer(b) || IsL0CBuffer(b);
  }

  static std::set<Buffer> CollectTouched(const Stmt &s) {
    BufferAccessCollector c;
    c.Run(s);
    return std::move(c.touched);
  }

  // Walk `s` and collect every buffer that any nested For has already claimed
  // via "multi_buffer_eligible". Used so that an outer For doesn't
  // re-claim a buffer that an inner For will multi-buffer — the storage is
  // owned by the innermost claimant.
  static std::set<Buffer> CollectInnerClaimed(const Stmt &s) {
    struct Visitor : public StmtVisitor {
      std::set<Buffer> claimed;
      void VisitStmt_(const ForNode *op) final {
        auto v = op->annotations.Get("multi_buffer_eligible");
        if (v.has_value()) {
          for (const auto &b : v.value().cast<Array<Buffer>>()) {
            claimed.insert(b);
          }
        }
        StmtVisitor::VisitStmt_(op);
      }
    } v;
    v(s);
    return std::move(v.claimed);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Pre-scan each child's touched set so we can augment used_buffers_ for
    // recursion into the others.
    const size_t n = op->seq.size();
    std::vector<std::set<Buffer>> per_child(n);
    for (size_t i = 0; i < n; ++i)
      per_child[i] = CollectTouched(op->seq[i]);

    Array<Stmt> new_seq;
    new_seq.reserve(n);
    auto saved = used_buffers_;
    for (size_t i = 0; i < n; ++i) {
      used_buffers_ = saved;
      for (size_t j = 0; j < n; ++j) {
        if (j == i)
          continue;
        used_buffers_.insert(per_child[j].begin(), per_child[j].end());
      }
      new_seq.push_back(VisitStmt(op->seq[i]));
    }
    used_buffers_ = saved;
    return SeqStmt(std::move(new_seq));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Only kSerial loops become ControlNodes in the AutoSchedule IR structure
    // and thus participate in multi-buffer scheduling. Parallel / unrolled /
    // vectorized loops are sunk into a leaf TaskNode and never scheduled, so
    // an annotation on them would be wasted (and may mislead downstream).
    if (op->kind != ForKind::kSerial) {
      return tvm::ffi::GetRef<For>(op);
    }

    auto parent_used = used_buffers_;

    Stmt new_body = VisitStmt(op->body);
    used_buffers_ = parent_used;

    // Frontend-provided annotation wins.
    if (op->annotations.count("multi_buffer_eligible")) {
      auto for_node = tvm::ffi::GetRef<For>(op);
      for_node.CopyOnWrite()->body = new_body;
      return for_node;
    }

    auto touched = CollectTouched(new_body);
    auto inner_claimed = CollectInnerClaimed(new_body);
    Array<Buffer> eligible;
    for (const auto &buf : touched) {
      if (!IsOnChipMultiBufferScope(buf))
        continue;
      if (parent_used.count(buf))
        continue;
      // An inner For has already claimed this buffer for multi-buffering;
      // the outer For must not also claim it.
      if (inner_claimed.count(buf))
        continue;
      WriteFirstClassifier cls;
      auto order = cls.Classify(new_body, buf);
      // kMaybeWriteFirst is also safe for multi-buffering: each iteration
      // either skips the buffer or writes-first to it.
      if (order == AccessOrder::kWriteFirst ||
          order == AccessOrder::kMaybeWriteFirst)
        eligible.push_back(buf);
    }

    For for_node = tvm::ffi::GetRef<For>(op);
    auto *n = for_node.CopyOnWrite();
    n->body = new_body;
    n->annotations.Set("multi_buffer_eligible", eligible);
    return for_node;
  }

  // Leaves in the IR structure: do not descend (matches AutoSchedule).
  //
  // Caveat for BlockNode: the AutoSchedule entry (TilelangRootBodyExtractor)
  // peels the outer `root` / `tilelang_root` blocks before building the IR
  // tree, so those wrappers must be passed through here. Only INNER blocks
  // (SIMT_VF / SIMD_VF / user-named blocks) become TaskNode leaves and should
  // not be descended into.
  Stmt VisitStmt_(const BlockNode *op) final {
    if (op->name_hint == "root" || op->name_hint == MainBlockName) {
      return StmtMutator::VisitStmt_(op);
    }
    return tvm::ffi::GetRef<Block>(op);
  }
  Stmt VisitStmt_(const WhileNode *op) final {
    return tvm::ffi::GetRef<While>(op);
  }
};

} // namespace

using namespace tir::transform;

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    auto *n = f.CopyOnWrite();
    n->body = MultiBufferAnnotator().Run(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateMultiBufferEligible", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMultiBufferEligible",
                        AnnotateMultiBufferEligible);
}

} // namespace tl
} // namespace tvm
