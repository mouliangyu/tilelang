/*!
 * \file annotate_multi_buffer_eligible.cc
 * \brief Pre-AutoSchedule analysis: for each For loop, decide which on-chip
 *        buffers can be multi-buffered and record the set on the For's
 *        annotations under "multi_buffer_eligible".
 *
 *  Runs after NormalizeControlFlowForSchedule (which rewrites `while` loops
 *  into bounded serial for loops and hoists complex if conditions into Bind
 *  variables). Operating after that rewrite lets this pass mark buffers inside
 * a former while body as multi-buffer eligible; the hoisted Bind conditions
 * keep the if-then-else intact so the write-first analysis is unaffected.
 *
 *  A buffer is eligible inside For L iff
 *    1) it lives in on-chip memory (shared / shared.l1 / shared.l0c, including
 *       their .dyn variants),
 *    2) it is not referenced anywhere outside L (siblings or ancestors),
 *    3) inside L's body it is always written before being read.
 *
 *  Frontend may pre-set the annotation. We treat it as a seed rather than a
 *  veto: the frontend-listed buffers are always kept, and any buffer the
 *  analysis additionally proves eligible is unioned in. A buffer the user does
 *  NOT want multi-buffered can be pinned to a single version via
 *  T.annotate_buffer_versions({buf: 1}) instead of being omitted here.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/buffer_version.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using StorageSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

Array<Var> NormalizeEligibleStorages(const Any &annotation) {
  Array<Var> result;
  StorageSet seen;
  for (const Any &item : Downcast<Array<Any>>(annotation)) {
    Var storage;
    if (auto buffer = item.try_cast<Buffer>()) {
      storage = buffer.value()->data;
    } else if (auto var = item.try_cast<Var>()) {
      storage = var.value();
    } else {
      LOG(FATAL) << "'" << kMultiBufferEligible
                 << "' entries must be Buffer or Var objects, got " << item;
    }
    if (seen.insert(storage).second)
      result.push_back(storage);
  }
  return result;
}

// ---------------------------------------------------------------------------
// WriteFirstClassifier: program-order classification of a storage's access
// pattern inside a Stmt. All Buffer aliases sharing buffer->data participate.
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
  AccessOrder Classify(const Stmt &s, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(s);
    return result_;
  }

private:
  Var target_storage_;
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
    if (target_storage_.same_as(op->buffer->data))
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
    if (target_storage_.same_as(op->buffer->data))
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

  void VisitStmt_(const BindNode *op) final { VisitExpr(op->value); }

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

  void VisitStmt_(const SBlockNode *op) final {
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

  void VisitStmt_(const SBlockRealizeNode *op) final {
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
    if (target_storage_.same_as(bl->buffer->data)) {
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
    if (target_storage_.same_as(bl->buffer->data)) {
      int rw = mask->value;
      if (rw & 1)
        result_ = AccessOrder::kReadFirst;
      else if (rw & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }
};

// ---------------------------------------------------------------------------
// StorageAccessCollector: one-shot scan of a Stmt to collect every on-chip
// storage that is read or written anywhere inside it, preserving first-access
// order.
// ---------------------------------------------------------------------------
class StorageAccessCollector : public StmtExprVisitor {
public:
  std::vector<Var> touched;

  void Run(const Stmt &s) { StmtExprVisitor::VisitStmt(s); }

private:
  StorageSet seen_;

  void Record(const Buffer &buffer) {
    if (!IsAscendOnChipBuffer(buffer))
      return;
    const Var &storage = buffer->data;
    if (seen_.insert(storage).second)
      touched.push_back(storage);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &r : op->reads)
      Record(r->buffer);
    for (const auto &w : op->writes)
      Record(w->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    static const auto region_op = Op::Get("tl.tileop.region");
    static const auto access_ptr_op = Op::Get("tl.access_ptr");
    if (op->op.same_as(region_op) && !op->args.empty()) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        Record(bl->buffer);
    } else if (op->op.same_as(access_ptr_op) && !op->args.empty()) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        Record(bl->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

public:
  // (no extra state needed; access_ptr already carries a BufferLoad)
};

// ---------------------------------------------------------------------------
// MultiBufferAnnotator: walk the PrimFunc body, mirroring the recursion that
// the initial scheduled-TIR builder will perform. Serial and unrolled loops
// become structural ControlNodes and may own an automatic version ring; other
// loops and blocks are opaque leaves.
//
// Recursion mirrors MaterializeScheduleUnits:
//   - SeqStmt          -> walk children
//   - For (serial/unrolled) -> ControlNode: annotate + walk body
//   - For (other kind)      -> TaskNode: leaf, do not walk
//   - IfThenElse       -> walk both branches
//   - LetStmt          -> leaf
//   - AttrStmt         -> walk tl.assume body, reject other attrs downstream
//   - Block / While    -> TaskNode: leaf
//   - other            -> leaf
// ---------------------------------------------------------------------------
class MultiBufferAnnotator : public StmtMutator {
public:
  explicit MultiBufferAnnotator(StorageSet manual = {})
      : manual_buffers_(std::move(manual)) {}

private:
  StorageSet used_storages_;
  // Data Vars of buffers the user manually multi-buffered (from
  // `tl.manual_multi_buffer`). These are excluded from eligibility: the pass
  // must not add a version dim / rewrite them; the barrier pass handles their
  // synchronization directly.
  StorageSet manual_buffers_;

  static std::vector<Var> CollectTouched(const Stmt &s) {
    StorageAccessCollector c;
    c.Run(s);
    return std::move(c.touched);
  }

  static StorageSet CollectTouchedStorages(const Stmt &s) {
    StorageSet storages;
    for (const Var &storage : CollectTouched(s))
      storages.insert(storage);
    return storages;
  }

  // Walk `s` and collect every storage that any nested For has already claimed
  // via kMultiBufferEligible. Used so that an outer For doesn't re-claim a
  // storage that an inner For will multi-buffer — the storage is
  // owned by the innermost claimant.
  static StorageSet CollectInnerClaimed(const Stmt &s) {
    struct Visitor : public StmtVisitor {
      StorageSet claimed;
      void VisitStmt_(const ForNode *op) final {
        auto v = op->annotations.Get(kMultiBufferEligible);
        if (v.has_value()) {
          for (const Var &storage : NormalizeEligibleStorages(v.value()))
            claimed.insert(storage);
        }
        StmtVisitor::VisitStmt_(op);
      }
    } v;
    v(s);
    return std::move(v.claimed);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // Pre-scan each child's touched set so we can augment used_storages_ for
    // recursion into the others.
    const size_t n = op->seq.size();
    std::vector<StorageSet> per_child(n);
    for (size_t i = 0; i < n; ++i)
      per_child[i] = CollectTouchedStorages(op->seq[i]);

    Array<Stmt> new_seq;
    new_seq.reserve(n);
    auto saved = used_storages_;
    for (size_t i = 0; i < n; ++i) {
      used_storages_ = saved;
      for (size_t j = 0; j < n; ++j) {
        if (j == i)
          continue;
        used_storages_.insert(per_child[j].begin(), per_child[j].end());
      }
      new_seq.push_back(VisitStmt(op->seq[i]));
    }
    used_storages_ = saved;
    return SeqStmt(std::move(new_seq));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Serial and unrolled loops are structural ControlNodes and may own an
    // automatic version ring. Parallel/vectorized loops remain opaque tasks.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      return GetRef<For>(op);
    }

    auto parent_used = used_storages_;

    Stmt new_body = VisitStmt(op->body);
    used_storages_ = parent_used;

    // Seed the eligible set with any frontend-provided annotation, then union
    // in everything the analysis proves safe. The frontend list is a hint of
    // buffers the user definitely wants multi-buffered; it does not suppress
    // additional candidates (the user pins unwanted ones to a single version
    // via T.annotate_buffer_versions instead).
    auto inner_claimed = CollectInnerClaimed(new_body);
    Array<Var> eligible;
    StorageSet already;
    if (auto v = op->annotations.Get(kMultiBufferEligible)) {
      for (const Var &storage : NormalizeEligibleStorages(v.value())) {
        if (!inner_claimed.count(storage) && already.insert(storage).second)
          eligible.push_back(storage);
      }
    }

    std::vector<Var> touched = CollectTouched(new_body);
    for (const Var &storage : touched) {
      if (already.count(storage))
        continue;
      // Manually multi-buffered buffers are never pass-versioned.
      if (manual_buffers_.count(storage))
        continue;
      if (parent_used.count(storage))
        continue;
      // An inner For has already claimed this buffer for multi-buffering;
      // the outer For must not also claim it.
      if (inner_claimed.count(storage))
        continue;
      WriteFirstClassifier cls;
      auto order = cls.Classify(new_body, storage);
      // kMaybeWriteFirst is also safe for multi-buffering: each iteration
      // either skips the buffer or writes-first to it.
      if (order == AccessOrder::kWriteFirst ||
          order == AccessOrder::kMaybeWriteFirst) {
        already.insert(storage);
        eligible.push_back(storage);
      }
    }

    For for_node = GetRef<For>(op);
    auto *n = for_node.CopyOnWrite();
    n->body = new_body;
    n->annotations.Set(kMultiBufferEligible, eligible);
    return for_node;
  }

  // Leaves in the IR structure: do not descend (matches AutoSchedule).
  //
  // The callback runs this visitor on tilelang_root's body. Every block below
  // that root is an opaque TaskNode leaf.
  Stmt VisitStmt_(const SBlockNode *op) final { return GetRef<SBlock>(op); }
  Stmt VisitStmt_(const WhileNode *op) final { return GetRef<While>(op); }
};

} // namespace

using namespace tirx::transform;

// Collect the data Vars named in the `tl.manual_multi_buffer` annotation on any
// SBlock in the function body.
static StorageSet CollectManualMultiBuffers(const Stmt &body) {
  struct Visitor : public StmtVisitor {
    StorageSet vars;
    void VisitStmt_(const SBlockNode *op) final {
      if (auto annotation = op->annotations.Get(kManualMultiBuffer)) {
        for (const auto &[data, _] :
             annotation.value().cast<BufferVersionMap>()) {
          vars.insert(data);
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  } v;
  v(body);
  return std::move(v.vars);
}

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "AnnotateMultiBufferEligible",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          auto manual = CollectManualMultiBuffers(root);
          root.CopyOnWrite()->body =
              MultiBufferAnnotator(std::move(manual))(root->body);
          return root;
        },
        /*require_kernel=*/false);
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
