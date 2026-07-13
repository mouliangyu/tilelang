#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "support/check.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm::tl {
using namespace tirx;

// Replaces every mutable read (BufferLoad / ProducerLoad / Reduce / opaque
// call) in an expression with a fresh, independent Var.
//
// Purpose: when an expression will be compared against another inside ONE
// analyzer (a producer vs consumer access, or two binds that could alias), a
// shared mutable read would be wrongly assumed equal on both sides — a store
// may have changed it in between. Freshening makes each read an independent
// unknown, which is always sound (it only weakens equalities).
class FreshenMutableReads : public ExprMutator {
public:
  using ExprMutator::operator();

private:
  PrimExpr Fresh(const PrimExpr &e) {
    auto it = memo_.find(e);
    if (it != memo_.end())
      return it->second;
    Var v("free" + std::to_string(memo_.size()), e.dtype());
    memo_.emplace(e, v);
    return v;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const ProducerLoadNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const ReduceNode *op) override {
    return Fresh(ffi::GetRef<PrimExpr>(op));
  }
  PrimExpr VisitExpr_(const CallNode *op) override {
    // Opaque / stateful calls are unknowns; pure calls recurse so their pure
    // sub-structure is preserved.
    if (SideEffect(ffi::GetRef<PrimExpr>(op)) > CallEffectKind::kPure)
      return Fresh(ffi::GetRef<PrimExpr>(op));
    return ExprMutator::VisitExpr_(op);
  }

  std::unordered_map<PrimExpr, PrimExpr, tvm::ffi::StructuralHash,
                     ExprDeepEqual>
      memo_;
};

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(Var var, PrimExpr val) : kind(kBindValue), var(var), value(val) {};
  Constr(Var var, Range range) : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  void Format(std::ostream &os) const {
    os << "Constr(kind=";
    switch (kind) {
    case kConstr:
      os << "kConstr";
      os << ", is_assume=" << (is_assume ? "true" : "false");
      os << ", value=" << value;
      break;
    case kBindValue:
      os << "kBindValue";
      os << ", var=" << var->name_hint;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name_hint;
      os << ", range=Range(min=" << range->min;
      os << ", extent=" << range->extent << ")";
      break;
    default:
      os << "Unknown";
    }
    os << ")";
  }

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      if (var.dtype().is_vector())
        return Bool(true);
      return var == value;
    case kBindRange:
      return And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  Constr Substitute(ffi::Map<Var, PrimExpr> subs) const {
    switch (kind) {
    case kConstr:
      return Constr(tirx::Substitute(value, subs), is_assume);
    case kBindValue:
    case kBindRange: {
      auto it = subs.find(var);
      if (it != subs.end() && !(*it).second.as<VarNode>())
        return Constr(tirx::Substitute(ToGenericConstr(), subs), is_assume);
      Var new_var =
          it != subs.end() ? ffi::GetRef<Var>((*it).second.as<VarNode>()) : var;
      if (kind == kBindValue)
        return Constr(new_var, tirx::Substitute(value, subs));
      return Constr(
          new_var, Range::FromMinExtent(tirx::Substitute(range->min, subs),
                                        tirx::Substitute(range->extent, subs)));
    }
    }
    LOG(FATAL) << "Unreachable";
    return Constr();
  }
  void Populate(arith::Analyzer &analyzer) const {
    // analyzer.Bind() installs a rewrite `var -> value`, giving strong
    // reasoning (const-int-bound, modular-set, simplifier substitution). But it
    // is UNSOUND when `value` contains a mutable read: binding two different
    // vars to the same read makes the analyzer prove them equal, even though a
    // store may change the read between the binds — and ConstrVisitor does not
    // track stores. So freshen the mutable reads to independent unknowns first,
    // then Bind.
    switch (kind) {
    case kConstr:
      analyzer.EnterConstraint(value, is_assume);
      break;
    case kBindValue:
      analyzer.Bind(var, FreshenMutableReads()(value));
      break;
    case kBindRange: {
      FreshenMutableReads freshen;
      analyzer.Bind(var, Range::FromMinExtent(freshen(range->min),
                                              freshen(range->extent)));
      break;
    }
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<Var, PrimExpr> subs) const {
    ConstrSet new_set;
    for (const auto &c : constrs_) {
      new_set.constrs_.push_back(c.Substitute(subs));
    }
    return new_set;
  }
  // Rename `from` and every bind defined at-or-after it (in definition order)
  // by appending `suffix`; binds defined *before* `from` are left shared. If
  // `from` is undefined, ALL binds are renamed (no shared prefix). Vars
  // already present in `subs` (caller-seeded, e.g. thread vars) are left as
  // seeded. New renames are accumulated into `subs` so the caller can also
  // apply them to external expressions (region bounds) and read back a renamed
  // var (e.g. subs[from]). Returns the substituted copy of this set.
  ConstrSet RenameFrom(const std::string &suffix, ffi::Map<Var, PrimExpr> &subs,
                       const Var &from = Var()) const {
    bool active = !from.defined();
    for (const auto &c : constrs_) {
      if (c.kind != Constr::kBindValue && c.kind != Constr::kBindRange)
        continue;
      if (from.defined() && c.var.same_as(from))
        active = true;
      if (active && !subs.count(c.var))
        subs.Set(c.var, Var(c.var->name_hint + suffix, c.var.dtype()));
    }
    return Substitute(subs);
  }
  // Return a new set = this ∪ other. A kConstr predicate is kept unless a
  // structurally-identical predicate is already present (dedup). A bind
  // (kBindValue/kBindRange) whose var is already bound is deduped by var.
  // is_assume is cleared on the result: an assume is only trusted at the single
  // program point that stated it (and, being trusted, may reference impure
  // reads), so it must not be applied as a global fact once two points are
  // merged.
  ConstrSet Merge(const ConstrSet &other) const {
    ConstrSet out = *this;
    std::unordered_map<const VarNode *, Constr> bound;
    std::unordered_set<PrimExpr, tvm::ffi::StructuralHash, ExprDeepEqual> preds;
    for (const auto &c : out.constrs_) {
      if (c.kind == Constr::kBindValue || c.kind == Constr::kBindRange)
        bound.emplace(c.var.get(), c);
      else if (c.kind == Constr::kConstr)
        preds.insert(c.value);
    }
    for (const auto &c : other.constrs_) {
      if (c.kind == Constr::kConstr) {
        if (preds.insert(c.value).second)
          out.constrs_.push_back(c);
        continue;
      }
      auto it = bound.find(c.var.get());
      if (it == bound.end()) {
        bound.emplace(c.var.get(), c);
        out.constrs_.push_back(c);
        continue;
      }
      const Constr &e = it->second;
      bool same = e.kind == c.kind &&
                  (c.kind == Constr::kBindValue
                       ? ExprDeepEqual()(e.value, c.value)
                       : (ExprDeepEqual()(e.range->min, c.range->min) &&
                          ExprDeepEqual()(e.range->extent, c.range->extent)));
      if (!same) {
        std::ostringstream os_e, os_c;
        e.Format(os_e);
        c.Format(os_c);
        LOG(WARNING) << "ConstrSet::Merge: var '" << c.var->name_hint
                     << "' bound to conflicting values across merged sets; "
                        "caller should rename per-side-varying vars. Dropping "
                        "the incoming bind. existing="
                     << os_e.str() << " incoming=" << os_c.str();
      }
    }
    for (auto &c : out.constrs_) {
      c.is_assume = false;
    }
    return out;
  }
  // Lower every bind to a generic predicate (kConstr), leaving existing kConstr
  // entries as-is. Use this when merging into an analyzer that ALSO Binds one
  // of the shared vars externally: keeping the binds would re-Bind that var and
  // trip the analyzer's re-bind check, so the whole set is downgraded to
  // predicates, which coexist with any external Bind.
  ConstrSet ToConstraints() const {
    ConstrSet out;
    for (const auto &c : constrs_)
      out.constrs_.push_back(Constr(c.ToGenericConstr(), c.is_assume));
    return out;
  }
  void Populate(arith::Analyzer &analyzer) const {
    for (const auto &c : constrs_) {
      c.Populate(analyzer);
    }
  }
  bool CanProve(const PrimExpr &expr) const {
    arith::Analyzer analyzer;
    Populate(analyzer);
    return analyzer.CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return Bool(true);
    PrimExpr result = constrs_[0].ToGenericConstr();
    for (size_t i = 1; i < constrs_.size(); ++i) {
      result = And(result, constrs_[i].ToGenericConstr());
    }
    return result;
  }

  void Format(std::ostream &os) const {
    os << "ConstrSet(size=" << constrs_.size() << ") {\n";
    for (size_t i = 0; i < constrs_.size(); ++i) {
      os << "  [" << i << "] ";
      constrs_[i].Format(os);
      os << "\n";
    }
    os << "}";
  }

  std::vector<Constr> constrs_;
};

struct ConstrVisitor : public StmtExprVisitor {
private:
  using Base = StmtExprVisitor;

protected:
  struct Guard {
    std::vector<Constr> *constrs;
    Guard(std::vector<Constr> &c) : constrs(&c) {}
    Guard(Guard &&other) noexcept : constrs(other.constrs) {
      other.constrs = nullptr;
    }
    Guard &operator=(Guard &&other) noexcept {
      if (this != &other) {
        if (constrs)
          constrs->pop_back();
        constrs = other.constrs;
        other.constrs = nullptr;
      }
      return *this;
    }
    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;
    ~Guard() {
      if (constrs)
        constrs->pop_back();
    }
  };

  template <typename... Args> Guard MakeGuard(const Args... args) {
    constr_stack_.push_back(Constr(args...));
    return Guard{constr_stack_};
  }

public:
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;
  void VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                           const PrimExpr false_value) {
    // Visit the condition first without any guard, as it is always evaluated
    // This ensures any buffer accesses in the condition are recorded
    Base::VisitExpr(cond);
    {
      auto guard = MakeGuard(cond);
      Base::VisitExpr(true_value);
    }
    {
      auto guard = MakeGuard(Not(cond));
      Base::VisitExpr(false_value);
    }
  }
  void VisitStmt_(const BindNode *op) override {
    auto guard = MakeGuard(op->var, op->value);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const SeqStmtNode *op) override {
    std::vector<Guard> bind_guards;
    for (const auto &stmt : op->seq) {
      VisitStmt(stmt);
      if (const auto *bind = stmt.as<BindNode>())
        bind_guards.push_back(MakeGuard(bind->var, bind->value));
    }
  }
  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tirx::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
      Base::VisitStmt_(op);
    } else if (op->attr_key == tirx::attr::thread_extent ||
               op->attr_key == s_tir::attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      Range dom = Range::FromMinExtent(make_zero(op->value.dtype()), op->value);
      auto guard = MakeGuard(iv->var, dom);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const AssertStmtNode *op) override {
    auto guard = MakeGuard(op->condition);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const IfThenElseNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      auto guard = MakeGuard(Not(op->condition));
      Base::VisitStmt(op->else_case.value());
    }
  }
  void VisitExpr_(const SelectNode *op) override {
    VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }
  void VisitExpr_(const CallNode *op) override {
    static auto op_if_then_else = Op::Get("tirx.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      Base::VisitExpr_(op);
    }
  }
  void VisitStmt_(const ForNode *op) override {
    auto guard_1 =
        MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    auto guard_2 = MakeGuard(op->extent > 0);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const WhileNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->body);
    }
  }
  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
