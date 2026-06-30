/*!
 * \file if_condition_extract.cc
 * \brief Extract complex if conditions into temporary Bind variables.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class IfConditionExtractor : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = IfConditionExtractor();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  IfConditionExtractor() = default;

  // counter to generate unique name for each IfStmt
  int counter_ = 0;

  //! \brief Check if the expression is a simple variable.
  bool IsSimpleVar(const PrimExpr &expr) {
    return expr.as<VarNode>() != nullptr;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }

    if (IsSimpleVar(condition)) {
      // The condition is already a simple variable; keep the if as-is.
      return IfThenElse(condition, then_case, else_case);
    }

    // Extract the complex condition into a temporary bind variable so its
    // value is computed once and consumers (e.g. AutoSchedule) see a flat var,
    // but keep the if-then-else as a single node. AutoSchedule's
    // IRStructureBuilder already handles both branches natively.
    std::string var_name = "__cond_" + std::to_string(counter_++);
    Var cond_var(var_name, DataType::Bool());
    Stmt new_if = IfThenElse(cond_var, then_case, else_case);
    return SeqStmt({tirx::Bind(cond_var, condition), new_if});
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (auto stmt : op->seq) {
      auto new_stmt = VisitStmt(stmt);
      if (!new_stmt.defined())
        continue;
      if (auto seq_node = new_stmt.as<SeqStmtNode>()) {
        seq.insert(seq.end(), seq_node->seq.begin(), seq_node->seq.end());
      } else {
        seq.push_back(new_stmt);
      }
    }
    return SeqStmt(std::move(seq));
  }
};

using namespace tirx::transform;
tvm::transform::Pass IfConditionExtract() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return IfConditionExtractor::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.IfConditionExtract", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.IfConditionExtract", IfConditionExtract);
}

} // namespace tl
} // namespace tvm
