/*!
 * \file restore_while_loops.cc
 * \brief Restore `while` loops that NormalizeControlFlowForSchedule turned into
 *        bounded serial for loops so AutoSchedule could pipeline their bodies.
 *
 * For every For tagged with the `synthetic_while` annotation, rewrite
 *   `for w in range(0, W): BODY`
 * back into
 *   `while(true): BODY[w := 0]`
 * The real termination test is the `if (not cond): loop_break()` guard that
 * NormalizeControlFlowForSchedule placed at the top of BODY; it is preserved.
 *
 * Substituting w:=0 is safe because the synthetic for is never itself
 * pipelined (num_stages unset), so no buffer is multi-versioned at the w level;
 * w appears in version/flag subscripts only through the inner Pipelined loops,
 * where its coefficient (M_inner * W, W = lcm(1..max_num_stages)) is divisible
 * by every num_versions and thus already dropped out under Simplify. Any
 * residual `w` term is a multiple of num_versions, so w:=0 preserves the
 * version ring index.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class WhileLoopRestorer : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = WhileLoopRestorer();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  WhileLoopRestorer() = default;

  Stmt VisitStmt_(const ForNode *op) final {
    // Recurse first so nested synthetic-while fors are restored bottom-up.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto tag = for_node->annotations.Get("synthetic_while");
    if (!tag.has_value()) {
      return for_node;
    }
    Stmt body = tirx::Substitute(
        for_node->body,
        Map<Var, PrimExpr>{
            {for_node->loop_var, IntImm(for_node->loop_var.dtype(), 0)}});
    return While(IntImm(DataType::Bool(), 1), body);
  }
};

using namespace tirx::transform;
tvm::transform::Pass RestoreWhileLoops() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return WhileLoopRestorer::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RestoreWhileLoops", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RestoreWhileLoops", RestoreWhileLoops);
}

} // namespace tl
} // namespace tvm
