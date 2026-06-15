#pragma once
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "./barrier.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Map;

struct WarpSpecializeConfig;

class VarRefCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> vars;
  void VisitExpr_(const VarNode *op) override { vars.insert(op); }
};

bool IsLetDeclTask(const TaskNode *task);
bool IsLetDeclNode(const IRStructure *node);
bool ContainsLetDecl(const IRStructure *node);

std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap);
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id);

std::shared_ptr<IRStructure>
RemoveUnusedLetDecls(std::shared_ptr<IRStructure> root);

Stmt ConvertIRStructureToStmt(IRStructure *root, Var sid = Var());

Stmt ReNestLetStmts(const Stmt &stmt);

} // namespace tl
} // namespace tvm
