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
// Entry point: filter the root ordered list for a warpgroup.
std::vector<std::shared_ptr<IRStructure>> CloneIRStructureWithWarpgroupFilter(
    const std::vector<std::shared_ptr<IRStructure>> &root, int warpgroup_id);

// Emit one warpgroup body
Stmt EmitWarpgroupBody(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                       Var sid = Var());

Stmt ReNestLetStmts(const Stmt &stmt);

} // namespace tl
} // namespace tvm
