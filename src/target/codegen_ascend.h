#ifndef TVM_TL_TARGET_CODEGEN_ASCEND_H_
#define TVM_TL_TARGET_CODEGEN_ASCEND_H_

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenC {
public:
  void Init(bool output_ssa);
  void AddFunction(const PrimFunc &f);
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void VisitStmt_(const tir::BlockNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  bool IsScopePartOfType() const final;

private:
  int simtvf_helper_counter_{0};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_ASCEND_H_
