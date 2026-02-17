#ifndef TVM_TL_TARGET_CODEGEN_ASCEND_H_
#define TVM_TL_TARGET_CODEGEN_ASCEND_H_

#include "codegen_cpp.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenTileLangCPP {
public:
  void Init(bool output_ssa);
  void AddFunction(const PrimFunc& f);
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void PrintStorageSync(const CallNode *op) final;

private:
  int simtvf_helper_counter_{0};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_ASCEND_H_
