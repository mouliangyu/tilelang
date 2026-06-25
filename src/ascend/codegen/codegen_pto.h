/*!
 * \file ascend/codegen/codegen_pto.h
 * \brief Utility to generate PTO Python source.
 */
#ifndef TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
#define TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_

#include "cuda/codegen/codegen_py.h"

#include <string>

namespace tvm {
namespace codegen {

class CodeGenTileLangPTO final : public CodeGenTileLangPY {
public:
  void AddFunction(const GlobalVar &gvar, const PrimFunc &func) override;
  std::string Finish() override;

protected:
  void PrintFuncDecorator_(std::ostream &os) override; // NOLINT(*)
  void PrintFunctionSignature_(const ffi::String &function_name,
                               const PrimFunc &func,
                               std::ostream &os) override; // NOLINT(*)

  void VisitStmt_(const DeclBufferNode *op) override;
  void VisitStmt_(const BindNode *op) override;
  void VisitStmt_(const AllocBufferNode *op) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const SBlockNode *op) override;
  void VisitStmt_(const IfThenElseNode *op) override;
  void VisitStmt_(const EvaluateNode *op) override;

  void VisitExpr_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const AndNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const OrNode *op, std::ostream &os) override;   // NOLINT(*)

private:
  std::string PtoScalarType(DataType t) const;
  std::string PtoPtrType(DataType t, const std::string &space) const;
  std::string GetPtoPointerExpr(const VarNode *buffer_var, DataType elem_dtype,
                                const PrimExpr &index);
  std::string GetPtoPointerExpr(const BufferNode *buffer,
                                const PrimExpr &index);
  std::string GetAddressOfExpr_(const CallNode *op);
  std::string GetAccessPtrExpr_(const CallNode *op);
  std::string GetAscendCopyGmUbExpr_(const CallNode *op);
  std::string GetAscendCopyUbGmExpr_(const CallNode *op);
  std::pair<std::string, std::string>
  ParseHardEventPair(const std::string &hard_event) const;

  std::string current_function_name_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
