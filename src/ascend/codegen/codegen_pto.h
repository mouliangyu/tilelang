/*!
 * \file ascend/codegen/codegen_pto.h
 * \brief Utility to generate PTO Python source.
 */
#ifndef TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
#define TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cuda/codegen/codegen_py.h"

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
  void VisitStmt_(const BufferStoreNode *op) override;
  void VisitStmt_(const BindNode *op) override;
  void VisitStmt_(const AllocBufferNode *op) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const SBlockNode *op) override;
  void VisitStmt_(const IfThenElseNode *op) override;
  void VisitStmt_(const EvaluateNode *op) override;

  void VisitExpr_(const BroadcastNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) override;                     // NOLINT(*)
  void VisitExpr_(const CastNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const AndNode *op, std::ostream &os) override;  // NOLINT(*)
  void VisitExpr_(const OrNode *op, std::ostream &os) override;   // NOLINT(*)
  void VisitExpr_(const SelectNode *op,
                  std::ostream &os) override; // NOLINT(*)

private:
  struct FragmentInfo {
    int lanes{0};
    DataType dtype;
  };

  struct PTOGemmEmitContext {
    bool initialized{false};
    int64_t tile_m{0};
    int64_t tile_n{0};
    int64_t tile_k{0};
    int64_t base_k{0};
    DataType input_dtype;
    DataType accum_dtype;
    std::string helper_name;
    std::string a_l0_name;
    std::string b_l0_name;
  };

  std::string PtoScalarType(DataType t) const;
  std::string PtoPtrType(DataType t, const std::string &space) const;
  std::string GetAccessPtrExpr_(const CallNode *op);
  std::string GetPtoPointerExpr(const VarNode *buffer_var, DataType elem_dtype,
                                const PrimExpr &index);
  std::string GetPtoPointerExpr(const BufferNode *buffer,
                                const PrimExpr &index);
  std::string GetAddressOfExpr_(const CallNode *op);
  std::string GetAscendCopyGmUbExpr_(const CallNode *op);
  std::string GetAscendCopyUbGmExpr_(const CallNode *op);
  std::string EmitPTOAllReduceExpr_(const std::string &func_name,
                                    const CallNode *op);
  ffi::Array<Var> CollectVFCaptures(const SBlockNode *op) const;
  void EmitSimtVFFunction(const SBlockNode *op, const ffi::Array<Var> &captures,
                          const std::string &helper_name, int64_t thread_x,
                          int64_t thread_y, int64_t thread_z);
  void ExtractSimtThreadExtents(const SBlockNode *op, int64_t *thread_x,
                                int64_t *thread_y, int64_t *thread_z) const;
  void EmitSimtVFLaunch(const ffi::Array<Var> &captures,
                        const std::string &helper_name, int64_t thread_x,
                        int64_t thread_y, int64_t thread_z);
  std::string ResolveVarName(const Var &v) const;
  std::string PtoScalarLoad(const BufferNode *buffer, const PrimExpr &index);
  void EmitPtoScalarStore(const BufferNode *buffer, const std::string &value,
                          const PrimExpr &index);
  void EmitPtoBufferAllocation(const Buffer &buffer);
  void EmitScalarizedLoad(const BufferLoadNode *op, std::ostream &os);
  void EmitScalarizedStore(const BufferStoreNode *op);
  std::string ScopeOfBuffer(const BufferNode *buffer) const;
  void PrintBinaryExpr_(const std::string &opstr, DataType dtype, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) override; // NOLINT(*)

  std::string current_function_name_;
  int simtvf_helper_counter_{0};
  bool inside_simtvf_body_{false};
  std::string GetPtoLocalPtrExpr(const PrimExpr &expr, const std::string &space,
                                 DataType fallback_dtype);
  std::string GetPtoLocalByteAddrExpr(const PrimExpr &index,
                                      DataType elem_dtype,
                                      const std::string &context);
  std::string GetPtoAccPtrExpr(const PrimExpr &expr, DataType dtype);
  void EnsurePTOGemmHelper(const CallNode *op);
  void EmitAscendCopyGmToCbuf(const CallNode *op);
  void EmitAscendGemmL1(const CallNode *op);
  void EmitAscendCopyMatrixCcToGm(const CallNode *op);
  void EmitPTOGemmRun(const std::string &a_mat, const std::string &b_mat,
                      const std::string &acc, const std::string &clear_accum,
                      const std::string &unit_flag_ctrl, int64_t hf32_mode);
  std::string PrintVmiAnnotationValue(const std::string &key,
                                      const ObjectRef &value);
  void PrintPtoVmiCall_(const CallNode *op, std::ostream &os);
  std::string LocalVarID(const VarNode *var);
  bool IsLocalVarBuffer(const VarNode *var) const;
  bool IsVmiLocalRegisterBuffer(const BufferNode *buffer) const;
  void CheckVmiLocalRegisterIndex(const BufferNode *buffer,
                                  const PrimExpr &index) const;
  bool HasAscendGemmL1(const PrimFunc &func) const;

  bool current_function_has_gemm_{false};
  bool has_gemm_l1_{false};
  std::unordered_map<const VarNode *, FragmentInfo> fragment_info_;
  std::unordered_set<const VarNode *> local_var_buffers_;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      hf32_mode_by_gemm_;
  PTOGemmEmitContext gemm_emit_ctx_;
  std::pair<std::string, std::string>
  ParseHardEventPair(const std::string &hard_event) const;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_PTO_CODEGEN_CODEGEN_PTO_H_
