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
  void VisitStmt_(const tir::LetStmtNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  bool IsScopePartOfType() const final;

  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;

  // Vector element access (use .x/.y/.z/.w like CUDA, not .s0/.s1 from base)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final;
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;

protected:
  // Scalarize vector math calls (like CUDA)
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final;

  // Scalarize vector binary ops (add/sub/mul/div) into per-element ops
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs, std::ostream &os) final;

private:
  int simtvf_helper_counter_{0};
  bool inside_simtvf_body_{false};
  int simdvf_helper_counter_{0};
  bool inside_simdvf_body_{false};
  bool has_gemm_l0_{false};
  bool has_gemm_l1_{false};
  bool has_gemm_l1_included_{false};
  bool has_nd2nz_copy_{false};
  bool has_nd2nz_copy_included_{false};
  bool has_simdvf_included_{false};
  bool has_limits_included_{false};
  bool has_bf16_cast_{false};
  bool has_bf16_cast_included_{false};

  enum class AscendKernelMode { kVector, kCube, kMix };
  AscendKernelMode kernel_mode_{AscendKernelMode::kVector};
  int mix_aiv_count_{2};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_ASCEND_H_
