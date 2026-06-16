#ifndef TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
#define TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_

#include "target/source/codegen_c.h"
#include <tvm/arith/analyzer.h>

namespace tvm {
namespace codegen {

class CodeGenTileLangAscend final : public CodeGenC {
public:
  using CodeGenC::PrintType;

  void Init(bool output_ssa);
  void AddFunction(const PrimFunc &f);
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintCCEVectorSuffix(DataType t, std::ostream &os);
  void VisitStmt_(const SBlockNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const BindNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  bool IsScopePartOfType() const final;

  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  std::string GetBufferRef(DataType t, const BufferNode *buffer,
                           PrimExpr index) final;

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
  bool has_simd_ops_{false};
  bool has_simd_inst_included_{false};

  ffi::Array<Var> CollectVFCaptures(const SBlockNode *op) const;
  void EmitVFFunction(const SBlockNode *op, const ffi::Array<Var> &captures,
                      const std::string &helper_name,
                      const std::string &func_attrs, bool &inside_vf_body);

  enum class AscendKernelMode { kVector, kCube, kMix };
  AscendKernelMode kernel_mode_{AscendKernelMode::kVector};
  int mix_aiv_count_{2};
  bool enable_fast_math_{false};

  // Accumulates constraints along the traversal
  arith::Analyzer analyzer_;

  bool VectorAccessIsAligned(const PrimExpr &index, int lanes);

  void EmitScalarizedLoad(const BufferLoadNode *op, std::ostream &os);
  bool EmitScalarizedStore(const BufferStoreNode *op);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_ASCEND_CODEGEN_CODEGEN_ASCEND_H_
