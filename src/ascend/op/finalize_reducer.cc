/*!
 * \file tl/ascend/op/finalize_reducer.cc
 * \brief Ascend implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "op/finalize_reducer.h"

#include "backend/common/target_utils.h"

#include <tvm/tirx/builtin.h>

#include <array>
#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

Stmt LowerFinalizeReducer(const FinalizeReducerOpNode &op, const LowerArgs &T,
                          arith::Analyzer *) {
  auto buffer = T.buffer_remap[op.reducer];
  auto opt_layout = T.layout_map.Get(op.reducer);
  ICHECK(opt_layout);
  ICHECK(opt_layout->as<Fragment>());
  auto layout = opt_layout->as<Fragment>().value();
  Array<PrimExpr> indices_0;
  indices_0.reserve(layout->OutputDim());
  for (int i = 0; i < layout->OutputDim(); ++i) {
    indices_0.push_back(Var("__finred_" + std::to_string(i)));
  }

  const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
  ICHECK(p_extent);
  int extent = *p_extent;
  ICHECK(extent == 1 || extent == *as_const_int(T.thread_bounds->extent))
      << "Illegal finalize_reducer: extent=" << extent
      << "; T.thread_bounds=" << T.thread_bounds;

  if (extent == 1) {
    return Evaluate(0);
  }

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
  auto op_str = op_names[static_cast<int>(op.op)];

  int reducing_threads = extent;

  int64_t layout_batch_size = 1;
  for (int i = 0; i < layout->OutputDim(); ++i) {
    const int64_t *p = as_const_int(layout->OutputShape()[i]);
    if (p == nullptr) {
      layout_batch_size = -1;
      break;
    }
    layout_batch_size *= *p;
  }

  int64_t effective_batch = static_cast<int64_t>(op.batch);
  if (effective_batch > 1 && layout_batch_size > 0) {
    ICHECK_LE(effective_batch, layout_batch_size)
        << "finalize_reducer: batch (" << effective_batch
        << ") exceeds total output elements (" << layout_batch_size << ")";
    ICHECK_EQ(layout_batch_size % effective_batch, 0)
        << "finalize_reducer: batch (" << effective_batch
        << ") must evenly divide total output elements (" << layout_batch_size
        << ")";
  }

  std::stringstream ss;
  auto thread_offset = T.thread_bounds->min;
  ss << "tl::AscendAllReduce<" << op_str << ", " << reducing_threads << ", "
     << 1 << ", " << thread_offset << ">::run";

  Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                        BufferLoad(buffer, indices_0)};
  bool is_pow2 = (reducing_threads & (reducing_threads - 1)) == 0;
  bool need_workspace = false;
  if (reducing_threads <= 32 && is_pow2) {
    need_workspace = false;
  } else if (reducing_threads > 32 && is_pow2) {
    need_workspace = true;
  } else {
    need_workspace = reducing_threads > 1;
  }
  if (need_workspace) {
    PrimExpr workspace =
        T.AddWorkspace(*as_const_int(T.thread_bounds->extent), buffer->dtype);
    thread_reduce_args.push_back(workspace);
  }
  auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
  Stmt body = BufferStore(buffer, call, indices_0);

  for (int i = layout->OutputDim() - 1; i >= 0; --i) {
    body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
               ForKind::kParallel, body);
  }

  return body;
}

} // namespace ascend

namespace {

bool MatchAscendFinalizeReducerTarget(Target target) {
  return TargetIsAscend(target);
}

bool RegisterAscendFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "ascend.FinalizeReducer",
      MatchAscendFinalizeReducerTarget,
      ascend::LowerFinalizeReducer,
  });
  return true;
}

const bool ascend_finalize_reducer_registered = RegisterAscendFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
