#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "./ir_structure.h"
#include "op/builtin.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_versions;
  Buffer new_buffer;
  std::vector<For> claim_loops;

  MultiVersionBufferInfo(Buffer buffer, int num_versions, Buffer new_buffer,
                         std::vector<For> claim_loops = {})
      : buffer(buffer), num_versions(num_versions), new_buffer(new_buffer),
        claim_loops(std::move(claim_loops)) {}
};

using MultiVersionBufferMap = std::unordered_map<Buffer, MultiVersionBufferInfo,
                                                 ObjectPtrHash, ObjectPtrEqual>;

struct FlagInfo {
  int warpgroup_id;
  PrimExpr flag_var;
  std::string hardevent;
  int mode;
};
using FlagInfoList = std::vector<FlagInfo>;

void AnalyzeAndInsertBarriers(std::vector<std::shared_ptr<IRStructure>> &root,
                              FlagInfoList &flag_infos,
                              std::vector<MultiVersionBufferInfo> &buffer_infos,
                              MultiVersionBufferMap &buffer_info_map,
                              const ManualMultiBufferTable &manual = {});

inline Stmt MakeAscendSetFlag(const std::string &hard_event,
                              PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_set_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendWaitFlag(const std::string &hard_event,
                               PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_wait_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendPipeBarrier(const std::string &pipe) {
  return Evaluate(
      Call(DataType::Handle(), ascend_pipe_barrier(), {StringImm(pipe)}));
}

inline Stmt MakeAscendCrossCoreSetFlag(int mode_id, const std::string &pipe,
                                       PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_set_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

inline Stmt MakeAscendCrossCoreWaitFlag(int mode_id, const std::string &pipe,
                                        PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_wait_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  explicit MultiBufferAccessRewriter(const MultiVersionBufferMap &multi_buffer)
      : multi_buffer_(multi_buffer) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = multi_buffer_.find(load->buffer);
    if (it != multi_buffer_.end()) {
      const auto &info = it->second;
      auto *n = load.CopyOnWrite();
      n->buffer = info.new_buffer;
      PrimExpr iteration = CalculateIterationCount(info.claim_loops);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info.num_versions)));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = multi_buffer_.find(store->buffer);
    if (it != multi_buffer_.end()) {
      const auto &info = it->second;
      auto *n = store.CopyOnWrite();
      n->buffer = info.new_buffer;
      PrimExpr iteration = CalculateIterationCount(info.claim_loops);
      n->indices.insert(n->indices.begin(),
                        indexmod(iteration, Integer(info.num_versions)));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        auto it = multi_buffer_.find(buffer_load->buffer);
        if (it != multi_buffer_.end()) {
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(VisitExpr(op->args[0]));
          new_args.push_back(VisitExpr(op->args[1]));
          new_args.push_back(IntImm(DataType::Int(32), 1));
          for (size_t i = 2; i < op->args.size(); i++) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          return Call(op->dtype, op->op, new_args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const MultiVersionBufferMap &multi_buffer_;
};

// Rewrites multi-buffered accesses that are outside the owning loop.
class BroadcastMultiBufferRewriter : public StmtExprMutator {
public:
  explicit BroadcastMultiBufferRewriter(
      const std::vector<MultiVersionBufferInfo> &buffer_infos) {
    for (const auto &info : buffer_infos) {
      multi_buffer_[info.buffer] = info.new_buffer;
    }
  }

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = multi_buffer_.find(load->buffer);
    if (it != multi_buffer_.end()) {
      auto *n = load.CopyOnWrite();
      n->buffer = it->second;
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = multi_buffer_.find(store->buffer);
    if (it != multi_buffer_.end()) {
      auto *n = store.CopyOnWrite();
      n->buffer = it->second;
      n->indices.insert(n->indices.begin(), IntImm(DataType::Int(32), 0));
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op) && op->args.size() >= 2) {
      if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
        auto it = multi_buffer_.find(buffer_load->buffer);
        if (it != multi_buffer_.end()) {
          auto num_versions = it->second->shape[0];
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(VisitExpr(op->args[0]));
          new_args.push_back(VisitExpr(op->args[1]));
          new_args.push_back(num_versions);
          for (size_t i = 2; i < op->args.size(); i++) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          return Call(op->dtype, op->op, new_args, op->annotations, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      multi_buffer_;
};

} // namespace tl
} // namespace tvm
