/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file normalize_per_core_task.cc
 * \brief Add canonical Task markers to implicit PerCoreTask bodies.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/transform/auto_schedule/memory_detector.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using ffi::GetRef;

namespace {

bool HasBufferAccess(const Stmt &stmt) {
  MemoryAccessDetector detector;
  detector.Analyze(stmt);
  return !detector.GetReadRegions().empty() ||
         !detector.GetWriteRegions().empty();
}

bool IsCandidateBoundary(const Stmt &stmt) {
  if (stmt.as<EvaluateNode>() || stmt.as<BufferStoreNode>() ||
      stmt.as<SBlockNode>() || stmt.as<WhileNode>()) {
    return true;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled;
  }
  return false;
}

class ImplicitCandidateMarker : public StmtMutator {
public:
  static Stmt Rewrite(const Stmt &stmt) {
    ImplicitCandidateMarker marker;
    return marker.VisitStmt(stmt);
  }

private:
  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // An explicit T.Task is already one canonical candidate. Do not recurse
      // into it and create nested markers, but keep inferring candidates from
      // the rest of the enclosing PerCoreTask body.
      if (attr->attr_key == tl::attr::kAscendTask)
        return stmt;
    }
    if (const auto *bind = stmt.as<BindNode>()) {
      TVM_FFI_CHECK(!HasBufferAccess(stmt), ValueError)
          << "An unmarked scalar Bind inside T.PerCoreTask may not access "
             "buffers: synchronization with statements outside the "
             "PerCoreTask is emitted only at T.Task candidate sites, so an "
             "external buffer dependency could not be synchronized at this "
             "Bind. Dependency-free scalar Bind control is allowed. Move the "
             "buffer access into a supported same-pipe T.Task candidate. Bind="
          << GetRef<Stmt>(bind);
      return stmt;
    }
    if (IsCandidateBoundary(stmt)) {
      if (!HasBufferAccess(stmt))
        return stmt;
      return AttrStmt(IntImm(DataType::Int(32), 0), tl::attr::kAscendTask,
                      IntImm(DataType::Int(32), 1), stmt, stmt->span);
    }
    return StmtMutator::VisitStmt(stmt);
  }
};

class PerCoreTaskNormalizer : public StmtMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    PerCoreTaskNormalizer normalizer;
    func.CopyOnWrite()->body = normalizer(func->body);
    return func;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tl::attr::kAscendPerCoreTask)
      return StmtMutator::VisitStmt_(op);
    Stmt body = ImplicitCandidateMarker::Rewrite(op->body);
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }
};

} // namespace

tvm::transform::Pass NormalizePerCoreTask() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return PerCoreTaskNormalizer::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizePerCoreTask", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizePerCoreTask",
                        NormalizePerCoreTask);
}

} // namespace tl
} // namespace tvm
