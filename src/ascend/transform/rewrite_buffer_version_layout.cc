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
 * \file tl/ascend/transform/rewrite_buffer_version_layout.cc
 * \brief Align each physical version of a multi-buffered UB allocation.
 */

#include "buffer_version.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "op/builtin.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr int kVersionAlignmentBits = 256;

using BufferVersionTable =
    std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual>;
// A single storage Var may back several Buffer objects (the allocation plus
// any T.view / T.reshape aliases), so collect every alias sharing the Var
// rather than assuming a unique owner.
using AllocationTable =
    std::unordered_map<Var, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>;

bool IsUnifiedBuffer(const Var &data) {
  String scope = GetPtrStorageScope(data);
  return scope == "shared" || scope == "shared.dyn";
}

class StorageMetadataCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      auto versions = op->node.try_cast<ascend::BufferVersionMap>();
      ICHECK(versions.has_value())
          << "'" << tl::attr::kBufferVersion
          << "' AttrStmt node must be a buffer version map";
      for (const auto &[data, version] : versions.value()) {
        const auto *count = version.as<IntImmNode>();
        ICHECK(count && count->value > 0)
            << "'" << tl::attr::kBufferVersion
            << "' values must be positive constant version counts";
        auto [it, inserted] = versions_.emplace(data, count->value);
        ICHECK(inserted || it->second == count->value)
            << "Conflicting buffer version counts for storage " << data << ": "
            << it->second << " vs " << count->value;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    allocations_[op->buffer->data].push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  BufferVersionTable versions_;
  AllocationTable allocations_;
};

// Physical layout of one versioned storage, kept in *bit* units so aliases
// with a different element dtype (e.g. a uint16 allocation viewed as uint8)
// can be rewritten against the same shared storage. Per-access quantities are
// converted back to the accessing buffer's element count in
// RewriteCompactOffset.
struct StorageLayout {
  PrimExpr storage_base_bits;
  PrimExpr compact_slot_bits;
  PrimExpr aligned_slot_bits;
  PrimExpr physical_extent_bits;
};

using StorageLayoutTable =
    std::unordered_map<Var, StorageLayout, ObjectPtrHash, ObjectPtrEqual>;

// Return an alias whose leading axis provably enumerates the `num_versions`
// physical versions (its trailing axes are then one logical slot). Such an
// alias means the versions occupy N equal, contiguous slots that this pass may
// realign. Returns nullopt when no alias exposes the version axis (e.g. a flat
// count-only manual buffer that hand-lays-out its own slots).
Optional<Buffer> FindVersionCarryingAlias(const std::vector<Buffer> &aliases,
                                          int64_t num_versions,
                                          arith::Analyzer *analyzer) {
  for (const Buffer &buffer : aliases) {
    if (buffer->shape.empty()) {
      continue;
    }
    if (analyzer->CanProveEqual(
            buffer->shape[0],
            make_const(buffer->shape[0].dtype(), num_versions))) {
      return buffer;
    }
  }
  return std::nullopt;
}

// Rewrite to a physical one-dimensional view before the generic FlattenBuffer
// pass. Merely changing a leading-axis stride would handle [versions, ...],
// but not manually versioned flat buffers or aliases that hide that axis.
class BufferVersionLayoutRewriter : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    StorageMetadataCollector collector;
    collector(func->body);
    if (collector.versions_.empty()) {
      return func;
    }

    arith::Analyzer layout_analyzer;
    StorageLayoutTable layouts;
    for (const auto &[data, num_versions] : collector.versions_) {
      // This pass only realigns Unified Buffer versions; L1/L0 versioned
      // storages keep their own layout. Skip before looking up the allocation
      // so that non-UB versioned buffers (which need no UB alloc here) do not
      // trip the owner-present check below.
      if (num_versions == 1 || !IsUnifiedBuffer(data)) {
        continue;
      }

      auto allocation = collector.allocations_.find(data);
      ICHECK(allocation != collector.allocations_.end())
          << "No allocation owner found for versioned storage " << data;

      // The pass only owns the layout of a storage whose accesses lay out the
      // versions as N equal slots along a leading axis (auto-versioned buffers
      // and the bare-buffer form of `annotate_manual_multi_buffer`, where the
      // allocation has a leading version dim equal to `num_versions`). A
      // count-only manual annotation (`{buf: N}`) on a flat buffer
      // intentionally decouples the ring size from the shape and hand-lays-out
      // disjoint slots plus padding/workspace; it has no such version-carrying
      // alias, so leave its layout untouched.
      const std::vector<Buffer> &aliases = allocation->second;
      Optional<Buffer> version_alias =
          FindVersionCarryingAlias(aliases, num_versions, &layout_analyzer);
      if (!version_alias.has_value()) {
        continue;
      }
      const Buffer &owner = version_alias.value();

      DataType index_dtype = owner->DefaultIndexType();
      ICHECK(index_dtype.is_int() || index_dtype.is_uint())
          << "Versioned buffer " << owner->name
          << " must use an integer index dtype";
      // The leading axis carries the versions; the trailing axes are one
      // logical slot. Compute the slot size in bits so aliases of a different
      // dtype resolve against the same shared storage.
      int elem_bits = owner->dtype.bits() * owner->dtype.lanes();
      ICHECK_EQ(kVersionAlignmentBits % elem_bits, 0)
          << "32-byte version alignment cannot be represented in "
          << owner->dtype << " elements";
      PrimExpr slot_elems = make_const(index_dtype, 1);
      for (size_t i = 1; i < owner->shape.size(); ++i) {
        PrimExpr axis = owner->shape[i];
        if (axis.dtype() != index_dtype) {
          axis = cast(index_dtype, axis);
        }
        slot_elems = slot_elems * axis;
      }
      slot_elems = layout_analyzer.Simplify(slot_elems);
      PrimExpr compact_slot_bits =
          slot_elems * make_const(index_dtype, elem_bits);
      PrimExpr aligned_slot_bits = layout_analyzer.Simplify(
          indexdiv(compact_slot_bits +
                       make_const(index_dtype, kVersionAlignmentBits - 1),
                   make_const(index_dtype, kVersionAlignmentBits)) *
          make_const(index_dtype, kVersionAlignmentBits));
      if (layout_analyzer.CanProveEqual(aligned_slot_bits, compact_slot_bits)) {
        continue;
      }
      // AutoSchedule accounts for the compact slot size when checking UB
      // capacity. We intentionally do not feed this per-version padding back
      // into that check: unaligned version slots are expected only in small
      // kernels where the extra allocation cannot approach the UB limit.
      PrimExpr physical_extent_bits = layout_analyzer.Simplify(
          aligned_slot_bits * make_const(index_dtype, num_versions));
      PrimExpr storage_base = owner->elem_offset.defined()
                                  ? owner->elem_offset
                                  : make_const(index_dtype, 0);
      if (storage_base.dtype() != index_dtype) {
        storage_base = cast(index_dtype, storage_base);
      }
      PrimExpr storage_base_bits = layout_analyzer.Simplify(
          storage_base * make_const(index_dtype, elem_bits));

      layouts.emplace(data,
                      StorageLayout{storage_base_bits,
                                    layout_analyzer.Simplify(compact_slot_bits),
                                    aligned_slot_bits, physical_extent_bits});
    }

    arith::Analyzer analyzer;
    BufferVersionLayoutRewriter rewriter(&analyzer, std::move(layouts));
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = rewriter(std::move(writer->body));
    return func;
  }

private:
  using IRMutatorWithAnalyzer::VisitExpr;
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt;
  using IRMutatorWithAnalyzer::VisitStmt_;

  BufferVersionLayoutRewriter(arith::Analyzer *analyzer,
                              StorageLayoutTable layouts)
      : IRMutatorWithAnalyzer(analyzer), layouts_(std::move(layouts)) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      return VisitStmt(op->body);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    auto node = Downcast<AllocBuffer>(StmtExprMutator::VisitStmt_(op));
    node.CopyOnWrite()->buffer = GetPhysicalBuffer(node->buffer);
    return node;
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    node.CopyOnWrite()->buffer = GetPhysicalBuffer(node->buffer);
    return node;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return RewriteBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return RewriteBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5U) {
      Var data = Downcast<Var>(op->args[1]);
      if (layouts_.count(data)) {
        Array<PrimExpr> args = op->args;
        int elem_bits = ElementBits(op->args[0].dtype());
        PrimExpr old_offset = VisitExpr(op->args[2]);
        PrimExpr extent = VisitExpr(op->args[3]);
        PrimExpr new_offset = RewriteCompactOffset(data, old_offset, elem_bits);
        ValidatePointerSpan(data, old_offset, extent, elem_bits);
        args.Set(2, std::move(new_offset));
        args.Set(3, std::move(extent));
        args.Set(4, VisitExpr(op->args[4]));
        return Call(op->dtype, op->op, std::move(args), op->span);
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  template <typename Node> Node RewriteBufferAccess(Node node) {
    const Var &data = node->buffer->data;
    if (!layouts_.count(data)) {
      return node;
    }

    Array<PrimExpr> compact_offsets = node->buffer->ElemOffset(node->indices);
    ICHECK_EQ(compact_offsets.size(), 1U)
        << "Versioned UB aliases must have one compact storage offset, but "
        << node->buffer << " produced " << compact_offsets;
    int elem_bits = ElementBits(node->buffer->dtype);
    PrimExpr new_offset =
        RewriteCompactOffset(data, compact_offsets[0], elem_bits);
    auto *writer = node.CopyOnWrite();
    writer->buffer = GetPhysicalBuffer(node->buffer);
    writer->indices = {std::move(new_offset)};
    return node;
  }

  // Remap a compact element offset (in `elem_bits`-wide elements of the
  // accessing buffer) so that version `v` starts at its 32-byte-aligned slot.
  // The stored layout is in bits, so aliases of any dtype resolve consistently.
  PrimExpr RewriteCompactOffset(const Var &data, PrimExpr compact_offset,
                                int elem_bits) {
    auto layout = layouts_.find(data);
    ICHECK(layout != layouts_.end());
    const StorageLayout &info = layout->second;
    DataType dtype = compact_offset.dtype();
    PrimExpr elem_bits_c = make_const(dtype, elem_bits);
    PrimExpr storage_base = CastIndex(info.storage_base_bits, dtype);
    PrimExpr compact_slot = CastIndex(info.compact_slot_bits, dtype);
    PrimExpr aligned_slot = CastIndex(info.aligned_slot_bits, dtype);
    PrimExpr relative = compact_offset * elem_bits_c - storage_base;
    PrimExpr version = indexdiv(relative, compact_slot);
    PrimExpr inner = indexmod(relative, compact_slot);
    PrimExpr new_bits = storage_base + version * aligned_slot + inner;
    return analyzer_->Simplify(indexdiv(new_bits, elem_bits_c));
  }

  void ValidatePointerSpan(const Var &data, PrimExpr compact_offset,
                           PrimExpr extent, int elem_bits) {
    auto layout = layouts_.find(data);
    ICHECK(layout != layouts_.end());
    DataType dtype = compact_offset.dtype();
    PrimExpr elem_bits_c = make_const(dtype, elem_bits);
    PrimExpr storage_base = CastIndex(layout->second.storage_base_bits, dtype);
    PrimExpr compact_slot = CastIndex(layout->second.compact_slot_bits, dtype);
    if (extent.dtype() != dtype) {
      extent = cast(dtype, extent);
    }
    PrimExpr inner =
        indexmod(compact_offset * elem_bits_c - storage_base, compact_slot);
    if (analyzer_->CanProve(inner + extent * elem_bits_c > compact_slot)) {
      LOG(FATAL) << "tvm_access_ptr for versioned buffer " << data->name_hint
                 << " crosses a 32-byte-aligned version slot: offset="
                 << compact_offset << ", extent=" << extent
                 << ", compact_slot_bits=" << compact_slot;
    }
  }

  static int ElementBits(DataType dtype) {
    return dtype.bits() * dtype.lanes();
  }

  PrimExpr CastIndex(PrimExpr value, DataType dtype) const {
    if (value.dtype() != dtype) {
      value = cast(dtype, value);
    }
    return value;
  }

  Buffer GetPhysicalBuffer(Buffer buffer) {
    auto cached = buffer_remap_.find(buffer);
    if (cached != buffer_remap_.end()) {
      return cached->second;
    }
    Buffer original = buffer;
    auto layout = layouts_.find(buffer->data);
    if (layout != layouts_.end()) {
      const StorageLayout &info = layout->second;
      // The physical extent is dtype-agnostic (kept in bits); express it in
      // this alias's own element count so a dtype-changing view resolves
      // against the same shared storage.
      int elem_bits = ElementBits(buffer->dtype);
      DataType extent_dtype = info.physical_extent_bits.dtype();
      PrimExpr extent_elems = analyzer_->Simplify(indexdiv(
          info.physical_extent_bits, make_const(extent_dtype, elem_bits)));
      buffer = Buffer(buffer->data, buffer->dtype, {extent_elems}, {},
                      make_const(extent_dtype, 0), buffer->name,
                      buffer->data_alignment, buffer->offset_factor,
                      buffer->buffer_type, {}, buffer->span);
    }
    buffer_remap_.emplace(std::move(original), buffer);
    return buffer;
  }

  StorageLayoutTable layouts_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

PrimFunc RewriteBufferVersionLayout(PrimFunc func) {
  return BufferVersionLayoutRewriter::Rewrite(std::move(func));
}

} // namespace

namespace transform {

tirx::transform::Pass RewriteAscendBufferVersionLayout() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tirx::transform::PassContext &ctx) {
    return RewriteBufferVersionLayout(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.RewriteAscendBufferVersionLayout", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteAscendBufferVersionLayout",
                        RewriteAscendBufferVersionLayout);
}

} // namespace transform
} // namespace tl
} // namespace tvm
