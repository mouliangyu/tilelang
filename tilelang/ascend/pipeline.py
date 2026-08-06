from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc, SBlock
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_autoschedule,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_race_check,
    should_force_let_inline,
)

from . import transform as ascend_transform


def _has_vf_region(mod: IRModule, names: set[str] | None = None) -> bool:
    names = names or {"SIMD_VF", "SIMT_VF"}
    for _, func in mod.functions.items():
        if not isinstance(func, PrimFunc):
            continue
        found = False

        def _visit(node):
            nonlocal found
            if found:
                return
            if isinstance(node, SBlock) and node.name_hint in names:
                found = True

        post_order_visit(func.body, _visit)
        if found:
            return True
    return False


def AscendPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    # Materialize the target-neutral kernel-launch nest (thread_binding For
    # loops emitted by T.Kernel) into thread_extent AttrStmts. Ascend's NPU
    # launch is a 1-D blockIdx.x grid with no threadIdx, so SIMT-style
    # materialization (lower_thread_binding=True) reproduces the previous
    # LaunchThread(blockIdx.x) behavior.
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline(pass_ctx=pass_ctx):
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check(pass_ctx=pass_ctx):
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    # MODE_MERGING preserves inactive destination lanes, so it is a
    # read-modify-write operation rather than a pure value expression. Expose
    # the destination as an rw access before scheduling and dependency analysis.
    mod = ascend_transform.LegalizeSimdMerging()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)

    # Rewrite UB->L1 copies into NZ buffers as scatter+post_copy, using the NZ
    # layouts just inferred. Must run before AutoSchedule so it can schedule the
    # emitted ops.
    mod = ascend_transform.InsertNd2Nz()(mod)

    # Run VFChecker after InsertNd2Nz: the UB->L1 copies it would otherwise
    # reject (e.g. casting copies) are by now rewritten into a SIMD_VF scatter.
    tilelang.ascend.analysis.VFChecker()(mod)

    # Clamp DMA copy OOB tails and emit GM->L1 padding fills (create_cbuf_matrix)
    # as standalone statements so AutoSchedule schedules each fill as its own
    # MTE2 task and orders it against the L1->L0 loads that read the padded
    # buffer. Must run before AutoSchedule.
    mod = ascend_transform.AscendInsertOOBPadding()(mod)

    if allow_autoschedule(pass_ctx=pass_ctx):
        mod = ascend_transform.NormalizeControlFlowForSchedule()(mod)
        mod = ascend_transform.AnnotateMultiBufferEligible()(mod)
        mod = ascend_transform.NormalizeNoConflictHints()(mod)
        mod = ascend_transform.NormalizeTask()(mod)
        mod = ascend_transform.LatencyEstimator()(mod)
        mod = ascend_transform.AutoSchedule()(mod)
        mod = ascend_transform.RestoreWhileLoops()(mod)
        mod = ascend_transform.NormalizeMixedKernelSid()(mod)
        mod = tilelang.transform.Simplify()(mod)

    mod = ascend_transform.NormalizeBufferVersion()(mod)
    mod = ascend_transform.AscendSimdVFLowerParallel()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = ascend_transform.RewriteAscendBufferVersionLayout()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    # Retype packed 4-bit float (float4_e2m1fn) storage into the 1-byte
    # packed-pair form (float4_e2m1fnx2) and halve fp4 offsets/indices, so
    # codegen only ever sees a valid 1-byte fp4 element. Must run after the
    # access_ptr/copy offsets are materialized and before MergeUBAllocations.
    mod = ascend_transform.RewriteFp4ToFp4x2()(mod)

    if not _has_vf_region(mod):
        mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = ascend_transform.AscendRemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse().passes[0](mod)
    mod = tirx.transform.Simplify()(mod)
    mod = ascend_transform.AscendRemoveNoOp()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    if allow_global_thread_synchronization(pass_ctx=pass_ctx):
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = ascend_transform.MarkScalarDcacheBypass()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = ascend_transform.MergeUBAllocations(
        align_bytes=32,
        disable_reuse=disable_reuse,
    )(mod)

    # Split each hard_event between its 8-slot flag namespace and the shared
    # get_buf/rls_buf mutex pool (knapsack) to minimize wasted flag_ids; a no-op
    # when no hard_event overflows 8 slots.
    mod = ascend_transform.RewriteFlagToBuf()(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)

register_pipeline(ascend_pipeline)
