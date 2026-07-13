"""Insert Ascend ND->NZ conversion for UB->L1 copies, driven by layout inference.

Runs after LayoutInference (which anchors Cube-side L1 buffers to the NZ
fractal layout) and before AutoSchedule. Any plain ``tl.tileop.copy`` from a UB
(``shared``) buffer into an L1 (``shared.l1``) buffer whose layout was inferred
as NZ is rewritten into the ``ascend_nd2nz_scatter`` + ``ascend_nd2nz_post_copy``
sequence. Emitting it here (rather than at trace time) lets AutoSchedule
sync/multi-buffer the two steps.
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import Call, Evaluate, SBlock, SeqStmt
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

_L1_SCOPES = ("shared.l1", "shared.l1.dyn")
_UB_SCOPES = ("shared", "shared.dyn")
_COPY_OP = "tl.tileop.copy"
_REGION_OP = "tl.tileop.region"
_SCATTER_OP = "tl.ascend_nd2nz_scatter"
_POST_COPY_OP = "tl.ascend_nd2nz_post_copy"


def _op_name(node):
    op = getattr(node, "op", None)
    return getattr(op, "name", None)


def _to_dtype_str(dtype) -> str:
    s = str(dtype)
    if "bfloat16" in s:
        return "bfloat16_t"
    if "float16" in s:
        return "half"
    if s in ("float32", "float"):
        return "float"
    # The UB->L1 software scatter only has C++ template specializations for
    # half / bfloat16_t / float in src/tl_templates/ascend/nd2nz_copy.h. Other
    # dtypes (fp8, int8, ...) need a new ascend_nd2nz_scatter_impl specialization
    # there first; without it codegen would hit the fallback static_assert.
    raise ValueError(
        f"Unsupported nd2nz dtype: {s}. The UB->L1 scatter supports only "
        "float16/bfloat16/float32; add a template specialization in "
        "src/tl_templates/ascend/nd2nz_copy.h to extend it."
    )


def _const_int(e):
    if isinstance(e, int):
        return e
    if hasattr(e, "value"):
        return int(e.value)
    return None


def _ann_get(ann, key):
    """Read annotation ``key`` from a Call's annotations map, or None."""
    return ann.get(key) if ann is not None else None


def _collect_nz_l1_vars(body) -> set:
    """buffer.data Vars of L1 buffers carrying a layout (only NZ is attached)."""
    nz_vars = set()

    def visit(node):
        if isinstance(node, SBlock):
            lm = node.annotations.get("layout_map")
            if lm is None:
                return
            for buf in lm.keys():
                if buf.scope() in _L1_SCOPES:
                    nz_vars.add(buf.data)

    post_order_visit(body, visit)
    return nz_vars


def _region_parts(region_call):
    """(buffer, BufferLoad, extents) from a tl.region call, or None."""
    if not (isinstance(region_call, Call) and _op_name(region_call) == _REGION_OP):
        return None
    load = region_call.args[0]
    if not isinstance(load, tirx.BufferLoad):
        return None
    return load.buffer, load, list(region_call.args[2:])


def _prod(extents):
    extent = tirx.const(1, "int32")
    for e in extents:
        extent = extent * e
    return extent


def _access_ptr(load, extents, rw_mask):
    # Emit tl.access_ptr (keeps the BufferLoad so sync/codegen can recover the
    # buffer + rw intent); LowerAccessPtr lowers it to tvm_access_ptr later.
    from tilelang.language.builtin import access_ptr as _tl_access_ptr

    return _tl_access_ptr(load, rw_mask, extent=_prod(extents))


def _rewrite_copy(value, nz_l1_vars, pending_allocs):
    """Return replacement Stmt for a UB(ND)->L1(NZ) copy, else None."""
    if not (isinstance(value, Call) and _op_name(value) == _COPY_OP):
        return None
    if len(value.args) < 2:
        return None
    src = _region_parts(value.args[0])
    dst = _region_parts(value.args[1])
    if src is None or dst is None:
        return None
    src_buf, src_load, src_ext = src
    dst_buf, dst_load, dst_ext = dst
    if src_buf.scope() not in _UB_SCOPES or dst_buf.scope() not in _L1_SCOPES:
        return None
    if dst_buf.data not in nz_l1_vars or len(dst_ext) < 2:
        return None

    # dual_copy detection: trust the ``double`` / ``dual_dst_ctl`` annotations
    # that the frontend dual_copy() attaches, NOT a 2x shape-ratio guess.
    # Shape inference needs constant extents; on symbolic shapes (e.g. M//2)
    # it silently misjudges is_dual as False, drops the AIV split, and lets
    # both sub-cores write the same region.
    ann = getattr(value, "annotations", None)
    is_dual = bool(_const_int(_ann_get(ann, "double")))
    split_dim_neg2 = 0
    half_extent = None
    if is_dual:
        dual_dst_ctl = _const_int(_ann_get(ann, "dual_dst_ctl"))
        if dual_dst_ctl not in (1, 2):
            raise ValueError(f"dual UB->L1 nd2nz copy into {dst_buf.name} requires dual_dst_ctl in {{1, 2}}, got {dual_dst_ctl}.")
        split_dim_neg2 = 0 if dual_dst_ctl == 1 else 1  # 1=row split, 2=col split
        half_extent = src_ext[-2 + split_dim_neg2]

    rows, cols = (src_ext[-2], src_ext[-1]) if is_dual else (dst_ext[-2], dst_ext[-1])
    full_rows = dst_buf.shape[-2]
    src_dtype_str = _to_dtype_str(src_buf.dtype)
    dst_dtype_str = _to_dtype_str(dst_buf.dtype)
    src_ptr = _access_ptr(src_load, src_ext, "r")
    dst_ptr = _access_ptr(dst_load, dst_ext, "w")

    elem_bytes = (dst_buf.dtype.bits + 7) // 8
    cols_i = _const_int(cols)
    same_dtype = str(src_buf.dtype) == str(dst_buf.dtype)
    post_copy_op = tirx.op.Op.get(_POST_COPY_OP)

    # Build the trailing dual extras (consumed by warpgroup_partition and
    # stripped before codegen).
    dual_extras = [tirx.const(split_dim_neg2, "int32"), half_extent] if is_dual else []

    if cols_i is not None and cols_i * elem_bytes == 32 and same_dtype and not is_dual:
        # Single D-group: NZ == ND, post_copy reads the ND source directly.
        return Evaluate(tirx.call_intrin("handle", post_copy_op, dst_ptr, src_ptr, rows, cols, full_rows, dst_dtype_str))

    # Multi-fractal scatter path. rows/cols may be symbolic; size the scratch
    # buffer symbolically (as the old copy_op._emit_nd2nz_seq did) instead of
    # bailing out on non-constant extents. Bailing would leave this copy as a
    # raw UB->L1 DMA (copy.cc dma_path 6) that writes ND data into a buffer the
    # GEMM treats as NZ -- a silent operand corruption with no assertion.
    # NOTE: pad one more row to avoid bank conflict (no vscatterb on Ascend 950).
    cols_pad = (rows + 1) * cols
    tmp = tirx.decl_buffer((cols_pad,), dst_buf.dtype, name="nz_scratch", scope="shared.dyn")
    pending_allocs.append(tmp)
    tmp_load = tirx.BufferLoad(tmp, [tirx.const(0, "int32")])
    tmp_w = _access_ptr(tmp_load, [cols_pad], "w")
    tmp_r = _access_ptr(tmp_load, [cols_pad], "r")

    scatter = tirx.call_intrin("handle", tirx.op.Op.get(_SCATTER_OP), src_ptr, tmp_w, rows, cols, dst_dtype_str, src_dtype_str)
    post = tirx.call_intrin("handle", post_copy_op, dst_ptr, tmp_r, rows, cols, full_rows, dst_dtype_str, *dual_extras)
    return SeqStmt([Evaluate(scatter), Evaluate(post)])


def InsertNd2Nz():
    """Rewrite UB->L1 copies into NZ buffers as scatter+post_copy."""

    def pass_fn(func, mod, ctx):
        nz_l1_vars = _collect_nz_l1_vars(func.body)
        if not nz_l1_vars:
            return func

        alloc_stack: list[list] = []

        def pre(node):
            if isinstance(node, SBlock):
                alloc_stack.append([])
            return None

        def post(node):
            if isinstance(node, Evaluate):
                rew = _rewrite_copy(node.value, nz_l1_vars, alloc_stack[-1])
                return rew if rew is not None else node
            if isinstance(node, SBlock):
                pending = alloc_stack.pop()
                if pending:
                    return SBlock(
                        node.iter_vars,
                        node.reads,
                        node.writes,
                        node.name_hint,
                        node.body,
                        node.init,
                        list(node.alloc_buffers) + pending,
                        node.match_buffers,
                        node.annotations,
                    )
            return node

        new_body = ir_transform(func.body, pre, post)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
