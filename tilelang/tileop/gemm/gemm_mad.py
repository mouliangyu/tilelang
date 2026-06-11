from .gemm_base import GemmBase
from tvm.target import Target
from tvm.ir import Range
from tvm import tir, DataType
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


def _to_int32(v):
    return tir.IntImm("int32", v) if isinstance(v, int) else v


def _compute_flat_offset(buf, region):
    offset = tir.const(0, "int32")
    ndim = len(region.region)
    for i, rng in enumerate(region.region):
        stride = tir.const(1, "int32")
        for j in range(i + 1, ndim):
            stride = stride * buf.shape[j]
        offset = offset + tir.Cast("int32", rng.min) * stride
    return offset


def _compute_flat_offset_excluding_last(buf, region):
    """Compute flat offset for all dims except the last (K dimension).

    The K-dimension offset is handled by the kStartPosition parameter
    inside load_cbuf_to_ca/cb, so we exclude it from the pointer offset.
    Mirrors the logic in copy.cc:941-948.
    """
    offset = tir.const(0, "int32")
    ndim = len(region.region)
    for i in range(ndim - 1):
        rng = region.region[i]
        stride = tir.const(1, "int32")
        for j in range(i + 1, len(buf.shape)):
            stride = stride * buf.shape[j]
        offset = offset + tir.Cast("int32", rng.min) * stride
    return offset


def _compute_extent(region):
    extent = tir.const(1, "int32")
    for rng in region.region:
        extent = extent * rng.extent
    return extent


def _make_access_ptr(buf, region, rw_mask):
    offset = _compute_flat_offset(buf, region)
    extent = _compute_extent(region)
    ptype = tir.type_annotation(buf.dtype)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tir.tvm_access_ptr"),
        ptype,
        buf.data,
        offset,
        extent,
        tir.IntImm("int32", rw_mask),
    )


_ASCEND_DTYPE_MAP = {
    "bfloat16": "bfloat16_t",
    "float16": "half",
    "float32": "float",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "float8_e4m3fn": "float8_e4m3_t",
}


class GemmMAD(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        assert not self.trans_A, f"Ascend Cube requires transpose_A=False, got {self.trans_A}"
        assert self.trans_B, f"Ascend Cube requires transpose_B=True, got {self.trans_B}"
        if self._is_l1_input():
            return self._lower_l1()
        return self._lower_l0()

    def _is_l1_input(self):
        return self.A.scope() == "shared.l1" and self.B.scope() == "shared.l1"

    def _compute_tile_k_sub(self):
        """Compute sub-K chunk size that fits in L0A/L0B (64KB each, double-buffered)."""
        elem_bytes = DataType(self.in_dtype).bits // 8
        C0 = 32 // elem_bytes  # 16 for bf16/fp16, 8 for fp32, 32 for int8
        l0_capacity = 65536  # 64KB per L0A/L0B
        num_l0_stages = 2  # double-buffered
        l0_per_stage = l0_capacity // num_l0_stages
        max_k_a = l0_per_stage // (self.M * elem_bytes)
        max_k_b = l0_per_stage // (self.N * elem_bytes)
        max_k = min(max_k_a, max_k_b, self.K)
        tile_k_sub = (max_k // C0) * C0
        assert tile_k_sub > 0, f"Cannot fit any sub-K tile in L0: M={self.M}, N={self.N}, dtype={self.in_dtype}, L0 capacity={l0_capacity}"
        while self.K % tile_k_sub != 0 and tile_k_sub > C0:
            tile_k_sub -= C0
        assert tile_k_sub > 0 and self.K % tile_k_sub == 0, f"K ({self.K}) is not divisible by any valid TILE_K_SUB <= {max_k}"
        return tile_k_sub

    def _lower_l0(self):
        """Original path: A and B in L0A/L0B scope, emit a single ascend_mad."""
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)
        a_ptr = _make_access_ptr(self.ARegion.buffer, self.ARegion, 1)
        b_ptr = _make_access_ptr(self.BRegion.buffer, self.BRegion, 1)

        call = tir.Call(
            "void",
            tir.op.Op.get("tl.ascend_mad"),
            [
                c_ptr,
                a_ptr,
                b_ptr,
                _to_int32(self.M),
                _to_int32(self.K),
                _to_int32(self.N),
                _to_int32(0),  # unit_flag_ctrl
                _to_int32(0),  # gemv_ctrl (false)
                _to_int32(0),  # BTbuf_ctrl (false)
                self.clear_accum,
            ],
        )

        @T.prim_func
        def _gemm_mad() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad, inline_let=True)

    def _lower_l1(self):
        """New path: A and B in L1 scope, emit ascend_gemm_l1 or ascend_blockscaled_gemm_l1."""
        if self.block_scaled:
            return self._lower_l1_blockscaled()
        tile_k_sub = self._compute_tile_k_sub()

        # L0C access pointer
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)

        # L1 source access pointers (base excluding K offset)
        a_buf = self.ARegion.buffer
        b_buf = self.BRegion.buffer
        a_base_offset = _compute_flat_offset_excluding_last(a_buf, self.ARegion)
        b_base_offset = _compute_flat_offset_excluding_last(b_buf, self.BRegion)
        a_total = _compute_extent(self.ARegion)
        b_total = _compute_extent(self.BRegion)

        a_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(a_buf.dtype),
            a_buf.data,
            a_base_offset,
            a_total,
            _to_int32(1),
        )
        b_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(b_buf.dtype),
            b_buf.data,
            b_base_offset,
            b_total,
            _to_int32(1),
        )

        dtype_str = _ASCEND_DTYPE_MAP.get(self.in_dtype)
        assert dtype_str is not None, f"Unsupported dtype for Ascend GEMM: {self.in_dtype}"

        call = tir.Call(
            "void",
            tir.op.Op.get("tl.ascend_gemm_l1"),
            [
                c_ptr,
                a_ptr,
                b_ptr,
                _to_int32(self.M),
                _to_int32(self.K),
                _to_int32(self.N),
                _to_int32(tile_k_sub),
                _to_int32(1 if self.trans_B else 0),
                self.clear_accum,
                tir.StringImm(dtype_str),
                _to_int32(0),  # buf_offset
            ],
        )

        @T.prim_func
        def _gemm_mad_l1() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1, inline_let=True)

    def _lower_l1_blockscaled(self):
        """Block-scaled GEMM path: A, B, SFA, SFB in L1 → emit ascend_blockscaled_gemm_l1."""
        tile_k_sub = self._compute_tile_k_sub()

        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)

        # L1 data access pointers
        a_buf = self.ARegion.buffer
        b_buf = self.BRegion.buffer
        a_base_offset = _compute_flat_offset_excluding_last(a_buf, self.ARegion)
        b_base_offset = _compute_flat_offset_excluding_last(b_buf, self.BRegion)
        a_total = _compute_extent(self.ARegion)
        b_total = _compute_extent(self.BRegion)

        a_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(a_buf.dtype),
            a_buf.data,
            a_base_offset,
            a_total,
            _to_int32(1),
        )
        b_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(b_buf.dtype),
            b_buf.data,
            b_base_offset,
            b_total,
            _to_int32(1),
        )

        # L1 SF access pointers
        sfa_region = self.sfaRegion
        sfb_region = self.sfbRegion
        assert sfa_region is not None and sfb_region is not None, "block_scaled=True requires sfa and sfb buffers"
        sfa_buf = sfa_region.buffer
        sfb_buf = sfb_region.buffer
        sfa_base_offset = _compute_flat_offset_excluding_last(sfa_buf, sfa_region)
        sfb_base_offset = _compute_flat_offset_excluding_last(sfb_buf, sfb_region)
        sfa_total = _compute_extent(sfa_region)
        sfb_total = _compute_extent(sfb_region)

        sfa_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(sfa_buf.dtype),
            sfa_buf.data,
            sfa_base_offset,
            sfa_total,
            _to_int32(1),
        )
        sfb_ptr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.tvm_access_ptr"),
            tir.type_annotation(sfb_buf.dtype),
            sfb_buf.data,
            sfb_base_offset,
            sfb_total,
            _to_int32(1),
        )

        in_dtype_str = _ASCEND_DTYPE_MAP.get(self.in_dtype)
        assert in_dtype_str is not None, f"Unsupported dtype: {self.in_dtype}"

        def _sf_dtype_str(dt) -> str:
            s = str(dt)
            if "int16" in s:
                return "int16_t"
            if "float8_e4m3" in s:
                return "float8_e4m3_t"
            return s

        sf_dtype_str = _sf_dtype_str(sfa_buf.dtype)

        sf_load_chunk_size = int(sfa_buf.shape[-2]) // (self.K // 64)

        call = tir.Call(
            "void",
            tir.op.Op.get("tl.ascend_blockscaled_gemm_l1"),
            [
                c_ptr,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                _to_int32(self.M),
                _to_int32(self.K),
                _to_int32(self.N),
                _to_int32(tile_k_sub),
                _to_int32(1 if self.trans_B else 0),
                self.clear_accum,
                tir.StringImm(in_dtype_str),
                tir.StringImm(sf_dtype_str),
                tir.StringImm("float"),
                _to_int32(0),  # buf_offset
                _to_int32(self.sf_k_within_chunk),
                _to_int32(sf_load_chunk_size),  # sf_load_chunk_size
            ],
        )

        @T.prim_func
        def _gemm_mad_l1_bs() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1_bs, inline_let=True)
