from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tvm import DataType, tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_MAD = "ascend.mad"


def _to_int32(value):
    return tirx.IntImm("int32", value) if isinstance(value, int) else value


def _compute_flat_offset(buf, region):
    offset = tirx.const(0, "int32")
    ndim = len(region.region)
    for i, rng in enumerate(region.region):
        stride = tirx.const(1, "int32")
        for j in range(i + 1, ndim):
            stride = stride * buf.shape[j]
        offset = offset + tirx.Cast("int32", rng.min) * stride
    return offset


def _compute_flat_offset_excluding_last(buf, region):
    offset = tirx.const(0, "int32")
    ndim = len(region.region)
    for i in range(ndim - 1):
        rng = region.region[i]
        stride = tirx.const(1, "int32")
        for j in range(i + 1, len(buf.shape)):
            stride = stride * buf.shape[j]
        offset = offset + tirx.Cast("int32", rng.min) * stride
    return offset


def _compute_extent(region):
    extent = tirx.const(1, "int32")
    for rng in region.region:
        extent = extent * rng.extent
    return extent


def _make_access_ptr(buf, region, rw_mask):
    return tirx.op.tvm_access_ptr(
        tirx.op.type_annotation(buf.dtype),
        buf.data,
        _compute_flat_offset(buf, region),
        _compute_extent(region),
        tirx.IntImm("int32", rw_mask),
    )


_ASCEND_DTYPE_MAP = {
    "bfloat16": "bfloat16_t",
    "float16": "half",
    "float32": "float",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "float8_e4m3": "float8_e4m3_t",
    "float8_e4m3fn": "float8_e4m3_t",
}


class GemmMAD(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        assert not self.trans_A, f"Ascend Cube requires transpose_A=False, got {self.trans_A}"
        assert self.trans_B, f"Ascend Cube requires transpose_B=True, got {self.trans_B}"
        if self._is_l1_input():
            if self.is_blockscaled:
                return self._lower_l1_blockscaled()
            return self._lower_l1()
        return self._lower_l0()

    def _is_l1_input(self):
        return self.A.scope() == "shared.l1" and self.B.scope() == "shared.l1"

    def _input_dtype(self):
        return str(self.a_dtype)

    def _compute_tile_k_sub(self):
        input_dtype = self._input_dtype()
        elem_bytes = DataType(input_dtype).bits // 8
        c0 = 32 // elem_bytes
        l0_capacity = 65536
        num_l0_stages = 2
        l0_per_stage = l0_capacity // num_l0_stages
        max_k_a = l0_per_stage // (self.M * elem_bytes)
        max_k_b = l0_per_stage // (self.N * elem_bytes)
        max_k = min(max_k_a, max_k_b, self.K)
        tile_k_sub = (max_k // c0) * c0
        assert tile_k_sub > 0, f"Cannot fit any sub-K tile in L0: M={self.M}, N={self.N}, dtype={input_dtype}, L0 capacity={l0_capacity}"
        while self.K % tile_k_sub != 0 and tile_k_sub > c0:
            tile_k_sub -= c0
        assert tile_k_sub > 0 and self.K % tile_k_sub == 0, f"K ({self.K}) is not divisible by any valid TILE_K_SUB <= {max_k}"
        return tile_k_sub

    def _lower_l0(self):
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)
        a_ptr = _make_access_ptr(self.ARegion.buffer, self.ARegion, 1)
        b_ptr = _make_access_ptr(self.BRegion.buffer, self.BRegion, 1)
        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_mad"),
            c_ptr,
            a_ptr,
            b_ptr,
            _to_int32(self.M),
            _to_int32(self.K),
            _to_int32(self.N),
            _to_int32(0),
            _to_int32(0),
            _to_int32(0),
            self.clear_accum,
        )

        @T.prim_func
        def _gemm_mad() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad, inline_let=True)

    def _lower_l1(self):
        tile_k_sub = self._compute_tile_k_sub()
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)

        a_buf = self.ARegion.buffer
        b_buf = self.BRegion.buffer
        a_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(a_buf.dtype),
            a_buf.data,
            _compute_flat_offset_excluding_last(a_buf, self.ARegion),
            _compute_extent(self.ARegion),
            tirx.IntImm("int32", 1),
        )
        b_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(b_buf.dtype),
            b_buf.data,
            _compute_flat_offset_excluding_last(b_buf, self.BRegion),
            _compute_extent(self.BRegion),
            tirx.IntImm("int32", 1),
        )

        input_dtype = self._input_dtype()
        dtype_str = _ASCEND_DTYPE_MAP.get(input_dtype)
        assert dtype_str is not None, f"Unsupported dtype for Ascend GEMM: {input_dtype}"

        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_gemm_l1"),
            c_ptr,
            a_ptr,
            b_ptr,
            _to_int32(self.M),
            _to_int32(self.K),
            _to_int32(self.N),
            _to_int32(tile_k_sub),
            _to_int32(1 if self.trans_B else 0),
            self.clear_accum,
            tirx.StringImm(dtype_str),
            _to_int32(0),  # buf_offset
        )

        @T.prim_func
        def _gemm_mad_l1() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1, inline_let=True)

    def _lower_l1_blockscaled(self):
        tile_k_sub = self._compute_tile_k_sub()
        c_ptr = _make_access_ptr(self.CRegion.buffer, self.CRegion, 2)

        a_buf = self.ARegion.buffer
        b_buf = self.BRegion.buffer
        a_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(a_buf.dtype),
            a_buf.data,
            _compute_flat_offset_excluding_last(a_buf, self.ARegion),
            _compute_extent(self.ARegion),
            tirx.IntImm("int32", 1),
        )
        b_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(b_buf.dtype),
            b_buf.data,
            _compute_flat_offset_excluding_last(b_buf, self.BRegion),
            _compute_extent(self.BRegion),
            tirx.IntImm("int32", 1),
        )

        sfa_region = self.SFARegion
        sfb_region = self.SFBRegion
        assert sfa_region is not None and sfb_region is not None, "blockscaled_gemm requires sfa and sfb buffers"
        sfa_buf = sfa_region.buffer
        sfb_buf = sfb_region.buffer
        sfa_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(sfa_buf.dtype),
            sfa_buf.data,
            _compute_flat_offset_excluding_last(sfa_buf, sfa_region),
            _compute_extent(sfa_region),
            tirx.IntImm("int32", 1),
        )
        sfb_ptr = tirx.op.tvm_access_ptr(
            tirx.op.type_annotation(sfb_buf.dtype),
            sfb_buf.data,
            _compute_flat_offset_excluding_last(sfb_buf, sfb_region),
            _compute_extent(sfb_region),
            tirx.IntImm("int32", 1),
        )

        input_dtype = self._input_dtype()
        in_dtype_str = _ASCEND_DTYPE_MAP.get(input_dtype)
        assert in_dtype_str is not None, f"Unsupported dtype for Ascend blockscaled GEMM: {input_dtype}"

        def _sf_dtype_str(dtype) -> str:
            dtype_str = str(dtype)
            if "int16" in dtype_str:
                return "int16_t"
            if "float8_e4m3" in dtype_str:
                return "float8_e4m3_t"
            return dtype_str

        sf_dtype_str = _sf_dtype_str(sfa_buf.dtype)
        sf_load_chunk_size = int(sfa_buf.shape[-2]) // (self.K // 64)

        call = tirx.call_intrin(
            "void",
            tirx.op.Op.get("tl.ascend_blockscaled_gemm_l1"),
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
            tirx.StringImm(in_dtype_str),
            tirx.StringImm(sf_dtype_str),
            tirx.StringImm("float"),
            _to_int32(0),  # buf_offset
            _to_int32(self.sf_k_within_chunk),
            _to_int32(sf_load_chunk_size),
        )

        @T.prim_func
        def _gemm_mad_l1_blockscaled() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_mad_l1_blockscaled, inline_let=True)
