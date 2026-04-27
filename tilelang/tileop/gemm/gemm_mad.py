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
        l0_capacity = 65536  # 64KB per L0A/L0B
        num_l0_stages = 2  # double-buffered
        l0_per_stage = l0_capacity // num_l0_stages
        max_k_a = l0_per_stage // (self.M * elem_bytes)
        max_k_b = l0_per_stage // (self.N * elem_bytes)
        max_k = min(max_k_a, max_k_b, self.K)
        tile_k_sub = (max_k // 16) * 16
        assert tile_k_sub > 0, f"Cannot fit any sub-K tile in L0: M={self.M}, N={self.N}, dtype={self.in_dtype}, L0 capacity={l0_capacity}"
        while self.K % tile_k_sub != 0 and tile_k_sub > 16:
            tile_k_sub -= 16
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
        """New path: A and B in L1 scope, emit ascend_gemm_l1 template call."""
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
