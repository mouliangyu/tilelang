from .gemm_base import GemmBase
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
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


class GemmMAD(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
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
