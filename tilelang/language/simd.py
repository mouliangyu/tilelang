"""T.simd.* — Raw CCE vector intrinsics for Ascend SIMD programming.

Function names match the underlying CCE intrinsics directly.
"""

from tvm import tirx
from tvm.tirx import BufferLoad
from tilelang._typing import DType, ShapeType
from .allocate import alloc_var as _alloc_var, alloc_local as _alloc_local
from .builtin import access_ptr

_Op = tirx.op.Op.get


class SimdPair:
    """Wraps a vintlv/vdintlv result, supports tuple unpacking: a, b = vintlv(x, y).

    Note: ``a, b = ...`` generates two pair_get calls referencing the same vintlv
    expression inline. The Ascend HoistSimdPairs pass eliminates the duplicate.
    """

    def __init__(self, pair):
        self._pair = pair

    @property
    def dtype(self):
        return self._pair.dtype

    def __iter__(self):
        yield tirx.call_intrin(str(self._pair.dtype), _Op("tl.simd.pair_get"), self._pair, 0)
        yield tirx.call_intrin(str(self._pair.dtype), _Op("tl.simd.pair_get"), self._pair, 1)


_ELEM_BITS = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "int16": 16,
    "uint16": 16,
    "int64": 64,
    "uint64": 64,
    "int8": 8,
    "uint8": 8,
    "float8_e4m3": 8,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0fnu": 8,
    "float4_e2m1fn": 4,
}


def _elem_bits(dtype_str: str) -> int:
    return _ELEM_BITS.get(str(dtype_str), 32)


def _vreg_lanes(elem_bits: int) -> int:
    return 2048 // elem_bits


def _vec_dtype(elem_dtype: str) -> str:
    lanes = _vreg_lanes(_elem_bits(elem_dtype))
    return f"{elem_dtype}x{lanes}"


def pset(elem_width: int, dist: str = "PAT_ALL"):
    """Create a predicate mask: pset_bXX(dist).

    Returns a vector_bool (boolx256).
    dist: "PAT_ALL", "PAT_VL1".."PAT_VL128", "PAT_M3", "PAT_M4", "PAT_H", "PAT_Q", etc.
    """
    return tirx.call_intrin("boolx256", _Op("tl.simd.pset"), elem_width, dist)


def pge(elem_width: int, dist: str = "PAT_ALL"):
    """Create a predicate mask from pge_bXX(dist)."""
    return tirx.call_intrin("boolx256", _Op("tl.simd.pge"), elem_width, dist)


def pand(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pand"), src0, src1, mask)


def por(src0, src1, mask):  # codespell:ignore
    return tirx.call_intrin("boolx256", _Op("tl.simd.por"), src0, src1, mask)  # codespell:ignore


def pxor(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pxor"), src0, src1, mask)


def pnot(src, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pnot"), src, mask)


def psel(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.psel"), src0, src1, mask)


def alloc_var(dtype: DType):
    """Allocate a single mutable SIMD register variable (return-value style).

    Uses ``local.var`` scope and behaves as one vector register value. For an
    addressable array of registers (``v[i]``), use :func:`alloc_local`.
    """
    return _alloc_var(_vec_dtype(dtype), scope="local.var")


def alloc_local(shape: ShapeType, dtype: DType):
    """Allocate an addressable array of mutable SIMD register variables.

    Uses ``local`` scope so each element is an individually addressable
    register, allowing indexed access like ``v[i]``::

        v = T.simd.alloc_local(4, "float32")
        for i in T.Unroll(4, explicit=True):
            v[i] = vld(s_ub[i * VL])
    """
    return _alloc_local(shape, _vec_dtype(dtype), scope="local")


# -- Data movement ---------------------------------------------------------------


def vld(addr, dist="NORM", off=None):
    """Vector load. Returns a typed vector register.

    addr can be a BufferLoad auto-wrapped as tl.access_ptr.
    """
    if isinstance(addr, BufferLoad):
        elem_dtype = str(addr.buffer.dtype)
        elem_bits = _elem_bits(elem_dtype)
        lanes = _vreg_lanes(elem_bits)
        extent = lanes
        if isinstance(dist, str):
            if dist.startswith("BRC_"):
                extent = 1
            elif dist in ("UNPK_B16", "UNPK4_B8"):
                extent = 64
        addr = access_ptr(addr, "r", extent=extent)
        vec_dtype = f"{elem_dtype}x{lanes}"
    else:
        inner_load = addr.args[0]
        elem_dtype = str(inner_load.buffer.dtype)
        vec_dtype = _vec_dtype(elem_dtype)

    args = [addr, dist]
    if off is not None:
        args.append(off)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vld"), *args)


def vsts(addr, src, mask, dist="NORM_B32", off=None):
    """Vector store: vsts(src, base, off, dist, mask).

    addr can be a BufferLoad auto-wrapped as tl.access_ptr.
    """
    if isinstance(addr, BufferLoad):
        elem_dtype = str(addr.buffer.dtype)
        elem_bits = _elem_bits(elem_dtype)
        lanes = _vreg_lanes(elem_bits)
        extent = lanes
        if isinstance(dist, str):
            if dist.startswith("ONEPT_"):
                extent = 1
            elif dist in ("PK_B32", "PK4_B32"):
                extent = 64
        addr = access_ptr(addr, "w", extent=extent)
    args = [addr, src, mask, dist]
    if off is not None:
        args.append(off)
    return tirx.call_intrin("void", _Op("tl.simd.vsts"), *args)


def vsstb(src, base, stride, mask):
    """Scatter-store 32B blocks with block stride: vsstb(src, base, stride, mask)."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "w", extent=_vreg_lanes(_elem_bits(elem_dtype)))
    return tirx.call_intrin("void", _Op("tl.simd.vsstb"), src, base, stride, mask)


# -- Binary arithmetic ------------------------------------------------------------


def vadd(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vadd"), src0, src1, mask, mode)


def vsub(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsub"), src0, src1, mask, mode)


def vmul(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmul"), src0, src1, mask, mode)


def vmula(acc, src0, src1, mask, mode="MODE_ZEROING"):
    """Fused multiply-add: dst = acc + src0 * src1."""
    return tirx.call_intrin(str(acc.dtype), _Op("tl.simd.vmula"), acc, src0, src1, mask, mode)


def vdiv(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vdiv"), src0, src1, mask, mode)


def vmax(src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmax"), src0, src1, mask, mode)


def vmin(src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmin"), src0, src1, mask, mode)


def vand(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vand"), src0, src1, mask, mode)


def vor(src0, src1, mask, mode="MODE_ZEROING"):  # codespell:ignore
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vor"), src0, src1, mask, mode)  # codespell:ignore


def vshl(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshl"), src0, src1, mask, mode)


def vshr(src0, src1, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshr"), src0, src1, mask, mode)


# -- Unary ------------------------------------------------------------------------


def vexp(src, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vexp"), src, mask, mode)


def vln(src, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vln"), src, mask, mode)


def vsqrt(src, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqrt"), src, mask, mode)


def vabs(src, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vabs"), src, mask, mode)


def vneg(src, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vneg"), src, mask, mode)


def vrelu(src, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vrelu"), src, mask, mode)


# -- Broadcast --------------------------------------------------------------------


def vdup(src, dtype_str, mask, mode="MODE_ZEROING"):
    """Broadcast scalar to all lanes: dst = vdup(scalar, dtype_str, mask).

    dtype_str specifies the target vector element type (e.g. "float32").
    """
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vdup"), src, mask, mode)


def vdupv(src, mask, pos="POS_LOWEST", mode="MODE_ZEROING"):
    """Broadcast lane N of src vector to all lanes: dst = vdupv(src, mask, pos).

    pos: "POS_LOWEST" (lane 0) or "POS_HIGHEST" (lane N).
    """
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vdupv"), src, mask, pos, mode)


# -- Cross-lane reduction ---------------------------------------------------------


def vcadd(src, mask, mode="MODE_ZEROING"):
    """Pairwise add reduction: dst = vcadd(src, mask)."""
    src_elem = str(src.dtype).split("x", 1)[0]
    out_dtype = {
        "int16": _vec_dtype("int32"),
        "uint16": _vec_dtype("uint32"),
    }.get(src_elem, str(src.dtype))
    return tirx.call_intrin(out_dtype, _Op("tl.simd.vcadd"), src, mask, mode)


def vcmax(src, mask, mode="MODE_ZEROING"):
    """Pairwise max reduction: dst = vcmax(src, mask)."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmax"), src, mask, mode)


def vcmin(src, mask, mode="MODE_ZEROING"):
    """Pairwise min reduction: dst = vcmin(src, mask)."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmin"), src, mask, mode)


def vcgadd(src, mask, mode="MODE_ZEROING"):
    """Grouped add reduction: dst = vcgadd(src, mask)."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgadd"), src, mask, mode)


def vcgmax(src, mask, mode="MODE_ZEROING"):
    """Grouped max reduction: dst = vcgmax(src, mask)."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgmax"), src, mask, mode)


def vsqz(src, mask, mode="MODE_STORED"):
    """Squeeze selected lanes toward the low lanes: dst = vsqz(src, mask)."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqz"), src, mask, mode)


# -- Index ramp / compare ---------------------------------------------------------


def vci(index, dtype_str, order="INC_ORDER"):
    """Index ramp: dst = vci(index, dtype_str, order).

    dst[lane] = index + lane (INC_ORDER) / index - lane (DEC_ORDER).
    dtype_str is the destination vector element type (e.g. "int32", "float32").
    """
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vci"), index, order)


def vcmp(src0, src1, mask, op="eq"):
    """Elementwise compare → vector_bool: dst = vcmp_<op>(src0, src1, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    return tirx.call_intrin("boolx256", _Op("tl.simd.vcmp"), src0, src1, mask, op)


def vcmps(src, scalar, mask, op="lt"):
    """Compare vector vs scalar → vector_bool: dst = vcmps_<op>(src, scalar, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    return tirx.call_intrin("boolx256", _Op("tl.simd.vcmps"), src, scalar, mask, op)


# -- Register permutation ---------------------------------------------------------


def vintlv(src0, src1):
    """Interleave two vector registers.

    Unpack: a, b = vintlv(x, y)
    """
    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vintlv"), src0, src1)
    return SimdPair(pair)


def vdintlv(src0, src1):
    """De-interleave two vector registers.

    Unpack: a, b = vdintlv(x, y)
    """
    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vdintlv"), src0, src1)
    return SimdPair(pair)


def pair_get(pair, index):
    """Extract element from a pair: reg = pair_get(pair, 0) or pair_get(pair, 1)."""
    inner = pair._pair if isinstance(pair, SimdPair) else pair
    return tirx.call_intrin(str(inner.dtype), _Op("tl.simd.pair_get"), inner, index)


def vpack(src, part=0):
    """Pack 32-bit lanes to 16-bit lanes: dst = vpack(src, LOWER/HIGHER)."""
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    src_elem = str(src.dtype).split("x", 1)[0]
    src_lanes = src.dtype.lanes
    pack_map = {"float32": "float16", "int32": "int16", "uint32": "uint16"}
    dst_elem = pack_map.get(src_elem, src_elem)
    vec_dtype = f"{dst_elem}x{src_lanes * 2}"
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vpack"), src, part)


def vgatherb(base, index, mask=None):
    """Gather 32B blocks from base using vector_u32 block offsets. Returns a vector."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "r", extent=_vreg_lanes(_elem_bits(elem_dtype)))
        vec_dtype = _vec_dtype(elem_dtype)
    else:
        inner_load = base.args[0]
        elem_dtype = str(inner_load.buffer.dtype)
        vec_dtype = _vec_dtype(elem_dtype)
    args = [base, index]
    if mask is not None:
        args.append(mask)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vgatherb"), *args)


def vgather2(base, index, mask):
    """Gather elements from base using per-lane offsets."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "r", extent=_vreg_lanes(_elem_bits(elem_dtype)))
    else:
        inner_load = base.args[0]
        elem_dtype = str(inner_load.buffer.dtype)
    out_elem = {
        "int8": "int16",
        "uint8": "uint16",
    }.get(elem_dtype, elem_dtype)
    return tirx.call_intrin(
        _vec_dtype(out_elem),
        _Op("tl.simd.vgather2"),
        base,
        index,
        mask,
    )


# -- Type conversion --------------------------------------------------------------


def vexpdif(src0, src1, mask, part=0):
    """Fused exp-sub: dst = vexpdif(src0, src1, mask, PART_X).

    Computes dst = exp(src0 - src1). part: 0=EVEN, 1=ODD.
    """
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vexpdif"), src0, src1, mask, part)


def vcvt(
    src,
    mask,
    target_dtype,
    round="ROUND_R",
    sat=True,
    part=0,
    mode="MODE_ZEROING",
):
    """Vector type conversion between float32, float16, bfloat16, float8, float4, and integers.

    Parameters
    ----------
    src : VReg
        Source vector register holding the input element values.
    mask : VReg (bool)
        Predicate mask; only lanes where mask[i] is true participate and are written.
    target_dtype : T.dtype
        Destination element type, e.g. "float32", "float16", "bfloat16", "float8_e4m3",
        "float8_e5m2", "float4_e2m1fn", "int32", "int16", "uint16", "int8", "uint8", "int64".
    round : str
        IEEE-754 rounding mode. Valid values:
        - ``"ROUND_R"`` (default): Round to nearest, ties to even (banker's rounding).
        - ``"ROUND_A"``: Round away from zero.
        - ``"ROUND_F"``: Round toward +∞ (ceiling).
        - ``"ROUND_C"``: Round toward -∞ (floor).
        - ``"ROUND_Z"``: Round toward zero (truncation).
        - ``"ROUND_O"``: Round to odd (only for f32→f16).
        - ``"ROUND_H"``: Round half away from zero (only for hif8 conversions).
        Not all modes are valid for every conversion pair; unused modes are ignored when absent.
    sat : bool
        saturation mode for narrow/overflow-prone conversions.
        - ``True`` (default): Saturate to target range on overflow.
        - ``False``: No sat; out-of-range values wrap.
        Appears in: all float→fp8, f32→f16/bf16, all float→int, integer narrowing.
        Absent from: widening conversions, f16→bf16, s16→f16, s32→f32 (no overflow possible).
    part : int
        Sub-register half/quarter selector for widening/narrowing conversions.
        For even/odd 2-way splits: ``"PART_EVEN"`` (0), ``"PART_ODD"`` (1).
        For fp8/fp4/int4 4-way splits: ``"PART_P0"`` (0), ``"PART_P1"`` (1), ``"PART_P2"`` (2), ``"PART_P3"`` (3).
        Not needed for same-width conversions (e.g. f16↔bf16, s32→f32, f32→s32, f16→s16).
    mode : str
        Write mode for masked-off lanes.
        - ``"MODE_ZEROING"`` (default): Inactive lanes are set to zero.
        - ``"MODE_MERGING"``: Inactive lanes preserve their prior value (only on mask-less int→int paths; 920R1 only for mask paths).

    Returns
    -------
    VReg
        Destination vector with the converted elements.

    Supported conversion pairs (simplified)
    ----------------------------------------
    * Float→Int: f32→s64/s32/s16, f16→s32/s16/s8/u8, bf16→s32.
    * Float→Float: f32→f16/bf16/fp8, f16→fp8/bf16/f32, bf16→fp8/f16/f32/fp4.
    * Int→Float: s16/s32/s64→f16/f32, s8/u8→f16.
    * Int→Int: Most s/u{8,16,32,64} widening/narrowing pairs.
    """

    src_dtype = str(src.dtype).split("x", 1)[0]
    fp16_dtypes = ("float16", "bfloat16")
    fp8_dtypes = ("float8_e4m3", "float8_e4m3fn", "float8_e5m2")
    fp4_dtypes = ("float4_e2m1fn",)

    rounds = ("ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z", "ROUND_O", "ROUND_H")
    if round not in rounds:
        raise ValueError(f"Invalid round '{round}'. Must be one of {rounds}")

    sat = "RS_ENABLE" if sat else "RS_DISABLE"

    part_x2 = ("PART_EVEN", "PART_ODD")
    part_x4 = ("PART_P0", "PART_P1", "PART_P2", "PART_P3")
    if isinstance(part, tirx.IntImm):
        part = int(part.value)

    modes = ("MODE_ZEROING", "MODE_MERGING")
    if mode not in modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {modes}")

    vec_dtype = _vec_dtype(target_dtype)

    def call(*args):
        return tirx.call_intrin(vec_dtype, _Op("tl.simd.vcvt"), src, mask, *args)

    # =========================================================================
    # 1. Float to Int
    # =========================================================================

    if src_dtype == "float32":
        if target_dtype == "int64":
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "int32":
            return call(round, sat, mode)
        if target_dtype == "int16":
            return call(round, sat, part_x2[part], mode)

    if src_dtype == "float16":
        if target_dtype == "int32":
            return call(round, part_x2[part], mode)
        if target_dtype == "int16":
            return call(round, sat, mode)
        if target_dtype in ("int8", "uint8"):
            return call(round, sat, part_x2[part], mode)

    if src_dtype == "bfloat16" and target_dtype == "int32":
        return call(round, sat, part_x2[part], mode)

    # =========================================================================
    # 2. Float to Float
    # =========================================================================

    if src_dtype == "float32":
        if target_dtype in fp16_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x4[part], mode)

    if src_dtype == "float16":
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "bfloat16":
            return call(round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)

    if src_dtype == "bfloat16":
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "float16":
            return call(sat, round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)
        if target_dtype in fp4_dtypes:
            return call(round, part_x4[part], mode)

    if src_dtype in fp8_dtypes and target_dtype == "float32":
        return call(part_x4[part], mode)

    if src_dtype in fp4_dtypes and target_dtype == "bfloat16":
        return call(part_x4[part], mode)

    # =========================================================================
    # 3. Int to Float
    # =========================================================================

    if src_dtype in ("int8", "uint8") and target_dtype == "float16":
        return call(part_x2[part], mode)

    if src_dtype == "int16":
        if target_dtype == "float16":
            return call(round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)

    if src_dtype == "int32" and target_dtype == "float32":
        return call(round, mode)

    if src_dtype == "int64" and target_dtype == "float32":
        return call(round, part_x2[part], mode)

    # =========================================================================
    # 4. Int to Int
    # =========================================================================

    if src_dtype == "int32":
        if target_dtype == "int64":
            return call(part_x2[part], mode)
        if target_dtype in ("int16", "uint16"):
            return call(sat, part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x4[part], mode)

    if src_dtype == "uint32":
        if target_dtype in ("int16", "uint16"):
            return call(sat, part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x4[part], mode)

    if src_dtype == "int16":
        if target_dtype in ("int32", "uint32"):
            return call(part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x2[part], mode)

    if src_dtype == "uint16":
        if target_dtype == "uint32":
            return call(part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x2[part], mode)

    if src_dtype == "int8":
        if target_dtype == "int16":
            return call(part_x2[part], mode)
        if target_dtype == "int32":
            return call(part_x4[part], mode)

    if src_dtype == "uint8":
        if target_dtype == "uint16":
            return call(part_x2[part], mode)
        if target_dtype == "uint32":
            return call(part_x4[part], mode)

    if src_dtype == "int64" and target_dtype == "int32":
        return call(sat, part_x2[part], mode)

    raise ValueError(f"Unsupported vcvt conversion {src_dtype}->{target_dtype}")


# -- Special ----------------------------------------------------------------------


def vsel(src0, src1, mask):
    """Bitwise select: mask ? src0 : src1."""
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsel"), src0, src1, mask)


def vselr(src, index):
    """Select lanes from src using per-lane indices."""
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vselr"), src, index)


def vmaxs(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmaxs"), src, scalar, mask, mode)


def vmins(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmins"), src, scalar, mask, mode)


def vmuls(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmuls"), src, scalar, mask, mode)


def vadds(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vadds"), src, scalar, mask, mode)


def vshls(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshls"), src, scalar, mask, mode)


def vshrs(src, scalar, mask, mode="MODE_ZEROING"):
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshrs"), src, scalar, mask, mode)


def mem_bar(mem_type):
    """Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD)."""
    return tirx.call_intrin("void", _Op("tl.simd.mem_bar"), mem_type)
