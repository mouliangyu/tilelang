"""T.simd.* — Raw CCE vector intrinsics for Ascend SIMD programming.

Function names match the underlying CCE intrinsics directly.
"""

from tvm import tir
from tvm.tir import BufferLoad
from tvm.script import tir as T

from tilelang.language.builtin import access_ptr

_Op = tir.op.Op.get


class SimdPair:
    """Wraps a vintlv/vdintlv result, supports tuple unpacking: a, b = vintlv(x, y).

    Note: ``a, b = ...`` generates two pair_get calls referencing the same vintlv
    expression inline.  The HoistSimdPairs pass eliminates the duplicate.
    """

    def __init__(self, pair):
        self._pair = pair

    @property
    def dtype(self):
        return self._pair.dtype

    def __iter__(self):
        yield tir.call_intrin(str(self._pair.dtype), _Op("tl.simd.pair_get"), self._pair, 0)
        yield tir.call_intrin(str(self._pair.dtype), _Op("tl.simd.pair_get"), self._pair, 1)


_ELEM_BITS = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "int16": 16,
    "uint16": 16,
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
    return tir.call_intrin("boolx256", _Op("tl.simd.pset"), elem_width, dist)


def alloc_var(dtype: str):
    """Allocate a mutable SIMD variable for return-value style SIMD code."""
    return T.alloc_buffer([1], _vec_dtype(dtype), scope="local.var")


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
    return tir.call_intrin(vec_dtype, _Op("tl.simd.vld"), *args)


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
    return tir.call_intrin("void", _Op("tl.simd.vsts"), *args)


def vsstb(src, base, stride, mask):
    """Scatter-store 32B blocks with block stride: vsstb(src, base, stride, mask)."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "w", extent=_vreg_lanes(_elem_bits(elem_dtype)))
    return tir.call_intrin("void", _Op("tl.simd.vsstb"), src, base, stride, mask)


# -- Binary arithmetic ------------------------------------------------------------


def vadd(src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vadd"), src0, src1, mask, mode)


def vsub(src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vsub"), src0, src1, mask, mode)


def vmul(src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vmul"), src0, src1, mask, mode)


def vdiv(src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vdiv"), src0, src1, mask, mode)


def vmax(src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vmax"), src0, src1, mask, mode)


def vmin(src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vmin"), src0, src1, mask, mode)


def vand(src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vand"), src0, src1, mask, mode)


def vor(src0, src1, mask, mode="MODE_ZEROING"):  # codespell:ignore
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vor"), src0, src1, mask, mode)  # codespell:ignore


# -- Unary ------------------------------------------------------------------------


def vexp(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vexp"), src, mask, mode)


def vln(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vln"), src, mask, mode)


def vsqrt(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vsqrt"), src, mask, mode)


def vrsqrt(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vrsqrt"), src, mask, mode)


def vabs(src, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vabs"), src, mask, mode)


def vneg(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vneg"), src, mask, mode)


def vrelu(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vrelu"), src, mask, mode)


def vrec(src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vrec"), src, mask, mode)


# -- Broadcast --------------------------------------------------------------------


def vdup(src, dtype_str, mask, mode="MODE_ZEROING"):
    """Broadcast scalar to all lanes: dst = vdup(scalar, dtype_str, mask).

    dtype_str specifies the target vector element type (e.g. "float32").
    """
    vec_dtype = _vec_dtype(dtype_str)
    return tir.call_intrin(vec_dtype, _Op("tl.simd.vdup"), src, mask, mode)


def vdupv(src, mask, pos="POS_LOWEST", mode="MODE_ZEROING"):
    """Broadcast lane N of src vector to all lanes: dst = vdupv(src, mask, pos).

    pos: "POS_LOWEST" (lane 0) or "POS_HIGHEST" (lane N).
    """
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vdupv"), src, mask, pos, mode)


# -- Cross-lane reduction ---------------------------------------------------------


def vcadd(src, mask, mode="MODE_ZEROING"):
    """Pairwise add reduction: dst = vcadd(src, mask)."""
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vcadd"), src, mask, mode)


def vcmax(src, mask, mode="MODE_ZEROING"):
    """Pairwise max reduction: dst = vcmax(src, mask)."""
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vcmax"), src, mask, mode)


def vcgadd(src, mask, mode="MODE_ZEROING"):
    """Grouped add reduction: dst = vcgadd(src, mask)."""
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vcgadd"), src, mask, mode)


# -- Register permutation ---------------------------------------------------------


def vintlv(src0, src1):
    """Interleave two vector registers.

    Unpack:  a, b = vintlv(x, y)
    """
    pair = tir.call_intrin(str(src0.dtype), _Op("tl.simd.vintlv"), src0, src1)
    return SimdPair(pair)


def vdintlv(src0, src1):
    """De-interleave two vector registers.

    Unpack:  a, b = vdintlv(x, y)
    """
    pair = tir.call_intrin(str(src0.dtype), _Op("tl.simd.vdintlv"), src0, src1)
    return SimdPair(pair)


def pair_get(pair, index):
    """Extract element from a pair: reg = pair_get(pair, 0) or pair_get(pair, 1)."""
    inner = pair._pair if isinstance(pair, SimdPair) else pair
    return tir.call_intrin(str(inner.dtype), _Op("tl.simd.pair_get"), inner, index)


def vpack(src, part=0):
    """Pack 32-bit lanes to 16-bit lanes: dst = vpack(src, LOWER/HIGHER)."""
    if isinstance(part, tir.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    src_elem = str(src.dtype).split("x", 1)[0]
    src_lanes = src.dtype.lanes
    pack_map = {"float32": "float16", "int32": "int16", "uint32": "uint16"}
    dst_elem = pack_map.get(src_elem, src_elem)
    vec_dtype = f"{dst_elem}x{src_lanes * 2}"
    return tir.call_intrin(vec_dtype, _Op("tl.simd.vpack"), src, part)


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
    return tir.call_intrin(vec_dtype, _Op("tl.simd.vgatherb"), *args)


# -- Type conversion --------------------------------------------------------------


def vexpdif(src0, src1, mask, part=0):
    """Fused exp-sub: dst = vexpdif(src0, src1, mask, PART_X).

    Computes dst = exp(src0 - src1).  part: 0=EVEN, 1=ODD.
    """
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vexpdif"), src0, src1, mask, part)


def vcvt(
    src,
    mask,
    target_dtype,
    round_mode=0,
    saturation=None,
    part=0,
    mode="MODE_ZEROING",
):
    """Vector cast. target_dtype is the destination element type (e.g. "float16")."""

    src_dtype = str(src.dtype).split("x", 1)[0]
    fp16_dtypes = ("float16", "bfloat16")
    fp8_dtypes = ("float8_e4m3", "float8_e4m3fn", "float8_e5m2")
    fp4_dtypes = ("float4_e2m1fn",)

    ROUNDS = ("ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z", "ROUND_O", "ROUND_H")
    if isinstance(round_mode, tir.IntImm):
        round_mode = int(round_mode.value)

    saturation = saturation or "RS_ENABLE"

    PART_F16 = ("PART_EVEN", "PART_ODD")
    PART_F8 = ("PART_P0", "PART_P1", "PART_P2", "PART_P3")
    if isinstance(part, tir.IntImm):
        part = int(part.value)

    mode = mode or "MODE_ZEROING"

    vec_dtype = _vec_dtype(target_dtype)

    def call(*args):
        return tir.call_intrin(vec_dtype, _Op("tl.simd.vcvt"), src, mask, *args)

    if src_dtype == "float32":
        if target_dtype == "int32":
            return call("ROUND_Z", saturation, mode)
        elif target_dtype in fp16_dtypes:
            return call(ROUNDS[round_mode], saturation, PART_F16[part], mode)
        elif target_dtype in fp8_dtypes:
            return call(ROUNDS[round_mode], saturation, PART_F8[part], mode)

    if target_dtype == "float32":
        if src_dtype == "int32":
            return call("ROUND_R", mode)
        elif src_dtype in fp16_dtypes:
            return call(PART_F16[part], mode)
        elif src_dtype in fp8_dtypes:
            return call(PART_F8[part], mode)

    if src_dtype in fp16_dtypes and target_dtype in fp8_dtypes:
        return call(ROUNDS[round_mode], saturation, PART_F16[part], mode)

    if src_dtype == "bfloat16" and target_dtype in fp4_dtypes:
        return call(ROUNDS[round_mode], PART_F8[part], mode)

    if src_dtype in fp4_dtypes and target_dtype == "bfloat16":
        return call(PART_F8[part], mode)

    raise ValueError(f"Unsupported vcvt conversion {src_dtype}->{target_dtype}")


# -- Special ----------------------------------------------------------------------


def vsel(src0, src1, mask):
    """Bitwise select: mask ? src0 : src1."""
    return tir.call_intrin(str(src0.dtype), _Op("tl.simd.vsel"), src0, src1, mask)


def vmuls(src, scalar, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vmuls"), src, scalar, mask, mode)


def vadds(src, scalar, mask, mode="MODE_ZEROING"):
    return tir.call_intrin(str(src.dtype), _Op("tl.simd.vadds"), src, scalar, mask, mode)


def mem_bar(mem_type):
    """Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD)."""
    return tir.call_intrin("void", _Op("tl.simd.mem_bar"), mem_type)
