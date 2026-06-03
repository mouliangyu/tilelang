"""T.simd.* — Raw CCE vector intrinsics for Ascend SIMD programming.

Function names match the underlying CCE intrinsics directly.
"""

from tvm import tir
from tvm.tir import BufferLoad

from tilelang.language.builtin import access_ptr

_Op = tir.op.Op.get


def _vreg_extent(dtype) -> int:
    """Compute VReg extent in elements for a given DataType."""
    return 8 * 32 // (dtype.bits // 8)


def alloc(dtype: str):
    """Allocate a SIMD vector register (vector_<T>).

    Returns a Var typed as vector_<T> (e.g. Float(32, 64)), not handle.
    """
    elem_bits = {
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
    }.get(dtype, 32)
    vreg_lanes = 8 * 32 // (elem_bits // 8)
    vec_dtype = f"{dtype}x{vreg_lanes}"
    return tir.call_intrin(vec_dtype, _Op("tl.simd.alloc"), dtype)


def pset(elem_width: int, dist: str = "PAT_ALL"):
    """Create a predicate mask: pset_bXX(dist).

    Returns a vector_bool (boolx256).
    dist: "PAT_ALL", "PAT_VL1".."PAT_VL128", "PAT_M3", "PAT_M4", "PAT_H", "PAT_Q", etc.
    """
    return tir.call_intrin("boolx256", _Op("tl.simd.pset"), elem_width, dist)


# -- Data movement -----------------------------------------------------------


def vld(dst, addr, dist="NORM", off=None):
    """Vector load: vld(dst, base, off, dist).

    addr can be a BufferLoad (e.g. temp[i*64]) auto-wrapped as tl.access_ptr.
    """
    if isinstance(addr, BufferLoad):
        extent = _vreg_extent(addr.buffer.dtype)
        if isinstance(dist, str):
            if dist.startswith("BRC_"):
                extent = 1
            elif dist in ("UNPK_B16", "UNPK4_B8"):
                extent = 64
        addr = access_ptr(addr, "r", extent=extent)
    args = [dst, addr, dist]
    if off is not None:
        args.append(off)
    return tir.call_intrin("void", _Op("tl.simd.vld"), *args)


def vsts(addr, src, mask, dist="NORM_B32", off=None):
    """Vector store: vsts(src, base, off, dist, mask).

    addr can be a BufferLoad auto-wrapped as tl.access_ptr.
    """
    if isinstance(addr, BufferLoad):
        extent = _vreg_extent(addr.buffer.dtype)
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


# -- Binary arithmetic --------------------------------------------------------


def vadd(dst, src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vadd"), dst, src0, src1, mask, mode)


def vsub(dst, src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vsub"), dst, src0, src1, mask, mode)


def vmul(dst, src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vmul"), dst, src0, src1, mask, mode)


def vdiv(dst, src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vdiv"), dst, src0, src1, mask, mode)


def vmax(dst, src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin("void", _Op("tl.simd.vmax"), dst, src0, src1, mask, mode)


def vmin(dst, src0, src1, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin("void", _Op("tl.simd.vmin"), dst, src0, src1, mask, mode)


def vand(dst, src0, src1, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vand"), dst, src0, src1, mask, mode)


def vor(dst, src0, src1, mask, mode="MODE_ZEROING"):  # codespell:ignore
    return tir.call_intrin("void", _Op("tl.simd.vor"), dst, src0, src1, mask, mode)  # codespell:ignore


# -- Unary --------------------------------------------------------------------


def vexp(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vexp"), dst, src, mask, mode)


def vln(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vln"), dst, src, mask, mode)


def vsqrt(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vsqrt"), dst, src, mask, mode)


def vrsqrt(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vrsqrt"), dst, src, mask, mode)


def vabs(dst, src, mask, mode="MODE_ZEROING"):  # noqa: A001
    return tir.call_intrin("void", _Op("tl.simd.vabs"), dst, src, mask, mode)


def vneg(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vneg"), dst, src, mask, mode)


def vrelu(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vrelu"), dst, src, mask, mode)


def vrec(dst, src, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vrec"), dst, src, mask, mode)


# -- Broadcast ----------------------------------------------------------------


def vdup(dst, src, mask, mode="MODE_ZEROING"):
    """Broadcast lane 0 to all lanes: vdup(dst, src, mask, MODE_ZEROING)."""
    return tir.call_intrin("void", _Op("tl.simd.vdup"), dst, src, mask, mode)


# -- Cross-lane reduction -----------------------------------------------------


def vcadd(dst, src, mask, mode="MODE_ZEROING"):
    """Pairwise add reduction: vcadd(dst, src, mask)."""
    return tir.call_intrin("void", _Op("tl.simd.vcadd"), dst, src, mask, mode)


def vcmax(dst, src, mask, mode="MODE_ZEROING"):
    """Pairwise max reduction: vcmax(dst, src, mask)."""
    return tir.call_intrin("void", _Op("tl.simd.vcmax"), dst, src, mask, mode)


# -- Register permutation -----------------------------------------------------


def vintlv(dst0, dst1, src0, src1):
    """Interleave two vector registers: vintlv(dst0, dst1, src0, src1)."""
    return tir.call_intrin("void", _Op("tl.simd.vintlv"), dst0, dst1, src0, src1)


def vdintlv(dst0, dst1, src0, src1):
    """De-interleave two vector registers: vdintlv(dst0, dst1, src0, src1)."""
    return tir.call_intrin("void", _Op("tl.simd.vdintlv"), dst0, dst1, src0, src1)


def vpack(dst, src, part=0):
    """Pack 32-bit lanes to 16-bit lanes: vpack(dst, src, LOWER/HIGHER)."""
    if isinstance(part, tir.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    return tir.call_intrin("void", _Op("tl.simd.vpack"), dst, src, part)


# -- Type conversion ----------------------------------------------------------


def vcvt(
    dst,
    src,
    mask,
    round_mode=0,
    saturation=None,
    part=0,
    mode="MODE_ZEROING",
):
    """Vector cast. The concrete CCE vcvt signature is inferred from dst/src dtype."""

    src_dtype = str(src.dtype).split("x", 1)[0]
    dst_dtype = str(dst.dtype).split("x", 1)[0]
    fp16_dtypes = ("float16", "bfloat16")
    fp8_dtypes = ("float8_e4m3", "float8_e4m3fn", "float8_e5m2")

    ROUNDS = ("ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z", "ROUND_O", "ROUND_H")
    if isinstance(round_mode, tir.IntImm):
        round_mode = int(round_mode.value)

    saturation = saturation or "RS_ENABLE"

    PART_F16 = ("PART_EVEN", "PART_ODD")
    PART_F8 = ("PART_P0", "PART_P1", "PART_P2", "PART_P3")
    if isinstance(part, tir.IntImm):
        part = int(part.value)

    mode = mode or "MODE_ZEROING"

    def call(*args):
        return tir.call_intrin("void", _Op("tl.simd.vcvt"), dst, src, mask, *args)

    if src_dtype == "float32":
        if dst_dtype == "int32":
            return call("ROUND_Z", saturation, mode)
        elif dst_dtype in fp16_dtypes:
            return call(ROUNDS[round_mode], saturation, PART_F16[part], mode)
        elif dst_dtype in fp8_dtypes:
            return call(ROUNDS[round_mode], saturation, PART_F8[part], mode)

    if dst_dtype == "float32":
        if src_dtype == "int32":
            return call("ROUND_R", mode)
        elif src_dtype in fp16_dtypes:
            return call(PART_F16[part], mode)
        elif src_dtype in fp8_dtypes:
            return call(PART_F8[part], mode)

    if src_dtype in fp16_dtypes and dst_dtype in fp8_dtypes:
        return call(ROUNDS[round_mode], saturation, PART_F16[part], mode)

    raise ValueError(f"Unsupported vcvt conversion {src_dtype}->{dst_dtype}")


# -- Special ------------------------------------------------------------------


def vsel(dst, src0, src1, mask, mode="MODE_ZEROING"):
    """Bitwise select: mask ? src0 : src1."""
    return tir.call_intrin("void", _Op("tl.simd.vsel"), dst, src0, src1, mask, mode)


def vmuls(dst, src, scalar, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vmuls"), dst, src, scalar, mask, mode)


def vadds(dst, src, scalar, mask, mode="MODE_ZEROING"):
    return tir.call_intrin("void", _Op("tl.simd.vadds"), dst, src, scalar, mask, mode)


def mem_bar(mem_type):
    """Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD)."""
    return tir.call_intrin("void", _Op("tl.simd.mem_bar"), mem_type)
