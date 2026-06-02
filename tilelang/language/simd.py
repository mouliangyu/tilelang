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
        addr = access_ptr(addr, "r", extent=_vreg_extent(addr.buffer.dtype))
    args = [dst, addr, dist]
    if off is not None:
        args.append(off)
    return tir.call_intrin("void", _Op("tl.simd.vld"), *args)


def vsts(addr, src, mask, dist="NORM_B32", off=None):
    """Vector store: vsts(src, base, off, dist, mask).

    addr can be a BufferLoad auto-wrapped as tl.access_ptr.
    """
    if isinstance(addr, BufferLoad):
        addr = access_ptr(addr, "w", extent=_vreg_extent(addr.buffer.dtype))
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


# -- Type conversion ----------------------------------------------------------


def vfcvt_narrow(dst, src, mask, round_mode=4, part=0, mode="MODE_MERGING"):
    """Narrowing cast f32->f16/bf16: vfcvt(dst, src, ROUND_X, PART_X, mode).

    round_mode: 0=R, 1=A, 2=F, 3=C, 4=Z, 5=O, 6=H.  part: 0=EVEN, 1=ODD.
    mode: "MODE_MERGING", "MODE_ZEROING", "MODE_ZEROING".
    """
    return tir.call_intrin("void", _Op("tl.simd.vfcvt_narrow"), dst, src, mask, round_mode, part, mode)


def vfcvt_widen(dst, src, part=0):
    """Widening cast f16/bf16->f32: vfcvt(dst, src, PART_X).

    part: 0=EVEN, 1=ODD.
    """
    return tir.call_intrin("void", _Op("tl.simd.vfcvt_widen"), dst, src, part)


def vfcvt_f2i(dst, src, mask, round_mode=4):
    """Float to int cast: vfcvt(dst, src, mask, ROUND_X, MODE_XX)."""
    return tir.call_intrin("void", _Op("tl.simd.vfcvt_f2i"), dst, src, mask, round_mode)


def vfcvt_i2f(dst, src, mask, round_mode=4):
    """Int to float cast: vfcvt(dst, src, mask, ROUND_X, MODE_XX)."""
    return tir.call_intrin("void", _Op("tl.simd.vfcvt_i2f"), dst, src, mask, round_mode)


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
