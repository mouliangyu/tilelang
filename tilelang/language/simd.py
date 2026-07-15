"""T.simd.* — Raw CCE vector intrinsics for Ascend SIMD programming.

Function names match the underlying CCE intrinsics directly.
"""

from tvm import tirx
from tvm.tirx import BufferLoad
from tvm.tirx.script.builder.ir import bind as _bind
from tvm.script.ir_builder import IRBuilder
from tilelang._typing import DType, ShapeType
from .allocate import alloc_var as _alloc_var, alloc_local as _alloc_local
from .builtin import access_ptr

_Op = tirx.op.Op.get


class SimdPair:
    """Wraps a vintlv/vdintlv/vld2 result, supports unpacking: a, b = vintlv(x, y).

    ``a, b = ...`` always emits two ``pair_get`` calls. For pure register
    permutations (``vintlv`` / ``vdintlv``) the Ascend ``HoistSimdPairs`` pass
    CSE's the duplicated pair expression. For opaque memory loads (``vld2``),
    the frontend binds the pair once at the call site so hoist never
    CSE/code-motions a load.
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


def _vreg_lanes(dtype_str: DType) -> int:
    key = str(dtype_str)
    if key == "bool":
        return 256
    if key not in _ELEM_BITS:
        raise ValueError(f"Unknown element dtype '{key}'. Must be one of {sorted(_ELEM_BITS)}")
    elem_bits = _ELEM_BITS[key]
    return 2048 // elem_bits


def _vec_dtype(elem_dtype: DType) -> str:
    lanes = _vreg_lanes(elem_dtype)
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


def _mask_bits(dtype_like) -> int:
    """CCE predicate width (b8/b16/b32/b64) for a dtype's element type."""
    elem = str(dtype_like).split("x", 1)[0]
    if elem not in _ELEM_BITS:
        raise ValueError(f"Cannot infer default mask width for dtype '{dtype_like}'")
    return max(_ELEM_BITS[elem], 8)  # CCE has no pset_b4; float4 -> b8


# Auto default-mask cache: one bound pset per element-bit width, reused across maskless
# ops as long as the frame it was bound into is still an enclosing scope. The TVMScript
# parser hands out a fresh Python wrapper per frame access, so identity is compared with
# ``same_as`` (underlying-object equality), not ``id()``. Reset per SimdVF block by vf.py.
_default_mask_cache: dict = {}  # elem_bits -> (declaring_frame, mask_var)


def _reset_default_mask_cache() -> None:
    _default_mask_cache.clear()


def _frame_on_stack(frame) -> bool:
    """True if ``frame`` is still an enclosing scope on the IRBuilder frame stack."""
    if frame is None or not IRBuilder.is_in_scope():
        return False
    return any(frame.same_as(f) for f in IRBuilder.current().frames)


def _default_mask_bits(bits: int):
    """Bind (once per scope) and return an all-lanes pset mask of the given predicate
    width (b8/b16/b32/b64).

    Emits ``vector_bool v = pset_bXX(PAT_ALL);`` into the current SimdVF frame so the
    Ascend codegen BindNode path renders it correctly (an inline pset is not handled in
    expression context). Repeated maskless ops reuse the same bound mask while its
    declaring scope is still open, instead of emitting a new pset each time.
    """
    cached = _default_mask_cache.get(bits)
    if cached is not None and _frame_on_stack(cached[0]):
        return cached[1]
    var = _bind(pset(bits, "PAT_ALL"))
    if IRBuilder.is_in_scope():
        frames = IRBuilder.current().frames
        _default_mask_cache[bits] = (frames[-1] if len(frames) else None, var)
    return var


def _default_mask(dtype_like):
    """Bind and return an all-lanes pset mask matching dtype_like's element width."""
    return _default_mask_bits(_mask_bits(dtype_like))


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
        lanes = _vreg_lanes(elem_dtype)
        extent = lanes
        if isinstance(dist, str):
            if dist.startswith("BRC_"):
                extent = 1
            elif dist == "E2B_B16":
                # Expand-to-block: 8 bf16/u16 values → 8×16-lane blocks.
                extent = 8
            elif dist in ("UNPK_B16", "UNPK4_B8"):
                extent = 64
        addr = access_ptr(addr, "r", extent=extent)
        vec_dtype = f"{elem_dtype}x{lanes}"
    else:
        inner_load = addr.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)

    args = [addr, dist]
    if off is not None:
        args.append(off)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vld"), *args)


_VLD2_DISTS = ("DINTLV_B8", "DINTLV_B16")


def vld2(addr, dist="DINTLV_B16", off=None):
    """Dual-dest vector load: ``a, b = vld2(x_ub[i, col], dist=\"DINTLV_B8\")``.

    Supported dists:
      - ``DINTLV_B8``: load 512×u8/fp8 → two 256-lane regs (even/odd bytes)
      - ``DINTLV_B16``: load 256×bf16/u16 → two 128-lane regs

    The access_ptr footprint is always 2× the single-vector width. Unlike
    ``vintlv``/``vdintlv``, this is an opaque memory load: the pair is bound
    once at this program point so ``HoistSimdPairs`` does not CSE or
    code-motion it.
    """
    if isinstance(dist, str) and dist not in _VLD2_DISTS:
        raise ValueError(f"vld2 only supports dist in {_VLD2_DISTS}, got {dist!r}")
    if isinstance(addr, BufferLoad):
        elem_dtype = str(addr.buffer.dtype)
        lanes = _vreg_lanes(elem_dtype)
        # Dual-dest deinterleave always consumes 2× vector width from UB.
        extent = lanes * 2
        addr = access_ptr(addr, "r", extent=extent)
        vec_dtype = f"{elem_dtype}x{lanes}"
    else:
        # Pre-wrapped access_ptr: this path does not auto-double the footprint
        # (same as vld). Callers must size extent to 2× vector width themselves.
        inner_load = addr.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)

    args = [addr, dist]
    if off is not None:
        args.append(off)
    pair = tirx.call_intrin(vec_dtype, _Op("tl.simd.vld2"), *args)
    # Bind at the original site (opaque load). Requires an active IRBuilder —
    # do not fall back to an unbound SimdPair (that reintroduces duplicate
    # inline loads under unpack).
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vld2 requires an active IRBuilder to bind the opaque load")
    return SimdPair(_bind(pair))


def vsts(addr, src, mask=None, dist="NORM_B32", off=None, extent=None):
    """Vector store: vsts(src, base, off, dist, mask).

    addr can be a BufferLoad auto-wrapped as tl.access_ptr.
    Optional ``extent`` overrides the default access_ptr footprint (e.g. 8 for
    a dense PAT_VL8 NORM_B16 recip pack).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    if isinstance(addr, BufferLoad):
        lanes = _vreg_lanes(addr.buffer.dtype)
        if extent is None:
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


def vsstb(src, base, stride, mask=None):
    """Scatter-store 32B blocks with block stride: vsstb(src, base, stride, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    if isinstance(base, BufferLoad):
        base = access_ptr(base, "w", extent=_vreg_lanes(base.buffer.dtype))
    return tirx.call_intrin("void", _Op("tl.simd.vsstb"), src, base, stride, mask)


# -- Binary arithmetic ------------------------------------------------------------


def vadd(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vadd"), src0, src1, mask, mode)


def vsub(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsub"), src0, src1, mask, mode)


def vmul(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmul"), src0, src1, mask, mode)


def vmula(dst, src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused multiply-add: dst = dst + src0 * src1."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vmula: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vmula"), dst, src0, src1, mask, mode)


def vmadd(dst, src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused multiply-add: dst = dst * src0 + src1."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vmadd: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vmadd"), dst, src0, src1, mask, mode)


def vaxpy(dst, src, scalar, mask=None, mode="MODE_ZEROING"):
    """Fused scalar multiply-add: dst = src * scalar + dst."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vaxpy: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vaxpy"), dst, src, scalar, mask, mode)


def vdiv(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vdiv"), src0, src1, mask, mode)


def vmax(src0, src1, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmax"), src0, src1, mask, mode)


def vmin(src0, src1, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmin"), src0, src1, mask, mode)


def vand(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vand"), src0, src1, mask, mode)


def vor(src0, src1, mask=None, mode="MODE_ZEROING"):  # codespell:ignore
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vor"), src0, src1, mask, mode)  # codespell:ignore


def vshl(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshl"), src0, src1, mask, mode)


def vshr(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshr"), src0, src1, mask, mode)


# -- Unary ------------------------------------------------------------------------


def vexp(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vexp"), src, mask, mode)


def vln(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vln"), src, mask, mode)


def vsqrt(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqrt"), src, mask, mode)


def vabs(src, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vabs"), src, mask, mode)


def vneg(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vneg"), src, mask, mode)


def vrelu(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vrelu"), src, mask, mode)


# -- Broadcast --------------------------------------------------------------------


def vdup(src, dtype_str: DType, mask=None, mode="MODE_ZEROING"):
    """Broadcast scalar to all lanes: dst = vdup(scalar, dtype_str, mask).

    dtype_str specifies the target vector element type (e.g. "float32").
    """
    if mask is None:
        mask = _default_mask(dtype_str)
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vdup"), src, mask, mode)


def vdupv(src, mask=None, pos="POS_LOWEST", mode="MODE_ZEROING"):
    """Broadcast lane N of src vector to all lanes: dst = vdupv(src, mask, pos).

    pos: "POS_LOWEST" (lane 0) or "POS_HIGHEST" (lane N).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vdupv"), src, mask, pos, mode)


# -- Cross-lane reduction ---------------------------------------------------------


def vcadd(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise add reduction: dst = vcadd(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    src_elem = str(src.dtype).split("x", 1)[0]
    out_dtype = {
        "int16": _vec_dtype("int32"),
        "uint16": _vec_dtype("uint32"),
    }.get(src_elem, str(src.dtype))
    return tirx.call_intrin(out_dtype, _Op("tl.simd.vcadd"), src, mask, mode)


def vcmax(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise max reduction: dst = vcmax(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmax"), src, mask, mode)


def vcmin(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise min reduction: dst = vcmin(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmin"), src, mask, mode)


def vcgadd(src, mask=None, mode="MODE_ZEROING"):
    """Grouped add reduction: dst = vcgadd(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgadd"), src, mask, mode)


def vcgmax(src, mask=None, mode="MODE_ZEROING"):
    """Grouped max reduction: dst = vcgmax(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgmax"), src, mask, mode)


def vcgmin(src, mask=None, mode="MODE_ZEROING"):
    """Grouped min reduction: dst = vcgmin(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgmin"), src, mask, mode)


def vsqz(src, mask=None, mode="MODE_STORED"):
    """Squeeze selected lanes toward the low lanes: dst = vsqz(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqz"), src, mask, mode)


# -- Index ramp / compare ---------------------------------------------------------


def vci(index, dtype_str: DType, order="INC_ORDER"):
    """Index ramp: dst = vci(index, dtype_str, order).

    dst[lane] = index + lane (INC_ORDER) / index - lane (DEC_ORDER).
    dtype_str is the destination vector element type (e.g. "int32", "float32").
    """
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vci"), index, order)


def vcmp(src0, src1, mask=None, op="eq"):
    """Elementwise compare → vector_bool: dst = vcmp_<op>(src0, src1, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin("boolx256", _Op("tl.simd.vcmp"), src0, src1, mask, op)


def vcmps(src, scalar, mask=None, op="lt"):
    """Compare vector vs scalar → vector_bool: dst = vcmps_<op>(src, scalar, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    if mask is None:
        mask = _default_mask(src.dtype)
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
    """Pack wider lanes to narrower lanes: dst = vpack(src, LOWER/HIGHER).

    Supports u32→u16 and u16→u8 (needed for dense UE8M0 scale packing).
    """
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    src_elem = str(src.dtype).split("x", 1)[0]
    src_lanes = src.dtype.lanes
    pack_map = {
        "float32": "float16",
        "int32": "int16",
        "uint32": "uint16",
        "uint16": "uint8",
        "int16": "int8",
    }
    dst_elem = pack_map.get(src_elem, src_elem)
    vec_dtype = f"{dst_elem}x{src_lanes * 2}"
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vpack"), src, part)


def vgatherb(base, index, mask=None):
    """Gather 32B blocks from base using vector_u32 block offsets. Returns a vector."""
    if isinstance(base, BufferLoad):
        elem_dtype = base.buffer.dtype
        base = access_ptr(base, "r", extent=_vreg_lanes(elem_dtype))
        vec_dtype = _vec_dtype(elem_dtype)
    else:
        inner_load = base.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)
    args = [base, index]
    if mask is not None:
        args.append(mask)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vgatherb"), *args)


def vgather2(base, index, mask=None):
    """Gather elements from base using per-lane offsets."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "r", extent=_vreg_lanes(elem_dtype))
    else:
        inner_load = base.args[0]
        elem_dtype = str(inner_load.buffer.dtype)
    if mask is None:
        # gather reads source elements, so the predicate aligns to the source (low-lane)
        # width -- e.g. an int8 gather widens the output to int16 but masks with pset_b8.
        mask = _default_mask_bits(_mask_bits(elem_dtype))
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


def vscatter(src, base, index, mask=None):
    """Scatter-store: base[index[lane]] = src[lane].

    Write-side counterpart to vgatherb/vgather2. ``index`` is a vector register of
    per-lane element offsets (uint32 for 32-bit elements, uint16 for 8/16-bit).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    if isinstance(base, BufferLoad):
        base = access_ptr(base, "w", extent=_vreg_lanes(base.buffer.dtype))
    return tirx.call_intrin("void", _Op("tl.simd.vscatter"), src, base, index, mask)


# -- Type conversion --------------------------------------------------------------


def vexpdif(src0, src1, mask=None, part=0):
    """Fused exp-sub: dst = vexpdif(src0, src1, mask, PART_X).

    Computes dst = exp(src0 - src1). part: 0=EVEN, 1=ODD.
    Supported types: float32, float16.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vexpdif"), src0, src1, mask, part)


def vabsdif(src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused abs-sub: dst = vabsdif(src0, src1, mask, mode).

    Computes dst = abs(src0 - src1) in a single instruction.
    Supported types: float32, float16.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vabsdif"), src0, src1, mask, mode)


def vcvt(
    src,
    target_dtype: DType,
    mask=None,
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
    target_dtype : T.dtype
        Destination element type, e.g. "float32", "float16", "bfloat16", "float8_e4m3",
        "float8_e5m2", "float4_e2m1fn", "int32", "int16", "uint16", "int8", "uint8", "int64".
    mask : VReg (bool)
        Predicate mask; only lanes where mask[i] is true participate and are written.
        Defaults to an all-lanes mask matching the narrower (low-lane) element: the
        source width when widening, the target width when narrowing.
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

    if mask is None:
        # vcvt masks by the narrower (low-lane) element: source when widening, target
        # when narrowing. So the predicate width is min(src, target), e.g. u8<->u16 both
        # use pset_b8 and u16<->u32 both use pset_b16.
        mask = _default_mask_bits(min(_mask_bits(src.dtype), _mask_bits(target_dtype)))

    target_dtype = str(target_dtype)
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
    src_dtype = str(src.dtype)
    index_dtype = str(index.dtype)
    # The index must be an unsigned integer vector.
    if index_dtype.startswith("int"):
        index = tirx.reinterpret("u" + index_dtype, index)
    # vselr supports f16/bf16 natively but not f32: reinterpret f32 to the same-width unsigned integer, select, then reinterpret back.
    if src_dtype.startswith("float32"):
        uint_dtype = src_dtype.replace("float", "uint")
        src = tirx.reinterpret(uint_dtype, src)
        res = tirx.call_intrin(uint_dtype, _Op("tl.simd.vselr"), src, index)
        return tirx.reinterpret(src_dtype, res)
    return tirx.call_intrin(src_dtype, _Op("tl.simd.vselr"), src, index)


def vmaxs(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmaxs"), src, scalar, mask, mode)


def vmins(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmins"), src, scalar, mask, mode)


def vmuls(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmuls"), src, scalar, mask, mode)


def vadds(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vadds"), src, scalar, mask, mode)


def vshls(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshls"), src, scalar, mask, mode)


def vshrs(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshrs"), src, scalar, mask, mode)


def mem_bar(mem_type):
    """Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD)."""
    return tirx.call_intrin("void", _Op("tl.simd.mem_bar"), mem_type)
