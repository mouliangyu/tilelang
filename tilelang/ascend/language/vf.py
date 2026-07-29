"""SimtVF (SIMT Vector Fixed) language interface for Ascend NPU."""

from __future__ import annotations

from tvm.ffi import register_object
from tvm.tirx.script.builder.frame import TIRFrame
from tilelang import _ffi_api


@register_object("tl.SimtVFFrame")
class SimtVFFrame(TIRFrame):
    """Frame for SimtVF (SIMT Vector Fixed) region.

    This frame wraps the SIMT vector-fragment execution region for Ascend NPU.
    The C++ side (SimtVFFrameNode) handles scope entry/exit and generates
    the Block("SIMT_VF") IR node with thread extent bindings.
    """

    def __enter__(self):
        super().__enter__()
        from tilelang.language.kernel import SimtVFContext, push_simtvf_context

        ctx = SimtVFContext(
            thread_vars=list(self.thread_vars),
            thread_extents=[int(e) for e in self.thread_extents],
        )
        push_simtvf_context(ctx)
        return self

    def __exit__(self, ptype, value, trace):
        from tilelang.language.kernel import pop_simtvf_context

        pop_simtvf_context()
        super().__exit__(ptype, value, trace)


def SimtVF(threads: int | list[int] | tuple = 128, latency: int = 0) -> SimtVFFrame:
    """Construct a dedicated SIMT vector-fragment region frame.

    Parameters
    ----------
    threads : int | list[int] | tuple
        Thread count for the SimtVF frame.
        Can be a single int (threadIdx.x only), or a list/tuple of up to 3
        integers for (threadIdx.x, threadIdx.y, threadIdx.z).
        Missing dimensions default to 1.
    latency : int
        Measured SimtVF latency in cycles. When non-zero, this value is
        attached to the generated SIMT_VF Block as a ``tl.vf_latency``
        annotation so the auto-schedule latency estimator can use it
        instead of the generic vector model. Default: 0 (use model).

    Returns
    -------
    res : SimtVFFrame
        The SimtVFFrame used to denote a SimtVF region.
    """
    if isinstance(threads, int):
        normalized = [threads, 1, 1]
    elif isinstance(threads, list):
        if len(threads) > 3:
            raise ValueError("SimtVF supports at most 3 thread dimensions")
        normalized = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        if len(threads) > 3:
            raise ValueError("SimtVF supports at most 3 thread dimensions")
        normalized = list(threads) + [1] * (3 - len(threads))
    else:
        raise TypeError(f"threads must be int, list, or tuple, got {type(threads)}")

    if any(t <= 0 for t in normalized):
        raise ValueError("SimtVF requires all thread dimensions > 0")

    return _ffi_api.SimtVF(normalized, int(latency))  # type: ignore[attr-defined] # pylint: disable=no-member


@register_object("tl.SimdVFFrame")
class SimdVFFrame(TIRFrame):
    """Frame for SimdVF (SIMD Vector Fragment) region on Ascend NPU.

    No thread dimensions — purely register-level SIMD model using MicroAPI.
    The C++ side (SimdVFFrameNode) handles scope entry/exit and generates
    the Block("SIMD_VF") IR node with tl.simdvf_scope marker.
    """

    def __enter__(self):
        if _inside_simdvf[0] == 0:
            from .simd import _reset_default_mask_cache

            _reset_default_mask_cache()
        _inside_simdvf[0] += 1
        super().__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)
        _inside_simdvf[0] -= 1
        if _inside_simdvf[0] == 0:
            from .simd import _reset_default_mask_cache

            _reset_default_mask_cache()


# Thread-local counter for nested SimdVF context
_inside_simdvf = [0]


def inside_simdvf() -> bool:
    """Return True if currently inside a SimdVF region."""
    return _inside_simdvf[0] > 0


def SimdVF(latency: int = 0) -> SimdVFFrame:
    """Construct a SimdVF region for Ascend NPU MicroAPI operations.

    Inside this region, the AscendLowerParallelToVector pass will lower
    T.Parallel loops into MicroAPI register-level SIMD operations.
    The region is emitted as a __simd_vf__ helper function and called inline.

    latency : int
        Measured SimdVF latency in cycles. When non-zero, this value is
        attached to the generated SIMD_VF Block as a ``tl.vf_latency``
        annotation so the auto-schedule latency estimator can use it
        instead of the generic vector model. Default: 0 (use model).

    Returns
    -------
    res : SimdVFFrame
        The SimdVFFrame used to denote a SimdVF region.
    """
    return _ffi_api.SimdVF(int(latency))  # type: ignore[attr-defined] # pylint: disable=no-member
