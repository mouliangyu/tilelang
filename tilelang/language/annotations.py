"""Annotation helpers exposed on the TileLang language surface."""

from collections.abc import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.tirx import FloatImm, IntImm, tvm_tuple
from tvm.tirx.script.parser import attr
from tvm.tirx.script.builder.ir import sblock_attr

_VALID_MEMORY_SCOPES = ("shared", "shared.l1", "shared.l0c", "shared.l0a", "shared.l0b")

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
    "annotate_min_blocks_per_sm",
    "annotate_buffer_versions",
    "annotate_manual_multi_buffer",
    "annotate_unlimit_memory",
]


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", tvm_tuple(device_func, panel_size))


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return sblock_attr({"layout_map": _layout_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return sblock_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return sblock_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_min_blocks_per_sm(n: int):
    """Annotate the minimum number of thread blocks per SM (multiprocessor).

    When set, this value is passed as the second argument of
    ``__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`` in
    the generated CUDA kernel.  A larger value hints the compiler to limit
    register usage so that more blocks can reside on each SM simultaneously,
    which can improve occupancy at the cost of potentially more register
    spilling.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(...):
    ...     with T.Kernel(...):
                T.annotate_min_blocks_per_sm(2)
    ...         ...
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    return attr(None, "tl.min_blocks_per_sm", n)


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return sblock_attr({"tl.non_restrict_params": data_vars})


def annotate_buffer_versions(buffer_versions_map: dict):
    """Manually control the number of multi-buffer versions for specific buffers.

    When using T.Persistent with num_stages > 1, the Z3 auto-scheduler computes
    optimal buffer version counts for each on-chip buffer. Use this annotation
    to override the Z3 solver and set a fixed version count for selected buffers.

    This is useful when you want to guarantee double-buffering for a critical
    buffer regardless of the Z3 solver's choice.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(x: T.Tensor((M, K), T.float16),
    ...               w: T.Tensor((N, K), T.float16),
    ...               y: T.Tensor((M, N), T.float16)):
    ...     with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
    ...         shared_buf = T.alloc_shared((block_M, block_K), T.float16)
    ...         T.annotate_buffer_versions({shared_buf: 2})
    ...         with T.Persistent(T.ceildiv(M, block_M), num_stages=4) as i:
    ...             ...
    """
    _buffer_versions_map = {}
    for buffer, num_versions in buffer_versions_map.items():
        if not isinstance(num_versions, int) or num_versions < 1:
            raise ValueError(f"num_versions must be a positive integer, got {num_versions} for buffer {buffer}")
        _buffer_versions_map[buffer.data] = num_versions
    return sblock_attr({"tl.buffer_versions_map": _buffer_versions_map})


def annotate_manual_multi_buffer(*items):
    """Declare buffers that the user has *manually* multi-buffered.

    Unlike :func:`annotate_buffer_versions` (which asks the Z3 auto-scheduler to
    version a buffer for you), this annotation tells AutoSchedule that the buffer
    is *already* multi-buffered by hand: the user writes the version index into
    every access themselves (e.g. ``buf[l % 2]`` / ``buf[(l + 1) % 2]``
    ping-pong), and the pass only needs to analyze dependencies and emit the
    ``set_flag`` / ``wait_flag`` synchronization.

    The per-buffer version count ``N`` is used as the ring size / the upper bound
    on the cross-iteration distance the analysis searches for — it does NOT
    reshape or reallocate the buffer and does NOT assume the version index lives
    in dimension 0. Each positional argument is either:

    - a bare buffer: ``N`` defaults to its leading dimension ``buffer.shape[0]``
      (which must then be a compile-time constant) — the common
      ``T.alloc_shared((N, ...))`` ping-pong case; or
    - a ``{buffer: N}`` dict: gives ``N`` explicitly, decoupled from the shape.
      Use this when the buffer has no dedicated leading version dim — e.g. a
      flattened buffer multi-buffered by disjoint sub-ranges — or when you want a
      version count that differs from ``shape[0]``.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(...):
    ...     with T.Kernel(...) as core_id:
    ...         buf = T.alloc_shared((2, N), T.bfloat16)     # leading version dim
    ...         flat = T.alloc_shared((3 * N,), T.bfloat16)  # flat, sub-ranges
    ...         T.annotate_manual_multi_buffer(buf, {flat: 3})
    ...         for l in T.serial(L):
    ...             ...  # buf[l % 2] ping-pong; flat[(l % 3) * N : ...] triple-buffer
    """

    def _as_count(n, buffer):
        n = int(n.value if isinstance(n, IntImm) else n)
        if n < 1:
            raise ValueError(f"manual multi-buffer version count must be >= 1, got {n} for buffer {buffer}")
        return n

    _buffer_versions_map = {}
    for item in items:
        if isinstance(item, dict):
            for buffer, num_versions in item.items():
                _buffer_versions_map[buffer.data] = _as_count(num_versions, buffer)
            continue
        buffer = item
        shape0 = buffer.shape[0]
        if not isinstance(shape0, (int, IntImm)):
            raise ValueError(
                f"annotate_manual_multi_buffer needs a constant leading (version) dimension "
                f"to infer N for buffer {buffer}; pass {{buffer: N}} to specify it explicitly"
            )
        _buffer_versions_map[buffer.data] = _as_count(shape0, buffer)
    return sblock_attr({"tl.manual_multi_buffer": _buffer_versions_map})


def annotate_unlimit_memory(*scopes: str):
    """Remove the memory capacity limit for the given scopes during auto-scheduling.

    When unlimit'd, a scope's capacity is effectively unbounded so Z3 will not
    constrain multi-buffer versions based on that memory pool. Use this when you
    know a particular scope has spare headroom.

    Valid scopes: ``"shared"``, ``"shared.l1"``, ``"shared.l0c"``, ``"shared.l0a"``, ``"shared.l0b"``.

    Example
    -------
    >>> @T.prim_func
    ... def my_kernel(x: T.Tensor((M, K), T.float16),
    ...               w: T.Tensor((N, K), T.float16),
    ...               y: T.Tensor((M, N), T.float16)):
    ...     with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
    ...         T.annotate_unlimit_memory("shared")
    ...         ...
    """
    valid = _VALID_MEMORY_SCOPES
    for scope in scopes:
        if scope not in valid:
            raise ValueError(f"Invalid memory scope {scope!r}, must be one of {valid}")
    return sblock_attr({"tl.unlimit_memory_scopes": list(scopes)})
