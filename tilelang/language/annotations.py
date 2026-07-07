"""Annotation helpers exposed on the TileLang language surface."""

from collections.abc import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment, to_buffer_region
from tvm.tirx import Buffer, FloatImm, IntImm, StringImm, tvm_tuple
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
    "annotate_unlimit_memory",
    "assume_no_conflict",
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


def assume_no_conflict(a, b=None, *, level=None, cross=None, group=None):
    """Assert that two buffer regions do not conflict, suppressing a false
    dependency the auto-scheduler would otherwise conservatively insert.

    The auto-scheduler's dependency analysis cannot prove that dynamically
    indexed accesses (e.g. a permutation ``state_cache[perm[idx], ...]``) are
    disjoint across loop iterations, so it inserts redundant synchronization.
    This hint lets the user declare that the regions do not overlap. It is
    consumed by the ``NormalizeNoConflictHints`` pass before auto-scheduling; at
    runtime it is a no-op.

    Args:
        a: first region -- ``buffer[indices...]`` or a bare ``buffer`` (the
            whole buffer). A region must cover the same footprint as the actual
            access it refers to: use slices for the accessed extents (e.g.
            ``buf[i, 0:M, 0:N]``), since a fully scalar-indexed ``buf[i, 0, 0]``
            denotes only a 1-element region and will not match a wider copy.
        b: second region; defaults to ``a`` (a self-dependency).
        level: which enclosing loop the non-overlap is asserted at, counted from
            the outermost enclosing loop (``0`` = outermost, ``1`` = next inner,
            ...). ``None`` (default) asserts it at each enclosing loop
            independently. For a ``group``, ``level`` indexes the loops enclosing
            both halves (their common ancestors), still counted from the
            outermost.
        cross: ``None`` = both same- and cross-iteration are conflict-free
            (default); ``True`` = only cross-iteration; ``False`` = only
            same-iteration.
        group: string tag pairing two separately-written half-declarations whose
            regions live in different scopes (so they cannot be written in one
            call). Must appear exactly twice; the combined hint is attached to
            the loops enclosing both. Rarely needed.

    Example:
        >>> # self-dependency across iterations: A[perm[i]] written each
        >>> # iteration lands in a disjoint slot (perm is a permutation), so
        >>> # there is no cross-iteration conflict
        >>> T.assume_no_conflict(A[perm[i], 0:N], level=0, cross=True)
        >>> # two distinct regions that never overlap -- both within the same
        >>> # iteration and across iterations (cross=None, the default)
        >>> T.assume_no_conflict(A[lo, 0:N], A[hi, 0:N], level=0)
        >>> # whole buffer: drop every dependency involving B at every loop
        >>> # enclosing this statement (level=None broadcasts to all of them)
        >>> T.assume_no_conflict(B)
        >>> # group: pair two regions written in separate (sibling) scopes
        >>> T.assume_no_conflict(A[wperm[i], 0:N], group="perm")   # in one loop
        >>> T.assume_no_conflict(A[rperm[j], 0:N], group="perm")   # in another
    """
    if cross is True:
        cross_code = 1
    elif cross is False:
        cross_code = 0
    elif cross is None:
        cross_code = -1
    else:
        raise ValueError(f"cross must be None/True/False, got {cross!r}")
    if level is not None and (not isinstance(level, int) or level < 0):
        raise ValueError(f"level must be None or a non-negative int, got {level!r}")
    if group is not None and b is not None:
        raise ValueError(
            "assume_no_conflict: a group's two half-declarations each carry one region, so `b` must be None when `group` is set"
        )

    # A bare Buffer (no indexing) declares non-conflict for the whole buffer; it
    # is carried as the Buffer object and matched by storage key. Everything else
    # is a concrete region.
    def operand(x):
        return x if isinstance(x, Buffer) else to_buffer_region(x)

    a_op = operand(a)
    b_op = operand(b) if b is not None else a_op
    # level None (all enclosing loops) is encoded as -1. All fields live in the
    # node list; the attr value is only a human-readable label (like T.assume).
    level_imm = IntImm("int32", level if level is not None else -1)
    node = [a_op, b_op, level_imm, StringImm(group or ""), IntImm("int32", cross_code)]
    return attr(node, "tl.assume_no_conflict", StringImm("no_conflict"))
