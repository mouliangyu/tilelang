"""Annotation helpers exposed on the TileLang language surface."""

from typing import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.script.parser.tir import attr, block_attr
from tvm.tir import FloatImm

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
    "annotate_buffer_versions",
]


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")


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

    return block_attr({"layout_map": _layout_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


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
    return block_attr({"tl.non_restrict_params": data_vars})


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
    return block_attr({"tl.buffer_versions_map": _buffer_versions_map})
