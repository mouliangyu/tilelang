"""Kernel launching language interface in TileLang."""

from __future__ import annotations
from collections import deque
from tvm import tir
from tvm.tir import Var
from tvm.target import Target
from tvm.script.ir_builder.tir.frame import TIRFrame, BlockFrame
from tvm.ffi import register_object
from tilelang import _ffi_api
from tilelang.jit.exceptions import JITNoBuilderError
import threading

# Ensure single-dimension kernel bindings can be unpacked like iterables.
# especially for issue https://github.com/tile-ai/tilelang/issues/830
if not hasattr(Var, "__iter__"):

    def _var_iter(self):
        yield self

    Var.__iter__ = _var_iter  # type: ignore[attr-defined]

if not hasattr(Var, "__len__"):
    Var.__len__ = lambda self: 1  # type: ignore[attr-defined]


class FrameStack:
    """
    A simple stack-like wrapper around a deque that provides
    push, pop, and top methods for convenience.
    """

    def __init__(self):
        self._stack = deque()

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        raise IndexError(f"{self.__class__.__name__} is empty")

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")

    def size(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __len__(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __bool__(self):
        """
        Allows truthy checks on the stack object itself,
        e.g., 'if stack: ...'
        """
        return bool(self._stack)


# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local = threading.local()


def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "kernel_launch_frame_stack"):
        _local.kernel_launch_frame_stack = FrameStack()
    return _local.kernel_launch_frame_stack


class SimtVFContext:
    """Stores thread binding info for an active SimtVF scope."""

    __slots__ = ("thread_vars", "thread_extents")

    def __init__(self, thread_vars, thread_extents):
        self.thread_vars = thread_vars
        self.thread_extents = thread_extents


_simtvf_local = threading.local()


def _get_simtvf_stack() -> FrameStack:
    if not hasattr(_simtvf_local, "simtvf_stack"):
        _simtvf_local.simtvf_stack = FrameStack()
    return _simtvf_local.simtvf_stack


def _get_current_simtvf() -> SimtVFContext | None:
    stack = _get_simtvf_stack()
    return stack.top() if stack else None


def push_simtvf_context(ctx: SimtVFContext):
    _get_simtvf_stack().push(ctx)


def pop_simtvf_context():
    _get_simtvf_stack().pop()


def _normalize_bindings(bindings: list[Var]) -> Var | list[Var]:
    """
    Return a bare Var when we only have a single binding so that users may write either
    `with T.Kernel(...) as pid:` or `with T.Kernel(...) as (pid,)`.
    Otherwise, keep the list semantics for multi-dimensional launches.
    """
    if len(bindings) == 1:
        return bindings[0]
    return bindings


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __enter__(self) -> Var | list[Var]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns one Var if we detect exactly 5 frames (meaning there is a single
        block dimension), or a list of Vars otherwise.
        """
        super().__enter__()
        _get_current_stack().push(self)

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame, BlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)
        maybe_npu = last_block_frame.annotations.get("tilelang.is_npu_kernel_frame", False)

        if maybe_cpu:
            return _normalize_bindings([frame.vars[0] for frame in self.frames[0:-1]])
        elif maybe_npu:
            return _normalize_bindings([frame.iter_var.var for frame in self.frames[0:-1]])
        else:
            return _normalize_bindings([frame.iter_var.var for frame in self.frames[0:-4]])

    def __exit__(self, ptype, value, trace):
        """
        Exits the KernelLaunchFrame scope and pops this frame from the stack,
        but only if it's indeed the topmost frame.
        """
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> KernelLaunchFrame | None:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def get_block_extent(self, dim: int) -> int:
        """
        Returns the block extent for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        iter_var = self.frames[dim].iter_var
        return int(iter_var.dom.extent)

    def get_block_extents(self) -> list[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(3)]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        iter_var = self.frames[-4 + dim].iter_var
        return int(iter_var.dom.extent)

    def get_thread_extents(self) -> list[int]:
        """
        Returns the thread extents for all three dimensions.
        """
        return [self.get_thread_extent(dim) for dim in range(3)]

    def get_thread_binding(self, dim: int = 0) -> Var:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return self.frames[-4 + dim].iter_var.var

    def get_thread_bindings(self) -> list[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return [frame.iter_var.var for frame in self.frames[-4:-1]]

    def get_num_threads(self) -> int:
        """
        Returns the thread indices from the topmost frame.
        """
        num_threads: int = 1
        for thread_dim in range(3):
            num_threads *= self.get_thread_extent(thread_dim)
        return num_threads

    def get_block_binding(self, dim: int = 0) -> Var:
        """
        Returns the block binding for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        return self.frames[dim].iter_var.var

    def get_block_bindings(self) -> list[Var]:
        """
        Returns all three block bindings.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def blocks(self) -> list[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def threads(self) -> list[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[-4:]]

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


def Kernel(
    *blocks: int | tir.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    blocks : int
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    is_cpu : bool
        Whether the kernel is running on CPU.
        Thus we will not bind threadIdx.x, threadIdx.y, threadIdx.z.
        and blockIdx.x, blockIdx.y, blockIdx.z.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.

    Examples
    --------
    Create a 1-D CUDA kernel launch and unpack the single block index:

    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            # bx is the blockIdx.x binding (also iterable as (bx,))
            ...

    Launch a 2-D grid while requesting two thread dimensions:

    .. code-block:: python

        with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
            tx, ty = T.get_thread_bindings()
            ...

    Emit a CPU kernel where thread bindings are skipped:

    .. code-block:: python

        with T.Kernel(loop_extent, is_cpu=True) as (i,):
            ...
    """
    # In eager mode, we construct AST directly without prim_func,
    # so there must be a Builder available. If not, this function
    # is being called outside of a JIT/prim_func context.
    # lazy import to avoid circular import
    from tilelang.language.eager.builder import Builder

    if Builder.current() is None:
        raise JITNoBuilderError("T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")

    attrs: dict = {}
    current_target = Target.current(allow_none=True)
    is_ascend_target = bool(current_target is not None and (current_target.kind.name == "ascend" or "ascend" in current_target.keys))

    if is_ascend_target and not is_cpu:
        if threads is not None:
            raise ValueError(
                "Ascend backend does not support `threads=` in `T.Kernel(...)`. Use `T.SimtVF(threads=...)` to define thread domains."
            )
        if len(blocks) != 1:
            raise ValueError(f"Ascend backend only supports 1-D grid in `T.Kernel(N)`. Got {len(blocks)}-D grid.")
        attrs["tilelang.is_npu_kernel_frame"] = True
    elif not is_cpu and threads is None:
        # Keep backward compatibility when target is not Ascend (or unknown).
        threads = 128  # default thread number

    if threads is None:
        normalized_threads = None
    elif isinstance(threads, int):
        normalized_threads = [threads, 1, 1]
    elif isinstance(threads, list):
        normalized_threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        normalized_threads = list(threads) + [1] * (3 - len(threads))
    else:
        assert is_cpu, "threads must be an integer or a list of integers"
        normalized_threads = None

    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, normalized_threads, attrs)


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_vars[dim]
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_binding(dim)


def get_thread_bindings() -> list[Var]:
    """Returns all three thread bindings."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_vars)
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_bindings()


def get_block_binding(dim: int = 0) -> Var:
    """Returns the block binding for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_binding(dim)


def get_block_bindings() -> list[Var]:
    """Returns all three block bindings."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_bindings()


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return simtvf.thread_extents[dim]
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extent(dim)


def get_thread_extents() -> list[int]:
    """Returns all three thread extents."""
    simtvf = _get_current_simtvf()
    if simtvf is not None:
        return list(simtvf.thread_extents)
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extents()


def get_block_extent(dim: int = 0) -> int:
    """Returns the block extent for the given dimension."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extent(dim)


def get_block_extents() -> list[int]:
    """Returns all three block extents."""
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extents()
