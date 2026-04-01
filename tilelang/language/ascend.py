"""Ascend-specific language primitives for TileLang."""

from __future__ import annotations

from tvm import tir


def ascend_pipe_barrier(pipe_t: str) -> None:
    """Insert an AscendC pipeline barrier.

    Generates: AscendC::PipeBarrier<pipe_t::XXX>();

    Parameters
    ----------
    pipe_t : str
        Pipeline type string, e.g., "PIPE_ALL", "PIPE_V", "PIPE_M",
        "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_S".

    Example
    -------
    >>> T.ascend_pipe_barrier("PIPE_ALL")
    """
    if not isinstance(pipe_t, str):
        raise TypeError(f"pipe_t must be a string, got {type(pipe_t)}")
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_pipe_barrier"), pipe_t)


def ascend_set_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC SetFlag for pipeline synchronization.

    Generates: AscendC::SetFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_set_flag("S_MTE3", event_id)
    """
    if not isinstance(hard_event, str):
        raise TypeError(f"hard_event must be a string, got {type(hard_event)}")
    if isinstance(event_id, int):
        event_id = tir.IntImm("int32", event_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_set_flag"), hard_event, event_id)


def ascend_wait_flag(hard_event: str, event_id) -> None:
    """Insert an AscendC WaitFlag for pipeline synchronization.

    Generates: AscendC::WaitFlag<AscendC::HardEvent::XXX>(eventID);

    Parameters
    ----------
    hard_event : str
        HardEvent enum name, e.g., "MTE2_MTE1", "S_MTE3", "V_MTE2".
    event_id : int or PrimExpr
        Event ID for synchronization.

    Example
    -------
    >>> T.ascend_wait_flag("S_MTE3", event_id)
    """
    if not isinstance(hard_event, str):
        raise TypeError(f"hard_event must be a string, got {type(hard_event)}")
    if isinstance(event_id, int):
        event_id = tir.IntImm("int32", event_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_wait_flag"), hard_event, event_id)


def ascend_threadfence() -> None:
    """Insert an AscendC thread fence.

    Generates: asc_threadfence();

    Example
    -------
    >>> T.ascend_threadfence()
    """
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_threadfence"))


def ascend_cross_core_set_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreSetFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreSetFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
    """
    if not isinstance(pipe, str):
        raise TypeError(f"pipe must be a string, got {type(pipe)}")
    if isinstance(mode_id, int):
        mode_id = tir.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tir.IntImm("int32", flag_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_cross_core_set_flag"), mode_id, pipe, flag_id)


def ascend_cross_core_wait_flag(mode_id: int, pipe: str, flag_id) -> None:
    """Insert an AscendC CrossCoreWaitFlag for cross-core synchronization.

    Generates: AscendC::CrossCoreWaitFlag<modeId, pipe>(flagId);

    Parameters
    ----------
    mode_id : int
        Cross-core sync mode: 0, 1, 2, or 4.
    pipe : str
        Pipeline type, e.g., "PIPE_MTE3", "PIPE_FIX".
    flag_id : int or PrimExpr
        Cross-core synchronization flag ID.

    Example
    -------
    >>> T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
    """
    if not isinstance(pipe, str):
        raise TypeError(f"pipe must be a string, got {type(pipe)}")
    if isinstance(mode_id, int):
        mode_id = tir.IntImm("int32", mode_id)
    if isinstance(flag_id, int):
        flag_id = tir.IntImm("int32", flag_id)
    return tir.call_intrin("void", tir.op.Op.get("tl.ascend_cross_core_wait_flag"), mode_id, pipe, flag_id)
