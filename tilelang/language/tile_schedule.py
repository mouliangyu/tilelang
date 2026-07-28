"""Tile schedulers built on TileLang ``meta_class``.

``@meta_class`` auto-inlines every method that emits a buffer store, so state
methods can freely read scalar state via ``self._x[0]`` and write via
``self._x[0] = ...`` (the latter lowers to a ``BufferStore``). Store-free
methods (``valid``, ``coord``) and the read-only state properties (``m_idx``,
``n_idx``, ``current_iter``) stay plain Python and just return ``PrimExpr``
values, so they work in conditions and can be reused statelessly. State lives
in ``T.alloc_var`` buffers allocated in ``__init__`` (only when ``stateful``).
Works in both eager (``@tilelang.jit``) and lazy (``@T.prim_func``) modes.

Note on compile-time branching: the lazy TVMScript parser does not constant-fold
a Python ``if`` on a compile-time value inside an inlined method (it would emit a
dead TIR branch). So all compile-time decisions (traversal order, clustering)
are made in ``__init__`` (plain Python); the inlined hot methods contain only
uniform runtime arithmetic. The store-free ``coord`` may use a compile-time
``if`` because, being plain Python, it is evaluated (not traced) per call.
"""

from __future__ import annotations

import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.language.meta import meta_class


@meta_class
class BaseTileScheduler:
    """Common state and tile-traversal skeleton.

    State (single-element ``T.alloc_var`` buffers): ``m_idx`` / ``n_idx`` are the
    current tile coordinates; ``linear_idx`` is the worker's global linear cursor;
    ``current_iter`` is the 0-based iteration count (how many tiles this worker has
    advanced past = the "wave" index). Each is exposed as a read-only property
    returning the scalar ``PrimExpr`` (``sched.m_idx``, ``sched.current_iter``);
    only scheduler methods mutate the underlying buffers (``self._m_idx[0] = ...``).
    ``current_iter`` is the single iteration clock the kernel can read for
    pipeline/double-buffer state (e.g. ``sched.current_iter & 1``), removing the
    need for a separate ``for w`` loop counter. Subclasses implement
    ``update_current_idx`` to decode ``linear_idx`` into ``(m_idx, n_idx)`` and
    set ``self._total_tiles`` (used by ``valid``).
    """

    def __init__(self, stateful: bool = True, name: str | None = None):
        # ``name`` is not read here: the ``meta_class`` __init__ wrapper binds
        # it from the signature to auto-name the state buffers below.
        self._stateful = stateful
        if stateful:
            self._m_idx = T.alloc_var(T.int32)
            self._n_idx = T.alloc_var(T.int32)
            self._linear_idx = T.alloc_var(T.int32)
            self._current_iter = T.alloc_var(T.int32)
        self._total_tiles = 0

    @property
    def m_idx(self):
        """Current tile row as a scalar ``PrimExpr``."""
        return self._m_idx[0]

    @property
    def n_idx(self):
        """Current tile column as a scalar ``PrimExpr``."""
        return self._n_idx[0]

    @property
    def linear_idx(self):
        """This worker's global linear tile cursor as a scalar ``PrimExpr``."""
        return self._linear_idx[0]

    @property
    def current_iter(self):
        """0-based iteration count (wave index) as a scalar ``PrimExpr``."""
        return self._current_iter[0]

    def update_current_idx(self, linear_idx):
        raise NotImplementedError("Subclasses must implement update_current_idx")

    def init(self, linear_init):
        self._current_iter[0] = 0
        self._linear_idx[0] = linear_init
        self.update_current_idx(self._linear_idx[0])

    def next_tile(self, step):
        self._current_iter[0] = self._current_iter[0] + 1
        self._linear_idx[0] = self._linear_idx[0] + step
        self.update_current_idx(self._linear_idx[0])

    def valid(self):
        return self._linear_idx[0] < self._total_tiles


@meta_class
class PersistentTileScheduler(BaseTileScheduler):
    """Persistent, grid-strided tile scheduler with L2 swizzle and clustering.

    The 2D tile grid (``num_m_tiles`` x ``num_n_tiles``) is flattened into a
    linear order and distributed across ``num_workers`` persistent workers:
    worker ``w`` processes linear indices ``w, w + num_workers, ...`` until they
    run past the total. The linear-to-2D decode is controlled by three
    independent, interacting factors:

    1. ``column_major`` -- traversal order. With ``swizzle_size == 1`` this is
       pure column-major (``m`` varies fastest) when ``True``, or pure row-major
       (``n`` varies fastest) when ``False``.
    2. ``swizzle_size`` -- L2-locality panel width. The "fast" axis is widened
       into panels of ``swizzle_size`` tiles along the "slow" axis: a full strip
       of the fast axis is swept for each group of ``swizzle_size`` slow-axis
       tiles before advancing. ``swizzle_size == 1`` disables swizzling. This is
       the CUTLASS-style threadblock swizzle used by the SM100 persistent GEMM
       examples. Non-divisible tail panels are handled (narrower last panel).
    3. ``cluster_size`` -- block clustering along M (x) only. The M tiles are
       grouped into clusters of ``cluster_size`` rows, and the scheduler runs at
       *cluster* granularity: it produces the cluster-row in ``m_idx`` over a
       grid of ``ceildiv(num_m_tiles, cluster_size)`` cluster-rows x
       ``num_n_tiles`` columns. The caller turns the cluster-row into a real
       block row by adding the in-cluster rank::

           bx = sched.m_idx * cluster_size + cta_rank_in_cluster

       (For ``cluster_size == 1`` this reduces to ``bx = sched.m_idx``.)

    Interaction: clustering reshapes the grid to ``M' x N'`` (with
    ``M' = ceildiv(num_m_tiles, cluster_size)``) *first*; the swizzle and
    traversal order then operate on that reshaped grid. ``num_workers`` is the
    number of resident workers (= clusters when ``cluster_size > 1``).

    Parameters
    ----------
    num_m_tiles : int | PrimExpr
        Number of tiles along M (``ceildiv(M, block_M)``).
    num_n_tiles : int | PrimExpr
        Number of tiles along N (``ceildiv(N, block_N)``).
    num_workers : int | PrimExpr, optional
        Persistent stride = number of resident workers/clusters. Defaults to
        ``driver.get_num_sms() // cluster_size`` (one block per SM, grouped into
        clusters).
    swizzle_size : int
        L2 swizzle panel width (default ``1`` = no swizzle).
    column_major : bool
        Traversal order (default ``True`` = column-major / M fastest).
    cluster_size : int
        Block-cluster size along M only (default ``1`` = no clustering).
    stateful : bool
        If ``True`` (default), allocate the ``m_idx`` / ``n_idx`` / ``linear_idx``
        / ``current_iter`` state buffers and enable ``init`` / ``next_tile`` /
        ``valid`` for a ``while`` persistent loop. If ``False``, allocate no state
        and expose only the pure ``coord(tile_id) -> (m, n)`` decode (for
        ``for w in range(waves)`` loops and auto warp-specialization, where the
        loop owns the iteration clock and the WS pass owns the pipeline phase).
    name : str, optional
        Optional name prefix for the state buffers in the generated IR
        (``{name}_m_idx`` / ``{name}_n_idx`` / ...). Default ``None`` leaves
        the buffers with generic auto-generated names; pass a name to make the
        IR easier to read when a kernel holds several scheduler instances.

    Examples
    --------
    Plain persistent GEMM (one block per SM)::

        m_blocks, n_blocks = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
        with T.Kernel(driver.get_num_sms(), threads=threads) as (block_id,):
            sched = T.PersistentTileScheduler(m_blocks, n_blocks)
            sched.init(block_id)
            while sched.valid():
                bx, by = sched.m_idx, sched.n_idx
                # ... compute tile (bx, by) ...
                sched.next_tile()

    With L2 swizzle (panel width 8) and named state buffers in the IR::

        sched = T.PersistentTileScheduler(
            m_blocks, n_blocks, swizzle_size=8, name="sched")

    With a 2-CTA cluster along M (e.g. SM100 2-SM MMA)::

        sm_num = driver.get_num_sms()
        cluster_size = 2
        with T.ClusterKernel(sm_num, threads=256, cluster_dims=cluster_size) as (block_id):
            cta_rank = T.block_rank_in_cluster()
            sched = T.PersistentTileScheduler(
                m_blocks, n_blocks,
                swizzle_size=8, cluster_size=cluster_size)
            sched.init(block_id // cluster_size)   # init with cluster id
            while sched.valid():
                bx = sched.m_idx * cluster_size + cta_rank
                by = sched.n_idx
                # ... compute tile (bx, by) ...
                sched.next_tile()

    Stateless form (``stateful=False``) for ``for w`` loops / auto-WS, where the
    scheduler is only a tile-coordinate decoder::

        sched = T.PersistentTileScheduler(
            m_blocks, n_blocks, swizzle_size=8, stateful=False)
        for w in range(waves):
            bx, by = sched.coord(num_workers * w + worker_id)
            if bx * block_M < M and by * block_N < N:
                # ... pipelined inner loop (T.Pipelined / T.copy / T.gemm) ...
                pass

    Manual warp-specialized kernels use ``current_iter`` as the single iteration clock
    (no separate ``for w`` loop): each warp role runs its own
    ``while sched.valid()`` loop and reads ``sched.current_iter`` for
    pipeline/double-buffer state (``sched.current_iter & 1`` etc.) while reading
    ``sched.m_idx`` / ``sched.n_idx`` for the tile::

        sched.init(block_id)
        while sched.valid():
            bx, by = sched.m_idx, sched.n_idx
            # ... use sched.current_iter for barrier phase / double-buffering ...
            sched.next_tile()

    Notes
    -----
    State is held in single-element ``T.alloc_var`` buffers behind read-only
    properties: ``sched.m_idx`` / ``sched.current_iter`` etc. return the scalar
    ``PrimExpr`` directly (no ``[0]``). Only scheduler methods mutate state, via
    ``self._x[0] = ...`` on the underlying buffers; the ``[0]`` is required
    there for the write to lower to a ``BufferStore``."""

    def __init__(
        self,
        num_m_tiles,
        num_n_tiles,
        num_workers=None,
        swizzle_size: int = 1,
        column_major: bool = True,
        cluster_size: int = 1,
        stateful: bool = True,
        name: str | None = None,
    ):
        super().__init__(stateful=stateful, name=name)
        self.cluster_size = cluster_size
        if num_workers is None:
            num_workers = driver.get_num_sms() // cluster_size
        self.num_workers = num_workers
        self.swizzle_size = swizzle_size
        self._column_major = column_major

        # Cluster reshapes the grid along M first: M' x N'.
        m_clusters = T.ceildiv(num_m_tiles, cluster_size)
        self._total_tiles = m_clusters * num_n_tiles

        # "primary" axis is swept fully within a swizzle panel; "secondary" axis
        # is chunked into panels of width ``swizzle_size``.
        if column_major:  # M fastest
            self._primary = m_clusters
            self._secondary = num_n_tiles
        else:  # N fastest
            self._primary = num_n_tiles
            self._secondary = m_clusters

    def coord(self, tile_id):
        """Decode a linear ``tile_id`` into ``(m_idx, n_idx)`` (pure PrimExpr).

        Stateless: builds expressions only, touches no state buffer, so it is
        usable on a ``stateful=False`` instance and can be called independently
        from any loop / warp role. ``m_idx`` is the cluster row when
        ``cluster_size > 1`` (caller adds the in-cluster rank).
        """
        primary = self._primary
        secondary = self._secondary
        swizzle = self.swizzle_size

        group = tile_id // (primary * swizzle)
        base = group * swizzle
        # Panel width along the slow axis; narrower for a non-divisible tail, and
        # clamped to >= 1 so decoding an out-of-range tile_id never divides by 0.
        width = T.max(1, T.min(swizzle, secondary - base))
        in_group = tile_id - group * (primary * swizzle)
        fast = in_group // width
        slow = base + in_group % width
        if self._column_major:
            return fast, slow  # (m, n)
        return slow, fast  # (m, n)

    def update_current_idx(self, linear_idx):
        m, n = self.coord(linear_idx)
        self._m_idx[0] = m
        self._n_idx[0] = n

    def init(self, worker_id):
        self._current_iter[0] = 0
        self._linear_idx[0] = worker_id
        self.update_current_idx(self._linear_idx[0])

    def next_tile(self):
        self._current_iter[0] = self._current_iter[0] + 1
        self._linear_idx[0] = self._linear_idx[0] + self.num_workers
        self.update_current_idx(self._linear_idx[0])


@meta_class
class AscendBaseTileScheduler(BaseTileScheduler):
    """Base for persistent GEMM block schedulers (GROUP_M swizzle + N-first snake).

    Holds everything shared across GEMM variants: the grid geometry, the stateless
    ``_swizzle`` / ``coord`` decode, the persistent-block-index advance, and the
    ``init`` / ``valid`` / ``next_block`` loop protocol. Concrete variants subclass
    this and implement ``_step`` (advance one block + set ``m_idx`` / ``n_idx`` /
    ``valid_flag``); grouped variants add their own state and ``_init_state``.

    Each persistent core independently walks the global ``(m_block, n_block)`` grid.
    The block index handed out on iteration ``current_iter`` is
    ``current_iter * num_cores + core_id``, mapped to a tile by ``_swizzle``. Usage::

        sched = T.AscendTileScheduler(block_m=BM, block_n=BN,
                                num_cores=NUM_CORES, shape_m=M, shape_n=N)
        sched.init(bx)
        while sched.valid():
            m_tile, n_tile = sched.m_idx[0], sched.n_idx[0]
            # ... compute tile (m_tile, n_tile) ...
            sched.next_block()

    Parameters
    ----------
    block_m, block_n : int
        Tile sizes along M and N.
    num_cores : int | PrimExpr
        Persistent stride = number of resident cores (grid dim).
    shape_m, shape_n : int | PrimExpr
        Problem M and N.
    group_m : int
        Swizzle panel height along M (4 or 8).
    xor_n : bool
        XOR the in-panel n-index with the m-index (default True).
    snake : bool
        Reverse the N-group order on odd superrows (default True).
    stateful : bool
        If True (default) allocate state and enable ``init`` / ``valid`` /
        ``next_block``. If False expose only the store-free ``coord`` decode.
    name : str, optional
        Optional state-buffer name prefix. The default ``None`` uses generic
        auto-generated names.

    Notes
    -----
    State lives in single-element ``T.alloc_var`` buffers -- read with
    ``sched.m_idx[0]`` and (inside methods) write with ``self.x[0] = ...``.
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        super().__init__(stateful=stateful, name=name)
        assert group_m in (4, 8), "group_m must be 4 or 8"

        self._block_m = block_m
        self._block_n = block_n
        self._num_cores = num_cores
        self._shape_m = shape_m
        self._shape_n = shape_n
        self._group_m = group_m
        self._group_n = 32 // group_m
        self._xor_n = xor_n
        self._snake = snake

        # Compile-time grid geometry (Normal / Batched share the full grid).
        self._num_m_blocks = T.ceildiv(shape_m, block_m)
        self._num_n_blocks = T.ceildiv(shape_n, block_n)
        self._num_blocks = self._num_m_blocks * self._num_n_blocks

        # ``coord`` (stateless swizzle) is available regardless of statefulness.
        self._total_tiles = self._num_blocks

        if not stateful:
            return

        # ``core_id`` (set by init) and ``valid_flag`` (data-dependent validity).
        self.core_id = T.alloc_var(T.int32)
        self.valid_flag = T.alloc_var(T.int32)
        self._alloc_state()

    # ---- hooks for subclasses ---------------------------------------------

    def _alloc_state(self):
        """Allocate variant-specific cursor buffers (default: none)."""

    def _init_state(self):
        """Initialize variant-specific cursors in ``init`` (default: nothing)."""

    def _step(self):
        """Advance one persistent block: set ``m_idx`` / ``n_idx`` / ``valid_flag``."""
        raise NotImplementedError

    # ---- stateless swizzle -------------------------------------------------

    def _swizzle(self, block_idx, local_num_m_blocks):
        """Map a (group-local) ``block_idx`` to ``(m_block, n_block)`` (store-free).

        Faithful port of DeepGEMM ``get_swizzled_block``: small-M row-major fast
        path, tail row-major fallback for the last ``< group_m`` M-rows, and the
        GROUP_M superrow swizzle (+ optional xor_n / snake) on the aligned region.
        ``local_num_m_blocks`` is compile-time for Normal/Batched (the whole grid)
        and a runtime PrimExpr for the grouped variants (the current group).
        Runtime conditions use ``T.if_then_else``; compile-time toggles use ``if``.
        """
        group_m = self._group_m
        group_n = self._group_n
        num_n = self._num_n_blocks
        n_groups = num_n // group_n
        blocks_per_superrow = group_m * num_n
        full_blocks = n_groups * 32

        # small-M row-major
        sm_m = block_idx // num_n
        sm_n = block_idx % num_n

        # tail boundaries (compile-time for int local, runtime otherwise)
        if isinstance(local_num_m_blocks, int):
            small_m = local_num_m_blocks < group_m
            tail_m_start = local_num_m_blocks - local_num_m_blocks % group_m if not small_m else 0
        else:
            small_m = None
            tail_m_start = local_num_m_blocks - local_num_m_blocks % group_m
        tail_start = tail_m_start * num_n

        # tail row-major
        tail = block_idx - tail_start
        tl_m = tail_m_start + tail // num_n
        tl_n = tail % num_n

        # aligned swizzle region
        superrow = block_idx // blocks_per_superrow
        within = block_idx - superrow * blocks_per_superrow
        # full panel
        grp = within // 32
        local = within - grp * 32
        m_local = local % group_m
        n_local = local // group_m
        if self._xor_n:
            n_local = n_local ^ m_local
        ng = grp
        if self._snake:
            ng = T.if_then_else(superrow % 2 == 1, n_groups - 1 - grp, grp)
        full_m = superrow * group_m + m_local
        full_n = ng * group_n + n_local
        # remainder columns (n not covered by full 32-blocks)
        rem = within - full_blocks
        rem_m = superrow * group_m + rem % group_m
        rem_n = n_groups * group_n + rem // group_m
        sw_m = T.if_then_else(within < full_blocks, full_m, rem_m)
        sw_n = T.if_then_else(within < full_blocks, full_n, rem_n)

        # tail-vs-swizzle
        al_m = T.if_then_else(block_idx >= tail_start, tl_m, sw_m)
        al_n = T.if_then_else(block_idx >= tail_start, tl_n, sw_n)

        if small_m is True:
            return sm_m, sm_n
        if small_m is False:
            return al_m, al_n
        # runtime local_num_m_blocks (grouped)
        m = T.if_then_else(local_num_m_blocks < group_m, sm_m, al_m)
        n = T.if_then_else(local_num_m_blocks < group_m, sm_n, al_n)
        return m, n

    def coord(self, block_idx):
        """Decode a linear ``block_idx`` into ``(m_block, n_block)`` (Normal grid)."""
        return self._swizzle(block_idx, self._num_m_blocks)

    def update_current_idx(self, linear_idx):
        m, n = self.coord(linear_idx)
        self._m_idx[0] = m
        self._n_idx[0] = n

    # ---- stateful walk -----------------------------------------------------

    def valid(self):
        return self.valid_flag[0] != 0

    def init(self, core_id):
        self._current_iter[0] = -1
        self.core_id[0] = core_id
        self._init_state()
        self.next_block()

    def next_block(self):
        self._step()

    def _advance_block_idx(self):
        """Bump the persistent iteration and return the new global block index."""
        self._current_iter[0] = self._current_iter[0] + 1
        self._linear_idx[0] = self._current_iter[0] * self._num_cores + self.core_id[0]
        return self._linear_idx[0]

    def get_actual_m(self, m_block_idx):
        return T.min(self._shape_m - m_block_idx * self._block_m, self._block_m)

    def get_actual_n(self, n_block_idx):
        return T.min(self._shape_n - n_block_idx * self._block_n, self._block_n)


@meta_class
class AscendTileScheduler(AscendBaseTileScheduler):
    """Single flat M x N grid, GROUP_M swizzle + N-first snake; the last
    ``< group_m`` M-rows fall back to row-major. This is the plain (non-grouped,
    non-batched) GEMM scheduler."""

    def _step(self):
        block_idx = self._advance_block_idx()
        m, n = self._swizzle(block_idx, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n
        self.valid_flag[0] = T.if_then_else(block_idx < self._num_blocks, 1, 0)


@meta_class
class AscendBatchedTileScheduler(AscendBaseTileScheduler):
    """Standard batched matmul: the Normal grid replicated ``num_groups`` (= batch)
    times. ``get_batch_idx()`` exposes the batch of the current block.

    Extra parameter ``num_groups`` (= batch count) beyond the base scheduler.
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        num_groups: int,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._num_groups = num_groups
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.batch_idx = T.alloc_var(T.int32)

    def _step(self):
        block_idx = self._advance_block_idx()
        batch = block_idx // self._num_blocks
        self.batch_idx[0] = batch
        in_batch = block_idx - batch * self._num_blocks
        m, n = self._swizzle(in_batch, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n
        self.valid_flag[0] = T.if_then_else(batch < self._num_groups, 1, 0)

    def get_batch_idx(self):
        return self.batch_idx[0]


@meta_class
class AscendMGroupedTileScheduler(AscendBaseTileScheduler):
    """MoE m-grouped "contiguous" layout. ``grouped_layout`` is the prefix-sum-of-rows
    array (length ``num_groups``); group ``g``'s valid rows are
    ``[align(psum[g-1], alignment), psum[g])`` in the global M space (group 0 starts at
    0). ``get_group_idx()`` exposes the current group for per-group B/SF offsets.

    Extra parameters beyond the base: ``grouped_layout`` (GM ``int32`` prefix-sum
    buffer), ``num_groups``, ``alignment`` (group-start row alignment, default 256).
    """

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        grouped_layout,
        num_groups: int,
        alignment: int = 256,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._grouped_layout = grouped_layout
        self._num_groups = num_groups
        self._alignment = alignment
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.group_idx = T.alloc_var(T.int32)
        self.cur_num_m_blocks = T.alloc_var(T.int32)
        self.last_psum_m = T.alloc_var(T.int32)
        self.cur_psum_m = T.alloc_var(T.int32)
        self.m_block_cumsum = T.alloc_var(T.int32)

    def _init_state(self):
        self.group_idx[0] = 0
        self.last_psum_m[0] = 0
        self.m_block_cumsum[0] = 0
        self.cur_psum_m[0] = self._grouped_layout[0]
        self.cur_num_m_blocks[0] = T.ceildiv(self.cur_psum_m[0], self._block_m)

    def _step(self):
        block_idx = self._advance_block_idx()
        num_n = self._num_n_blocks
        self.valid_flag[0] = 1
        # advance groups until block_idx lands in the current group's block range
        while self.group_idx[0] < self._num_groups:
            if block_idx < (self.m_block_cumsum[0] + self.cur_num_m_blocks[0]) * num_n:
                T.loop_break()
            self.group_idx[0] = self.group_idx[0] + 1
            if self.group_idx[0] == self._num_groups:
                self.valid_flag[0] = 0
                T.loop_break()
            self.last_psum_m[0] = T.ceildiv(self.cur_psum_m[0], self._alignment) * self._alignment
            self.cur_psum_m[0] = self._grouped_layout[self.group_idx[0]]
            self.m_block_cumsum[0] = self.m_block_cumsum[0] + self.cur_num_m_blocks[0]
            self.cur_num_m_blocks[0] = T.ceildiv(self.cur_psum_m[0] - self.last_psum_m[0], self._block_m)
        in_group = block_idx - self.m_block_cumsum[0] * num_n
        m, n = self._swizzle(in_group, self.cur_num_m_blocks[0])
        self._m_idx[0] = m + self.last_psum_m[0] // self._block_m
        self._n_idx[0] = n

    def get_group_idx(self):
        return self.group_idx[0]

    # m-grouped truncates the group's last m-block via the caller's own tail
    # handling; every scheduled m-block is a full BLOCK_M.
    def get_actual_m(self, m_block_idx):
        return self._block_m


@meta_class
class AscendKGroupedTileScheduler(AscendBaseTileScheduler):
    """MoE k-grouped. All groups share one M x N grid but differ in K length;
    ``grouped_layout[i]`` is the cumulative end-K of group ``i``. ``get_group_idx()``,
    ``get_k_idx_base()`` / ``get_sf_idx_base()`` give the group and its A/B and SF K
    offsets; ``get_shape_k()`` is the current group's K length.

    Extra parameters beyond the base: ``grouped_layout`` (GM ``int32`` prefix-sum
    buffer), ``num_groups``, ``alignment`` (group-start K alignment, default 256).
    """

    # One scale-factor pair covers this many K elements (MX FP hardware constant).
    _MX_SF_DIVISOR = 64

    def __init__(
        self,
        *,
        block_m,
        block_n,
        num_cores,
        shape_m,
        shape_n,
        grouped_layout,
        num_groups: int,
        alignment: int = 256,
        group_m: int = 4,
        xor_n: bool = True,
        snake: bool = True,
        stateful: bool = True,
        name: str | None = None,
    ):
        self._grouped_layout = grouped_layout
        self._num_groups = num_groups
        self._alignment = alignment
        super().__init__(
            block_m=block_m,
            block_n=block_n,
            num_cores=num_cores,
            shape_m=shape_m,
            shape_n=shape_n,
            group_m=group_m,
            xor_n=xor_n,
            snake=snake,
            stateful=stateful,
            name=name,
        )

    def _alloc_state(self):
        self.group_idx = T.alloc_var(T.int32)
        self.cur_shape_k = T.alloc_var(T.int32)
        self.last_psum_k = T.alloc_var(T.int32)
        self.cur_psum_k = T.alloc_var(T.int32)
        self.sf_k_cumsum = T.alloc_var(T.int32)
        self.k_block_cumsum = T.alloc_var(T.int32)

    def _init_state(self):
        self.group_idx[0] = 0
        self.last_psum_k[0] = 0
        self.sf_k_cumsum[0] = 0
        self.k_block_cumsum[0] = 0
        self.cur_psum_k[0] = self._grouped_layout[0]
        self.cur_shape_k[0] = self.cur_psum_k[0]

    def _step(self):
        block_idx = self._advance_block_idx()
        self.valid_flag[0] = 1
        # all groups share the M x N grid; each spans exactly num_blocks
        while self.group_idx[0] < self._num_groups:
            if self.cur_shape_k[0] > 0 and block_idx < self.k_block_cumsum[0] + self._num_blocks:
                T.loop_break()
            self.group_idx[0] = self.group_idx[0] + 1
            if self.group_idx[0] == self._num_groups:
                self.valid_flag[0] = 0
                T.loop_break()
            self.last_psum_k[0] = T.ceildiv(self.cur_psum_k[0], self._alignment) * self._alignment
            self.cur_psum_k[0] = self._grouped_layout[self.group_idx[0]]
            self.sf_k_cumsum[0] = self.last_psum_k[0] // self._MX_SF_DIVISOR
            if self.cur_shape_k[0] > 0:
                self.k_block_cumsum[0] = self.k_block_cumsum[0] + self._num_blocks
            self.cur_shape_k[0] = self.cur_psum_k[0] - self.last_psum_k[0]
        in_group = block_idx - self.k_block_cumsum[0]
        m, n = self._swizzle(in_group, self._num_m_blocks)
        self._m_idx[0] = m
        self._n_idx[0] = n

    def get_group_idx(self):
        return self.group_idx[0]

    def get_k_idx_base(self):
        return self.last_psum_k[0]

    def get_sf_idx_base(self):
        return self.sf_k_cumsum[0]

    def get_shape_k(self):
        return self.cur_shape_k[0]
