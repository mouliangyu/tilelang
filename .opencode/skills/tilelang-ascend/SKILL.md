---
name: ascend-programming
description: >
Guide for writing TileLang programs targeting the Huawei Ascend NPU.
Covers Ascend-specific concepts (AIC/AIV dual-core, Cube/Vector engines,
memory hierarchy, thread model, synchronization) and how they differ
from CUDA/GPU programming. Use this skill whenever the user asks about
Ascend, NPU, Huawei, or writes TileLang code for the Ascend backend.
---

# TileLang Ascend NPU Programming

You are helping the user write TileLang programs for the Huawei Ascend NPU.
The Ascend architecture differs substantially from CUDA GPUs. This skill
provides the concepts, API reference, programming patterns, and common
pitfalls you need to produce correct, high-performance Ascend kernels.

---

## 1. Mental Model: Ascend vs CUDA

| Concept | CUDA | Ascend NPU |
|---|---|---|
| Kernel launch | `T.Kernel(grid, threads=128)` | `T.Kernel(N)` -- **no threads arg** |
| Thread dimensions | 1-D, 2-D, or 3-D | 1-D grids only |
| Thread access | `T.get_thread_binding()` | No `threadIdx` directly; use `T.SimtVF(threads=N)` |
| Compute cores | SM (unified) | **Two distinct cores**: AIC (Cube) + AIV (Vector) |
| Matrix multiply | `T.gemm(...)` | `T.gemm(..., transpose_B=True)` -- **mandatory** |
| Shared memory | `T.alloc_shared(...)` ("shared") | `T.alloc_shared(...)` maps to UB (Unified Buffer) |
| Global memory | Global | GM (Global Memory / HBM) |
| L1 cache / data | N/A | `T.alloc_l1(...)` -- Cube Buffer (CBuf) |
| Register fragments | wmma/MMA | `T.alloc_l0a/l0b/l0c(...)` -- Cube engine buffers |
| Warp-level ops | `__shfl_sync` | `T.alloc_reducer` / `T.ascend_reduce_add` |
| Synchronization | `__syncthreads()` | Flag-based: `T.ascend_set_flag` / `T.ascend_wait_flag` |
| Auto-Schedule | `T.Pipelined(...)` | `T.Pipelined(...)` with `num_stages=...` |
| Compiler | nvcc | Bisheng (Huawei Ascend compiler) |

---

## 2. Thread Hierarchy

### 2.1 Kernel Launch

Ascend supports **only 1-D grids** and does **not** accept a `threads=`
parameter on `T.Kernel`. Thread domains are defined separately.

```python
# CORRECT - Ascend style:
with T.Kernel(NUM_CORES) as bx:
    ...

# WRONG - will raise ValueError:
with T.Kernel(NUM_CORES, threads=128) as bx:
    ...
```

### 2.2 SIMT VF: Thread-Parallel Compute

Use `T.SimtVF(threads=N)` to define a thread-parallel region. Inside
this block, `T.Parallel` loops distribute work across threads. The
codegen automatically inserts `threadIdx.x` mappings.

```python
with T.SimtVF(threads=128):
    for i in T.Parallel(N):
        out[i] = a[i] + b[i]
```

Auto-sync: The compiler inserts `asc_syncthreads()` when it detects
cross-thread data hazards (see `example_simtvf_auto_sync.py`). Thread
binding is accessed via `T.get_thread_binding()`.

### 2.3 SIMD VF: Register-Level Vector (No Threads)

`T.SimdVF()` maps to CCE MicroAPI vector instructions (2048-bit vectors,
32 x float32 or 64 x float16). There is **no thread concept** -- the
code runs as scalar instructions on the vector unit.

```python
with T.SimdVF():
    for i in range(N // 64):
        r0 = T.simd.vld(buf[i * 64])
        r1 = T.simd.vadd(r0, r0, mask)
        T.simd.vsts(out[i * 64], r1, mask)
```

---

## 3. Core Architecture: AIC vs AIV

The Ascend NPU has two distinct compute cores on each die:

| Core | Engine | Purpose | TileLang Scope |
|---|---|---|---|
| AIC | Cube unit | Matrix multiply (GEMM, convolutions) | `T.Cube()` |
| AIV | Vector unit | Element-wise ops, reductions, data movement | `T.Vector()` |

### 3.1 Pure Cube-Only Kernel

Use `T.Kernel(N)` with no core split. All code runs on the AIC core.
Suitable for GEMM that stays entirely on the cube engine.

```python
with T.Kernel(NUM_BLOCKS) as bx:
    x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
    w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
    res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
    # ... T.copy into L1, T.gemm, T.copy out ...
```

### 3.2 MixedKernel: Cube + Vector

Use `T.MixedKernel(NUM_BLOCKS, sids=2)` to split work across AIC (Cube)
and AIV (Vector) cores. The `sid` variable identifies which AIV sub-core
(0 or 1). Use `T.Cube()` and `T.Vector()` scopes to place code on each
core respectively.

```python
with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
    # AIC side: GEMM
    with T.Cube():
        T.gemm(x_l1, w_l1, res, transpose_B=True)

    # AIV side: element-wise + data movement
    with T.Vector():
        T.dual_copy(res, temp)
        T.copy(temp[sid * HALF:...], output[...])
```

**When to use MixedKernel:**
- FlashAttention variants (Cube does QK^T and PV, Vector does softmax)
- Any GEMM with post-processing (activation, bias, etc.)
- When you need to overlap Cube compute with Vector data movement

Auto-scheduled MixedKernel: The compiler can auto-schedule sync between
AIC and AIV. Use `T.MixedKernel` with `T.Pipelined`/`T.Persistent` --
the compiler inserts `CrossCoreSetFlag`/`CrossCoreWaitFlag` automatically.

### 3.3 Manual Mixed: T.Cube() and T.Vector() with Manual Sync

The older/explicit pattern. Use when you need full control over
cross-core synchronization flags:

```python
with T.Cube():
    T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_aiv_flag)
    ...
    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_aic_flag)

with T.Vector() as sid:
    my_aic_flag = 6 + sid * 16
    my_aiv_flag = 4 + sid * 16
    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_aic_flag)
    ...
```

---

## 4. Memory Hierarchy and Allocation

Ascend memory hierarchy (top to bottom):

```
GM (HBM)                       -- global memory, large capacity
  ↕ DMA (MTE pipes)
UB (Unified Buffer, "shared")  -- T.alloc_shared(...)
  ↕ DMA
L1 (CBuf, Cube Buffer)         -- T.alloc_l1(...)
  ↕ load
L0A / L0B (Cube inputs)       -- T.alloc_l0a/l0b(...)
  → Cube MAD unit
L0C (accumulator)              -- T.alloc_l0c(...)
```

### 4.1 Allocation API

```python
# Unified Buffer (UB) -- general-purpose on-chip memory
buf = T.alloc_shared((STAGES, TILE_M, TILE_K), "bfloat16")

# Cube Buffer (L1) -- fast buffer for Cube engine input
buf = T.alloc_l1((TILE_M, TILE_K), "bfloat16")

# Cube L0 buffers -- direct Cube engine inputs/outputs
buf_a = T.alloc_l0a((STAGES, TILE_M, TILE_K_SUB), "bfloat16")
buf_b = T.alloc_l0b((STAGES, TILE_N, TILE_K_SUB), "bfloat16")
buf_c = T.alloc_l0c((TILE_M, TILE_N), "float32")

# Fragment (register-level, inside SimtVF/SIMD)
frag = T.alloc_fragment((TILE,), "float32")

# Reducer (warp-level reduction accumulator)
reducer = T.alloc_reducer((1,), "float32", op="sum", replication="all")
```

### 4.2 Buffer Versioning

The auto-schedule pass estimates operation latency to determine buffer
lifetimes and pipelining strategy. When the automatic estimation is
inaccurate (e.g. for non-standard operations or complex control flow),
use `T.annotate_buffer_versions` to manually specify version counts:

```python
buf1 = T.alloc_shared((TILE,), "float32")
buf2 = T.alloc_shared((TILE,), "float32")
T.annotate_buffer_versions({buf1: 2, buf2: 2})
```

Without manual annotation, the compiler may misjudge buffer lifetimes,
leading to suboptimal pipelining or correctness issues.

Old-style manual double buffering: explicitly allocate a leading
dimension for stages and manage with manual flags (see
`examples/ascend/example_gemm.py`).

---

## 5. GEMM Constraints

### 5.1 Mandatory transpose_B=True

The Ascend Cube MAD unit requires `transpose_B=True`. This is a hardware
constraint -- the B matrix must be in transposed layout in L1.

```python
# ALWAYS use transpose_B=True on Ascend
T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(kt == 0))
```

### 5.2 Weight Layout Convention

Because of `transpose_B=True`, weights are typically stored as
`[N, K]` in global memory (transposed relative to standard `[K, N]`):

```python
# Weight buffer shape: [N, K], not [K, N]
W: T.Buffer((N_DIM, K_DIM), dtype)
```

### 5.3 Clear Accumulator

```python
T.gemm(a, b, c, transpose_B=True, clear_accum=True)   # for first k-step
T.gemm(a, b, c, transpose_B=True, clear_accum=False)  # for subsequent
```

### 5.4 HF32 Mode (fp32 throughput tradeoff)

For fp32 GEMM, enable HF32 to trade precision for ~2x throughput:

```python
T.set_hf32_mode("nearest_even")   # or "nearest_zero"
# ... T.gemm calls ...
T.set_hf32_mode(None)              # restore full fp32
```

---

## 6. Data Movement

### 6.1 T.copy

Basic DMA copy. Ascend-specific parameters:

```python
# GM → L1
T.copy(X[row_slice, col_slice], x_l1)

# L1 → L0A/L0B (used in L0-staged GEMM)
T.copy(x_l1[f, :, k_slice], x_l0[sf, :, :])

# L0C → UB (via dual_copy for M-split) or L0C → GM
T.copy(res, C[row, col])

# Ascend-specific params:
T.copy(src, dst, l2_cache_ctrl="NOTALLOC_KEEP")   # L2 cache bypass
T.copy(src, dst, transpose=True)                   # transposed DMA
```

**L2 cache control values:** `"NORMAL_FV"` (default), `"NOTALLOC_KEEP"`,
`"NOTALLOC_PW"`, `"NOTALLOC_CLEAN"`, etc.

### 6.2 T.dual_copy (Ascend-Only)

Splits L0C data across 2 AIV sub-cores. The split direction is inferred
from the shape mismatch:

- **M-split** (`src[M,N] -> dst[M/2,N]`): AIV0 gets rows `[0,M/2)`,
  AIV1 gets rows `[M/2,M)`. Used in standard GEMM + MixedKernel.
- **N-split** (`src[M,N] -> dst[M,N/2]`): splits across columns.

```python
# L0C → UB (M-split for 2 AIVs)
T.dual_copy(res_l0c, temp_ub)

# UB → L1 with ND-to-NZ format conversion
T.dual_copy(p_ub, p_l1, nd2nz=True)

# UB → GM (M-split write)
T.dual_copy(o_ub, O[row_start:row_end, 0:D])

# With L2 cache control
T.dual_copy(temp, C[row, col], l2_cache_ctrl="NOTALLOC_PW")
```

---

## 7. Synchronization

### 7.1 Modern: Auto-Schedule (Recommended)

Use `T.Pipelined` for loop-level double buffering. Use `T.Persistent`
for outer persistent loops. The compiler auto-manages all flags. This
is enabled by default (`TL_ENABLE_AUTO_SCHEDULE: True`).

```python
for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
    m_tile, n_tile = compute_tile(tile_idx)
    for kt in T.Pipelined(K_TILES, num_stages=2):
        T.copy(X[m_slice, k_slice], x_l1)
        T.copy(W[n_slice, k_slice], w_l1)
        T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
    T.dual_copy(res, temp)
    T.dual_copy(temp, C[m_slice, n_slice])
```

Key points:
- Buffers do **not** need a leading `STAGES` dimension; the compiler
  auto-multi-buffers.
- No manual SetFlag/WaitFlag calls needed.
- Reference: `examples/ascend/example_gemm_auto.py`

**Buffer version annotation**: The auto-schedule pass estimates
operation latency to determine buffer lifetimes and pipelining strategy.
When the automatic latency estimation is inaccurate (e.g. for
non-standard operations or complex control flow), use
`T.annotate_buffer_versions` to manually specify buffer version counts:

```python
buf1 = T.alloc_shared((TILE,), "float32")
buf2 = T.alloc_shared((TILE,), "float32")
T.annotate_buffer_versions({buf1: 2, buf2: 2})
```

Without this hint, the compiler may misjudge buffer lifetimes, leading
to suboptimal pipelining or correctness issues.

### 7.2 Legacy: Manual Flag Pipelining

Used in older examples. Requires explicit flags coordinated between
DMA engines (MTE1, MTE2, MTE3) and compute engines (V, M):

```python
# Pipeline events (hardware signal names):
#   MTE1_MTE2, MTE2_MTE1  -- Cube Engine ↔ MTE2 DMA
#   MTE2_V, V_MTE2         -- Vector Engine ↔ MTE2 DMA
#   MTE3_V, V_MTE3         -- Vector Engine ↔ MTE3 DMA
#   FIX_M, M_FIX           -- Cube Engine fixed-point sync

for f in range(NUM_STAGES):
    T.ascend_set_flag("MTE1_MTE2", f)   # signal "ready to produce"

for kt in range(K_TILES):
    f = kt % NUM_STAGES
    T.ascend_wait_flag("MTE1_MTE2", f)  # wait for DMA done
    T.copy(X[...], x_l1[f, :, :])
    T.ascend_set_flag("MTE2_MTE1", f)   # signal "data ready"

    T.ascend_wait_flag("MTE2_MTE1", f)  # wait for data
    T.gemm(x_l1[f], w_l1[f], res, transpose_B=True)
    T.ascend_set_flag("MTE1_MTE2", f)   # signal "buffer free"
```

Reference: `examples/ascend/example_gemm.py`

### 7.3 Cross-Core Synchronization

Used in MixedKernel with manual sync:

```python
T.ascend_cross_core_set_flag(4, "PIPE_MTE3", flag_id)
T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", flag_id)
T.ascend_pipe_barrier("PIPE_V")
```

### 7.4 Pipe Barrier

```python
T.ascend_pipe_barrier("PIPE_ALL")   # all pipes
T.ascend_pipe_barrier("PIPE_V")     # vector pipe
T.ascend_pipe_barrier("PIPE_MTE1")  # MTE1 pipe
```

---

## 8. SIMD MicroAPI (T.simd.*)

Low-level CCE vector intrinsics for `T.SimdVF()` blocks. These map
directly to Ascend CCE MicroAPI instructions (2048-bit vectors).

For detailed instruction semantics (lane widths, rounding modes, saturation
behavior, mask predicates), refer to the PTO Micro-Instruction Spec:
https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-micro-Instruction-SPEC.md

### 8.1 Core Operations

```python
with T.SimdVF():
    mask = T.simd.pset(32)            # set all 32 lanes active

    # Load/Store
    r = T.simd.vld(buf[offset])       # 2048-bit vector load
    T.simd.vsts(buf[offset], r, mask) # 2048-bit vector store

    # Arithmetic
    r = T.simd.vadd(a, b, mask)
    r = T.simd.vmul(a, b, mask)
    r = T.simd.vmax(a, b, mask)
    r = T.simd.vsub(a, b, mask)

    # Math
    r = T.simd.vexp(a, mask)
    r = T.simd.vln(a, mask)

    # Type conversion
    r = T.simd.vcvt(a, "float32")     # cast to float32
    r = T.simd.vcvt(a, "bfloat16")    # cast to bfloat16

    # Interleave/Deinterleave
    a0, a1 = T.simd.vintlv(x, y)      # interleave even/odd lanes
    x, y = T.simd.vdintlv(a0, a1)     # deinterleave back
```

Full list of available SIMD intrinsics: `vld`, `vsts`, `vsstb`, `vadd`, `vsub`,
`vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vexp`, `vln`, `vsqrt`,
`vrsqrt`, `vabs`, `vneg`, `vrelu`, `vrec`, `vdup`, `vdupv`, `vcadd`,
`vcmax`, `vcgadd`, `vintlv`, `vdintlv`, `vpack`, `vgatherb`, `vcvt`,
`vsel`, `vmuls`, `vadds`, `pset`, `mem_bar`.

### 8.2 SIMD Lower vs High-Level SIMD

Two levels of SIMD programming:

- **SIMD Lower (Recommended)**: Use raw `T.simd.*` intrinsics with
  manual loop unrolling (`for i in range(TILE // 64)`). This is the
  reliable approach -- you have full control over which instructions
  are emitted. See `examples/ascend/example_simdvf_vecadd_lower.py`
  and `examples/ascend/example_simdvf_vecadd_lower_auto.py`.

- **High-level SIMD**: Use `T.Parallel` inside `T.SimdVF()`, letting the
  compiler auto-lower to vector operations. See
  `examples/ascend/example_simdvf_vecadd.py`.

  **Warning: High-level SIMD lowering is not fully reliable.** The
  `AscendSimdVFLowerParallel` pass may fail on non-trivial patterns
  beyond simple element-wise operations. When in doubt, use SIMD Lower
  with explicit `T.simd.*` calls.

### 8.3 VF Latency Annotation

Both `T.SimtVF` and `T.SimdVF` accept a `latency=` parameter (in cycles)
that feeds into the auto-schedule pass. The Z3 pipeline scheduler uses
this value to determine buffer lifetimes and orchestrate DMA/compute
overlap. Inaccurate latency estimates can lead to suboptimal pipelining
or correctness issues.

**Automatic measurement** with `measure_vf_latency.py`:

```bash
# Measure all VF blocks in a kernel and write latency= back to source
python measure_vf_latency.py examples/ascend/example_simdvf_vecadd.py

# Preview without modifying the file
python measure_vf_latency.py examples/ascend/example_simdvf_vecadd.py --dry-run
```

This tool compiles the kernel, extracts generated VF functions from
the `.asc` output, runs each in isolation under `cannsim` (Ascend
cycle-accurate simulator), and injects the measured cycle count as
`latency=<N>` back into the Python source file.

**Manual annotation**: You can also directly set `latency=` on the
scope if you already know the cycle count:

```python
with T.SimdVF(latency=128):
    ...

with T.SimtVF(threads=256, latency=512):
    ...
```

The auto-schedule pass will use the manually provided value instead of
estimating, bypassing the potentially inaccurate built-in estimator.

---

## 9. Programming Patterns (with Example References)

### Pattern 1: Simple GEMM with Auto-Schedule

The recommended modern pattern for basic GEMM.

```python
@T.prim_func
def main(X: T.Buffer((M, K), dtype),
         W: T.Buffer((N, K), dtype),
         C: T.Buffer((M, N), accum_dtype)):
    with T.Kernel(NUM_BLOCKS) as bx:
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
        res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
        temp = T.alloc_shared((TILE_M // 2, TILE_N), accum_dtype)

        for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = compute_tile(tile_idx)
            for kt in T.Pipelined(K_TILES, num_stages=2):
                T.copy(X[m_slice, k_slice], x_l1)
                T.copy(W[n_slice, k_slice], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
            T.dual_copy(res, temp)
            T.dual_copy(temp, C[m_slice, n_slice])
```

Reference: `examples/ascend/example_gemm_auto.py`

### Pattern 2: MixedKernel GEMM (Auto-Scheduled Cube + Vector)

Same as Pattern 1 but uses `T.MixedKernel` to automatically split
work between AIC and AIV.

```python
with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
    x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
    w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
    res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
    temp = T.alloc_shared((TILE_M // 2, TILE_N), accum_dtype)

    for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
        for kt in T.Pipelined(K_TILES, num_stages=2):
            T.copy(X[...], x_l1)
            T.copy(W[...], w_l1)
            T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.dual_copy(res, temp)
        T.copy(temp[sid * (TILE_M // 2):...], C[...])
```

Reference: `examples/ascend/example_gemm_mixedkernel.py`

### Pattern 3: Manual Mixed (Cube + Vector with Manual Flags)

Explicit `T.Cube()` and `T.Vector()` scopes with manual cross-core
sync. More code but full control.

Reference: `examples/ascend/example_gemm_mix.py`

### Pattern 4: SIMD VF (Register-Level Vector)

Element-wise operations using `T.SimdVF()`. No threads -- uses CCE
MicroAPI vector instructions directly.

```python
with T.Kernel(NUM_BLOCKS) as bx:
    for iter in T.Pipelined(NUM_TILES, num_stages=2):
        T.copy(A[begin:end], temp1)
        T.copy(B[begin:end], temp2)
        with T.SimdVF():
            for i in T.Parallel(TILE_ELEMS):
                temp3[i] = temp1[i] * (temp1[i] + temp2[i])
        T.copy(temp3, C[begin:end])
```

Reference: `examples/ascend/example_simdvf_vecadd.py`

### Pattern 5: SIMT VF (Thread-Parallel Vector)

Element-wise operations distributed across threads.

```python
with T.SimtVF(threads=NUM_THREADS):
    for i in T.Parallel(TILE_ELEMS):
        temp3[i] = temp1[i] + temp2[i]
```

Reference: `examples/ascend/example_simtvf_vecadd.py`

### Pattern 6: RMSNorm with Fragment Trick

Uses `T.alloc_fragment` for register-level vectorized loads (float4
chunks) and `T.alloc_reducer` + `T.finalize_reducer` for cross-warp
reduction of `sum(x^2)`.

Reference: `examples/ascend/example_rmsnorm.py`

### Pattern 7: L0-Staged GEMM

Double-buffers L0A and L0B within each K-tile for maximum Cube
utilization. Adds an inner `SUB_K` tiling loop.

Reference: `examples/ascend/example_gemm_l0.py` (manual),
`examples/ascend/example_gemm_l0_auto.py` (auto-scheduled)

---

## 10. Common Pitfalls

1. **Forgetting `transpose_B=True`**: The single most common Ascend
   error. The Cube MAD unit requires it. GEMM without it fails at
   compile time or produces garbage.

2. **Using `threads=` on `T.Kernel`**: Raises `ValueError` immediately.
   Thread domains go inside `T.SimtVF(threads=N)` scopes.

3. **Multi-dimensional grids**: Ascend only supports `T.Kernel(N)`.
   For 2-D decomposition, compute mapping yourself: e.g.,
   `row = tile_idx // N_TILES; col = tile_idx % N_TILES`.

4. **Weight layout**: Because `transpose_B=True` is mandatory, store
   weight matrices as `[N, K]` (not `[K, N]` in global memory).

5. **Buffer dimension with auto-schedule**: With `T.Pipelined` or
   `T.Persistent`, buffers should **not** have a leading `STAGES`
   dimension. The compiler multi-buffers automatically.

6. **L0C is float32-only**: The cube accumulator only supports
   `"float32"` dtype. L1 and L0A/L0B can be `"bfloat16"` or `"float16"`,
   but L0C must be `"float32"`.

7. **Reducer usage**: `T.alloc_reducer` requires both `T.clear(reducer)`
   before accumulation and `T.finalize_reducer(reducer)` before reading
   results.

8. **SIMT VF thread sync**: The compiler auto-inserts `asc_syncthreads()`
   based on data-flow analysis. When writing to UB in one thread and
   reading in another, sync is automatic. No manual barriers needed.

9. **dual_copy shape mismatch**: `T.dual_copy` splits data across AIV
   sub-cores. The destination must be exactly half the source dimension
   (for M-split) or the compiler raises `ValueError`.

10. **AIV sid indexing in output**: In `T.MixedKernel`, `T.copy` from
    UB to GM uses `sid` to partition output rows. Forgetting the `sid`
    offset causes both AIV sub-cores to write the same region.

---

## 11. Compilation and Execution

```python
import tilelang

# Standard compilation (auto-schedule enabled by default)
kernel = tilelang.compile(program, out_idx=-1)

# Disable auto-scheduling for manual-flag kernels
kernel = tilelang.compile(program, out_idx=-1,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})

# Multiple outputs
kernel = tilelang.compile(program, out_idx=[1, 3])

# Inspect generated Ascend C source
print(kernel.get_kernel_source())

# Run on NPU
x, w = torch.randn(...), torch.randn(...)
result = kernel(x, w)
torch.npu.synchronize()
```

Run scripts with: `npu-env python your_script.py`

### Post-Processing Callback

Register a callback to modify generated Ascend C source before
compilation:

```python
from tilelang.engine.callback import register_ascend_postproc

@register_ascend_postproc
def customize_source(code, target):
    # Modify 'code' (str), return modified source
    return code
```

Reference: `examples/ascend/example_ascend_postproc_callback.py`

---

## 12. Example Files Reference

All examples are in `examples/ascend/`. Key files organized by category:

### GEMM variants
| File | Description |
|---|---|
| `example_gemm.py` | Manual flag pipelining (legacy reference) |
| `example_gemm_auto.py` | **Recommended**: auto-scheduled with T.Persistent + T.Pipelined |
| `example_gemm_mix.py` | Manual MixedKernel (Cube + Vector with cross-core flags) |
| `example_gemm_mixedkernel.py` | Auto-scheduled MixedKernel GEMM |
| `example_gemm_l0.py` | L0-staged GEMM (L1 → L0A/L0B sub-tiling) |
| `example_gemm_l0_auto.py` | Auto-scheduled L0-staged GEMM |
| `example_gemm_bypass_l2.py` | GEMM with L2 cache bypass |
| `example_gemm_padded_copy.py` | Padded DMA copy for alignment |
| `example_gemm_ub_merge.py` | UB shared memory merging optimization |
| `example_gemm_various_shapes.py` | Different tile shape configurations |

### SIMD VF (register-level vector operations)
| File | Description |
|---|---|
| `example_simdvf_vecadd.py` | High-level SIMD vector add (T.Parallel in SimdVF) |
| `example_simdvf_vecadd_lower.py` | Raw T.simd.* intrinsics (manual unrolling) |
| `example_simdvf_vecadd_lower_auto.py` | Auto-scheduled SIMD lower |
| `example_simdvf_vintlv.py` | SIMD vintlv/vdintlv interleave operations |
| `example_simdvf_per_token_cast_to_fp8.py` | Per-token fp8 type conversion |
| `example_simd_mem_bar.py` | SIMD memory barrier usage |

### SIMT VF (thread-parallel operations)
| File | Description |
|---|---|
| `example_simtvf_vecadd.py` | Thread-parallel vector add |
| `example_simtvf_vecadd_auto.py` | Auto-scheduled SIMT vector add |
| `example_simtvf_vecadd_mutex.py` | SIMT mutex synchronization |
| `example_simtvf_auto_sync.py` | Auto thread sync demonstration |
| `example_simtvf_copy.py` | SIMT copy patterns |
| `example_simtvf_ubuf.py` | UB buffer usage in SIMT |
| `example_simtvf_ubuf_multi.py` | Multi-buffer UB in SIMT |
| `example_simtvf_per_token_cast_to_fp8.py` | Per-token fp8 cast in SIMT |
| `example_simtvf_vector_add.py` | Minimal SIMT vector add |

### Normalization
| File | Description |
|---|---|
| `example_rmsnorm.py` | RMSNorm with fragment trick + reducer |
| `example_rmsnorm_auto.py` | Auto-scheduled RMSNorm |

### Attention
| File | Description |
|---|---|
| `example_flash_attn.py` | FlashAttention variant |

### Misc
| File | Description |
|---|---|
| `example_buffer_version_annotation.py` | Buffer version annotation for pipelining |
| `example_ascend_postproc_callback.py` | Post-processing codegen callback |
