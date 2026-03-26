# Symmetric Batched Matrix Multiply Transpose (Y = X @ X^T)

Blackwell (SM100) optimized kernels for computing Y = X @ X^T, leveraging TileLang's TCGEN05 instructions, warp specialization, and 2-CTA cluster cooperation.

## Key Optimizations

1. **Symmetry-aware tiling** -- only computes the upper triangle (bx <= col_tile), then mirrors the result to the lower triangle via transposed write-back. This cuts total compute nearly in half.
2. **Warp specialization** -- warp 0 acts as TMA producer (global -> shared), warp 1 as TCGEN05 GEMM consumer, remaining warps handle the epilogue. A multi-stage async pipeline (6 stages) hides memory latency.
3. **2-CTA clusters** -- two CTAs share shared memory to double the effective N-tile size (block_N=256, each CTA loads 128 rows of B).
4. **`T.transpose` for coalesced write-back** -- the lower-triangle symmetric write uses `T.transpose` to transpose the tile in shared memory first, then writes back with a coalesced `T.copy`. This replaces the previous strided (non-coalesced) global store and delivers ~3x speedup on the epilogue.

## Files

| File | Description |
|------|-------------|
| `symmetric_mmt_2cta.py` | Non-batched: X(M, K) @ X^T -> Y(M, M) |
| `batched_symmetric_mmt_2cta.py` | Batched: X(B, M, K) @ X^T -> Y(B, M, M) |

## Performance

All benchmarks run on **NVIDIA GB200 (SM100)**, FP16 input, FP32 accumulation.

Tile config: `block_M=128, block_N=256, block_K=64, stages=6, threads=128`.

### Non-Batched (Y = X @ X^T)

Shapes derived from 1T+ model dimensions (DSV3 hidden=7168, MoE expert FFN=18432, dense 1T hidden=12288).

| M | K | Kernel (ms) | PyTorch (ms) | vs PyTorch | Kernel TFLOPS | PyTorch TFLOPS |
|------:|------:|------:|------:|------:|------:|------:|
| 4096 | 7168 | 0.1026 | 0.1518 | **1.48x** | 2343.7 | 1584.0 |
| 8192 | 7168 | 0.3561 | 0.5518 | **1.55x** | 2701.3 | 1743.6 |
| 4096 | 18432 | 0.2293 | 0.3616 | **1.58x** | 2696.9 | 1710.6 |
| 8192 | 12288 | 0.5738 | 1.1002 | **1.92x** | 2874.5 | 1499.0 |
| 16384 | 16384 | 3.2825 | 5.8868 | **1.79x** | 2679.7 | 1494.2 |

### Batched (Y = X @ X^T, with batch dim)

| B | M | K | Kernel (ms) | PyTorch (ms) | vs PyTorch | Kernel TFLOPS | PyTorch TFLOPS |
|----:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 4096 | 7168 | 0.1027 | 0.1540 | **1.50x** | 2342.1 | 1562.0 |
| 1 | 8192 | 7168 | 0.3553 | 0.5742 | **1.62x** | 2707.4 | 1675.5 |
| 4 | 4096 | 7168 | 0.3965 | 0.5511 | **1.39x** | 2426.5 | 1745.6 |
| 4 | 8192 | 7168 | 1.4840 | 2.6335 | **1.77x** | 2593.2 | 1461.3 |
| 1 | 4096 | 18432 | 0.2306 | 0.3676 | **1.59x** | 2682.6 | 1682.4 |
| 4 | 4096 | 18432 | 0.9562 | 1.6148 | **1.69x** | 2587.2 | 1532.0 |
| 1 | 8192 | 12288 | 0.6318 | 1.0735 | **1.70x** | 2610.5 | 1536.4 |
| 4 | 8192 | 12288 | 2.4746 | 4.3699 | **1.77x** | 2665.9 | 1509.7 |
| 4 | 16384 | 16384 | 13.8669 | 24.5074 | **1.77x** | 2537.3 | 1435.7 |

### Summary

- **1.39x -- 1.92x faster** than PyTorch (cuBLAS) across all tested shapes.
- Peak throughput reaches **2874 TFLOPS** (FP16 with FP32 accum) on GB200.
- The speedup comes from two sources:
  - **~2x from symmetry**: only computing the upper triangle halves the FLOPs.
  - **Efficient epilogue**: `T.transpose` in shared memory ensures coalesced global writes for the mirrored lower triangle.

## Usage

```python
import tilelang
import tilelang.language as T

# Non-batched
from symmetric_mmt_2cta import matmul_transpose_tcgen05_ws_2cta
kernel = matmul_transpose_tcgen05_ws_2cta(M=4096, K=7168)
Y = kernel(X)  # X: (4096, 7168) -> Y: (4096, 4096)

# Batched
from batched_symmetric_mmt_2cta import batched_matmul_transpose_tcgen05_ws_2cta
kernel = batched_matmul_transpose_tcgen05_ws_2cta(B=4, M=8192, K=7168)
Y = kernel(X)  # X: (4, 8192, 7168) -> Y: (4, 8192, 8192)
```

## Requirements

- NVIDIA Blackwell GPU (SM100, e.g. GB200)
- TileLang with `T.transpose` support
- PyTorch with CUDA
