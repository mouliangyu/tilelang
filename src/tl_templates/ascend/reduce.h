#pragma once

#include "simt_api/asc_simt.h"

#ifndef HUGE_VALF
#define HUGE_VALF __builtin_huge_valf()
#endif

namespace tl {
struct SumOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) { return x + y; }
};
struct MaxOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) {
    return fmaxf(x, y);
  }
};
struct MinOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) {
    return fminf(x, y);
  }
};

/*!
 * \brief Cross-thread reduction on Ascend using XOR-butterfly pattern.
 *
 * Template parameters:
 *   Reducer       - binary reduction functor (SumOp, MaxOp, MinOp).
 *   threads       - number of thread positions that span the reduce dimension,
 *                   equal to extent * scale.
 *   scale         - stride of participating threads in the thread index space.
 *                   When the thread-to-data mapping is normalized as
 *                     threadIdx = source * scale + ...
 *                   `scale` is the stride between consecutive logical
 *                   participants in the reduce dimension.
 *                   The recursion terminates when threads == scale, meaning
 *                   each reduce group has been collapsed to a single thread.
 *   thread_offset - base thread index offset within the block.
 *
 * Example: for a 3D tensor (1,4,4) with 16 elements mapped to threads as
 *   thread = j * 4 + k, reducing along j (dim=1):
 *   extent=4, scale=4, threads=16.
 *   XOR offsets: 8, 4 → terminates at scale=4.
 *   This correctly reduces threads {0,4,8,12}, {1,5,9,13}, etc.
 */
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  // Allow other instantiations to call butterfly_reduce recursively.
  template <class, int, int, int> friend struct AscendAllReduce;

private:
  static constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

  // Recursive XOR-butterfly reduce.
  // At each level, XOR with offset = threads/2, then recurse with threads/2.
  // Terminates when threads == scale (each group collapsed to one thread).
  static __simt_callee__ float butterfly_reduce(float x) {
    if constexpr (threads <= scale) {
      return x;
    } else {
      constexpr int offset = threads / 2;
      Reducer op;
      x = op(x, asc_shfl_xor(x, offset));
      return AscendAllReduce<Reducer, offset, scale,
                             thread_offset>::butterfly_reduce(x);
    }
  }

  // Fallback for non-power-of-2 groups or large reductions using UB buffer.
  static __simt_callee__ float ub_reduce(float x, __ubuf__ float *red_buf) {
    int tx = threadIdx.x - thread_offset;
    // Compute group membership: threads within the same group of size `threads`
    // share a reduction, but only every `scale`-th thread holds a unique value.
    int group = tx / threads;
    int lane = tx % threads;
    Reducer op;

    red_buf[tx] = x;
    asc_syncthreads();

    float result = x;
    // Only threads at positions that are multiples of `scale` participate
    if (lane % scale == 0) {
      int base = group * threads + (lane % scale);
      result = red_buf[group * threads + lane];
      for (int i = scale; i < threads; i += scale) {
        int idx = group * threads + (lane % scale) + i;
        result = op(result, red_buf[idx]);
      }
    }
    asc_syncthreads();

    // Broadcast: store result for the group, each sub-lane reads it
    if (lane % scale == 0 && lane / scale == 0) {
      red_buf[group * threads + (lane % scale)] = result;
    }
    asc_syncthreads();
    // Each thread reads the result of its group
    result = red_buf[group * threads + (tx % scale)];
    asc_syncthreads();
    return result;
  }

public:
  static __simt_callee__ float run(float x, __ubuf__ float *red_buf = nullptr) {
    if constexpr (threads <= scale) {
      // No reduction needed: each group has exactly one thread.
      return x;
    } else if constexpr (threads <= 32 && is_pow2(threads) && is_pow2(scale)) {
      // Fast path: power-of-2, fits in warp, use shfl_xor butterfly.
      return butterfly_reduce(x);
    } else if constexpr (threads <= 32 && !is_pow2(threads / scale)) {
      // Non-power-of-2 extent: use UB fallback.
      return ub_reduce(x, red_buf);
    } else if constexpr (threads > 32) {
      // Large reduction: warp-level reduce + cross-warp via UB.
      constexpr int num_warps = threads / 32;
      int tx = threadIdx.x - thread_offset;
      int wid = tx >> 5, lid = tx & 31;

      if constexpr (scale <= 32 && is_pow2(scale)) {
        // Scale fits within a warp: first do intra-warp butterfly,
        // then cross-warp via shared memory.
        float warp_val = AscendAllReduce<Reducer, 32, scale,
                                         thread_offset>::butterfly_reduce(x);
        if (lid == 0)
          red_buf[wid] = warp_val;
        asc_syncthreads();
        float result;
        if (tx < 32) {
          float v = (lid < num_warps)                ? red_buf[lid]
                    : std::is_same_v<Reducer, SumOp> ? 0.0f
                    : std::is_same_v<Reducer, MaxOp> ? -HUGE_VALF
                                                     : HUGE_VALF;
          if constexpr (std::is_same_v<Reducer, SumOp>)
            result = asc_reduce_add(v);
          else if constexpr (std::is_same_v<Reducer, MaxOp>)
            result = asc_reduce_max(v);
          else
            result = asc_reduce_min(v);
        }
        asc_syncthreads();
        if (tx == 0)
          red_buf[0] = result;
        asc_syncthreads();
        return red_buf[0];
      } else {
        // General large + large-scale case: fallback to UB.
        return ub_reduce(x, red_buf);
      }
    } else {
      // Catch-all fallback.
      return ub_reduce(x, red_buf);
    }
  }
};
} // namespace tl
