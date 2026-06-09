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

// ─── Helper: identity value for each reducer ────────────────────────────────
template <class Reducer>
static __simt_callee__ constexpr float reduce_identity() {
  if constexpr (std::is_same_v<Reducer, SumOp>)
    return 0.0f;
  else if constexpr (std::is_same_v<Reducer, MaxOp>)
    return -HUGE_VALF;
  else
    return HUGE_VALF; // MinOp
}

// ─── Helper: dispatch to hardware warp-reduce intrinsic ─────────────────────
template <class Reducer> static __simt_callee__ float hw_reduce(float v) {
  if constexpr (std::is_same_v<Reducer, SumOp>)
    return asc_reduce_add(v);
  else if constexpr (std::is_same_v<Reducer, MaxOp>)
    return asc_reduce_max(v);
  else
    return asc_reduce_min(v);
}

/*!
 * \brief Cross-thread reduction on Ascend.
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
 * Two warp-internal strategies are used depending on the number of independent
 * reduce groups per warp:
 *
 *   1. hw_reduce (asc_reduce_add/max/min + identity masking):
 *      Used when extent >= 16 && scale == 1, giving <= 2 groups per warp.
 *      Cost: groups_per_warp rounds of a single hardware reduce instruction.
 *
 *   2. butterfly (asc_shfl_xor):
 *      Used otherwise.  Cost: log2(extent) rounds of shfl_xor + reducer op.
 *
 * For cross-warp reductions (threads > 32), each warp first performs an
 * intra-warp reduce, writes the result to UB (shared memory), synchronizes,
 * and then a single warp finalizes via hw_reduce.
 *
 * Example: for a 3D tensor (1,4,4) with 16 elements mapped to threads as
 *   thread = j * 4 + k, reducing along j (dim=1):
 *   extent=4, scale=4, threads=16.
 *   XOR offsets: 8, 4 → terminates at scale=4.
 *   This correctly reduces threads {0,4,8,12}, {1,5,9,13}, etc.
 */
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  template <class, int, int, int> friend struct AscendAllReduce;

  static_assert(threads >= 1);
  static_assert(scale >= 1);
  static_assert(threads % scale == 0);

private:
  static constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }
  static constexpr int extent = threads / scale;

  // ─── Warp-internal: XOR-butterfly ─────────────────────────────────────────
  // Recursive butterfly reduce.  At each level, XOR with offset = cur/2,
  // then recurse with cur/2.  Terminates when cur == stop.
  template <int cur, int stop> static __simt_callee__ float butterfly(float x) {
    if constexpr (cur <= stop) {
      return x;
    } else {
      constexpr int offset = cur / 2;
      Reducer op;
      x = op(x, asc_shfl_xor(x, offset));
      return butterfly<cur / 2, stop>(x);
    }
  }

  // ─── Warp-internal: hardware reduce + identity masking ────────────────────
  // Precondition: scale == 1.
  // Each group is a contiguous block of `threads` lanes; non-participating
  // lanes feed the identity value so the hardware full-warp reduce yields
  // the correct per-group result.
  // Cost: groups_per_warp rounds of asc_reduce_*.
  static __simt_callee__ float warp_hw_reduce(float x) {
    constexpr float id = reduce_identity<Reducer>();
    constexpr int groups = 32 / threads;

    if constexpr (groups == 1) {
      // Entire warp is one group.
      return hw_reduce<Reducer>(x);
    } else {
      int tx = threadIdx.x - thread_offset;
      int my_group = (tx & 31) / threads;
      float result = x;
// Each round: one group contributes real values, others feed identity.
// Unroll for small group counts.
#pragma unroll
      for (int g = 0; g < groups; ++g) {
        float v = hw_reduce<Reducer>((my_group == g) ? result : id);
        if (my_group == g)
          result = v;
      }
      return result;
    }
  }

  // ─── Warp-internal: choose best strategy ──────────────────────────────────
  static __simt_callee__ float warp_reduce(float x) {
    if constexpr (extent <= 1) {
      return x;
    } else if constexpr (extent >= 16 && scale == 1) {
      return warp_hw_reduce(x);
    } else {
      return butterfly<threads, scale>(x);
    }
  }

  // ─── Fallback for non-power-of-2 using UB buffer ─────────────────────────
  static __simt_callee__ float ub_reduce(float x, __ubuf__ float *red_buf) {
    int tx = threadIdx.x - thread_offset;
    int group = tx / threads;
    int lane = tx % threads;
    Reducer op;

    red_buf[tx] = x;
    asc_syncthreads();

    if (lane < scale) {
      float result = red_buf[group * threads + lane];
      for (int i = scale; i < threads; i += scale) {
        result = op(result, red_buf[group * threads + lane + i]);
      }
      red_buf[group * threads + lane] = result;
    }
    asc_syncthreads();
    float result = red_buf[group * threads + (lane % scale)];
    asc_syncthreads();
    return result;
  }

  // ─── Cross-warp reduction via UB ──────────────────────────────────────────
  static __simt_callee__ float cross_warp_reduce(float x,
                                                 __ubuf__ float *red_buf) {
    constexpr int num_warps = threads / 32;
    int tx = threadIdx.x - thread_offset;
    int wid = tx >> 5, lid = tx & 31;

    // Step 1: intra-warp reduce using best warp-level strategy.
    float warp_val =
        AscendAllReduce<Reducer, 32, scale, thread_offset>::warp_reduce(x);

    // Step 2: each warp writes its result(s) to UB.
    // With scale == 1, each warp produces 1 value (lane 0).
    // With scale > 1, each warp produces `scale` values (lanes 0..scale-1).
    if (lid < scale)
      red_buf[wid * scale + lid] = warp_val;
    asc_syncthreads();

    // Step 3: cross-warp reduce — single warp reads and reduces.
    float result;
    if constexpr (scale == 1) {
      // num_warps scalar values to reduce.  All of warp 0 participates,
      // unused lanes get identity.  One hw_reduce call suffices.
      if (tx < 32) {
        float v = (lid < num_warps) ? red_buf[lid] : reduce_identity<Reducer>();
        result = hw_reduce<Reducer>(v);
      }
    } else if constexpr (scale * num_warps <= 32) {
      // All cross-warp data fits in one warp.
      // Warp 0 reads scale * num_warps values, then does a warp-level
      // reduce with the appropriate (scale * num_warps, scale) parameters.
      if (tx < 32) {
        constexpr int total = scale * num_warps;
        float v = (lid < total) ? red_buf[lid] : reduce_identity<Reducer>();
        result = AscendAllReduce<Reducer, total, scale, 0>::warp_reduce(v);
      }
    } else {
      // Rare case: scale * num_warps > 32.  Fallback to serial in warp 0.
      if (tx < 32) {
        Reducer op;
        result = reduce_identity<Reducer>();
        int my_slot = lid % scale;
        for (int w = 0; w < num_warps; ++w) {
          int idx = w * scale + my_slot;
          result = (lid < scale) ? op(result, red_buf[idx]) : result;
        }
      }
    }
    asc_syncthreads();

    // Step 4: broadcast result back to all threads.
    if (tx < scale)
      red_buf[tx] = result;
    asc_syncthreads();
    result = red_buf[tx % scale];
    asc_syncthreads();
    return result;
  }

public:
  static __simt_callee__ float run(float x, __ubuf__ float *red_buf = nullptr) {
    if constexpr (threads <= scale) {
      // No reduction needed.
      return x;
    } else if constexpr (threads <= 32 && is_pow2(threads) && is_pow2(scale)) {
      // Fits in a single warp, power-of-2: use register-only strategy.
      return warp_reduce(x);
    } else if constexpr (threads <= 32) {
      // Non-power-of-2 or other edge case: UB fallback.
      return ub_reduce(x, red_buf);
    } else if constexpr (threads > 32 && is_pow2(threads) && scale <= 32 &&
                         is_pow2(scale)) {
      // Cross-warp, well-behaved parameters.
      return cross_warp_reduce(x, red_buf);
    } else {
      // General fallback.
      return ub_reduce(x, red_buf);
    }
  }
};
} // namespace tl
