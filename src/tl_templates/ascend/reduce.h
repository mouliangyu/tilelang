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

template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
private:
  static constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

  static __simt_callee__ float shfl_reduce(float x) {
    if constexpr (threads <= 1)
      return x;
    Reducer op;
    if constexpr (threads >= 32)
      x = op(x, asc_shfl_xor(x, 16));
    if constexpr (threads >= 16)
      x = op(x, asc_shfl_xor(x, 8));
    if constexpr (threads >= 8)
      x = op(x, asc_shfl_xor(x, 4));
    if constexpr (threads >= 4)
      x = op(x, asc_shfl_xor(x, 2));
    if constexpr (threads >= 2)
      x = op(x, asc_shfl_xor(x, 1));
    return x;
  }

  // Fallback for non-power-of-2 groups using UB buffer.
  static __simt_callee__ float ub_reduce(float x, __ubuf__ float *red_buf) {
    int tx = threadIdx.x - thread_offset;
    int group = tx / threads;
    int lane = tx % threads;
    Reducer op;

    red_buf[tx] = x;
    asc_syncthreads();

    float result = x;
    if (lane == 0) {
      for (int i = 1; i < threads; i++)
        result = op(result, red_buf[group * threads + i]);
    }
    asc_syncthreads();

    if (lane == 0)
      red_buf[group] = result;
    asc_syncthreads();
    result = red_buf[group];
    asc_syncthreads();
    return result;
  }

public:
  static __simt_callee__ float run(float x, __ubuf__ float *red_buf = nullptr) {
    if constexpr (threads <= 32) {
      if constexpr (is_pow2(threads)) {
        return shfl_reduce(x);
      } else {
        return ub_reduce(x, red_buf);
      }
    } else {
      constexpr int num_warps = threads / 32;
      int tx = threadIdx.x - thread_offset;
      int wid = tx >> 5, lid = tx & 31;
      float warp_val;
      if constexpr (std::is_same_v<Reducer, SumOp>)
        warp_val = asc_reduce_add(x);
      else if constexpr (std::is_same_v<Reducer, MaxOp>)
        warp_val = asc_reduce_max(x);
      else
        warp_val = asc_reduce_min(x);
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
    }
  }
};
} // namespace tl
