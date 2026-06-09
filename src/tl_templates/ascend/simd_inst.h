#pragma once

#include "kernel_operator.h"

namespace simd_inst {
template <typename T> struct vec {};

template <> struct vec<float> {
  using type = vector_f32;
};
template <> struct vec<half> {
  using type = vector_f16;
};

#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> {
  using type = vector_bf16;
};
template <> struct vec<fp8_e4_t> {
  using type = vector_f8e4m3;
};
template <> struct vec<fp8_e5_t> {
  using type = vector_f8e5m2;
};
template <> struct vec<float8_e4m3_t> {
  using type = vector_f8e4m3;
};
template <> struct vec<float8_e5m2_t> {
  using type = vector_f8e5m2;
};
template <> struct vec<float8_e8m0_t> {
  using type = vector_f8e8m0;
};
template <> struct vec<float4_e2m1x2_t> {
  using type = vector_f4e2m1x2;
};
template <> struct vec<float4_e1m2x2_t> {
  using type = vector_f4e1m2x2;
};
#endif

template <> struct vec<uint32_t> {
  using type = vector_u32;
};

template <> struct vec<uint16_t> {
  using type = vector_u16;
};

template <> struct vec<int32_t> {
  using type = vector_s32;
};

template <> struct vec<int16_t> {
  using type = vector_s16;
};

template <typename SrcVec> struct vec_pair {
  SrcVec v0;
  SrcVec v1;
};

template <typename T> using vec_t = typename vec<T>::type;

template <typename T, typename Dist>
__simd_callee__ inline vec_t<T> vld(__ubuf__ T *src, Dist dist) {
  vec_t<T> dst;
  ::vld(dst, src, dist);
  return dst;
}

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_t<T> vld(__ubuf__ T *src, Offset offset, Dist dist) {
  vec_t<T> dst;
  ::vld(dst, src, offset, dist);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline void vscatter(vec_t<T> data, __ubuf__ T *base,
                                     IdxVec idx, vector_bool mask) {
  ::vscatter(data, base, idx, mask);
}

// f16→f32, f8→f32, i32→f32: ::vcvt(dst, src, mask, part/round, mode)
template <typename U, typename SrcVec, typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask, Part part,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, part, mode);
  return dst;
}

// f32→i32: ::vcvt(dst, src, mask, round, rs, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, mode);
  return dst;
}

// f32→f16, f32→f8, f16→f8: ::vcvt(dst, src, mask, round, rs, part, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Part part, Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, part, mode);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vdintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vdintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

// -- Binary arithmetic
// ----------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vadd(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vadd(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsub(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vsub(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmul(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmul(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vdiv(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vdiv(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmax(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmax(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmin(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmin(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vand(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vand(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vor(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                  Mode mode) {
  SrcVec dst;
  ::vor(dst, src_0, src_1, mask, mode);
  return dst;
}

// -- Unary
// ----------------------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vln(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vln(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqrt(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vsqrt(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vrsqrt(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  vrsqrt(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vabs(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vabs(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vneg(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vneg(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vrelu(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vrelu(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vrec(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  vrec(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vexp(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vexp(dst, src, mask, mode);
  return dst;
}

// -- Cross-lane reductions
// ------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcmax(dst, src, mask, mode);
  return dst;
}

// -- Broadcast
// ------------------------------------------------------------------

// scalar broadcast: T = float / half / int32_t / ...
template <typename T, typename Mode>
__simd_callee__ inline vec_t<T> vdup(T src, vector_bool mask, Mode mode) {
  vec_t<T> dst;
  ::vdup(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Pos, typename Mode>
__simd_callee__ inline SrcVec vdupv(SrcVec src, vector_bool mask, Pos pos,
                                    Mode mode) {
  SrcVec dst;
  ::vdup(dst, src, mask, pos, mode);
  return dst;
}

// -- Select
// ---------------------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsel(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vsel(dst, src_0, src_1, mask, mode);
  return dst;
}

// -- Scalar-vector ops
// ----------------------------------------------------------

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vadds(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vadds(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmuls(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmuls(dst, src, scalar, mask, mode);
  return dst;
}

// -- Exponential difference
// -----------------------------------------------------

template <typename SrcVec, typename Part>
__simd_callee__ inline SrcVec vexpdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask, Part part) {
  SrcVec dst;
  ::vexpdif(dst, src_0, src_1, mask, part);
  return dst;
}

template <typename U, typename SrcVec, typename Part>
__simd_callee__ inline vec_t<U> vpack(SrcVec src, Part part) {
  vec_t<U> dst;
  ::vpack(dst, src, part);
  return dst;
}

template <typename T>
__simd_callee__ inline void vsstb(vec_t<T> src, __ubuf__ T *base,
                                  int32_t stride, vector_bool mask) {
  ::vsstb(src, base, stride, mask);
}

template <typename T, typename Post>
__simd_callee__ inline void vsstb(vec_t<T> src, __ubuf__ T *&base,
                                  int32_t stride, vector_bool mask, Post post) {
  ::vsstb(src, base, stride, mask, post);
}

template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> data, __ubuf__ T *base,
                                 int32_t offset, Dist dist, vector_bool mask) {
  ::vsts(data, base, offset, dist, mask);
}

template <typename T> __simd_callee__ inline void mem_bar(T mem_type) {
  ::mem_bar(mem_type);
}

} // namespace simd_inst
