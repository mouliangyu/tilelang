#pragma once

#include "kernel_operator.h"

namespace simd_inst {
template <typename T> struct vec {};

template <> struct vec<float> {
  using type = vector_f32;
};

#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> {
  using type = vector_bf16;
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

template <typename U, typename SrcVec, typename Part>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     Part part) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, part);
  return dst;
}

template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Part>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Part part) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, part);
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

template <typename SrcVec>
__simd_callee__ inline SrcVec vor(SrcVec src_0, SrcVec src_1,
                                  vector_bool mask) {
  SrcVec dst;
  ::vor(dst, src_0, src_1, mask);
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

template <typename SrcVec>
__simd_callee__ inline SrcVec vexp(SrcVec src, vector_bool mask) {
  SrcVec dst;
  ::vexp(dst, src, mask);
  return dst;
}

template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> data, __ubuf__ T *base,
                                 int32_t offset, Dist dist, vector_bool mask) {
  ::vsts(data, base, offset, dist, mask);
}

} // namespace simd_inst
