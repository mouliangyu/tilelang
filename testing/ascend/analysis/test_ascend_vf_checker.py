import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower
import pytest


def test_copy_matching_dtypes_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            T.copy(A[:256], temp)

    lower(kernel, target="ascend")


def test_copy_dtype_mismatch_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            T.copy(A[:256], temp)

    with pytest.raises(ValueError, match="DMA copies cannot perform type casting"):
        lower(kernel, target="ascend")


def test_copy_dtype_mismatch_inside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            with T.SimtVF(threads=128):
                T.copy(A[:256], temp)

    lower(kernel, target="ascend")


def test_parallel_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float32")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            for i in T.Parallel(256):
                temp[i] = A[i]

    with pytest.raises(ValueError, match="Parallel loops outside VF blocks"):
        lower(kernel, target="ascend")


def test_simd_vf_load_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    temp[i] = A[i]

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simd_vf_store_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    A[i] = temp[i]

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simd_vf_copy_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    T.copy(A[i], temp[i])

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simt_vf_access_global_allowed():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimtVF(threads=128):
                for i in T.Parallel(256):
                    temp[i] = A[i]

    lower(kernel, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
