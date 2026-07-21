from tilelang.ascend.kernel_cache import AscendCythonKernelCache, AscendTVMFFIKernelCache
from tilelang.cache import _dispatch_map, _resolve_cache_dispatch


def test_ascend_tvm_ffi_cache_uses_target_specific_source_suffixes():
    cache, target, backend, _ = _resolve_cache_dispatch("ascend", "tvm_ffi", False)

    assert type(cache) is AscendTVMFFIKernelCache
    assert cache is not _dispatch_map["tvm_ffi"]
    assert "ascend" in target.keys
    assert backend == "tvm_ffi"
    assert cache.device_kernel_path == "device_kernel.asc"
    assert cache.host_kernel_path == "host_kernel.c"


def test_ascend_cython_cache_uses_bisheng_source_suffixes():
    cache, target, backend, _ = _resolve_cache_dispatch("ascend", "cython", False)

    assert type(cache) is AscendCythonKernelCache
    assert cache is not _dispatch_map["cython"]
    assert "ascend" in target.keys
    assert backend == "cython"
    assert cache.device_kernel_path == "device_kernel.asc"
    assert cache.host_kernel_path == "host_kernel.asc"


def test_cuda_cache_source_suffixes_are_unchanged():
    cache, _, backend, _ = _resolve_cache_dispatch({"kind": "cuda", "arch": "sm_90"}, "tvm_ffi", False)

    assert cache is _dispatch_map["tvm_ffi"]
    assert backend == "tvm_ffi"
    assert cache.device_kernel_path == "device_kernel.cu"
    assert cache.host_kernel_path == "host_kernel.cu"
