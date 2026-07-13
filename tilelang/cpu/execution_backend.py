from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


# tvm_ffi is available for both plain CPU/C targets and the Ascend NPU
# backend (target kind "c" with key "ascend"). The Ascend runnable runtime
# module is provided by src/ascend (AscendModule); see
# tilelang/ascend/target.py for the kDLExtDev device-type tagging that lets
# the FFI ABI binder accept torch_npu tensors. Note: "cython" is registered
# first for "c", so it remains the default for execution_backend="auto";
# tvm_ffi is selected only when requested explicitly.
register_execution_backend("c", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "c",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)

# llvm: only tvm_ffi (no cython) so execution_backend="auto" resolves to tvm_ffi (#2519).
register_execution_backend(
    "llvm",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
