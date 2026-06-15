from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


def _is_not_ascend_target(target: Target) -> bool:
    return "ascend" not in target.keys


for _target_kind in ("c", "llvm"):
    register_execution_backend(_target_kind, ExecutionBackendSpec("cython"), override=True)
    register_execution_backend(
        _target_kind,
        ExecutionBackendSpec(
            "tvm_ffi",
            supports_target=_is_not_ascend_target,
            enable_host_codegen=True,
            enable_device_compile=True,
        ),
        override=True,
    )
