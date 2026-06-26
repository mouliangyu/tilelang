from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


def _is_pto_target(target: Target) -> bool:
    # PTO and Ascend targets share kind "c"; "pto" selects the PTO codegen.
    return "pto" in target.keys


def _is_ascend_target(target: Target) -> bool:
    # Plain Ascend codegen (Ascend semantics without the PTO backend selector).
    return ("ascend" in target.keys or target.kind.name == "ascend") and "pto" not in target.keys


register_device_codegen(
    "c",
    DeviceCodegen(
        "pto",
        build=global_func_device_codegen("target.build.tilelang_pto"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_pto_without_compile"),
        supports_target=_is_pto_target,
    ),
    override=True,
)

register_device_codegen(
    "c",
    DeviceCodegen(
        "ascend",
        build=global_func_device_codegen("target.build.tilelang_ascend"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_ascend_without_compile"),
        supports_target=_is_ascend_target,
    ),
    override=True,
)
