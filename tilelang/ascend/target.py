from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

# torch_npu represents NPU tensors in DLPack as kDLExtDev (device type 12).
# The Ascend target uses target kind "c" (whose default device type is kDLCPU),
# so we explicitly tag it with kDLExtDev to match the runtime DLPack tensors;
# this lets the tvm_ffi ABI binder's device_type check pass without relaxation.
_kDLExtDev = 12


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def _with_ascend_key(target: Target) -> Target:
    target_dict = dict(target.export())
    target_dict["keys"] = list(dict.fromkeys([*target_dict.get("keys", ()), "ascend"]))
    target_dict.setdefault("target_device_type", _kDLExtDev)
    return Target(target_dict)


def _make_ascend_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    keys = target_dict.get("keys", ())
    if isinstance(keys, str):
        keys = (keys,)
    target_dict["kind"] = "c"
    target_dict["keys"] = list(dict.fromkeys([*keys, "ascend"]))
    target_dict.setdefault("target_device_type", _kDLExtDev)
    return Target(target_dict)


def check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _detect_ascend_target() -> Target | None:
    if check_ascend_availability():
        return _make_ascend_target()
    return None


def normalize_ascend_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if "ascend" not in target.keys:
            return None
        return _with_ascend_key(target)

    if isinstance(target, dict):
        target_dict = dict(target)
        if target_dict.get("kind") == "ascend":
            try:
                return _make_ascend_target(target_dict)
            except Exception:
                return None

        try:
            parsed_target = Target(target_dict)
        except Exception:
            return None
        if "ascend" in parsed_target.keys:
            return _with_ascend_key(parsed_target)
        return None

    normalized_target = target.strip()
    if normalized_target == "c -keys=ascend":
        try:
            return _make_ascend_target()
        except Exception:
            return None
    if normalized_target != "ascend":
        return None

    try:
        return _make_ascend_target()
    except Exception:
        return None


def target_is_ascend(target: Target) -> bool:
    return _target_ffi_api().TargetIsAscend(target)


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
