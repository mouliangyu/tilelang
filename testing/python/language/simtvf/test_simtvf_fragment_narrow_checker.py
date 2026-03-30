import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.utils.target import determine_target


def _run_to_optimize(func):
    target = determine_target("ascend", return_object=True)
    mod = tvm.IRModule({"main": func})
    with target:
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)
        return OptimizeForTarget(lowered, target)


def test_fragment_narrow_checker_rejects_narrowed_fragment():
    @T.prim_func
    def narrowed(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1, is_ascend=True):
            frag = T.alloc_fragment((16,), "float32")
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    frag[i] = A[i]
                    B[i] = frag[i] + T.float32(1)

    with pytest.raises(ValueError, match="outside any T.SimtVF region"):
        _run_to_optimize(narrowed)


def test_fragment_narrow_checker_rejects_fragment_leak():
    @T.prim_func
    def leaked(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1, is_ascend=True):
            frag = T.alloc_fragment((16,), "float32")
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    frag[i] = A[i]
            B[0] = frag[0]

    with pytest.raises(ValueError, match="outside any T.SimtVF region"):
        _run_to_optimize(leaked)
