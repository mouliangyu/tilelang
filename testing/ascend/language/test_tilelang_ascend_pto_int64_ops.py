"""PTO codegen for 64-bit scalar edge cases (cann/pto-as issue #14).

A full-width right shift and a 64-bit atomic add both used to produce
unusable PTO source: the shift lowered to a division by a wrapped constant,
and the atomic add was rejected outright.  The guarded scalar weight load is
the shape that used to be emitted as an assignment whose right-hand side was
an if-statement, which is not valid Python.
"""

import ast

import pytest

import tilelang.language as T
from tilelang.engine.lower import lower


def _shift_right_full_width():
    @T.prim_func
    def main(x: T.Tensor((1,), "int64"), y: T.Tensor((1,), "int64")):
        with T.Kernel(1), T.SimtVF(threads=1):
            y[0] = x[0] >> 63

    return main


@pytest.mark.pto
def test_pto_shift_right_full_width_keeps_sign():
    """A shift by bits - 1 must not divide by the wrapped power of two."""
    source = lower(_shift_right_full_width(), target="pto").kernel_source

    assert "-9223372036854775808" not in source
    assert "scalar.select(" in source


def _shift_right_31():
    @T.prim_func
    def main(x: T.Tensor((1,), "int32"), y: T.Tensor((1,), "int32")):
        with T.Kernel(1), T.SimtVF(threads=1):
            y[0] = x[0] >> 31

    return main


@pytest.mark.pto
def test_pto_shift_right_full_width_int32_keeps_sign():
    source = lower(_shift_right_31(), target="pto").kernel_source

    assert "-2147483648" not in source
    assert "scalar.select(" in source


def _atomic_add_int64():
    @T.prim_func
    def main(counter: T.Tensor((1,), "int64"), prev: T.Tensor((1,), "int64")):
        with T.Kernel(1), T.SimtVF(threads=1):
            prev[0] = T.atomic_add(counter[0], 1, return_prev=True)

    return main


@pytest.mark.pto
def test_pto_atomic_add_int64():
    """ASC counts clamps with a 64-bit atomic add, so PTO must emit it."""
    source = lower(_atomic_add_int64(), target="pto").kernel_source

    assert "pto.atomic_add(" in source
    assert 'signedness="signed"' in source


def _guarded_weight_load():
    extent = T.dynamic("num_expanded_tokens")

    @T.prim_func
    def main(
        weights: T.Tensor((extent,), "float32"),
        idx: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1), T.SimtVF(threads=1):
            token_id = idx[0]
            out[0] = weights[token_id] * 2.5

    return main


@pytest.mark.pto
def test_pto_guarded_scalar_load_is_parseable_python():
    """The emitted PTODSL must stay valid Python for guarded scalar loads."""
    source = lower(_guarded_weight_load(), target="pto").kernel_source

    ast.parse(source)
