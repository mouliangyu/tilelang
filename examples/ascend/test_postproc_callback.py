"""pytest test for example_ascend_postproc_callback.py — verifies the
register_ascend_postproc_callback hook injects its marker into generated source."""

import tilelang

from example_ascend_postproc_callback import CUSTOM_MARKER, vector_add


def test_postproc_marker_in_source():
    kernel = tilelang.compile(vector_add(1024), target="ascend", out_idx=-1)
    source = kernel.get_kernel_source()
    assert CUSTOM_MARKER in source, "postproc callback did not inject CUSTOM_MARKER"


if __name__ == "__main__":
    test_postproc_marker_in_source()
    print("PASS: test_postproc_marker_in_source")
