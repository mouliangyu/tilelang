import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def test_ascend_dcache_bypass_scalar_write():
    """Buffer with scalar write → scalar read/write emits bypass;
    pure-read buffer uses normal dcache access."""

    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),  # has scalar write → bypass
        ro_buf: T.Buffer((16,), "float32"),  # pure-read → dcache
    ):
        with T.Kernel(1) as _:
            rw_buf[0] = T.float32(1)  # scalar store → marks rw_buf as write
            a = rw_buf[0]  # scalar load → bypass
            b = ro_buf[0]  # scalar load → dcache (ro_buf has NO writes)
            rw_buf[0] = a + b  # scalar store → bypass (rw_buf already in write set)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)

    # rw_buf has scalar write → read must bypass
    assert "tl::read_gm_bypass_dcache" in source, "write buffer scalar read should emit tl::read_gm_bypass_dcache"
    # rw_buf has scalar write → write must bypass
    assert "tl::write_gm_bypass_dcache" in source, "write buffer scalar write should emit tl::write_gm_bypass_dcache"
    # bypass header must be included
    assert "#include <tl_templates/ascend/dcache_bypass.h>" in source, "bypass header must be included when bypass intrinsics are used"

    # Every bypass call must reference rw_buf, never ro_buf
    import re

    bypass_calls = re.findall(r"tl::(?:read|write)_gm_bypass_dcache\([^;]+\)", source)
    for call in bypass_calls:
        assert "ro_buf" not in call, f"ro_buf must NOT use bypass: {call}"
        assert "rw_buf" in call, f"bypass call must reference rw_buf: {call}"

    # ro_buf uses normal array access (dcache), not wrapped in bypass
    assert "ro_buf[0]" in source, "ro_buf must use normal array access (dcache)"


def test_ascend_dcache_bypass_mte_write():
    """Buffer with MTE copy → scalar read emits bypass; pure-read buffer uses dcache."""

    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),  # MTE copy destination → bypass
        ro_buf: T.Buffer((16,), "float32"),  # pure-read → dcache
    ):
        with T.Kernel(1) as _:
            ub = T.alloc_shared((16,), "float32")
            T.copy(ro_buf, ub)  # MTE read from ro_buf
            a = rw_buf[0]  # scalar read from rw_buf → should become bypass
            ub[0] = a + T.float32(1)
            T.copy(ub, rw_buf)  # MTE write to rw_buf → marks it as write

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)

    # rw_buf has an MTE copy (ub → rw_buf), so its scalar read must bypass.
    # Note: the scalar read `a = rw_buf[0]` happens BEFORE the MTE copy,
    # but the TIR pass does a pre-scan over the whole body, so rw_buf is
    # detected as a write buffer regardless of order.
    assert "tl::read_gm_bypass_dcache" in source, "MTE-write buffer scalar read should emit tl::read_gm_bypass_dcache"

    # Every bypass call must reference rw_buf, never ro_buf
    import re

    bypass_calls = re.findall(r"tl::(?:read|write)_gm_bypass_dcache\([^;]+\)", source)
    for call in bypass_calls:
        assert "ro_buf" not in call, f"ro_buf must NOT use bypass: {call}"
        assert "rw_buf" in call, f"bypass call must reference rw_buf: {call}"


if __name__ == "__main__":
    tilelang.testing.main()
