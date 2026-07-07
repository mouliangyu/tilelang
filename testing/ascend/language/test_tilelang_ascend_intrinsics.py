import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def test_ascend_pipe_barrier():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_pipe_barrier("PIPE_ALL")
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::PipeBarrier<pipe_t::PIPE_ALL>();" in source


def test_ascend_set_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0)" in source
    assert "AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0)" in source


def test_ascend_threadfence():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_threadfence()
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "asc_threadfence();" in source


def test_ascend_cross_core_set_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8)" in source


def test_ascend_cross_core_wait_flag():
    @T.prim_func
    def func(A: T.Buffer((16,), "float32")):
        with T.Kernel(1) as _:
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 0x8)
            A[0] = T.float32(1)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)
    assert "AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(8)" in source


def test_ascend_simd_3510_intrinsics_codegen():
    @T.prim_func
    def func(
        C: T.Buffer((128,), "float32"),
        O: T.Buffer((64,), "int32"),
        H: T.Buffer((128,), "float16"),
        S: T.Buffer((128,), "int16"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((128,), "float32")
            c_ub = T.alloc_shared((128,), "float32")
            i_ub = T.alloc_shared((64,), "int32")
            o_ub = T.alloc_shared((64,), "int32")
            h_ub = T.alloc_shared((128,), "float16")
            s_ub = T.alloc_shared((128,), "int16")

            with T.SimdVF():
                full = T.simd.pset(32)
                half = T.simd.pset(16)
                p8 = T.simd.pge(8, "PAT_VL128")
                p16 = T.simd.pge(16, "PAT_VL64")
                p32 = T.simd.pge(32, "PAT_VL16")
                pred = T.simd.pand(p32, full, full)
                pred = T.simd.por(pred, p8, full)
                pred = T.simd.pxor(pred, p16, full)
                pred = T.simd.pnot(pred, full)
                pred = T.simd.psel(pred, full, p32)

                a = T.simd.vld(a_ub[0])
                b = T.simd.vld(a_ub[64])
                a = T.simd.vadd(a, b, pred)
                a = T.simd.vsub(a, T.simd.vdup(T.float32(1), "float32", pred), pred)
                a = T.simd.vmul(a, T.simd.vdup(T.float32(2), "float32", pred), pred)
                a = T.simd.vdiv(a, T.simd.vdup(T.float32(2), "float32", pred), pred)
                a = T.simd.vmax(a, b, pred)
                a = T.simd.vmin(a, b, pred)
                a = T.simd.vexp(a, pred)
                a = T.simd.vln(a, pred)
                a = T.simd.vabs(a, pred)
                a = T.simd.vsqrt(a, pred)
                a = T.simd.vneg(a, pred)
                a = T.simd.vrelu(a, pred)
                a = T.simd.vmaxs(a, T.float32(1), pred)
                a = T.simd.vmins(a, T.float32(3), pred)
                a = T.simd.vmuls(a, T.float32(2), pred)
                a = T.simd.vadds(a, T.float32(1), pred)
                a = T.simd.vsel(
                    a,
                    b,
                    T.simd.pand(
                        T.simd.vcmp(a, b, pred, "ge"),
                        T.simd.vcmps(a, T.float32(0), pred, "gt"),
                        pred,
                    ),
                )
                b = T.simd.vexpdif(a, b, pred)
                b = T.simd.vadd(b, T.simd.vdupv(T.simd.vcmax(a, pred), pred), pred)
                b = T.simd.vadd(b, T.simd.vcadd(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcmin(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgadd(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgmax(a, pred), pred)
                b = T.simd.vadd(b, T.simd.vcgmin(a, pred), pred)
                a, b = T.simd.vintlv(a, b)
                a, b = T.simd.vdintlv(a, b)
                a = T.simd.vadd(a, b, pred)
                a = T.simd.vadd(
                    a,
                    T.simd.vgatherb(a_ub[0], T.simd.vci(T.uint32(0), "uint32"), pred),
                    pred,
                )
                a = T.simd.vadd(
                    a,
                    T.simd.vgather2(a_ub[0], T.simd.vci(T.uint32(0), "uint32"), pred),
                    pred,
                )
                a = T.simd.vadd(
                    a,
                    T.simd.vcvt(
                        T.simd.vld(h_ub[0], dist="UNPK_B16"),
                        "float32",
                        half,
                        part=0,
                    ),
                    pred,
                )
                b = T.simd.vsqz(a, pred)
                T.simd.vsts(c_ub[0], a, pred)
                T.simd.vsts(c_ub[64], b, pred)
                b = T.simd.vcvt(a, "float16", pred)
                T.simd.vsts(h_ub[0], b, half, dist="PK_B32")
                b = T.simd.vpack(a)
                T.simd.vsts(h_ub[64], b, half, dist="PK_B32")
                T.simd.vsstb(a, c_ub[0], T.int32(1), pred)
                T.simd.mem_bar("VST_VLD")

                a = T.simd.vld(i_ub[0])
                b = T.simd.vdup(T.int32(1), "int32", pred)
                a = T.simd.vand(a, T.simd.vci(T.int32(0), "int32"), pred)
                a = T.simd.vor(a, b, pred)
                a = T.simd.vshl(a, b, pred)
                a = T.simd.vshr(a, b, pred)
                a = T.simd.vselr(a, T.simd.vci(T.uint32(0), "uint32"))
                a = T.simd.vshls(a, T.int32(1), pred)
                a = T.simd.vshrs(a, T.int32(1), pred)
                T.simd.vsts(o_ub[0], a, pred)
                b = T.simd.vpack(a)
                T.simd.vsts(s_ub[0], b, half, dist="NORM_B16")
                a = T.simd.vcadd(b, half)
                T.simd.vsts(o_ub[0], a, pred)

            T.copy(c_ub, C)
            T.copy(o_ub, O)
            T.copy(h_ub, H)
            T.copy(s_ub, S)

    artifact = lower(func, target="ascend")
    source = artifact.kernel_source
    print(source)


def test_copy_pad_value_gm_to_ub():
    # 30 float32 = 120B per row (not 32B-aligned); dst UB is over-allocated to
    # 32 cols so each row can be right-padded up to 128B.
    @T.prim_func
    def func(A: T.Buffer((4, 30), "float32"), B: T.Buffer((4, 32), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((4, 32), "float32")
            T.copy(A[:, :], a_ub[:, :30], pad_value=0.0)
            T.copy(a_ub[:, :], B[:, :])

    source = lower(func, target="ascend").kernel_source
    print(source)
    pad_idx = source.find("set_mov_pad_val")
    copy_idx = source.find("copy_gm_to_ubuf_align_v2")
    assert pad_idx != -1 and copy_idx != -1
    # The pad-register write must precede the padded copy.
    assert pad_idx < copy_idx
    # Padded copy: rightPadding=2 (128B-120B)/4, dataSelect=1, dst_stride=128.
    assert "0, 4, 120, 0, 2, 1," in source


if __name__ == "__main__":
    tilelang.testing.main()
