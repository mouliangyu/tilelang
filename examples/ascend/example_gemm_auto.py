import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_ascend_postproc_callback


@register_ascend_postproc_callback
def my_ascend_postproc(code, target):
    return code
    return """
#include <tl_templates/ascend/gemm.h>
extern "C"
__global__ __mix__(1, 2) void main_kernel(__gm__ float* C, __gm__ bfloat16_t* W, __gm__ bfloat16_t* X) {
  int32_t bx = blockIdx.x;
  int32_t n_tile[1];
  int32_t m_tile[1];
  __ubuf__ float *temp = (__ubuf__ float *)0;
  __cc__ float *res = (__cc__ float *)0;
  __cbuf__ bfloat16_t *w_l1 = (__cbuf__ bfloat16_t *)0;
  if ASCEND_IS_AIC {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    for (int32_t out_tile = 0; out_tile < 32; ++out_tile) {
      bool __cond_0 = (out_tile < 28);
      if (__cond_0) {
        n_tile[0] = (((out_tile & 3) * 8) + (bx >> 2));
      }
      if (!__cond_0) {
        n_tile[0] = (((out_tile * 8) + (bx >> 2)) - 224);
      }
      if (__cond_0) {
        m_tile[0] = (((out_tile >> 2) * 4) + (bx & 3));
      }
      bool __cond_1 = (3 < (out_tile & 7));
      if (__cond_1) {
        n_tile[0] = (31 - n_tile[0]);
      }
      if (!__cond_0) {
        m_tile[0] = ((bx & 3) + 28);
      }
      int32_t m_tile_1 = m_tile[0];
      int32_t n_tile_1 = n_tile[0];
      // AscendC::PipeBarrier<pipe_t::PIPE_ALL>();
      for (int32_t kt = 0; kt < 32; ++kt) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((kt & 1));
        bisheng::cce::set_mte2_nz_para(uint64_t(1) | (uint64_t(1) << 16) | (uint64_t(256) << 32));
        bisheng::cce::copy_gm_to_cbuf_multi_nd2nz((__cbuf__ bfloat16_t*)(w_l1 + ((kt & 1) * 65536)), (&(W[((n_tile_1 * 2097152) + (kt * 256))])), 0, 16384, 0, 256, 256, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((kt & 1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(((kt & 1) + 2));
        bisheng::cce::set_mte2_nz_para(uint64_t(1) | (uint64_t(1) << 16) | (uint64_t(256) << 32));
        bisheng::cce::copy_gm_to_cbuf_multi_nd2nz((__cbuf__ bfloat16_t*)(w_l1 + (((kt & 1) * 65536) + 131072)), (&(X[((m_tile_1 * 2097152) + (kt * 256))])), 0, 16384, 0, 256, 256, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(((kt & 1) + 2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(((kt & 1) + 2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((kt & 1));
        ascend_gemm_l1<256, 256, 256, 64, true, bfloat16_t>((__cc__ float*)(res), (__cbuf__ bfloat16_t*)(w_l1 + (((kt & 1) * 65536) + 131072)), (__cbuf__ bfloat16_t*)(w_l1 + ((kt & 1) * 65536)), (kt == 0), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((kt & 1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(((kt & 1) + 2));
      }
      // AscendC::PipeBarrier<pipe_t::PIPE_ALL>();
      AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(18);
      AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(2);
      // AscendC::PipeBarrier<pipe_t::PIPE_ALL>();
      copy_matrix_cc_to_ub((&(temp[0])), (__cc__ float*)(res), 0, 256, 256, 256, 256, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      // AscendC::PipeBarrier<pipe_t::PIPE_ALL>();
      AscendC::CrossCoreSetFlag<4, PIPE_FIX>(1);
      AscendC::CrossCoreSetFlag<4, PIPE_FIX>(17);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
  }
  if ASCEND_IS_AIV {
    int32_t sid = get_subblockid();
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(((sid * 16) + 1));
    for (int32_t out_tile_1 = 0; out_tile_1 < 32; ++out_tile_1) {
      bool __cond_0_1 = (out_tile_1 < 28);
      if (__cond_0_1) {
        n_tile[0] = (((out_tile_1 & 3) * 8) + (bx >> 2));
      }
      if (!__cond_0_1) {
        n_tile[0] = (((out_tile_1 * 8) + (bx >> 2)) - 224);
      }
      if (__cond_0_1) {
        m_tile[0] = (((out_tile_1 >> 2) * 4) + (bx & 3));
      }
      bool __cond_1_1 = (3 < (out_tile_1 & 7));
      if (__cond_1_1) {
        n_tile[0] = (31 - n_tile[0]);
      }
      if (!__cond_0_1) {
        m_tile[0] = ((bx & 3) + 28);
      }
      int32_t m_tile_2 = m_tile[0];
      int32_t n_tile_2 = n_tile[0];
      AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>((sid * 16));
      bisheng::cce::copy_ubuf_to_gm_align_v2((__gm__ void*)((&(C[(((m_tile_2 * 2097152) + (n_tile_2 * 256)) + 1048576)]))), (__ubuf__ void*)((&(temp[0]))), 0, 128, 1024, 0, 32768, 1024);
      AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(((sid * 16) + 1));
    }
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(((sid * 16) + 1));
  }
}
"""


def gemm(M_DIM=8192, K_DIM=8192, N_DIM=8192, MIXED=True):
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = 256
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    TILES_PER_CORE = OUT_TILES // NUM_BLOCKS
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer((M_DIM, K_DIM), "bfloat16"),
        W: T.Buffer((N_DIM, K_DIM), "bfloat16"),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), "bfloat16")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "bfloat16")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")

            for out_tile in T.Serial(TILES_PER_CORE):
                tile_idx = out_tile * NUM_BLOCKS + bx
                # m_tile = tile_idx // N_TILES
                # n_tile = tile_idx % N_TILES
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)  # Cube
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)  # Cube
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))  # Cube
                if MIXED:
                    T.dual_copy(res, temp)  # Vector
                    T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])  # Vector
                    # T.copy(temp, C[m_tile * TILE_M + sid * (TILE_M // 2), n_tile * TILE_N]) # Vector
                    # T.copy(temp, C[m_tile * TILE_M + (TILE_M // 2), n_tile * TILE_N])  # Vector
                else:
                    T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N])  # Cube

    return main


def ref_program(x, w):
    """Reference implementation: C = A @ B"""
    return x.float() @ w.float().T


NUM_REPEATS = 500
M_DIM = 8192
K_DIM = 8192
N_DIM = 8192

if __name__ == "__main__":
    import torch

    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print("Compiling gemm kernel...")
    program = gemm(M_DIM, K_DIM, N_DIM)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Create test data
    x = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16, device=device)
    w = torch.randn(N_DIM, K_DIM, dtype=torch.bfloat16, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(x, w)
    torch.npu.synchronize()
    print("Done")

    # Verify
    expected = ref_program(x, w)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  {'PASS' if max_diff < 1e-2 else 'FAIL'}  rel={max_diff:.2e}")
    if not max_diff < 1e-2:
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")

    # Benchmark
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(NUM_REPEATS):
        c = kernel(x, w)
    t1.record()
    torch.npu.synchronize()

    ms = t0.elapsed_time(t1)
    flops = 2.0 * M_DIM * N_DIM * K_DIM * NUM_REPEATS
    ref = flops / (ms / 1e3) / 1e12
    print(f"{ms / 1e3:.1f}s total  |  {ms / NUM_REPEATS:.3f} ms/iter  |  {ref:.1f} TFLOPS")
