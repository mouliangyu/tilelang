"""pytest test for Ascend NPU do_bench event timing."""

import torch
import tilelang

from example_gemm import gemm, ref_program
from tilelang.profiler import do_bench


def test_do_bench_event():
    M, K, N = 8192, 256, 256
    kernel = tilelang.compile(
        gemm(M, K, N),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )

    device = torch.device("npu")
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()

    expected = ref_program(x, w)
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-2, f"max_diff={max_diff:.2e}"

    kernel_latency = kernel.get_profiler().do_bench(backend="event")
    ref_latency = do_bench(lambda: ref_program(x, w), backend="event")

    assert isinstance(kernel_latency, float)
    assert isinstance(ref_latency, float)
    assert kernel_latency >= 0.0
    assert ref_latency >= 0.0
    print(kernel_latency)
    print(ref_latency)


if __name__ == "__main__":
    test_do_bench_event()
    print("PASS: test_do_bench_event")
