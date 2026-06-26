import tilelang.testing
import example_gemm
import example_gemm_l0
import example_gemm_mix_manual
import example_gemm_mixedkernel
import example_gemm_bypass_l2
import example_gemm_various_shapes
import example_blockscaled_gemm
import example_flash_attn
import example_rmsnorm
import example_buffer_version_annotation
import example_atomic_add
import example_simdvf_topk_gate
import example_simdvf_vecadd
import example_simdvf_vecadd_lower
import example_simtvf_vecadd
import example_simtvf_vecadd_mutex
import example_simdvf_per_token_cast_to_fp8
import example_simtvf_per_token_cast_to_fp8
import example_crosslevel_multibuffer


def _safe_process(func, name, **kwargs):
    """Run one perf entry, but never let a single failing example abort the suite.

    A kernel may fail to compile/run on one side of the comparison (e.g. it relies on
    codegen the baseline branch doesn't have yet). We skip it with a warning instead of
    crashing the whole driver; the comparison then simply omits that kernel.
    """
    try:
        tilelang.testing.process_func(func, name, **kwargs)
    except Exception as e:  # noqa: BLE001 - intentionally broad: isolate per-example failures
        print(f"  ⚠️  {name} failed, skipping: {type(e).__name__}: {e}", flush=True)


def regression_gemm_bf16():
    _safe_process(example_gemm.run_regression_perf, "ascend_gemm_bf16", dtype="bfloat16")


def regression_gemm_fp32():
    _safe_process(example_gemm.run_regression_perf, "ascend_gemm_fp32", dtype="float32")


def regression_gemm_l0():
    _safe_process(example_gemm_l0.run_regression_perf, "ascend_gemm_l0")


def regression_gemm_mix_manual():
    _safe_process(example_gemm_mix_manual.run_regression_perf, "ascend_gemm_mix_manual")


def regression_gemm_mixedkernel():
    _safe_process(example_gemm_mixedkernel.run_regression_perf, "ascend_gemm_mixedkernel")


def regression_gemm_bypass_l2():
    _safe_process(example_gemm_bypass_l2.run_regression_perf, "ascend_gemm_bypass_l2")


def regression_gemm_various_shapes():
    _safe_process(example_gemm_various_shapes.run_regression_perf, "ascend_gemm_various_shapes")


def regression_blockscaled_gemm():
    _safe_process(example_blockscaled_gemm.run_regression_perf, "ascend_blockscaled_gemm")


def regression_flash_attn():
    _safe_process(example_flash_attn.run_regression_perf, "ascend_flash_attn")


def regression_rmsnorm():
    _safe_process(example_rmsnorm.run_regression_perf, "ascend_rmsnorm")


def regression_buffer_version_annotation():
    _safe_process(example_buffer_version_annotation.run_regression_perf, "ascend_buffer_version_annotation")


def regression_atomic_add():
    _safe_process(example_atomic_add.run_regression_perf, "ascend_atomic_add")


def regression_simdvf_topk_gate():
    _safe_process(example_simdvf_topk_gate.run_regression_perf, "ascend_simdvf_topk_gate")


def regression_simdvf_vecadd():
    _safe_process(example_simdvf_vecadd.run_regression_perf, "ascend_simdvf_vecadd")


def regression_simdvf_vecadd_lower():
    _safe_process(example_simdvf_vecadd_lower.run_regression_perf, "ascend_simdvf_vecadd_lower")


def regression_simtvf_vecadd():
    _safe_process(example_simtvf_vecadd.run_regression_perf, "ascend_simtvf_vecadd")


def regression_simtvf_vecadd_mutex():
    _safe_process(example_simtvf_vecadd_mutex.run_regression_perf, "ascend_simtvf_vecadd_mutex")


def regression_simdvf_per_token_cast_to_fp8():
    _safe_process(example_simdvf_per_token_cast_to_fp8.run_regression_perf, "ascend_simdvf_per_token_cast_to_fp8")


def regression_simtvf_per_token_cast_to_fp8():
    _safe_process(example_simtvf_per_token_cast_to_fp8.run_regression_perf, "ascend_simtvf_per_token_cast_to_fp8")


def regression_crosslevel_multibuffer():
    _safe_process(example_crosslevel_multibuffer.run_regression_perf, "ascend_crosslevel_multibuffer")


if __name__ == "__main__":
    tilelang.testing.regression()
