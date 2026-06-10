#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../target/utils.h"
#include "./ir_structure.h"
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::String;

// Latency estimator for GPU with configurable hardware parameters
class LatencyEstimator {
public:
  // GPU latency parameters (H100 defaults, in cycles)
  struct GPUParams {
    // Base latencies
    int64_t global_memory_read = 1000; // Global memory read latency
    int64_t global_memory_write =
        800; // Global memory write latency (usually lower)
    int64_t shared_memory_read = 26;  // Shared memory read latency
    int64_t shared_memory_write = 26; // Shared memory write latency
    int64_t register_access = 1;      // Register access latency
    int64_t cuda_core_operation =
        4;                       // Basic CUDA core operation (add, mul, etc.)
    int64_t tma_operation = 100; // TMA operation latency
    int64_t tmem_read = 0;       // Tensor memory read latency (0 = not present)
    int64_t tmem_write = 0;     // Tensor memory write latency (0 = not present)
    int64_t tmem_bandwidth = 0; // Tensor memory bandwidth in bytes/cycle

    // Tensor Core shape-aware parameters
    int64_t tensor_core_base_latency =
        32; // Base latency for tensor core operation
    int64_t tensor_core_per_element_latency =
        1; // Additional latency per matrix element
    int64_t tensor_core_throughput =
        4096; // Number of tensor core operations per cycle (throughput)
    int64_t wgmma_base_latency = 40;    // Base latency for WGMMA operation
    int64_t wgmma_per_tile_latency = 2; // Additional latency per tile

    // Tensor Core II (Initiation Interval) parameters
    int64_t tensor_core_min_ii = 4; // Minimum II for tensor core operations
    int64_t tensor_core_ii_per_tile = 1;        // Additional II per WGMMA tile
    int64_t tensor_core_max_parallel_tiles = 8; // Max parallel tiles per SM

    // Bandwidth parameters (bytes per cycle)
    int64_t global_memory_bandwidth = 64;  // bytes per cycle
    int64_t shared_memory_bandwidth = 512; // bytes per cycle

    // Pipeline initiation capabilities
    int64_t max_memory_ops_per_cycle =
        1; // Max memory ops that can start per cycle

    // Different operation throughputs (operations per cycle)
    int64_t add_throughput = 4; // Addition throughput (ops/cycle)
    int64_t sub_throughput = 4; // Subtraction throughput (ops/cycle)
    int64_t mul_throughput = 4; // Multiplication throughput (ops/cycle)
    int64_t div_throughput =
        1; // Division throughput (ops/cycle, usually slower)
    int64_t mod_throughput = 1;     // Modulo throughput (ops/cycle)
    int64_t min_max_throughput = 4; // Min/Max throughput (ops/cycle)
    int64_t cmp_throughput = 4;     // Comparison throughput (ops/cycle)
    int64_t logic_throughput = 4;   // Logical operations throughput (ops/cycle)
    int64_t bitwise_throughput = 4; // Bitwise operations throughput (ops/cycle)
    int64_t shift_throughput = 4;   // Shift operations throughput (ops/cycle)
    int64_t special_func_throughput =
        1; // Special functions throughput (ops/cycle, exp2, log2, sin, cos,
           // etc.)

    // ------------------------------------------------------------------
    // Ascend-only fields (ignored on CUDA/HIP targets, default 0 = "unused").
    // Populated by AscendParams below. The Estimate() body does not consume
    // them yet — they exist so the consumer pass (latency / II calculation)
    // can be rewritten incrementally without re-touching every target.
    // ------------------------------------------------------------------

    // Per-pipe issue latency (cycles) and steady-state bandwidth (bytes/cycle).
    // These cover Ascend's physically-distinct DMA channels; the legacy
    // global_memory_* / shared_memory_* fields stay as a coarse fallback for
    // code paths that have not been refactored yet.
    //
    // The `mte*_bandwidth` fields are the *headline* per-pipe number (= the
    // fastest sub-path that pipe drives, typically L0A/L0B for MTE1 and
    // UB<->GM for MTE2/MTE3). When a task's actual src/dst scope is known,
    // the more precise per-path fields below should be preferred.
    int64_t mte1_base_latency = 0;    // L1 -> L0A/L0B issue latency
    int64_t mte1_bandwidth = 0;       // headline (L0A/L0B path)
    int64_t mte2_base_latency = 0;    // GM -> UB / GM -> L1 issue latency
    int64_t mte2_bandwidth = 0;       // headline (AIC GM->L1 path)
    int64_t mte3_base_latency = 0;    // UB -> GM / UB -> L1 issue latency
    int64_t mte3_bandwidth = 0;       // headline (UB->GM path, per AIV)
    int64_t fixpipe_base_latency = 0; // L0C -> UB / L0C -> GM issue latency
    int64_t fixpipe_bandwidth = 0;    // Fixpipe egress bandwidth

    // Per-destination MTE1 paths (AIC). All four destinations are independent
    // physical ports on the L1 Buffer; aggregate read bandwidth when all
    // ports fire concurrently is the sum.
    int64_t l1_to_l0a_bandwidth = 0;    // L1 -> L0A (MTE1)
    int64_t l1_to_l0b_bandwidth = 0;    // L1 -> L0B (MTE1)
    int64_t l1_to_bt_bandwidth = 0;     // L1 -> BT (bias table, MTE1)
    int64_t l1_to_fp_buf_bandwidth = 0; // L1 -> FP Buf (MTE1)

    // MTE2 sub-paths. AIC and AIV each have an MTE2 with different bandwidths
    // against HBM/L2, and egress HBM through separate ports so their aggregate
    // is the sum.
    int64_t aic_mte2_to_l1_bandwidth = 0; // AIC: GM -> L1 Buffer
    int64_t aiv_mte2_to_ub_bandwidth = 0; // AIV: GM -> UB (per AIV)

    // Two-segment (knee) MTE latency model. A single MTE copy is split by the
    // LSU into ceil(bytes/512) micro-ops (the 512B cacheline granularity).
    // Below the split threshold (`knee`) it is one outstanding op with a flat
    // latency; above it the request saturates LSU outstanding and grows
    // linearly. So the cost switches at the knee:
    //   lat(bytes) = (bytes <= knee) ? flat
    //                                : seg_base + bytes / seg_bw
    // A path with seg_bw == 0 falls back to the legacy single-segment
    // (base + bytes / BandwidthForPath) — i.e. unmeasured paths are unchanged.
    // NOTE: the 512B..~16KB mid-range (micro-op split ramp, a high plateau
    // ~900-1000 cyc that sits ABOVE both the flat and the large-packet line)
    // is under-estimated by ~20% — acceptable because real kernels hit either
    // small scalar copies (<=knee) or large bulk transfers (>=64KB), not the
    // mid-range. Source: cannsim trace_tools size sweep (single
    // MOV_SRC_TO_DST_ALIGNv2 dur, 16B..256KB) cross-checked against the
    // "ISSUE QUEUE and LSU Outstanding" doc (512B split, ~105 B/cyc saturated
    // read after NoC throttling).
    int64_t mte2_gm_to_ub_knee = 0;     // micro-op split threshold (bytes)
    int64_t mte2_gm_to_ub_flat = 0;     // <=knee single-op latency (cyc)
    int64_t mte2_gm_to_ub_seg_base = 0; // large-packet line intercept (cyc)
    int64_t mte2_gm_to_ub_seg_bw = 0;   // large-packet slope (B/cyc)

    // MTE3 UB->GM two-segment (knee) fit. Same structure/rationale as the
    // GM->UB params above (512B micro-op split, ~103 B/cyc saturated write).
    int64_t mte3_ub_to_gm_knee = 0;
    int64_t mte3_ub_to_gm_flat = 0;
    int64_t mte3_ub_to_gm_seg_base = 0;
    int64_t mte3_ub_to_gm_seg_bw = 0;

    // MTE2 GM->L1 (AIC) two-segment fit — DISTINCT from GM->UB (AIV). This is
    // the path GEMM's x_l1/w_l1 loads use via copy_gm_to_cbuf_multi_nd2nz. The
    // AIC MTE2 has a deeper issue queue (32 vs AIV's 16) AND pays nd2nz format
    // conversion + splits each copy into 2-4 MTE2 ops, so the curve is shaped
    // differently: a much wider flat region (~986 cyc up to ~8KB, not 512B)
    // and a slower saturated rate (~98 B/cyc vs AIV's 105). Measured via
    // cannsim trace_tools nd2nz sweep (n_rows x 256 bf16, 1KB..256KB); fit over
    // the GEMM working range 32KB..128KB gives 837 + bytes/98 (0% error at the
    // 128KB tile GEMM actually uses). The legacy base=634/bw=256
    // under-estimated this path by ~47% at 128KB.
    int64_t mte2_gm_to_l1_knee = 0;
    int64_t mte2_gm_to_l1_flat = 0;
    int64_t mte2_gm_to_l1_seg_base = 0;
    int64_t mte2_gm_to_l1_seg_bw = 0;

    // MTE3 sub-paths (AIV-initiated, three independent destinations).
    int64_t aiv_mte3_to_gm_bandwidth = 0;         // UB -> Out (per AIV)
    int64_t aiv_mte3_to_aic_l1_bandwidth = 0;     // UB -> AIC L1 (per AIV)
    int64_t aiv_mte3_to_aic_fp_buf_bandwidth = 0; // UB -> AIC FP Buf (per AIV)

    // VRF -> VALU consumption rate on AIV (bytes/cycle). dtype-agnostic — to
    // convert into ops/cycle, divide by element size (fp32 = 4, fp16/bf16 = 2,
    // int8 = 1).
    int64_t valu_bandwidth = 0;

    // Scalar L1 D-Cache <-> Out bandwidth (separate from the L1 Buffer above;
    // both AIC and AIV have this same 32 B/cycle path).
    int64_t scalar_dcache_bandwidth = 0;

    // On-chip buffer bandwidths (bytes/cycle). UB bandwidth is the legacy
    // shared_memory_bandwidth; these split L1/L0A/L0B/L0C off as their own
    // ports. Note these are *port* bandwidths (fill or drain rate from the
    // associated pipe), not internal banking limits.
    int64_t l1_bandwidth = 0;  // L1 Buffer access bandwidth (aggregate)
    int64_t l0a_bandwidth = 0; // L0A fill rate from L1 via MTE1
    int64_t l0b_bandwidth = 0; // L0B fill rate from L1 via MTE1
    int64_t l0c_bandwidth = 0; // L0C drain rate via FixPipe

    // Cube unit (single mad) shape — typically the smallest mad granularity
    // emitted by codegen (e.g. 16x16x16 fp16 on dav-3510). Used to refine the
    // tile-based II formula that is currently commented out around :271-315.
    int64_t cube_unit_m = 0;
    int64_t cube_unit_k = 0;
    int64_t cube_unit_n = 0;

    // Per-dtype Cube throughput (MACs/cycle). tensor_core_throughput is the
    // generic fallback; these let the estimator pick the right number once
    // TaskNode carries dtype info.
    int64_t cube_throughput_fp16 = 0;
    int64_t cube_throughput_bf16 = 0;
    int64_t cube_throughput_fp32 = 0;

    // Core clocks (Hz). Bandwidth fields above are bytes/cycle at the
    // associated pipe's clock; convert via B/s = B/cycle * Hz. Two distinct
    // clocks because AIC (Cube) and AIV (Vector) run independently.
    double aic_frequency_hz = 0.0;
    double aiv_frequency_hz = 0.0;

    // HBM / L2 / on-chip memory capacities (bytes) and raw bandwidths.
    int64_t hbm_bandwidth = 0;     // bytes/cycle, per AI Core's share
    int64_t hbm_latency = 0;       // cycles, raw DRAM access
    int64_t l2_capacity_bytes = 0; // shared L2 size
    int64_t l2_bandwidth = 0;      // bytes/cycle
    int64_t l2_hit_latency = 0;    // cycles
    int64_t l1_capacity_bytes = 0; // AIC L1 Buffer size
    int64_t ub_capacity_bytes = 0; // AIV Unified Buffer size
    int64_t l0a_capacity_bytes = 0;
    int64_t l0b_capacity_bytes = 0;
    int64_t l0c_capacity_bytes = 0;
    int64_t bt_capacity_bytes = 0;        // Bias Table
    int64_t fp_buf_capacity_bytes = 0;    // FP Buffer (AIC)
    int64_t ss_buffer_capacity_bytes = 0; // SSBuffer (scalar)

    // Per-pipe issue queue depth (max in-flight ops the pipe can hold). Used
    // to bound effective II under back-to-back issue.
    int64_t mte1_queue_depth = 0;
    int64_t mte2_queue_depth = 0;
    int64_t mte3_queue_depth = 0;
    int64_t cube_queue_depth = 0;
    int64_t fixpipe_queue_depth = 0;
    int64_t vector_queue_depth = 0;
    int64_t scalar_queue_depth = 0;

    // Cube MMAD pipeline depth (cycles between consecutive MMAD issues to
    // reach steady state). unit_flag enables back-to-back mads without
    // explicit set/wait — depth bounds how many can be in flight.
    int64_t cube_pipeline_depth = 0;
    int64_t cube_unit_flag_depth = 0;

    // BT / FP buffer line depths (number of independent banks/lines).
    int64_t bt_buffer_depth = 0;
    int64_t fp_buffer_depth = 0;

    // Additional Cube dtype throughputs (MACs/cycle). 0 means "use the
    // generic tensor_core_throughput fallback".
    int64_t cube_throughput_fp8 = 0;
    int64_t cube_throughput_mxfp8 = 0;
    int64_t cube_throughput_mxfp4 = 0;
    int64_t cube_throughput_int8 = 0;
    int64_t cube_throughput_int4 = 0;

    // Per-dtype VALU throughput (ops/cycle). When 0, derive from
    // valu_bandwidth / dtype.bytes(). These let the estimator account for
    // narrow-dtype speedups (fp16 ≈ 2× fp32 on most lanes).
    int64_t valu_throughput_fp32 = 0;
    int64_t valu_throughput_fp16 = 0;
    int64_t valu_throughput_bf16 = 0;
    int64_t valu_throughput_int32 = 0;
    int64_t valu_throughput_int16 = 0;
    int64_t valu_throughput_int8 = 0;

    // Special function throughput on AIV (exp/ln/sqrt/rsqrt). Separate from
    // the generic special_func_throughput so we can model per-instr cost.
    int64_t valu_special_func_throughput = 0;

    // Scalar pipe op throughput. Distinct from cuda_core_operation because
    // Ascend's S pipe has its own issue width and ALU shape.
    int64_t scalar_op_throughput = 0;

    // Front-end / control-flow penalties on the scalar pipe.
    int64_t branch_mispredict_cycles = 0;
    int64_t icache_miss_cycles = 0;
    int64_t scalar_load_use_cycles = 0;

    // MTE descriptor issue overhead (cycles charged once per copy command on
    // top of the bandwidth-limited bulk transfer time).
    int64_t mte_descriptor_cycles = 0;
  };

  // Ascend NPU parameters (dav-3510 / Ascend 950 / A5).
  //
  // Every field is explicitly assigned `= 0`. Values must be measured /
  // confirmed from the architecture document before the estimator is
  // trustworthy on Ascend. The formula bodies in EstimateAscend() are
  // already written against these fields — once any field gets a real
  // number, the formulas immediately consume it.
  //
  // Convention: bandwidth = bytes/cycle (at the pipe's clock), latency =
  // cycles. Cube/Vector throughput = ops/cycle (MACs/cycle for Cube).
  struct AscendParams : GPUParams {
    AscendParams() {
      // ================================================================
      // Coarse legacy fields. The Ascend estimator does NOT consume these
      // (it routes through the per-path fields below), so they stay at 0
      // to avoid leaking H100 numbers via the generic Estimate() if we
      // ever fall through to it.
      // ================================================================
      global_memory_read = 0;
      global_memory_write = 0;
      shared_memory_read = 0;
      shared_memory_write = 0;
      register_access = 0;
      cuda_core_operation = 0;
      tma_operation = 0;
      tmem_read = 0;
      tmem_write = 0;
      tmem_bandwidth = 0;
      global_memory_bandwidth = 0;
      shared_memory_bandwidth = 0;
      max_memory_ops_per_cycle = 0;

      // ================================================================
      // Cube (MMAD) — generic / legacy fields.
      // tensor_core_throughput is the FALLBACK consulted by
      // CubeThroughputFor() when the dtype-specific field is 0.
      // We seed it with the FP16/BF16 number (8192 MACs/cycle) since
      // that is the most common operand on Ascend GEMMs.
      // ================================================================
      tensor_core_base_latency =
          8; // Cube MMAD issue → L0C-ready latency (cycles).
             // Measured: cubecore trace shows MMAD 128×128×128 fp16 =
             // 524 cyc total. At 8192 MACs/cyc, 4.2M MACs → 512 cyc
             // bandwidth + ~12 cyc base ≈ 524 total. Base of 8 cyc
             // is the dispatch overhead for a single minimal mad.
      tensor_core_per_element_latency = 0;
      tensor_core_throughput = 8192; // = FP16/BF16 (16x16x16x2)
      tensor_core_min_ii = 1;        // single MMAD per cycle (back-to-back)
      tensor_core_ii_per_tile = 0;
      tensor_core_max_parallel_tiles = 0;
      wgmma_base_latency = 0; // N/A on Ascend
      wgmma_per_tile_latency = 0;

      // ================================================================
      // VALU throughputs (ops/cycle per AIV). Measured via cannsim Cube
      // kernel trace (AIV0/AIV1 RVEC per-instruction timing, n=72 events):
      //   RV_PLT (predicated load):  5.0 cyc/instr (n=32, σ=0.0)
      //   RV_VST (vector store):     8.0 cyc/instr (n=32, σ=0.0)
      //   RV_PSET (predicate set):   5.0 cyc/instr (n=2)
      //   RV_VDUPS (scalar dup):     5.0 cyc/instr (n=2)
      // A 256-lane VALU instruction processes 64 fp32 elements.
      // Peak measured fp32 throughput =
      //   64 elems / 5.0 cyc × 4 concurrent slices ≈ 51 ops/cyc.
      // fp16 throughput ≈ 2× fp32 = 102 ops/cyc (wider per-element).
      // Div/special-func use iterative approximation; set conservatively.
      add_throughput = 51;     // measured: RV_PLT 5.0 cyc × 64 fp32 × 4 slices
      sub_throughput = 51;     // same functional unit as add
      mul_throughput = 51;     // same functional unit as add
      div_throughput = 13;     // ~4× slower (iterative Newton-Raphson)
      mod_throughput = 13;     // ~4× slower
      min_max_throughput = 51; // same functional unit as add
      cmp_throughput = 51;     // same functional unit
      logic_throughput = 51;   // bitwise ops on same VALU
      bitwise_throughput = 51; // bitwise ops on same VALU
      shift_throughput = 51;   // shift ops on same VALU
      special_func_throughput = 8; // exp/sqrt/rsqrt: iterative ~6-8× slower

      // ================================================================
      // Per-pipe headline bandwidths (B/cycle).
      // Source: A5 PDF p.4 davinci data path table + p.5 AIC diagram.
      // The "headline" number is the fastest sub-path the pipe drives;
      // EstimateAscend uses BandwidthForPath() to pick the precise
      // per-port number when scope info is available.
      //   MTE1 headline = L1->L0A/L0B = 256
      //   MTE2 headline = AIC GM->L1  = 256
      //   MTE3 headline = UB->GM      = 128 per AIV
      //   FixPipe       = L0C->{UB,L1,GM} = 128
      // ================================================================
      mte1_base_latency = 5; // Measured via cannsim Cube kernel trace:
                             // LOAD_2Dv2 (L1→L0A, 32KB fp16) = 108-133 cyc.
                             // At 256 B/cycle, 32KB takes 128 cyc bandwidth →
                             // base = total - bw ≈ 0-5 cyc. The ~5 cyc covers
                             // descriptor setup / issue overhead.
      mte1_bandwidth = 256;
      mte2_base_latency =
          634; // Measured via cannsim trace_tools (direct cycles,
               // n=10 sizes): MTE2 for 32-128B plateau at 633-635.
               // Previous 365 used broken cannsim-report scale.
      mte2_bandwidth = 256;
      mte3_base_latency = 418; // Scaled from MTE2 correction (634/365 ≈ 1.74×
                               // applied to old value 240). Previous 240 used
                               // broken cannsim-report scale. TODO: re-measure
                               // with trace_tools sweep like MTE2.
      mte3_bandwidth = 128;
      fixpipe_base_latency =
          730; // Measured via cannsim Cube kernel trace:
               // FIX_L0C_TO_DST (64KB fp32 L0C→GM) = 1244 cyc.
               // AT 128 B/cycle, 64KB takes 512 cyc bandwidth
               // → base ≈ 1244-512 = 732 cyc. This includes
               // L0C readout + formatting + MTE3 GM write setup.
      fixpipe_bandwidth = 128;

      // ================================================================
      // MTE1 per-destination ports (AIC: L1 -> Cube input buffers).
      // Source: A5 PDF p.4 davinci column.
      //   L1 -> L0A     : 256 B/cycle
      //   L1 -> L0B     : 256 B/cycle
      //   L1 -> BT      :  32 B/cycle (combined "BT+FP buf" line)
      //   L1 -> FP Buf  :  32 B/cycle
      // ================================================================
      l1_to_l0a_bandwidth = 256;
      l1_to_l0b_bandwidth = 256;
      l1_to_bt_bandwidth = 32;
      l1_to_fp_buf_bandwidth = 32;

      // ================================================================
      // MTE2 per-source (GM -> on-chip). Source: A5 PDF p.4.
      //   AIC OUT->L1   : 256 B/cycle
      //   AIV OUT->UB   : 128 B/cycle per AIV (×2 AIV = 256 total per Core)
      // A5 has separate L2 ports for AIC and AIV (see p.4 path table — both
      // rows count concurrently); the L2->HBM contention happens one level
      // higher and is not modelled here.
      // ================================================================
      aic_mte2_to_l1_bandwidth = 256;
      aiv_mte2_to_ub_bandwidth = 128;

      // Two-segment GM->UB (AIV MTE2) fit. Measured single-copy MOV dur via
      // cannsim trace_tools sweep (16B..256KB): flat ~656 cyc for <=512B (the
      // micro-op split threshold), then a linear large-packet segment
      // dur = 743 + bytes/105 (fit over >=16KB, max_err ~4% on >=32KB). The
      // 105 B/cyc saturated rate matches the NoC-throttled read bandwidth in
      // the LSU-outstanding doc (theory 550 GB/s, throttled in practice).
      // GM->L1 (AIC) is NOT yet measured separately — it still uses the legacy
      // single-segment model.
      mte2_gm_to_ub_knee = 512;
      mte2_gm_to_ub_flat = 656;
      mte2_gm_to_ub_seg_base = 743;
      mte2_gm_to_ub_seg_bw = 105;

      // MTE3 UB->GM two-segment fit. Same cannsim trace_tools sweep method
      // (single MOV dur, 16B..256KB): flat ~445 cyc for <=512B, then
      // dur = 478 + bytes/103 for >=16KB (max_err ~6% on >=24KB). The ~103
      // B/cyc saturated write rate matches the GM->UB read rate (both bounded
      // by the same NoC-throttled LSU outstanding). Mid-range 640B..32KB
      // (~680-776 cyc plateau) under-estimated ~28%, same accepted tradeoff.
      mte3_ub_to_gm_knee = 512;
      mte3_ub_to_gm_flat = 445;
      mte3_ub_to_gm_seg_base = 478;
      mte3_ub_to_gm_seg_bw = 103;

      // AIC GM->L1 (nd2nz). Wide flat (issue queue 32) up to ~8KB, then
      // 837 + bytes/98 over 32KB..128KB. See field comment above for method.
      mte2_gm_to_l1_knee = 8192;
      mte2_gm_to_l1_flat = 986;
      mte2_gm_to_l1_seg_base = 837;
      mte2_gm_to_l1_seg_bw = 98;

      // ================================================================
      // MTE3 per-destination (AIV: UB -> off-AIV). Source: A5 PDF p.4.
      //   UB -> Out (L2/GM)    : 128 B/cycle per AIV
      //   UB -> AIC L1         : 128 B/cycle per AIV (A5-only path)
      //   UB -> AIC FP Buf     : 128 B/cycle per AIV (A5-only path)
      // ================================================================
      aiv_mte3_to_gm_bandwidth = 128;
      aiv_mte3_to_aic_l1_bandwidth = 128;
      aiv_mte3_to_aic_fp_buf_bandwidth = 128;

      // ================================================================
      // VALU / scalar bandwidth (B/cycle). Source: A5 PDF p.8 AIV diagram.
      //   VRF <-> UB       : 256 B/cycle (the "256B/cycle" arrow under VALU)
      //   Scalar L1 DCache : 32 B/cycle (S pipe, both AIC and AIV)
      // ================================================================
      valu_bandwidth = 256;
      scalar_dcache_bandwidth = 32;

      // ================================================================
      // On-chip buffer aggregate / port bandwidths.
      // L1 aggregate = sum of concurrent egress ports
      //              = L0A(256) + L0B(256) + BT(32) + FP(32) + UB(128)
      //              = 704 B/cycle worst-case fan-out.
      // L0A/L0B fill rate = matching MTE1 port = 256 B/cycle.
      // L0C drain rate = FixPipe out = 128 B/cycle.
      // ================================================================
      l1_bandwidth = 704;
      l0a_bandwidth = 256;
      l0b_bandwidth = 256;
      l0c_bandwidth = 128;

      // ================================================================
      // Cube physical mad granularity. Source: A5 PDF p.7.
      //   FP16/BF16 mad = 16 x 16 x 16 (one cycle per mad pair).
      // ================================================================
      cube_unit_m = 16;
      cube_unit_k = 16;
      cube_unit_n = 16;

      // ================================================================
      // Per-dtype Cube throughput (MACs/cycle). Source: A5 PDF p.7.
      //   FP16/BF16          : 16 x 16 x 16 x 2 = 8192
      //   HiF8/FP8/MXFP8     : 16 x 32 x 16 x 2 = 16384
      //   MXFP4              : 16 x 64 x 16 x 2 = 32768
      //   FP32 / INT8 / INT4 : not documented for A5; leave at 0 so the
      //                        fallback (tensor_core_throughput) kicks in.
      // ================================================================
      cube_throughput_fp16 = 8192;
      cube_throughput_bf16 = 8192;
      cube_throughput_fp32 =
          0; // A5 lacks native fp32 MMAD (confirmed: bisheng only emits
             // fp16/bf16/fp8 mad instructions). Falls back to
             // tensor_core_throughput for scheduling.
      cube_throughput_fp8 = 16384;
      cube_throughput_mxfp8 = 16384;
      cube_throughput_mxfp4 = 32768;
      cube_throughput_int8 = 0; // A5 MMAD does not support INT8 (fp16/bf16/fp8
                                // only). Leave 0 → fallback to generic.
      cube_throughput_int4 = 0; // A5 MMAD does not support INT4. Leave 0.

      // ================================================================
      // Core clocks (Hz). cannsim is a cycle-accurate simulator that
      // operates in abstract cycles without real-time correspondence;
      // clock frequency cannot be determined from cannsim alone.
      // The cannsim.log reports "cycle: 420, speed: 0.48 KHz" for the
      // SIMULATOR speed (420 sim cycles / 0.87s wall clock), not the
      // hardware clock. Hardware clock needs silicon datasheet or
      // white paper. Set to 0 = "do not convert to s/B".
      // ================================================================
      aic_frequency_hz = 0.0;
      aiv_frequency_hz = 0.0;

      // ================================================================
      // HBM / L2 / on-chip capacities (bytes). Sources:
      //   A5 PDF p.5 (AIC diagram): L1=512KB, L0A=L0B=64KB, BT=FP=4KB,
      //                              L0C=256KB.
      //   A5 PDF p.8 (AIV diagram): UB = 256 KB per AIV. Each AI Core
      //                              has 2 AIVs => 512 KB UB per Core,
      //                              but the field is per-AIV.
      //   A5 PDF p.4 (overview):    SSBuffer = 3 KB.
      // HBM / L2 parameters.
      // These are system-level (DRAM controller, HBM PHY, shared L2)
      // and NOT measurable via AI-core-level cannsim which models only
      // the MTE ports (GM↔UB/L1), not the DRAM or L2 internals.
      //   hbm_bandwidth:  measure via memory-bound kernel on real NPU
      //                   (e.g. STREAM triad), not cannsim.
      //   hbm_latency:    measure via pointer-chasing microbench on NPU.
      //   l2_*:           needs white paper or L2-specific microbench.
      // ================================================================
      hbm_bandwidth = 0;     // System-level; needs NPU DRAM benchmark
      hbm_latency = 0;       // System-level; needs NPU DRAM benchmark
      l2_capacity_bytes = 0; // System-level; needs white paper
      l2_bandwidth = 0;      // System-level; needs white paper
      l2_hit_latency = 0;    // System-level; needs L2 microbench on NPU
      l1_capacity_bytes = 512 * 1024;
      ub_capacity_bytes = 256 * 1024; // per AIV
      l0a_capacity_bytes = 64 * 1024;
      l0b_capacity_bytes = 64 * 1024;
      l0c_capacity_bytes = 256 * 1024;
      bt_capacity_bytes = 4 * 1024;
      fp_buf_capacity_bytes = 4 * 1024;
      ss_buffer_capacity_bytes = 3 * 1024;

      // ================================================================
      // Per-pipe issue queue depths (max in-flight ops per pipe).
      //   A5 PDF p.8: AIV MTE3 Queue = 16 entry (confirmed).
      //   Other pipes: diagram boxes shown but entry counts not labelled.
      //   Default estimate of 16 is consistent with MTE3 and conservative
      //   for scheduling (lower depth → higher II floor).
      //   Scalar pipe: narrower issue width; 4 is conservative.
      // ================================================================
      mte1_queue_depth = 16;
      mte2_queue_depth = 16;
      mte3_queue_depth = 16;
      cube_queue_depth = 16;
      fixpipe_queue_depth = 16;
      vector_queue_depth = 16;
      scalar_queue_depth = 4;

      // ================================================================
      // Cube pipeline / buffer depths.
      //   cube_pipeline_depth = min cycles between back-to-back MMAD issues
      //     to reach steady state. Observed: single MMAD per cycle at
      //     steady state → depth = 1.
      //   cube_unit_flag_depth = max in-flight MMADs with unit_flag.
      //     A5 PDF p.7: unit_flag supports back-to-back mads; 16 is a
      //     conservative bound (matches mte3_queue_depth).
      // ================================================================
      cube_pipeline_depth = 1;
      cube_unit_flag_depth = 16;
      bt_buffer_depth = 0;
      fp_buffer_depth = 0;

      // ================================================================
      // Per-dtype VALU throughputs. Derive from valu_bandwidth lazily
      // via ValuThroughputFor() when these are 0; explicit numbers
      // would come from microbench of A5 SimdVF / SimtVF kernels.
      //   fp32 (4B): 256/4 = 64 ops/cycle  per AIV
      //   fp16 (2B): 256/2 = 128
      //   bf16 (2B): 256/2 = 128
      //   int32(4B): 256/4 = 64
      //   int16(2B): 256/2 = 128
      //   int8 (1B): 256/1 = 256
      // (lazy fallback already does this; explicit values below let us
      //  override per-dtype if VALU has narrow-int speedups beyond pure
      //  bandwidth — currently we trust the lazy path.)
      // ================================================================
      valu_throughput_fp32 = 0;  // lazy fallback => 64
      valu_throughput_fp16 = 0;  // lazy fallback => 128
      valu_throughput_bf16 = 0;  // lazy fallback => 128
      valu_throughput_int32 = 0; // lazy fallback => 64
      valu_throughput_int16 = 0; // lazy fallback => 128
      valu_throughput_int8 = 0;  // lazy fallback => 256
      valu_special_func_throughput =
          8; // VALU exp/ln/sqrt/rsqrt throughput.
             // Derived from valu_bandwidth: special
             // functions use iterative approximation
             // (~8-32 cycles per element vs 1 for
             // add/mul). 8 ops/cycle is conservative.

      // ================================================================
      // Scalar pipe throughput. Measured via cannsim Cube kernel trace
      // (SCALAR per-instruction timing, n=590 instructions):
      //   Arithmetic (ADD/SUB/MUL/MADD/CMP/AND/OR/SHL/SHR): 3.0 cyc each
      //   MOV (MOV_XD_IMM/MOV_XD_XN/MOVK): 3.0 cyc each (n=198 total)
      //   MOV_SPR_XN: 1.5 cyc; WAIT_FLAG: 1.0 cyc
      //   Memory (LD/ST/LDP/STP): 90-740 cyc (cache/DMA)
      // Peak scalar throughput = 1/3 ≈ 0.33 ops/cyc for compute ops.
      // Using 1 op/cycle (peak single-issue) is optimistic but gives
      // the scheduler headroom for descriptor-setup code.
      // ================================================================
      scalar_op_throughput = 1; // S pipe: 1 scalar op/cycle (peak)
      branch_mispredict_cycles = 0;
      icache_miss_cycles = 0;
      scalar_load_use_cycles = 0;

      // ================================================================
      // MTE descriptor overhead. Charged once per copy command on top
      // of the bandwidth-limited bulk transfer. Measured via cannsim:
      // 8×256B copies vs 1×2048B copy show negligible per-descriptor
      // difference (≤13 cycles on 2000+ total). Descriptor setup cost
      // is absorbed into the base_latency for scheduling purposes.
      // ================================================================
      mte_descriptor_cycles = 0;
    }
  };

  struct B200Params : GPUParams {
    B200Params() {
      global_memory_read = 714;
      global_memory_write = 571;
      shared_memory_read = 19;
      shared_memory_write = 19;
      cuda_core_operation = 3;
      tma_operation = 71;
      tensor_core_base_latency = 23;
      tensor_core_throughput = 9300;
      wgmma_base_latency = 29;
      wgmma_per_tile_latency = 1;
      global_memory_bandwidth = 109;
      add_throughput = 6;
      sub_throughput = 6;
      mul_throughput = 6;
      min_max_throughput = 6;
      cmp_throughput = 6;
      logic_throughput = 6;
      bitwise_throughput = 6;
      shift_throughput = 6;
      tmem_read = 1;
      tmem_write = 1;
      tmem_bandwidth = 0;
    }
  };

  explicit LatencyEstimator(Target target = Target()) {
    if (target.defined() && TargetIsAscend(target)) {
      params_ = AscendParams();
      is_ascend_ = true;
    } else if (target.defined() && TargetIsSm100(target)) {
      params_ = B200Params();
    }
  }

  // Set thread count for parallel execution
  void SetThreadCount(int64_t thread_count) { thread_count_ = thread_count; }

  // Estimate latency for a TaskNode
  void Estimate(TaskNode *task) {
    if (is_ascend_) {
      EstimateAscend(task);
      return;
    }
    int64_t total_latency = 0;
    int64_t memory_latency = 0;
    int64_t compute_latency = 0;

    // Count memory operations and track bytes by memory type
    int num_memory_ops = 0;
    int64_t global_memory_bytes = 0;
    int64_t shared_memory_bytes = 0;
    int64_t register_bytes = 0;
    int64_t tmem_bytes = 0;

    // Estimate latency from memory accesses and track bandwidth usage
    for (const auto &region : task->GetReadRegions()) {
      int64_t region_latency =
          EstimateMemoryAccessLatency(region, true); // read
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer &buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
      case MemoryType::kGlobal:
        global_memory_bytes += access_bytes;
        break;
      case MemoryType::kShared:
        shared_memory_bytes += access_bytes;
        break;
      case MemoryType::kRegister:
        register_bytes += access_bytes;
        break;
      case MemoryType::kTmem:
        tmem_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes;
        break;
      }
    }

    for (const auto &region : task->GetWriteRegions()) {
      int64_t region_latency =
          EstimateMemoryAccessLatency(region, false); // write
      memory_latency += region_latency;
      num_memory_ops++;

      // Track bandwidth usage by memory type
      const Buffer &buffer = region->buffer;
      String scope = buffer.scope();
      MemoryType mem_type = GetMemoryTypeFromScope(scope);
      int64_t access_bytes = CalculateAccessBytes(region);

      switch (mem_type) {
      case MemoryType::kGlobal:
        global_memory_bytes += access_bytes;
        break;
      case MemoryType::kShared:
        shared_memory_bytes += access_bytes;
        break;
      case MemoryType::kRegister:
        register_bytes += access_bytes;
        break;
      case MemoryType::kTmem:
        tmem_bytes += access_bytes;
        break;
      default:
        global_memory_bytes += access_bytes;
        break;
      }
    }

    // Estimate compute latency based on operation counting
    if (task->UsesPipe(ResourcePipe::kVector)) {
      // Use OperationCounter visitor to count operations and estimate latency
      OperationCounter counter(thread_count_, &params_);

      // Visit all statements in the task
      for (const auto &stmt : task->stmts) {
        counter(stmt);
      }

      // Get estimated latency
      compute_latency = counter.GetEstimatedLatency();
    }

    if (task->UsesPipe(ResourcePipe::kCube)) {
      // Shape-aware matmul latency estimation
      int64_t matmul_latency = 32;

      if (task->HasTensorCoreShape()) {
        // Calculate total operations across all shapes
        int64_t total_ops = task->GetTotalTensorCoreOps();

        // Simple model: latency = base + ops / throughput
        matmul_latency = params_.tensor_core_base_latency +
                         (total_ops + params_.tensor_core_throughput - 1) /
                             params_.tensor_core_throughput;

        // Clamp to reasonable values
        matmul_latency =
            std::max(matmul_latency, params_.tensor_core_base_latency);
      }

      compute_latency = std::max(compute_latency, matmul_latency);
    }

    // Total latency is sum of memory and compute (assuming sequential for now)
    total_latency = memory_latency + compute_latency;

    // Calculate initiation interval (II)
    int64_t ii = 1; // Default minimum II

    bool has_dma = (task->GetPipeMask() &
                    (static_cast<uint16_t>(ResourcePipe::kMTE1) |
                     static_cast<uint16_t>(ResourcePipe::kMTE2) |
                     static_cast<uint16_t>(ResourcePipe::kMTE3) |
                     static_cast<uint16_t>(ResourcePipe::kFixpipe))) != 0;
    bool has_cube = task->UsesPipe(ResourcePipe::kCube);

    if (has_dma && !has_cube) {
      // Case 1: Only data movement (no compute)
      // DMA latency can be hidden by pipelining
      // II is determined by bandwidth constraints only
      if (global_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (global_memory_bytes + params_.global_memory_bandwidth - 1) /
            params_.global_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
      if (shared_memory_bytes > 0) {
        int64_t bandwidth_ii =
            (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
            params_.shared_memory_bandwidth;
        ii = std::max(ii, bandwidth_ii);
      }
    } else if (!has_dma && has_cube) {
      ii = total_latency;
      // Case 2: Only Tensor Core operations (no TMA)
      // Tensor Core operations are highly pipelined

      // Start with minimum II
      // ii = params_.tensor_core_min_ii;

      // If we have shape information, calculate more accurate II
      // if (task->HasTensorCoreShape()) {
      // Calculate total number of WGMMA tiles across all shapes
      // int64_t total_wgmma_tiles = task->GetTotalWGMMATiles();

      // II based on tile count and parallelism
      // More tiles may require larger II due to resource constraints
      // int64_t tile_based_ii =
      //     params_.tensor_core_min_ii +
      //     (total_wgmma_tiles + params_.tensor_core_max_parallel_tiles - 1) /
      //         params_.tensor_core_max_parallel_tiles;

      // ii = std::max(ii, tile_based_ii);

      // For very small operations, II can be smaller
      //   if (total_wgmma_tiles <= 2) {
      //     ii = std::min(ii, static_cast<int64_t>(2));
      //   }
      // }

      // II must be at least the bandwidth-limited II
      // if (global_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (global_memory_bytes + params_.global_memory_bandwidth - 1) /
      //       params_.global_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // if (shared_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
      //       params_.shared_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // For Tensor Core, II should be reasonable compared to latency
      // Typically II << latency for pipelined operations
      // int64_t max_ii_ratio = 4; // II should not exceed latency/4
      // int64_t max_reasonable_ii =
      //     (total_latency + max_ii_ratio - 1) / max_ii_ratio;
      // ii =
      //     std::min(ii, std::max(max_reasonable_ii,
      //     params_.tensor_core_min_ii));

    } else {
      // Case 3: Other cases (both TMA and Tensor Core, or neither)
      // Force II = total_latency for conservative scheduling
      ii = total_latency;

      // // Special case: single memory operation
      // if (num_memory_ops == 1 && task->stmts.size() == 1) {
      //   // Single operation that is a memory access
      //   // Check if this is likely a memory operation (has read/write
      //   regions) if (!task->GetReadRegions().empty() ||
      //       !task->GetWriteRegions().empty()) {
      //     ii = memory_latency;
      //   }
      // }

      // // Additional II constraints from bandwidth limitations
      // // II must be at least the time needed to transfer data based on
      // bandwidth if (global_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (global_memory_bytes + params_.global_memory_bandwidth - 1) /
      //       params_.global_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }

      // if (shared_memory_bytes > 0) {
      //   int64_t bandwidth_ii =
      //       (shared_memory_bytes + params_.shared_memory_bandwidth - 1) /
      //       params_.shared_memory_bandwidth;
      //   ii = std::max(ii, bandwidth_ii);
      // }
    }

    // II must be at least 1 cycle
    if (ii < 1)
      ii = 1;

    // Store results in task node
    task->SetLatency(total_latency);
    task->SetII(ii);
  }

private:
  GPUParams params_;
  int64_t thread_count_ = 1; // Default to 1 (no parallelism)
  bool is_ascend_ = false;

  // ====================================================================
  // Ascend-specific estimator.
  //
  // The formulas below are written in full against the (currently zero)
  // AscendParams fields. Every divisor is guarded with `safe_div(a, b)` so
  // that an unfilled (= 0) bandwidth/throughput parameter degenerates to a
  // zero contribution rather than a divide-by-zero. Once a parameter gets
  // a real value, the formula immediately starts producing a real number.
  //
  // The estimator is structured by physical resource:
  //   1. Memory latency: walk read/write regions, classify (src,dst) scope
  //      pair to a physical path, charge `base + bytes/bandwidth`.
  //   2. Compute latency:
  //        Cube   -> (M*K*N) / cube_throughput_<dtype> + base
  //        Vector -> max(op-count via OperationCounter,
  //                      bytes / valu_bandwidth)
  //        Scalar -> op count / scalar_op_throughput
  //   3. II by pipe: take the max of:
  //        - bandwidth-limited II per physical port
  //        - latency-limited II (= total_latency when pipe can't overlap)
  //        - queue-depth-limited II (= ops / queue_depth)
  //   4. Total latency = memory_latency + compute_latency
  //      under sequential assumption; pipeline overlap is handled by the
  //      scheduler (Z3) via the II we emit.
  // ====================================================================

  static int64_t safe_div_ceil(int64_t numer, int64_t denom) {
    return denom > 0 ? (numer + denom - 1) / denom : 0;
  }
  static int64_t safe_div(int64_t numer, int64_t denom) {
    return denom > 0 ? numer / denom : 0;
  }

  // Read `tl.vf_latency` annotation from SIMT_VF / SIMD_VF Block nodes within
  // the task. Returns 0 when no annotation is present.
  static int64_t ReadVFLatencyFromBlocks(const TaskNode *task) {
    struct VFLatencyReader : public StmtVisitor {
      void VisitStmt_(const BlockNode *op) override {
        if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
          auto it = op->annotations.find("tl.vf_latency");
          if (it != op->annotations.end()) {
            if (auto *imm = (*it).second.as<IntImmNode>()) {
              latency = std::max(latency, imm->value);
            }
          }
        }
        StmtVisitor::VisitStmt_(op);
      }
      int64_t latency{0};
    };

    VFLatencyReader reader;
    for (const auto &stmt : task->stmts) {
      reader(stmt);
    }
    return reader.latency;
  }

  // Classification of (scope -> scope) for an MTE-style copy. Matches the
  // PipeMask the IRStructureBuilder will have set on the task, but lets
  // EstimateAscend pick the *specific* physical port (e.g. L1->L0A vs
  // L1->L0B) from the region's buffer.
  enum class AscendPath {
    kUnknown,
    kGmToL1,    // MTE2 (AIC)
    kGmToUb,    // MTE2 (AIV)
    kUbToGm,    // MTE3
    kUbToL1,    // MTE3 (cross-core)
    kUbToFpBuf, // MTE3 (cross-core)
    kL1ToL0a,   // MTE1
    kL1ToL0b,   // MTE1
    kL1ToBt,    // MTE1
    kL1ToFpBuf, // MTE1
    kL0cToUb,   // Fixpipe
    kL0cToGm,   // Fixpipe
  };

  // Pick the bandwidth (bytes/cycle) for a physical path. Returns 0 when
  // the path or the parameter is unknown — `safe_div` then collapses that
  // term to zero in the formulas below.
  int64_t BandwidthForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
      return params_.aic_mte2_to_l1_bandwidth;
    case AscendPath::kGmToUb:
      return params_.aiv_mte2_to_ub_bandwidth;
    case AscendPath::kUbToGm:
      return params_.aiv_mte3_to_gm_bandwidth;
    case AscendPath::kUbToL1:
      return params_.aiv_mte3_to_aic_l1_bandwidth;
    case AscendPath::kUbToFpBuf:
      return params_.aiv_mte3_to_aic_fp_buf_bandwidth;
    case AscendPath::kL1ToL0a:
      return params_.l1_to_l0a_bandwidth;
    case AscendPath::kL1ToL0b:
      return params_.l1_to_l0b_bandwidth;
    case AscendPath::kL1ToBt:
      return params_.l1_to_bt_bandwidth;
    case AscendPath::kL1ToFpBuf:
      return params_.l1_to_fp_buf_bandwidth;
    case AscendPath::kL0cToUb:
      return params_.fixpipe_bandwidth;
    case AscendPath::kL0cToGm:
      return params_.fixpipe_bandwidth;
    default:
      return 0;
    }
  }

  // Two-segment (knee) parameters for a path: {knee, flat, seg_base, seg_bw}.
  // seg_bw == 0 means "no two-segment fit for this path" and the caller falls
  // back to the legacy single-segment base + bytes/bw model.
  struct SegParams {
    int64_t knee;
    int64_t flat;
    int64_t seg_base;
    int64_t seg_bw;
  };
  SegParams SegParamsForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToUb:
      return {params_.mte2_gm_to_ub_knee, params_.mte2_gm_to_ub_flat,
              params_.mte2_gm_to_ub_seg_base, params_.mte2_gm_to_ub_seg_bw};
    case AscendPath::kUbToGm:
      return {params_.mte3_ub_to_gm_knee, params_.mte3_ub_to_gm_flat,
              params_.mte3_ub_to_gm_seg_base, params_.mte3_ub_to_gm_seg_bw};
    case AscendPath::kGmToL1:
      return {params_.mte2_gm_to_l1_knee, params_.mte2_gm_to_l1_flat,
              params_.mte2_gm_to_l1_seg_base, params_.mte2_gm_to_l1_seg_bw};
    default:
      return {0, 0, 0, 0};
    }
  }

  // Latency (cycles) for moving `bytes` over `path`. Uses the two-segment knee
  // model when the path has a measured fit, else the legacy single segment.
  int64_t MemLatencyForPath(AscendPath path, int64_t bytes) const {
    SegParams sp = SegParamsForPath(path);
    if (sp.seg_bw > 0) {
      if (bytes <= sp.knee)
        return sp.flat;
      return sp.seg_base + safe_div_ceil(bytes, sp.seg_bw);
    }
    return BaseLatencyForPath(path) +
           safe_div_ceil(bytes, BandwidthForPath(path));
  }

  int64_t BaseLatencyForPath(AscendPath path) const {
    switch (path) {
    case AscendPath::kGmToL1:
    case AscendPath::kGmToUb:
      return params_.mte2_base_latency;
    case AscendPath::kUbToGm:
    case AscendPath::kUbToL1:
    case AscendPath::kUbToFpBuf:
      return params_.mte3_base_latency;
    case AscendPath::kL1ToL0a:
    case AscendPath::kL1ToL0b:
    case AscendPath::kL1ToBt:
    case AscendPath::kL1ToFpBuf:
      return params_.mte1_base_latency;
    case AscendPath::kL0cToUb:
    case AscendPath::kL0cToGm:
      return params_.fixpipe_base_latency;
    default:
      return 0;
    }
  }

  // Best-effort classification of a single region into an AscendPath from
  // its buffer scope. For reads, `path = ? -> dst_of_pipe`; for writes,
  // `path = src_of_pipe -> ?`. The task's pipe mask narrows it further.
  AscendPath ClassifyRegion(const String &scope, uint16_t pipe_mask,
                            bool is_read) const {
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);

    std::string s = scope.operator std::string();
    if (has_mte2) {
      if (s == "global" && !is_read)
        return AscendPath::kUnknown;
      if (s == "global" && is_read)
        return AscendPath::kGmToUb; // default
      if (s == "shared.l1")
        return AscendPath::kGmToL1;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kGmToUb;
    }
    if (has_mte3) {
      if (s == "global")
        return AscendPath::kUbToGm;
      if (s == "shared.l1")
        return AscendPath::kUbToL1;
    }
    if (has_mte1) {
      if (s == "shared.l0a")
        return AscendPath::kL1ToL0a;
      if (s == "shared.l0b")
        return AscendPath::kL1ToL0b;
      if (s == "shared.bt")
        return AscendPath::kL1ToBt;
    }
    if (has_fix) {
      if (s == "shared.l0c" && is_read) {
        // L0C is the source; dest scope comes from the *other* region.
        return AscendPath::kL0cToUb; // refined by paired region below
      }
      if (s == "global")
        return AscendPath::kL0cToGm;
      if (s == "shared" || s == "shared.dyn")
        return AscendPath::kL0cToUb;
    }
    return AscendPath::kUnknown;
  }

  AscendPath ClassifyCopy(const String &src_scope,
                          const String &dst_scope) const {
    std::string src = src_scope.operator std::string();
    std::string dst = dst_scope.operator std::string();
    if (src == "global" && dst == "shared.l1")
      return AscendPath::kGmToL1;
    if (src == "global" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kGmToUb;
    if ((src == "shared" || src == "shared.dyn") && dst == "global")
      return AscendPath::kUbToGm;
    if ((src == "shared" || src == "shared.dyn") && dst == "shared.l1")
      return AscendPath::kUbToL1;
    if (src == "shared.l1" && dst == "shared.l0a")
      return AscendPath::kL1ToL0a;
    if (src == "shared.l1" && dst == "shared.l0b")
      return AscendPath::kL1ToL0b;
    if (src == "shared.l1" && dst == "shared.bt")
      return AscendPath::kL1ToBt;
    if (src == "shared.l1" && dst == "shared.fp")
      return AscendPath::kL1ToFpBuf;
    if (src == "shared.l0c" && (dst == "shared" || dst == "shared.dyn"))
      return AscendPath::kL0cToUb;
    if (src == "shared.l0c" && dst == "global")
      return AscendPath::kL0cToGm;
    return AscendPath::kUnknown;
  }

  // Per-dtype Cube throughput (MACs/cycle). Falls back to the generic
  // tensor_core_throughput when no dtype-specific number is available.
  int64_t CubeThroughputFor(DataType dtype) const {
    if (dtype.is_float16())
      return params_.cube_throughput_fp16 ? params_.cube_throughput_fp16
                                          : params_.tensor_core_throughput;
    if (dtype.is_bfloat16())
      return params_.cube_throughput_bf16 ? params_.cube_throughput_bf16
                                          : params_.tensor_core_throughput;
    if (dtype.is_float() && dtype.bits() == 32)
      return params_.cube_throughput_fp32 ? params_.cube_throughput_fp32
                                          : params_.tensor_core_throughput;
    if (dtype.is_float8() || (dtype.is_float() && dtype.bits() == 8))
      return params_.cube_throughput_fp8 ? params_.cube_throughput_fp8
                                         : params_.tensor_core_throughput;
    if (dtype.is_int() && dtype.bits() == 8)
      return params_.cube_throughput_int8 ? params_.cube_throughput_int8
                                          : params_.tensor_core_throughput;
    return params_.tensor_core_throughput;
  }

  // Per-dtype VALU throughput (ops/cycle). When the dtype-specific field is
  // 0, fall back to valu_bandwidth / dtype.bytes() if known, then to 0.
  int64_t ValuThroughputFor(DataType dtype) const {
    int64_t per_dtype = 0;
    if (dtype.is_float() && dtype.bits() == 32)
      per_dtype = params_.valu_throughput_fp32;
    else if (dtype.is_float16())
      per_dtype = params_.valu_throughput_fp16;
    else if (dtype.is_bfloat16())
      per_dtype = params_.valu_throughput_bf16;
    else if (dtype.is_int() && dtype.bits() == 32)
      per_dtype = params_.valu_throughput_int32;
    else if (dtype.is_int() && dtype.bits() == 16)
      per_dtype = params_.valu_throughput_int16;
    else if (dtype.is_int() && dtype.bits() == 8)
      per_dtype = params_.valu_throughput_int8;
    if (per_dtype > 0)
      return per_dtype;
    int64_t bytes = dtype.bytes();
    return bytes > 0 ? safe_div(params_.valu_bandwidth, bytes) : 0;
  }

  // The Ascend latency / II estimation entry point. Mirrors the structure
  // of the generic Estimate() but consults the per-path / per-dtype fields.
  void EstimateAscend(TaskNode *task) {
    uint16_t pipe_mask = task->GetPipeMask();
    bool has_mte1 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE1);
    bool has_mte2 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE2);
    bool has_mte3 = pipe_mask & static_cast<uint16_t>(ResourcePipe::kMTE3);
    bool has_fix = pipe_mask & static_cast<uint16_t>(ResourcePipe::kFixpipe);
    bool has_cube = pipe_mask & static_cast<uint16_t>(ResourcePipe::kCube);
    bool has_vec = pipe_mask & static_cast<uint16_t>(ResourcePipe::kVector);
    bool has_scl = pipe_mask & static_cast<uint16_t>(ResourcePipe::kScalar);
    bool has_dma = has_mte1 || has_mte2 || has_mte3 || has_fix;
    int64_t measured_vf_latency = ReadVFLatencyFromBlocks(task);

    // -----------------------------------------------------------------
    // 1. Memory latency: per-physical-path, base + bytes / bandwidth.
    //    Bytes are aggregated per path for the II step.
    // -----------------------------------------------------------------
    std::unordered_map<int, int64_t> bytes_per_path;
    int64_t memory_latency = 0;

    auto add_memory_access = [&](AscendPath path, int64_t bytes) {
      int64_t lat =
          params_.mte_descriptor_cycles + MemLatencyForPath(path, bytes);
      memory_latency += lat;
      bytes_per_path[static_cast<int>(path)] += bytes;
    };

    if (task->HasCopyInfo()) {
      for (const auto &info : task->GetCopyInfos()) {
        add_memory_access(ClassifyCopy(info.src_scope, info.dst_scope),
                          info.bytes);
      }
    } else {
      auto visit_region = [&](const BufferRegion &region, bool is_read) {
        const Buffer &buffer = region->buffer;
        String scope = buffer.scope();
        AscendPath path = ClassifyRegion(scope, pipe_mask, is_read);
        int64_t bytes = CalculateAccessBytes(region);
        add_memory_access(path, bytes);
      };
      for (const auto &r : task->GetReadRegions())
        visit_region(r, /*is_read=*/true);
      for (const auto &r : task->GetWriteRegions())
        visit_region(r, /*is_read=*/false);
    }

    // -----------------------------------------------------------------
    // 2. Compute latency.
    // -----------------------------------------------------------------
    int64_t compute_latency = 0;
    int64_t vector_latency = 0;
    int64_t cube_latency = 0;
    int64_t scalar_latency = 0;

    if (has_vec) {
      if (measured_vf_latency > 0) {
        // User-annotated latency — use it directly, skip model estimation.
        vector_latency = measured_vf_latency;
      } else {
        // Op-count path (TIR visitor).
        OperationCounter counter(thread_count_, &params_);
        for (const auto &stmt : task->stmts)
          counter(stmt);
        int64_t op_count_latency = counter.GetEstimatedLatency();

        // Bandwidth-bound path: total bytes the VALU consumes / valu_bandwidth.
        int64_t valu_bytes = 0;
        for (const auto &r : task->GetReadRegions()) {
          String scope = r->buffer.scope();
          if (scope == "local" || scope == "local.fragment" ||
              scope == "shared" || scope == "shared.dyn") {
            valu_bytes += CalculateAccessBytes(r);
          }
        }
        int64_t bw_latency = safe_div_ceil(valu_bytes, params_.valu_bandwidth);

        vector_latency = std::max(op_count_latency, bw_latency);
      }
      compute_latency = std::max(compute_latency, vector_latency);
    }

    if (has_cube) {
      // matmul latency = base + total_MACs / per-dtype throughput.
      // total_MACs comes from TaskNode if HasTensorCoreShape(); else fall
      // back to a single unit-mad.
      int64_t matmul_latency = params_.tensor_core_base_latency;
      if (task->HasTensorCoreShape()) {
        matmul_latency = params_.tensor_core_base_latency;
        for (size_t i = 0; i < task->GetTensorCoreShapeCount(); ++i) {
          const auto &shape = task->GetTensorCoreShape(i);
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          matmul_latency += safe_div_ceil(ops, throughput);
        }
        matmul_latency =
            std::max(matmul_latency, params_.tensor_core_base_latency);
      } else if (params_.cube_unit_m > 0 && params_.cube_unit_k > 0 &&
                 params_.cube_unit_n > 0) {
        // Single unit mad fallback.
        int64_t unit_ops =
            params_.cube_unit_m * params_.cube_unit_k * params_.cube_unit_n;
        int64_t throughput = CubeThroughputFor(DataType::Float(16));
        matmul_latency = params_.tensor_core_base_latency +
                         safe_div_ceil(unit_ops, throughput);
      }
      cube_latency = matmul_latency;
      compute_latency = std::max(compute_latency, cube_latency);
    }

    if (has_scl && !has_vec) {
      // Pure scalar tasks (set/wait_flag bodies, address compute). Charge
      // a flat per-stmt cost — scalar pipe throughput is the divisor.
      int64_t scalar_ops = static_cast<int64_t>(task->stmts.size());
      scalar_latency =
          params_.scalar_op_throughput > 0
              ? safe_div_ceil(scalar_ops, params_.scalar_op_throughput)
              : scalar_ops;
      compute_latency = std::max(compute_latency, scalar_latency);
    }

    int64_t total_latency = memory_latency + compute_latency;

    // -----------------------------------------------------------------
    // 4. II: per-pipe analysis. Take the max across all active pipes.
    // -----------------------------------------------------------------
    int64_t ii = 1;
    int64_t memory_ii = 1;
    for (const auto &kv : bytes_per_path) {
      AscendPath path = static_cast<AscendPath>(kv.first);
      // Steady-state throughput: the saturated large-packet rate (seg_bw)
      // when the path has a two-segment fit, else the headline bandwidth.
      SegParams sp = SegParamsForPath(path);
      int64_t bw = sp.seg_bw > 0 ? sp.seg_bw : BandwidthForPath(path);
      memory_ii = std::max(memory_ii, safe_div_ceil(kv.second, bw));
    }

    int64_t cube_ii = params_.tensor_core_min_ii;
    if (has_cube) {
      if (task->HasTensorCoreShape()) {
        cube_ii = params_.tensor_core_min_ii;
        for (size_t i = 0; i < task->GetTensorCoreShapeCount(); ++i) {
          const auto &shape = task->GetTensorCoreShape(i);
          int64_t ops = shape.m * shape.n * shape.k * 2;
          int64_t throughput = CubeThroughputFor(shape.input_dtype);
          cube_ii = std::max(cube_ii, safe_div_ceil(ops, throughput));
        }
      }
      cube_ii = std::max(cube_ii, params_.cube_pipeline_depth);
    }

    int64_t vector_ii = 1;
    if (has_vec) {
      if (measured_vf_latency > 0) {
        vector_ii = measured_vf_latency;
      } else {
        int64_t denom =
            params_.vector_queue_depth > 0 ? params_.vector_queue_depth : 1;
        vector_ii = std::max(vector_ii, safe_div_ceil(vector_latency, denom));
      }
    }

    int64_t scalar_ii = 1;
    if (has_scl) {
      int64_t denom =
          params_.scalar_queue_depth > 0 ? params_.scalar_queue_depth : 1;
      scalar_ii = std::max(scalar_ii, safe_div_ceil(scalar_latency, denom));
    }

    if (has_dma && !has_cube && !has_vec) {
      // DMA-only: II = max bandwidth-limited II across active paths.
      ii = std::max(ii, memory_ii);
    } else if (has_cube && !has_dma && !has_vec) {
      // Cube-only: II = max(cube_pipeline_depth, total_macs / throughput,
      //                     tensor_core_min_ii).
      ii = std::max(ii, cube_ii);
    } else if (has_vec && !has_dma && !has_cube) {
      // Vector-only: II = compute_latency / vector_queue_depth (fallback to
      // compute_latency when queue depth is unknown).
      ii = std::max(ii, vector_ii);
    } else {
      // Mixed / fallback: keep task granularity, but bound II by the slowest
      // active pipe instead of forcing it to the full sequential latency.
      if (has_dma)
        ii = std::max(ii, memory_ii);
      if (has_cube)
        ii = std::max(ii, cube_ii);
      if (has_vec)
        ii = std::max(ii, vector_ii);
      if (has_scl)
        ii = std::max(ii, scalar_ii);
    }

    // Queue-depth-limited floor: a pipe with finite queue depth can never
    // issue faster than ops/queue_depth. Approximate "ops" by number of
    // stmts on that pipe — coarse but at least monotone in workload.
    auto queue_bound = [&](int64_t depth, int64_t ops) {
      return depth > 0 ? safe_div_ceil(ops, depth) : 0;
    };
    int64_t num_stmts = static_cast<int64_t>(task->stmts.size());
    if (has_mte1)
      ii = std::max(ii, queue_bound(params_.mte1_queue_depth, num_stmts));
    if (has_mte2)
      ii = std::max(ii, queue_bound(params_.mte2_queue_depth, num_stmts));
    if (has_mte3)
      ii = std::max(ii, queue_bound(params_.mte3_queue_depth, num_stmts));
    if (has_fix)
      ii = std::max(ii, queue_bound(params_.fixpipe_queue_depth, num_stmts));
    if (has_cube)
      ii = std::max(ii, queue_bound(params_.cube_queue_depth, num_stmts));
    if (has_vec)
      ii = std::max(ii, queue_bound(params_.vector_queue_depth, num_stmts));
    if (has_scl)
      ii = std::max(ii, queue_bound(params_.scalar_queue_depth, num_stmts));

    if (ii < 1)
      ii = 1;

    task->SetLatency(total_latency);
    task->SetII(ii);
  }

  // Operation counter visitor with latency estimation
  class OperationCounter : public StmtExprVisitor {
  public:
    // Loop dimension information
    struct LoopDimension {
      const VarNode *var;
      int64_t trip_count;
      int depth;
    };

    int64_t total_latency = 0;
    int64_t thread_count = 1; // Thread count for parallel execution
    const LatencyEstimator::GPUParams *params = nullptr;

    // Track loop dimensions for loop-invariant detection
    std::vector<LoopDimension> loop_stack;
    std::unordered_map<const VarNode *, int> var_to_depth;

    OperationCounter(int64_t thread_count = 1,
                     const LatencyEstimator::GPUParams *params = nullptr)
        : thread_count(thread_count), params(params) {}

    // Operator to visit a statement
    void operator()(const Stmt &stmt) { VisitStmt(stmt); }

    // Get estimated latency
    int64_t GetEstimatedLatency() const { return total_latency; }

    // Helper function to update latency for an operation based on throughput
    void update_operation(
        int64_t throughput,
        const std::unordered_set<const VarNode *> &contained_vars) {
      int64_t effective_factor =
          CalculateEffectiveParallelFactor(contained_vars);

      // Throughput is operations per cycle
      // For 1 operation: cycles = 1 / throughput
      // For effective_factor operations: cycles = effective_factor / throughput
      // With thread-level parallelism: cycles = (effective_factor / throughput)
      // / thread_count

      if (throughput > 0) {
        // Calculate cycles needed for effective_factor operations with given
        // throughput
        int64_t cycles_for_operations =
            (effective_factor + throughput - 1) / throughput;

        if (cycles_for_operations >= thread_count) {
          total_latency +=
              (cycles_for_operations + thread_count - 1) / thread_count;
        } else {
          total_latency += 1; // At least 1 cycle
        }
      } else {
        // Zero or negative throughput, assume 1 cycle
        total_latency += 1;
      }
    }

    // Analyze which loop variables are contained in an expression
    std::unordered_set<const VarNode *>
    AnalyzeContainedLoopVars(const PrimExpr &expr) {
      class VarCollector : public ExprVisitor {
      public:
        const std::unordered_map<const VarNode *, int> &var_to_depth;
        std::unordered_set<const VarNode *> collected_vars;

        VarCollector(
            const std::unordered_map<const VarNode *, int> &var_to_depth)
            : var_to_depth(var_to_depth) {}

        void VisitExpr_(const VarNode *op) final {
          if (var_to_depth.count(op)) {
            collected_vars.insert(op);
          }
          ExprVisitor::VisitExpr_(op);
        }
      };

      VarCollector collector(var_to_depth);
      collector(expr);
      return collector.collected_vars;
    }

    // Calculate effective parallel factor based on contained loop variables
    int64_t CalculateEffectiveParallelFactor(
        const std::unordered_set<const VarNode *> &contained_vars) {
      if (contained_vars.empty()) {
        return 1; // Completely loop-invariant
      }

      // Find the maximum depth of contained loop variables
      int max_depth = -1;
      for (const VarNode *var : contained_vars) {
        auto it = var_to_depth.find(var);
        if (it != var_to_depth.end()) {
          max_depth = std::max(max_depth, it->second);
        }
      }

      // Calculate effective parallel factor: product of trip counts for loops
      // with depth <= max_depth
      int64_t effective_factor = 1;
      for (const auto &loop : loop_stack) {
        if (loop.depth <= max_depth) {
          effective_factor *= loop.trip_count;
        }
      }

      return effective_factor;
    }

    void VisitExpr_(const AddNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->add_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->sub_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MulNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mul_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const DivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const ModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorDivNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->div_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const FloorModNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->mod_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const LENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GTNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const GENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const EQNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NENode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->cmp_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const AndNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const OrNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const NotNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->logic_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MinNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const MaxNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));
      update_operation(params->min_max_throughput, contained_vars);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      auto contained_vars = AnalyzeContainedLoopVars(ffi::GetRef<PrimExpr>(op));

      // Check for special math functions by name
      if (op->op.as<OpNode>()) {
        auto op_node = op->op.as<OpNode>();
        std::string op_name = op_node->name;

        // Check for special math functions
        if (op_name == "exp2" || op_name == "log2" || op_name == "exp" ||
            op_name == "log" || op_name == "sin" || op_name == "cos" ||
            op_name == "tan" || op_name == "asin" || op_name == "acos" ||
            op_name == "atan" || op_name == "sinh" || op_name == "cosh" ||
            op_name == "tanh" || op_name == "sqrt" || op_name == "rsqrt" ||
            op_name == "pow" || op_name == "erf" || op_name == "sigmoid") {
          // Special math functions have lower throughput
          update_operation(params->special_func_throughput, contained_vars);
        } else if (op_name.find("copy") != std::string::npos ||
                   op_name.find("gemm") != std::string::npos ||
                   op_name.find("tma_load") != std::string::npos ||
                   op_name.find("tma_store") != std::string::npos) {
          // Ignore copy, gemm, tma_load, tma_store operations
          // Just visit arguments but don't count the call itself
        } else {
          // Default: assume it's a regular function call with basic throughput
          update_operation(params->cuda_core_operation, contained_vars);
        }
      } else {
        // Not an OpNode, use default throughput
        update_operation(params->cuda_core_operation, contained_vars);
      }
      // Visit arguments
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ForNode *op) final {
      // Calculate trip count for the loop
      int64_t trip_count = 1;
      PrimExpr loop_extent = op->extent;
      PrimExpr loop_step = op->step.has_value() ? op->step.value()
                                                : IntImm(DataType::Int(32), 1);

      // Try to get constant values
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        if (const int64_t *step_ptr = as_const_int(loop_step)) {
          int64_t extent = *extent_ptr;
          int64_t step = *step_ptr;
          if (step > 0) {
            // ceil(extent / step) = (extent + step - 1) / step
            trip_count = (extent + step - 1) / step;
          } else {
            trip_count = extent; // Invalid step, use extent
          }
        } else {
          trip_count = 100; // Non-constant step, use default
        }
      } else {
        trip_count = 100; // Non-constant extent, use default
      }

      // Create loop dimension information
      LoopDimension loop_dim{.var = op->loop_var.get(),
                             .trip_count = trip_count,
                             .depth = static_cast<int>(loop_stack.size())};

      // Push loop onto stack and update mapping
      loop_stack.push_back(loop_dim);
      var_to_depth[op->loop_var.get()] = loop_dim.depth;

      // Visit loop body
      StmtExprVisitor::VisitStmt_(op);

      // Pop loop from stack
      var_to_depth.erase(op->loop_var.get());
      loop_stack.pop_back();
    }

    void VisitStmt_(const EvaluateNode *op) final {
      if (op->value.defined()) {
        StmtExprVisitor::VisitExpr(op->value);
      }
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      // Count operations in indices
      for (const auto &index : op->indices) {
        StmtExprVisitor::VisitExpr(index);
      }
      // Count operations in value
      StmtExprVisitor::VisitExpr(op->value);
    }

    void VisitStmt_(const SeqStmtNode *op) final {
      for (const auto &child : op->seq) {
        StmtExprVisitor::VisitStmt(child);
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const LetStmtNode *op) final {
      // Let binding: the value expression is evaluated once
      StmtExprVisitor::VisitExpr(op->value);
      StmtExprVisitor::VisitStmt(op->body);
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      VisitExpr(op->condition);

      // For if-then-else branches, we need to take the maximum latency of both
      // paths Save current latency
      int64_t old_latency = total_latency;

      // Count latency in then branch
      StmtExprVisitor::VisitStmt(op->then_case);
      int64_t then_latency = total_latency;

      // Restore latency and count else branch
      total_latency = old_latency;
      if (op->else_case) {
        StmtExprVisitor::VisitStmt(op->else_case.value());
      }
      int64_t else_latency = total_latency;

      // Take the maximum of both branches
      total_latency = old_latency + std::max(then_latency - old_latency,
                                             else_latency - old_latency);
    }

    void VisitStmt_(const BlockNode *op) final {
      StmtExprVisitor::VisitStmt(op->body);
    }
  };

  // Helper function to calculate total bytes accessed in a region
  int64_t CalculateAccessBytes(const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    const Region &ranges = region->region;

    // Calculate total number of elements
    int64_t total_elements = 1;
    for (size_t i = 0; i < ranges.size(); ++i) {
      const auto &range = ranges[i];
      // Try to get constant extent if possible
      if (const auto *extent_int = range->extent.as<IntImmNode>()) {
        total_elements *= extent_int->value;
      } else if (i < buffer->shape.size()) {
        // For symbolic extents, use buffer shape as fallback (more
        // accurate than assuming extent=1). Loop-context arith bounds
        // would be even better but require a scope-aware visitor.
        if (const int64_t *shape_extent = as_const_int(buffer->shape[i])) {
          total_elements *= *shape_extent;
        } else {
          total_elements *= 1;
        }
      } else {
        // For non-constant extents, use a conservative estimate
        // Assume at least 1 element per dimension
        // TODO: Better estimation for symbolic extents
        total_elements *= 1;
      }
    }

    // Get data type size in bytes
    DataType dtype(buffer->dtype);
    int64_t element_size = dtype.bytes();

    return total_elements * element_size;
  }

  // Estimate latency for a single memory access
  int64_t EstimateMemoryAccessLatency(const BufferRegion &region,
                                      bool is_read) {
    const Buffer &buffer = region->buffer;
    String scope = buffer.scope();
    MemoryType mem_type = GetMemoryTypeFromScope(scope);

    int64_t access_bytes = CalculateAccessBytes(region);

    switch (mem_type) {
    case MemoryType::kGlobal:
      // Global memory latency depends on data size
      // Base latency + bandwidth-limited component
      // Latency = base_latency + bytes / bytes_per_cycle
      // Subtract cache line size (32 bytes) since first cache line has base
      // latency
      if (is_read) {
        // Base read latency + bandwidth component
        return params_.global_memory_read +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      } else {
        // Write latency usually lower
        return params_.global_memory_write +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
    case MemoryType::kShared:
      // Shared memory has high bandwidth, less sensitive to size
      // Subtract typical burst size (128 bytes) for base latency
      if (is_read) {
        return params_.shared_memory_read +
               (access_bytes - 128) / params_.shared_memory_bandwidth;
      } else {
        return params_.shared_memory_write +
               (access_bytes - 128) / params_.shared_memory_bandwidth;
      }
    case MemoryType::kRegister:
      return params_.register_access;
    case MemoryType::kTmem:
      if (params_.tmem_read == 0) {
        return params_.global_memory_read +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
      return is_read ? params_.tmem_read : params_.tmem_write;
    default:
      // Unknown memory type, use global memory as conservative estimate
      if (is_read) {
        return params_.global_memory_read +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      } else {
        return params_.global_memory_write +
               (access_bytes - 32) / params_.global_memory_bandwidth;
      }
    }
  }
};

} // namespace tl
} // namespace tvm
