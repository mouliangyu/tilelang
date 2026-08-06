/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_LATENCY_ESTIMATOR_PARAMS_H_
#define TVM_TL_ASCEND_TRANSFORM_LATENCY_ESTIMATOR_PARAMS_H_

#include <cstdint>

namespace tvm {
namespace tl {

struct AscendLatencyParams {
  // Convention: latency is in cycles, bandwidth in bytes/cycle, and compute
  // throughput in operations/cycle (MACs/cycle for Cube).

  // Cube (MMAD). The physical FP16/BF16 granularity and throughput come from
  // the A5 architecture guide p.7: 16x16x16x2 = 8192 operations/cycle. The
  // dispatch overhead is measured; one MMAD can be issued each cycle.
  int64_t cube_base_latency{8};
  int64_t cube_fallback_throughput{8192};
  int64_t cube_min_ii{1};
  int64_t cube_pipeline_depth{1};
  int64_t cube_unit_m{16};
  int64_t cube_unit_k{16};
  int64_t cube_unit_n{16};
  int64_t cube_throughput_fp16{8192};
  int64_t cube_throughput_bf16{8192};
  // A5 has no native FP32 MMAD; zero selects the fallback throughput.
  int64_t cube_throughput_fp32{0};
  // HiF8/FP8 uses 16x32x16x2 operations/cycle.
  int64_t cube_throughput_fp8{16384};
  // INT8 MMAD is not supported on A5; zero selects the fallback.
  int64_t cube_throughput_int8{0};

  // VALU throughput measured with cannsim RVEC instruction timing. Common
  // fp32 operations sustain about 51 ops/cycle; division, modulo, and special
  // functions use iterative implementations and are slower.
  int64_t add_throughput{51};
  int64_t sub_throughput{51};
  int64_t mul_throughput{51};
  int64_t div_throughput{13};
  int64_t mod_throughput{13};
  int64_t min_max_throughput{51};
  int64_t cmp_throughput{51};
  int64_t logic_throughput{51};
  int64_t special_func_throughput{8};
  // Unknown operations use the OperationCounter one-cycle fallback.
  int64_t default_operation_throughput{0};
  // A5 architecture guide p.8: VRF <-> UB is 256 bytes/cycle per AIV.
  int64_t valu_bandwidth{256};
  int64_t vector_queue_depth{16};

  // DMA/FixPipe issue latency. MTE1 and MTE2 are measured with cannsim
  // trace_tools; MTE3 is scaled from the MTE2 correction; FixPipe is measured
  // on a 64KB FP32 L0C-to-destination transfer.
  int64_t mte1_base_latency{5};
  int64_t mte2_base_latency{634};
  int64_t mte3_base_latency{418};
  int64_t fixpipe_base_latency{730};
  // Descriptor overhead is already absorbed into measured base latency.
  int64_t mte_descriptor_cycles{0};

  // Physical path bandwidths from the A5 data-path table (p.4): L1 can feed
  // L0A/L0B at 256 B/cycle and BT/FP at 32 B/cycle; AIC MTE2 reaches L1 at
  // 256 B/cycle; each AIV MTE2/MTE3 path sustains 128 B/cycle.
  int64_t l1_to_l0a_bandwidth{256};
  int64_t l1_to_l0b_bandwidth{256};
  int64_t l1_to_bt_bandwidth{32};
  int64_t l1_to_fp_buf_bandwidth{32};
  int64_t aic_mte2_to_l1_bandwidth{256};
  int64_t aiv_mte2_to_ub_bandwidth{128};
  int64_t aiv_mte3_to_gm_bandwidth{128};
  int64_t aiv_mte3_to_aic_l1_bandwidth{128};
  int64_t fixpipe_bandwidth{128};

  // Measured two-segment DMA latency model:
  //   latency(bytes) = bytes <= knee ? flat : seg_base + bytes / seg_bw.
  // GM->UB and UB->GM are fitted from 16B..256KB cannsim sweeps. GM->L1 has a
  // wider flat region because its issue queue hides transfers through 8KB.
  int64_t mte2_gm_to_ub_knee{512};
  int64_t mte2_gm_to_ub_flat{656};
  int64_t mte2_gm_to_ub_seg_base{743};
  int64_t mte2_gm_to_ub_seg_bw{105};
  int64_t mte3_ub_to_gm_knee{512};
  int64_t mte3_ub_to_gm_flat{445};
  int64_t mte3_ub_to_gm_seg_base{478};
  int64_t mte3_ub_to_gm_seg_bw{103};
  int64_t mte2_gm_to_l1_knee{8192};
  int64_t mte2_gm_to_l1_flat{986};
  int64_t mte2_gm_to_l1_seg_base{837};
  int64_t mte2_gm_to_l1_seg_bw{98};
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_LATENCY_ESTIMATOR_PARAMS_H_
