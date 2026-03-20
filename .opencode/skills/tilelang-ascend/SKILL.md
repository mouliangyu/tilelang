---
name: tilelang-ascend
description: >
  TileLang Ascend NPU backend development guide. Covers SimtVF programming model,
  bisheng compiler integration, codegen customization, and debugging techniques
  for Ascend 910 series NPUs. Use this skill when developing or debugging
  TileLang kernels for Huawei Ascend hardware.
---

# TileLang Ascend 后端开发指南

本文档涵盖 TileLang 在华为昇腾 NPU 上的后端开发，包括 SimtVF 编程模型、bisheng 编译器集成、代码生成定制和调试技术。

---

## 1. 架构概览

### 1.1 Ascend 编译流程

```
TileLang TIR
    ↓ lower (phase.py)
SplitHostDevice → MakePackedAPI → LowerDeviceKernelLaunch
    ↓
Ascend CodeGen (codegen_ascend.cc)
    ↓
.asc 源文件 (bisheng CCE-C 方言)
    ↓ bisheng 编译
.so 动态库
    ↓ CythonKernelAdapter
PyTorch 调用
```

### 1.2 关键文件位置

| 组件 | 路径 |
|------|------|
| Target 规范化 | `tilelang/utils/target.py` |
| 编译流程 | `tilelang/engine/phase.py`, `tilelang/engine/lower.py` |
| C++ 代码生成 | `src/target/codegen_ascend.cc`, `src/target/codegen_ascend.h` |
| Runtime 入口 | `src/target/rt_mod_ascend.cc` |
| bisheng 集成 | `tilelang/contrib/bisheng.py` |
| Kernel 适配器 | `tilelang/jit/adapter/wrapper.py` (TLAscendSourceWrapper) |
| 设备适配 | `tilelang/jit/adapter/cython/adapter.py` |
| 模板头文件 | `src/tl_templates/ascend/common.h` |
| Lowering 修改 | `src/transform/lower_device_kernel_launch.cc`, `src/transform/split_host_device.cc` |

---

## 2. SimtVF 编程模型

### 2.1 SimtVF 概念

SimtVF (SIMT Vector Function) 是昇腾的 SIMT 风格编程模型。与 CUDA 的关键区别：

| 特性 | CUDA | Ascend SimtVF |
|------|------|---------------|
| 线程指定 | `T.Kernel(grid, threads=N)` | `T.SimtVF(threads=N)` |
| 线程索引 | `threadIdx.x/y/z` 直接使用 | `threadIdx` 在 SimtVF 区域内使用 |
| Block 维度 | `<<<grid, block>>>` | `<<<grid, 0, stream>>>` (无 block 参数) |
| Kernel 签名 | `__global__ void` | `__global__ __vector__ void` |
| 内存限定符 | 无 | `__gm__` (global), `__ubuf__` (shared) |

SimtVF 单独抽出的设计允许一个 kernel 中有多个 VF（Vector Function），类似 TMA 的 MTE 访存操作可以在 VF 外部调用。这种分离使得访存和计算可以更好地重叠执行。

### 2.2 基础 SimtVF Kernel

```python
import tilelang
import tilelang.language as T

def vector_add(N):
    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1) as _:           # grid 维度
            with T.SimtVF(threads=128):   # 线程数
                for i in T.Parallel(N):
                    C[i] = A[i] + B[i]
    return main

# 编译执行
kernel = tilelang.compile(vector_add(1024), target="ascend", out_idx=-1)
c = kernel(a, b)  # a, b 是 NPU 上的 tensor
```

### 2.3 生成的 Ascend 代码示例

```cpp
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

__simt_vf__ __launch_bounds__(128) inline void simt_vf_0(
    __gm__ float* C, __gm__ float* A, __gm__ float* B
) {
  int32_t simtvf_tx = threadIdx.x;
  for (int32_t i = 0; i < 2; ++i) {
    *(__gm__ float4*)(C + offset) = *(__gm__ float4*)(A + offset) 
                                  + *(__gm__ float4*)(B + offset);
  }
}

__global__ __vector__ void main_kernel(
    __gm__ float* A, __gm__ float* B, __gm__ float* C
) {
  pipe_barrier(pipe_t::PIPE_ALL);
  asc_vf_call<simt_vf_0>(cce::dim3(128), C, A, B);
  pipe_barrier(pipe_t::PIPE_ALL);
}
```

---

## 3. 内存限定符

### 3.1 Ascend 内存类型

| 限定符 | 对应内存 | 用途 |
|--------|---------|------|
| `__gm__` | Global Memory (HBM) | 输入输出 buffer |
| `__ubuf__` | Unified Buffer (片上) | 共享内存/临时 buffer |

### 3.2 代码生成中的处理

在 `codegen_ascend.cc` 中，`PrintStorageScope` 负责映射：

```cpp
void CodeGenTileLangAscend::PrintStorageScope(
    const std::string& scope, std::ostream& os
) {
  if (scope == "global") {
    os << "__gm__ ";
  } else if (scope == "shared") {
    os << "__ubuf__ ";
  } else if (!scope.empty()) {
    LOG(FATAL) << "Unsupported storage scope: " << scope;
  }
}
```

### 3.3 常见问题

**问题**: 编译错误 "pointer arguments must reside in __gm__ or __ubuf__"

**原因**: SimtVF helper 函数的指针参数缺少内存限定符。

**解决**: 确保 `codegen_ascend.cc` 中为 captures 注册了正确的 scope：

```cpp
for (const auto &v : captures) {
  auto scope = GetPtrStorageScope(v);
  alloc_storage_scope_[v.get()] = scope;
}
```

---

## 4. bisheng 编译器集成

### 4.1 编译命令

```bash
bisheng -fPIC --shared source.asc \
  --npu-arch=dav-3101 \
  -I<tilelang>/src \
  -o output.so
```

### 4.2 环境变量

| 变量 | 说明 |
|------|------|
| `ASCEND_NPU_ARCH` | NPU 架构版本，如 `dav-3101` |
| `PATH` | 需要包含 bisheng 编译器路径 |

### 4.3 架构版本

| 芯片 | npu-arch |
|------|----------|
| Ascend 910B | `dav-3101` |

---

## 5. 代码生成定制

### 5.1 修改 Kernel 函数签名

在 `codegen_ascend.cc` 中：

```cpp
void CodeGenTileLangAscend::AddFunction(const PrimFunc &f) {
  // 1. 注册参数 scope
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      auto scope = GetPtrStorageScope(param);
      alloc_storage_scope_[param.get()] = scope;
    }
  }

  // 2. 生成函数签名
  this->stream << "__global__ __vector__ ";
  for (size_t i = 0; i < f->params.size(); ++i) {
    // 使用 PrintStorageScope 添加内存限定符
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, stream);
    }
    // ...
  }
}
```

### 5.2 Lowering Pass 修改

**lower_device_kernel_launch.cc**: Ascend 跳过 same_device_type 优化

```cpp
bool is_ascend_callee = callee_target->HasKey("ascend");
if (same_device_type && !is_ascend_callee) {
  // call_extern path
}
// Ascend 走 call_packed 路径
```

**split_host_device.cc**: Ascend kernel 返回 void

```cpp
bool can_propagate_errors = [&]() {
  if (device_target->HasKey("ascend")) {
    return false;  // void return type
  }
  auto kind = device_target->GetTargetDeviceType();
  return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
}();
```

---

## 6. 运行时适配

### 6.1 设备适配 (adapter.py)

```python
def _process_buffer_device(self):
    if is_cpu_target(self.target):
        if is_ascend_target(self.target):
            device = torch.device("npu")
        else:
            device = torch.device("cpu")
```

### 6.2 Wrapper 适配 (wrapper.py)

```python
class TLAscendSourceWrapper(TLCUDASourceWrapper):
    def get_stream_type(self):
        return None  # Ascend 不需要显式 stream

    def create_dispatch_func(self, code, function_informations):
        # Ascend: <<<grid, smem, stream>>>
        grid_dim = f"({grid_x} * {grid_y} * {grid_z})"
        kernel_launch_code += (
            f"\t{name}<<<{grid_dim}, {smem}, nullptr>>>({args});\n"
        )
```

---

## 7. 调试技巧

### 7.1 常见编译错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `expecting parameter to have __gm__` | 函数参数缺少内存限定符 | 检查 `AddFunction` 中的 scope 输出 |
| `no viable conversion from 'dim3' to 'unsigned int'` | grid 参数类型错误 | 使用整数而非 dim3 |
| `use of undeclared identifier 'asc_vf_call'` | 缺少头文件 | `common.h` 包含 `simt_api/asc_simt.h` |
| `casting '__gm__ float *' to 'float4 *' not allowed` | 向量化转换缺限定符 | 实现 `PrintStorageScope` |
| `calling_conv != kDeviceKernelLaunch` | calling convention 不匹配 | 检查 `rt_mod_ascend.cc` |

### 7.2 NPU 状态检查

```bash
# 查看 NPU 信息
sudo npu-smi info

# 查看显存使用
sudo npu-smi info -m
```

### 7.3 运行调试

```bash
# 设置 Ascend 日志
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 运行测试
sudo -E PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  ASCEND_NPU_ARCH=dav-3101 \
  python test.py
```

---

## 8. 测试用例

### 8.1 最小测试

```python
"""examples/ascend/example_simtvf_vector_add.py"""
import tilelang
import tilelang.language as T
import torch
import torch_npu

def vector_add(N):
    @T.prim_func
    def main(A: T.Buffer((N,), "float32"),
             B: T.Buffer((N,), "float32"),
             C: T.Buffer((N,), "float32")):
        with T.Kernel(1) as _:
            with T.SimtVF(threads=128):
                for i in T.Parallel(N):
                    C[i] = A[i] + B[i]
    return main

if __name__ == "__main__":
    N = 1024
    device = torch.device("npu")
    kernel = tilelang.compile(vector_add(N), target="ascend", out_idx=-1)
    
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    c = kernel(a, b)
    
    expected = a + b
    assert torch.equal(c, expected), "Verification failed!"
    print("SUCCESS!")
```

---

## 9. 开发工作流

### 9.1 初次配置

```bash
# 0. 配置 Ascend 环境变量 (进入开发环境前执行)
source /usr/local/Ascend/cann-9.0.T530/set_env.sh

# === 检查环境是否已就绪 ===
cd tilelang
if [ -d .venv-ascend ] && python -c "import tilelang; import torch_npu" 2>/dev/null; then
    echo "Environment already configured, skipping setup"
    source .venv-ascend/bin/activate
else
    # 1. 安装 uv 包管理器
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    # 2. 安装独立 Python (避免系统 Python 头文件缺失问题)
    uv python install 3.11

    # 3. 创建虚拟环境
    uv venv --python 3.11.15 .venv-ascend
    source .venv-ascend/bin/activate

    # 4. 安装 PyTorch 和 torch_npu (使用项目预编译的 wheel)
    uv pip install torch==2.8.0
    uv pip install /path/to/torch_npu-2.8.0.post3.dev20260127-cp311-cp311-manylinux_2_28_aarch64.whl

    # 5. 安装构建依赖
    uv pip install scikit-build-core cython z3-solver patchelf cmake ninja

    # 6. Editable install (禁用 CUDA，启用 Ascend)
    USE_CUDA=OFF CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install -e . --no-build-isolation
fi

# 验证安装
python -c "import tilelang; print('tilelang:', tilelang.__version__)"
python -c "import torch; import torch_npu; print('torch_npu OK')"
```

### 9.2 添加新功能

```bash
# 1. 修改代码生成
vim src/target/codegen_ascend.cc

# 2. 重建
USE_CUDA=OFF uv pip install -e . --no-build-isolation

# 3. 测试
sudo -E ASCEND_NPU_ARCH=dav-3101 python test.py
```

### 9.3 常见修改点

| 需求 | 文件 |
|------|------|
| 修改 kernel 签名 | `codegen_ascend.cc:AddFunction` |
| 修改 SimtVF 区域 | `codegen_ascend.cc:VisitStmt_(const ForNode*)` |
| 修改内存限定符 | `codegen_ascend.cc:PrintStorageScope` |
| 修改 host wrapper | `wrapper.py:TLAscendSourceWrapper` |
| 修改设备类型 | `adapter.py:CythonKernelAdapter` |
| 修改编译参数 | `tilelang/contrib/bisheng.py` |
| 修改头文件 | `src/tl_templates/ascend/common.h` |

---

## 10. 已知限制

1. **仅支持 SimtVF**: 当前仅实现了 `T.SimtVF` 编程模型
2. **无 warp 级操作**: 不支持 CUDA 风格的 warp shuffle
3. **无共享内存分配**: 暂不支持 `T.alloc_shared`
4. **有限的数据类型**: 主要支持 float32/bfloat16/float16
5. **需要 sudo**: NPU 操作需要管理员权限

---

## 11. 相关资源

- **TileLang 主文档**: 参考 `tilelang` skill
- **示例代码**: `examples/ascend/`
- **测试代码**: `testing/python/language/simtvf/`
