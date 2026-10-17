# misha W8A8 MoE + LoRA 迁移

源码来自目标容器 `ds-ltc-v25-ep` 中 vllm-ascend 的 `misha` 分支，固定提交
`72b7912616e0e296e64f0ca0003afcc2ef7ce6bc`。迁移仅使用已提交源码；容器里其他未跟踪的实验实现不属于本组件。
按用户确认的范围，不引入其他分支的算子、辅助函数或构建配置，也不在本次迁移中修复跨分支实验代码的依赖。容器现有实验文件保留原状。
文件来源、原始/迁移后 SHA256 见
[`provenance.json`](../ops/ascendc/moe_lora/provenance.json)。保留原文件的版权和许可声明。

## 执行与源码

```text
vllm_ascend.lora.ops_backend.get_lora_ops()
  → ops_repo.<同名接口>
  → torch.ops.ops_repo.<同名接口>
  → _lora_C + libops_repo_moe_lora_kernels.so
  → AscendC direct launch / 基础 CANN aclnnGroupedMatmulV4
```

7 个内核位于 `ops/ascendc/moe_lora/`：`bgmv_shrink`、`bgmv_expand`、
`sgmv_shrink`、`sgmv_expand`、`combined_lora_idx`、`add_lora_fused`、
`add_lora_swiglu_quant`。这些是原分支的直接启动内核，不使用 msopgen/OPP；
`scripts/ops.py` 继续管理有 `operator.json` 的 OPP 算子。

C++ 框架适配位于 `torch_custom/ops_repo/csrc/moe_lora/`，构建入口为相邻
`CMakeLists.txt`，Python 入口为 `ops_repo/moe_lora.py`。GMM fallback 使用基础
`libopapi.so`，无 vLLM Python 或原 vllm-ascend `.so` 依赖。
C++ namespace、设备入口名称及动态库名称均与原库隔离，可并存。

## 构建与安装

要求目标 Python 环境已有匹配的 torch、torch_npu、setuptools、wheel、CMake ≥ 3.19、
支持 C++17 的编译器与完整 CANN AscendC 工具链；不会自动升级框架。当前仅配置 A2/A3，
不支持 A5/310P。LoRA 组件无需 OpsRepoAdd OPP，可以单独构建 wheel：

```bash
git clone https://github.com/SkychenLee/ops-repo.git
cd ops-repo
# 在需要运行算子的 Python 环境中编译，CANN 路径按服务器实际安装位置调整。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910_93 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps --no-index \
  --wheel-dir dist/ascend910_93/lora-wheel .
python -m pip install --no-deps --force-reinstall \
  dist/ascend910_93/lora-wheel/ops_repo-0.1.0+ascend910.93-*.whl
python -c 'import ops_repo; print(ops_repo.__version__)'
```

无需复制原开发容器或其 `build/`、`dist/` 目录。源码与构建输入均在 Git 仓库中，
wheel 在目标服务器本地生成；CANN、驱动及匹配的 PyTorch/torch_npu 由目标环境提供。
若在容器中部署，请在该容器内执行以上命令。

`OPS_REPO_SOC` 指定产品族；直接编译所需的具体型号默认 A2=`ascend910b1`、
A3=`ascend910_9391`，可以用 `OPS_REPO_SOC_VERSION` 覆盖同产品族型号。
`MAX_JOBS` 默认为 4。包内 `_lora_build_info.json` 记录具体型号、Python、架构、
torch/torch_npu 版本，导入检查完全匹配的框架版本。导入不选择或初始化 NPU。
构建产物不是跨 PyTorch ABI、跨芯片通用包。

如需 Add + LoRA 的组合包，在既有完整 wheel 命令前增加
`OPS_REPO_BUILD_LORA=1`：

```bash
OPS_REPO_BUILD_LORA=1 python scripts/build_wheel.py \
  --soc ascend910_93 --ops ops_repo_add --jobs 4
```

原 Add-only 构建方式保持有效。LoRA-only wheel 不包含 `ops_repo.add` 的原生实现，
调用该接口会明确提示缺少 Add 组件。完整 Add+LoRA OPP 构建仍需另行验收。

## 对外接口与约束

| API | 行为 |
| --- | --- |
| `bgmv_shrink` / `sgmv_shrink` | 将 LoRA A 投影写入传入的 `y`，返回 `None` |
| `bgmv_expand` / `sgmv_expand` | 将 LoRA B 投影累加到 `y` 的指定切片，返回同一个 `y` |
| `add_lora` | 根据原分支条件选择融合 kernel、GMM 或 BGMV，原地更新 `y` |
| `add_lora_shrink` / `add_lora_expand` | 分阶段 MoE/fully-sharded LoRA，保持原路由参数和阈值 |
| `add_lora_expand_delta` | 生成独立 LoRA delta，不改写 `gate_up` |
| `add_lora_swiglu_quant` | delta add + SwiGLU + per-token INT8 quant，返回 `(act, y, scale)` |
| `build_combined_lora_idx` | 生成 adapter/expert 合并索引，无效 token 输出 `-1` |

函数参数名、顺序、默认值与迁移的 host 接口一致。可通过
`torch.ops.ops_repo.<name>.default._schema` 检查完整签名。
bgmv/sgmv 的 schema 已修正为只有输出 `y` 可变，expand 返回 alias；10 个接口均注册
C++ Meta 实现，支持 FakeTensor 的元数据推导。该修正不会改变数值算法。

保持上游推理约束：FP16/BF16，融合 LoRA rank 为 8/16/32/64，融合路径最多 4 个切片，
相应维度按 16 对齐且 `offset_start=0`；不满足融合条件时沿用原 fallback。
SwiGLU 输入 `[T, 2W]` 连续，`W % 64 == 0`；量化输出 INT8、scale FP32。
调用者仍需提供与原 vllm-ascend 一致的 dtype、shape、layout、路由和设备。
这是推理组件，没有反向实现。

## vllm-ascend 适配

当前容器中已修改实际调用点。可交付补丁见
[`integrations/vllm_ascend`](../integrations/vllm_ascend/README.md)。
backend 在首次实际 LoRA 调用时选择新库并缓存；没有安装 `ops_repo` 时使用旧 namespace，
安装了但缺少 LoRA 组件/加载失败时明确报错，不静默掩盖坏包。
no-LoRA 模块导入不会触发新库加载。改动后的 worker 需要重新启动才能加载新 Python 文件。

## 验证

```bash
# 主机离线测试，不要求 NPU
python3 -m unittest discover -s tests -v
python3 scripts/ops.py check

# 安装后的真实 C++ schema / Meta / FakeTensor，禁止设备初始化
OPS_REPO_REQUIRE_LORA_TESTS=1 python -m unittest discover -s tests -p test_moe_lora.py -v

# 后续有空闲设备时显式启用；本轮未运行
OPS_REPO_RUN_NPU_TESTS=1 OPS_REPO_TEST_DEVICE=npu:0 \
  python -m pytest tests/npu -q
```

硬件用例覆盖 rank 8/16/32/64、FP16/BF16、无 LoRA/无效 token、原地/alias、
分阶段调用、MoE adapter/expert 展平、路由索引和 SwiGLU 量化。
离线编译和 Fake/Meta 通过不代表 NPU 精度、ACL Graph 捕获、性能或完整模型通过。
本轮按用户要求不运行 NPU 测试。

### 本次验证记录（2026-10-17）

目标容器 `ds-ltc-v25-ep`，Python 3.12 / aarch64，torch `2.10.0+cpu`、
torch_npu `2.10.0.post4`、CANN 9.1.0；实际编译型号 `ascend910_9391`。

| 检查 | 结果 |
| --- | --- |
| 7 个 AscendC 内核与独立 PyTorch 扩展 | 真实编译、wheel 打包与 pip 安装成功 |
| wheel 内容、RECORD SHA256、16 份迁移源码指纹 | 通过 |
| 原生库链接 | `RUNPATH=$ORIGIN`；无 vLLM 依赖，设备入口有独立前缀 |
| 仓库 `unittest` | 80 tests，OK，2 项因主机缺少框架/原生组件跳过 |
| 已安装组件的 loader/schema/Meta/Fake/Dynamo | 13 tests，全部通过 |
| vllm-ascend backend 与 quant_moe 单测 | 25 tests，全部通过 |
| 已安装新库与实际 vLLM wrapper 联调 | 选择 `ops_repo`，FakeTensor/fullgraph 图中实际出现 `torch.ops.ops_repo.bgmv_*` |
| NPU 精度、ACL Graph、性能、完整模型 | 未执行 |

离线 Dynamo 测试关闭 accelerator stream 探测，并硬拦 NPU lazy initialization；
最终元数据及联调验证中 `torch.npu.is_initialized()` 保持 `False`。

产物：`dist/ascend910_93/lora-wheel/ops_repo-0.1.0+ascend910.93-cp312-cp312-linux_aarch64.whl`，
SHA256 `44fb412d32cfc7f51223e3dedc7e6d28b0d4eb79ca18a79f2d2b64335ee78168`。
同目录的 `build-receipt.json` 保存构建环境与校验记录。日志位于
`build/lora-build.log`、`build/offline-tests.log`、`build/native-metadata-tests.log`、
`build/migration/vllm-adapter-cpu-tests.log`、`build/migration/vllm-selector-fake-compile.log`。

当前容器原有 `vllm_ascend_C` 加载仍报
`undefined symbol: vllm_ascend_recover_ub_bytes`，且两个旧 `test_lora.py` 用例在
原始 HEAD 上也因 mock 返回值失败。缺符号来自现有未跟踪实验代码对其他分支辅助函数的引用，不属于本次迁移范围；未补入这些函数或对应构建配置。
这些环境及基线测试问题仅记录为完整模型验证的限制，详见适配目录 README。新库本身不依赖该旧动态库。
