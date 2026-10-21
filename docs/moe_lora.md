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
不支持 A5/310P。LoRA 组件无需 OpsRepoAdd OPP，可以单独构建 wheel。
先在目标机器上获取源码，并进入运行 vLLM 的同一个 Python 环境：

```bash
git clone https://github.com/SkychenLee/ops-repo.git
cd ops-repo
# 在需要运行算子的 Python 环境中编译，CANN 路径按服务器实际安装位置调整。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### A2 编译与安装

A2 使用 `OPS_REPO_SOC=ascend910b`。下面以 `ascend910b1` 为例；如果实际芯片为
910B2，将命令中的 `OPS_REPO_SOC_VERSION=ascend910b1` 改为
`OPS_REPO_SOC_VERSION=ascend910b2`，其他型号同样按实际芯片设置。

```bash
OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910b \
  OPS_REPO_SOC_VERSION=ascend910b1 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps --no-index \
  --wheel-dir dist/ascend910b/lora-wheel .
python -m pip install --no-deps --force-reinstall \
  dist/ascend910b/lora-wheel/ops_repo-0.1.0+ascend910b-*.whl
python -c 'import ops_repo; print(ops_repo.__version__, ops_repo.__file__)'
```

A2 wheel 在目标 A2 环境中生成，不复用 A3 的 wheel。A2（910B3）已完成本机
构建、安装与数值/图验证，见下方 2026-10-20 验证记录。

### A3 编译与安装

```bash
OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910_93 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps --no-index \
  --wheel-dir dist/ascend910_93/lora-wheel .
python -m pip install --no-deps --force-reinstall \
  dist/ascend910_93/lora-wheel/ops_repo-0.1.0+ascend910.93-*.whl
python -c 'import ops_repo; print(ops_repo.__version__)'
```

### 构建参数与部署

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
no-LoRA 模块导入不会触发新库加载。首次选择后每个 worker 输出一条库路径日志。
改动后的 worker 需要重新启动才能加载新 Python 文件。

当前模型联调场景为 `/mnt/share/litianchen/v25_ep/ds_16_v25.sh`：
vLLM `0.25.1`、model runner v1、TP8 + EP、MoE LoRA EP allgather、
rank16、max-seqs8、DSpark5 和 FULL_DECODE_ONLY，未启用 fully-sharded LoRA。
补丁支持 `additional_config.moe_lora_ep_backend="allgather"`，让该场景在 A3 上实际选用
已有的 EP AllGather 通信与本地专家路由。默认 `"auto"` 保持 misha 原硬件选择逻辑；
其他值在配置阶段报错。所有代码均来自当前已跟踪实现的适配，不引入实验分支文件。

脚本中的 `VLLM_ASCEND_DISABLE_PIN_MEMORY=1` 已接入平台能力判断，必须在启动进程前设置。
它让真实 LoRA checkpoint 加载跳过 pinned host memory，避免本机八 worker 加载大适配器时
出现的 `aclrtMallocHostWithCfg` 分配失败；默认值 `0` 保持原有 pinning 行为。

## 验证

```bash
# 主机离线测试，不要求 NPU
python3 -m unittest discover -s tests -v
python3 scripts/ops.py check

# 安装后的真实 C++ schema / Meta / FakeTensor，禁止设备初始化
OPS_REPO_REQUIRE_LORA_TESTS=1 python -m unittest discover -s tests -p test_moe_lora.py -v

# 有空闲设备时显式启用；ASCEND_RT_VISIBLE_DEVICES 可限制到指定物理卡
OPS_REPO_RUN_NPU_TESTS=1 OPS_REPO_TEST_DEVICE=npu:0 \
  python -m pytest tests/npu -q
```

硬件用例覆盖 rank 8/16/32/64、FP16/BF16、无 LoRA/无效 token、原地/alias、
分阶段调用、MoE adapter/expert 展平、路由索引和 SwiGLU 量化。
图用例在非默认流捕获后更新固定地址的输入，重复回放 8 次，并比较独立 CPU 参考值。
TP8 专项用单卡拼接模拟 rank2 shrink → rank16 expand 的 all-gather 布局；
它不替代实际跨卡通信验证。EP 专项采用本地 32 个 expert、topk6、H4096/I2048/rank16、
max-loras3，覆盖 8/48 token decode、64 token GMM prefill、空 expert 分组和无 LoRA。
fully-sharded w2 专项覆盖输入分片的部分 shrink 求和（all-reduce 布局）、
partial_expand 的 tp_rank 输出分片、expand 对 shrink 缓冲的只读约束，
以及 rank8 适配器在 max-lora-rank 16 下的零填充等价。
算子正确性和图测试不代表完整模型精度或性能通过。

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
| EP 配置与通信选择器 | 新增 27 项 + 原有 25 项，全部通过；显式 allgather 与默认策略均覆盖 |
| pinned memory 策略及真实 LoRA loader | 5 项通过，包含独立进程启用/禁用验证；上述 CPU 联合回归共 82 项 |
| 已安装新库与实际 vLLM wrapper 联调 | 选择 `ops_repo`，FakeTensor/fullgraph 图中实际出现 `torch.ops.ops_repo.bgmv_*` |
| NPU 数值与行为 | 132 项通过（FP16/BF16、融合与 fallback、路由、量化） |
| NPUGraph 捕获与更新输入后回放 | 23 项通过，包括组合图、TP8 rank2 → rank16 布局与 EP 真实维度 |
| 干净 misha 原算子与新库同输入差分 | 16 项输出逐位一致，最大绝对差 0 |
| v1 TP8 + EP 模型启动 | 14 个 full-decode 图捕获成功；8 个 worker 均加载新库的 binding 和 kernel `.so` |
| `ds` / `lora1` HTTP 请求 | 32/32 通过，各模型 16 个；包含长短输入、多轮对话及三轮 8 并发混合请求 |
| 性能基准 | 未执行 |

离线 Dynamo 测试关闭 accelerator stream 探测，并硬拦 NPU lazy initialization；
最终元数据及联调验证中 `torch.npu.is_initialized()` 保持 `False`。

产物：`dist/ascend910_93/lora-wheel/ops_repo-0.1.0+ascend910.93-cp312-cp312-linux_aarch64.whl`，
SHA256 `44fb412d32cfc7f51223e3dedc7e6d28b0d4eb79ca18a79f2d2b64335ee78168`。
同目录的 `build-receipt.json` 保存构建环境与校验记录。日志位于
`build/lora-build.log`、`build/offline-tests.log`、`build/native-metadata-tests.log`、
`build/migration/vllm-adapter-cpu-tests.log`、`build/migration/vllm-selector-fake-compile.log`。

容器原有 `vllm_ascend_C` 曾因未跟踪实验源码引用其他分支的辅助函数，报
`undefined symbol: vllm_ascend_recover_ub_bytes`。已从相同 `misha` 提交及其锁定子模块
导出干净源码，重编原 binding/direct kernels 并备份替换旧二进制，未引入其他分支代码。
原实验文件保持不变；新旧 namespace 可同时加载，10 个 legacy API 注册完整。
具体复现步骤与备份目录见 [适配说明](../integrations/vllm_ascend/README.md)。
两个旧 `test_lora.py` 用例在原始 HEAD 上因 mock 返回值失败的问题仍按基线问题记录。

NPU 和图日志位于 `build/migration/npu-validation/`，`all-155-npu.log` 记录全部 155 项联合通过，
`metrics.json` 保存图回放误差。
`build/migration/legacy-vs-ops-repo-npu9.{py,log,json}` 保存原库/新库差分及复现脚本。
这些日志是本机验证产物，不随源码提交。

### 验证记录（2026-10-20，A2/910B3）

目标机器 8×Ascend 910B3，Python 3.12 / aarch64，torch `2.10.0+cpu`、
torch_npu `2.10.0.post2`、CANN 9.0.1；本机以
`OPS_REPO_SOC=ascend910b OPS_REPO_SOC_VERSION=ascend910b3` 构建并安装 wheel。

| 检查 | 结果 |
| --- | --- |
| 仓库 `unittest`（offline） | 85 tests，OK（1 项跳过） |
| 已安装组件的 loader/schema/Meta/Fake | 13 tests，全部通过 |
| NPU 数值与图 | 164 项通过（155 项既有 + 9 项新增，见下） |
| 新增精度用例 | fully-sharded w2 输入分片（bgmv/GMM）、partial_expand tp_rank 分片与求和、expand 对 shrink 缓冲只读、rank8 零填充至 max-rank16 等价 |
| 与 vllm-ascend 集成回归 | `tests/ut/lora` 289 通过 / 5 xfailed；4 项失败用例在同一提交的 legacy 后端上同样失败，确认为用例契约过期而非迁移缺陷 |
| 干净 misha 原算子与新库同输入差分 | 14 条生产路径输出逐位一致（fused w13/w2、two-op bgmv/GMM、SwiGLU 量化、combined 索引） |

新增用例位于 `tests/npu/test_fully_sharded_accuracy.py` 与
`tests/npu/test_moe_lora.py::test_rank8_zero_padded_to_max_rank16`，自 vllm-ascend
`tests/ut/lora/test_fully_sharded_two_op.py`、`test_add_lora_fused_moe.py` 移植；
跨卡通信仍以单卡 cat/求和模拟布局，不替代实际跨卡验证。

### `ds_16_v25.sh` 模型联调

使用 `/mnt/share/litianchen/v25_ep/ds_16_v25.sh` 的启动配置，模型为
`/mnt/weight/DeepSeek-V4-Flash-0731-w8a8`，物理卡 8–15，端口 8888。
本次 vLLM 源码为 `752a3a504485790a2e8491cacbb35c137339ad34`（0.25.1）。
原脚本 LoRA 路径 `/mnt/share/remote_share/litianchen/ds_lora_16` 不存在，
启动副本中替换为实际存在的 `/mnt/share/litianchen/ds_lora_16`；原脚本未修改。
保留 max-model-len128000、max-batched-tokens16384、max-seqs8、rank16、max-loras3、
DSpark5、EP allgather 及 FULL_DECODE_ONLY 设置。

`/health` 返回 200，`/v1/models` 注册 `ds` 和 `lora1`。32 个请求均返回非空结果，
`finish_reason` 均为 `stop`，成功运行的服务日志没有 ERROR。
长输入约 1949 prompt tokens；这次联调没有进行 128K 压测、完整模型精度评估或性能基准。
8 个 worker 的 `/proc/<pid>/maps` 均包含 `ops_repo/_lora_C*.so` 和
`ops_repo/libops_repo_moe_lora_kernels.so`。本次 worker 启动后才修正了 backend INFO
日志的 logger，因此新库加载证据使用进程映射记录；后续重启会输出一次选择日志。

复现请求使用 [`examples/check_vllm_lora_http.py`](../examples/check_vllm_lora_http.py)，
具体启动与验证命令见 [适配说明](../integrations/vllm_ascend/README.md)。本次本机记录位于
`build/migration/model-validation/v1-ep-20261017-113246/`：`launch.sh` 保存启动副本，
`server.log` 保存模型日志，`backend-library-maps.json` 保存全部 worker 的新库映射，
`http-smoke.json` 保存完整请求与响应，`http-smoke.log` 保存请求摘要。
