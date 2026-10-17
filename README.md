# Ascend 算子仓库

独立的 Ascend 自定义算子工程，面向 **A2 / A3 / A5** 管理编译目标。参考 FLA 的 AscendC / Triton 后端组织，提供供 PyTorch、vllm-ascend 调用的 `ops_repo` Python 包。

包含 `OpsRepoAdd`（同 shape、非空、连续 FP32 张量逐元素相加），以及从 vllm-ascend `misha` 迁移的 **W8A8 MoE + LoRA 组件**（7 个直接启动 AscendC 内核、10 个 PyTorch 接口）。LoRA 组件可独立构建 wheel，接入目标容器中的 vllm-ascend，见 [迁移与使用说明](docs/moe_lora.md)。原 Add OPP 的三类芯片上机验收仍未完成，LoRA 本轮按要求不运行 NPU 验证；Triton 继续保留开发骨架。

下文的 msopgen、OPP、`OPS_REPO_BUILD_TORCH` 流程针对 Add 组件。MoE + LoRA 的独立构建入口是 `OPS_REPO_BUILD_LORA=1`，无需 Add OPP，具体命令及验证范围见上述迁移说明。

## 在其他服务器编译 MoE + LoRA

在已安装匹配的 torch、torch_npu、完整 CANN AscendC 工具链，以及 CMake ≥ 3.19、
C++17 编译器、setuptools 和 wheel 的 Python 环境中执行。以下以 A3 为例：

```bash
git clone https://github.com/SkychenLee/ops-repo.git
cd ops-repo
source /usr/local/Ascend/ascend-toolkit/set_env.sh
OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910_93 MAX_JOBS=4 \
  python3 -m pip wheel --no-build-isolation --no-deps --no-index \
  --wheel-dir dist/ascend910_93/lora-wheel .
python3 -m pip install --no-deps dist/ascend910_93/lora-wheel/*.whl
python3 -c 'import ops_repo; print(ops_repo.__version__)'
```

A2 使用 `OPS_REPO_SOC=ascend910b`；具体芯片型号可用 `OPS_REPO_SOC_VERSION` 指定。
该组件无需 Add OPP 或原 vllm-ascend 编译产物。A3 已完成真实编译与离线验证，
NPU 精度与性能仍待验证。详细环境、参数、硬件测试和 vllm-ascend 适配见
[MoE + LoRA 说明](docs/moe_lora.md)。

## 目录与调用路径

```text
ops/                             # 对应 FLA 的 fla/ops，按实现后端组织
  ascendc/<category>/<name>/
    operator.json                 # 接口定义、配置目标、硬件验证记录
    op_host/                      # 算子注册、shape/dtype 推导、tiling
    op_kernel/                    # AscendC 实现，可在内部按架构拆分
  triton/triton_core/             # Triton kernel 源码预留位置
torch_custom/ops_repo/            # 对应 FLA 的 torch_custom/fla_npu
  csrc/                          # PyTorch dispatcher → ACLNN 绑定
  ops_repo/                      # 安装后公开 import ops_repo
    ops/ascendc/                  # Python AscendC 入口
    ops/triton/                   # Python Triton 入口，当前未实现
setup.py                         # 根目录统一 Python 打包入口
scripts/ops.py                    # AscendC 开发与构建 CLI，命令见下文
scripts/build_wheel.py            # 编译 OPP、原生适配并生成完整 wheel
build.sh                         # scripts/ops.py build 的快捷入口
examples/                        # 原生 ACLNN、PyTorch、vllm-ascend 调用示例
docs/                            # 后端开发与 vllm-ascend 集成说明
tests/                           # 构建工具与 Python 接口测试
build/<soc>/cann_project/         # 本机 msopgen 生成的工程，不提交
dist/<soc>/                      # OPP run 包及 SHA256 构建记录，不提交
  wheel/                         # 包含 OPP 与原生适配的完整 wheel，不提交
.local/<soc>/opp/                # 本地 OPP 安装目录，不提交
```

```text
vllm-ascend 中的模型/算子调用点
  → ops_repo.add(x, y, backend="ascendc")
  → torch.ops.ops_repo.add
  → aclnnOpsRepoAdd
  → 自定义 OPP 中的 AscendC kernel
```

Python 包没有 vLLM 硬依赖。安装它不会自动替换模型计算，需要在选定模型的实际调用点使用接口。参考 [vllm-ascend 集成说明](docs/vllm_ascend.md) 和 [后端开发说明](docs/backends.md)。

现有示例位于 `ops/ascendc/elementwise/ops_repo_add/`。算子源码、框架适配、测试与调用示例分开维护，分组关系与 FLA 对照见 [目录说明](docs/layout.md)。Triton 源码只在 `ops/triton/triton_core/` 维护，打包映射到 `ops_repo.ops.triton.triton_core`；仓库根目录不会遮蔽已安装的 `ops_repo` 包。

## 开发与构建工具

`scripts/` 保存仓库的开发工具，核心入口是 [`scripts/ops.py`](scripts/ops.py)。它使用 Python 标准库，读取 `ops/ascendc/<category>/<name>/operator.json` 和对应源码，管理 AscendC 算子的创建、检查、工程生成、编译与安装。目前只管理 AscendC；运行时调用 `ops_repo.add()` 不经过该脚本。

在仓库根目录执行 `python3 scripts/ops.py <命令>`，可用 `--help` 或 `<命令> --help` 查看参数：

| 命令 | 用途与产物 |
| --- | --- |
| `list` | 校验清单并列出算子、分类、启用状态、配置目标和硬件验证记录 |
| `check` | 离线检查清单字段、命名与源码目录；通过不代表 NPU 编译或运行成功 |
| `doctor` | 检查 Python、CMake、C++ 编译器、make、CANN 头文件、AscendC 编译器与 msopgen |
| `new <name>` | 在 `ops/ascendc/<category>/<name>/` 创建清单、README 和 host/kernel 占位源码，默认禁用 |
| `generate` | 调用本机 msopgen 生成工程，放入仓库自有源码，输出到 `build/<soc>/cann_project/` |
| `build` | 检查环境、生成工程并用 CMake 编译；将 OPP `.run` 包及 SHA256 构建记录保存到 `dist/<soc>/` |
| `install` | 校验该芯片最近一次构建的安装包，将 OPP 安装到 `.local/<soc>/opp/` |

[`build.sh`](build.sh) 将参数原样转交给 `python3 scripts/ops.py build`。根 [`setup.py`](setup.py) 负责 Python 包打包，以及通过 `OPS_REPO_BUILD_TORCH=1` 启用的 PyTorch 原生适配层编译；编译原生适配层前，需要先用上述工具构建并安装 OPP。[`scripts/build_wheel.py`](scripts/build_wheel.py) 将 OPP 编译、临时安装、原生适配编译和 wheel 打包串成一次构建。新增算子源码后，Python 接口和 PyTorch 绑定仍需单独实现，详见下文「新增自己的算子」。

### 与 FLA 工具的对应关系

以下对照基于 `flashserve/flash-linear-attention-npu` 的固定提交 [`54ae09b`](https://github.com/flashserve/flash-linear-attention-npu/tree/54ae09b1243cd851d986c2624a57bbb5ee63dc27)。FLA 将相关职责分布在多个脚本和仓库内维护的 CMake 工程中：

| 本仓入口 | FLA 对应入口 | 对应范围与区别 |
| --- | --- | --- |
| `doctor` | [`scripts/check_npu_env.py`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/scripts/check_npu_env.py) | 构建与运行环境预检；FLA 默认还检查框架和 NPU 运行环境，本仓 `doctor` 不要求 torch |
| `list` / 部分 `check` | [`scripts/check_build_ops.py`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/scripts/check_build_ops.py)、`bash build.sh --list-ops` | 从源码中的 CMake 文件发现算子，列出或校验 `--ops` / `FLA_NPU_OPS` 名称；与本仓的 `operator.json` 清单检查范围不同 |
| `new` | [`scripts/opgen/opgen_standalone.py`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/scripts/opgen/opgen_standalone.py) | 复制内置 Add 模板、替换名称并生成源码和 CMake 骨架；对应创建新算子 |
| `generate` | 仓库内的 [`CMakeLists.txt`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/CMakeLists.txt) 与 `cmake/` | FLA 直接使用维护好的构建工程；本仓为已有算子调用 msopgen 生成临时 CANN 工程，与创建源码骨架是两个步骤 |
| `build` | [`build.sh`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/build.sh) | 通过 `--pkg`、`--soc`、`--ops` 等参数选择芯片与算子，编译 OPP `.run` 包 |
| `install` | 生成的 `.run` 安装器 | FLA 支持更新已安装 wheel 内嵌的 OPP，或指定安装路径；本仓 `ops.py install` 用于外置 OPP，完整 wheel 的 OPP 随 pip 安装 |
| 完整 wheel：`scripts/build_wheel.py` 与根 `setup.py` | [`scripts/build_wheel.py`](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/scripts/build_wheel.py) 与根 `setup.py` | 两者均提供 OPP、原生适配与 Python 层的一体打包；本仓使用普通 PyTorch C++ 扩展，要求匹配构建时的框架版本 |

FLA 的具体命令见其[开发者指南](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/docs/开发者指南.md)。本仓借鉴源码职责划分，自行实现了 `scripts/ops.py`，使用自己的构建参数和产物目录；本仓操作请使用下文命令。

## 目标芯片与环境

| 产品 | `--soc` | 默认生成器型号 | 当前验证状态 |
|---|---|---|---|
| A2 | `ascend910b` | `Ascend910B1` | 工程生成已验证，上机未验证 |
| A3 | `ascend910_93` | `Ascend910_93` | 工程生成已验证，上机未验证 |
| A5 | `ascend950` | `Ascend950` | 工程生成已验证，上机未验证 |

`--soc-version` 可覆盖同一目标下的具体型号，不允许跨芯片混配。`scripts/ops.py` 的构建、安装建议显式传 `--soc`，不传时默认 A2；`scripts/build_wheel.py` 则要求显式指定 `--soc`。每次构建只选择该目标下启用的算子，显式指定未配置该目标的算子会报错。

需要匹配目标芯片的完整 **CANN Toolkit、ops 包和 msopgen**，以及 Python 3.9+、CMake、C++ 编译器、make 等官方构建依赖。所需 CMake 版本以本机 CANN 生成工程为准。FLA 参考提交 `54ae09b` 的环境说明采用 A2/A3 CANN ≥ 8.5.2、A5 CANN ≥ 9.0.0，可作为选型参考；本仓尚未验证这些版本组合。Triton 后续需要与 CANN 匹配的 `triton-ascend`。

调用 PyTorch 接口还需要与 vllm-ascend 环境匹配的 `torch`、`torch_npu`。当前采用标准 PyTorch C++ 扩展绑定，需要随匹配的 PyTorch/torch_npu 环境编译，未实现 FLA 的 Stable ABI 解耦运行时。安装脚本不会自动下载或升级这些依赖。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按实际安装路径调整
python3 scripts/ops.py doctor
python3 scripts/ops.py list
python3 scripts/ops.py check
```

非默认路径可传 `--cann-path /path/to/toolkit`、`--msopgen /path/to/msopgen`。`doctor` 检查基本构建环境；API/目标芯片兼容性仍需实际构建和运行确认。

## 方式一：构建并安装完整 wheel

适合把算子交付给其他机器或容器。构建机器需要上文的完整开发环境，以及已安装的 `setuptools>=61`、`wheel` 和匹配的 `torch`、`torch_npu`。构建脚本不会自动安装或升级这些依赖。

```bash
# 在匹配部署环境的 Python/容器中，从仓库根目录执行。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 scripts/build_wheel.py --soc ascend910_93 --ops ops_repo_add --jobs 8
```

该命令会检查环境、编译本轮 OPP、校验 `.run` 包，将其安装到临时目录，再在干净的源码副本中编译原生适配并打包。成功后将完整 wheel 保存到 `dist/ascend910_93/wheel/`，并打印准确的安装命令。`--soc` 必填；`--cann-path`、`--msopgen`、`--soc-version` 可指定构建工具与芯片型号。目前公开绑定只有 `ops_repo_add`，完整 wheel 的算子选择必须包含它。

完整 wheel 包含：

- `ops_repo` Python 接口、Fake/Meta 适配和编译好的 `_C` 原生扩展。
- `ops_repo/opp/vendors/ops_repo/` 下的 OPP，包括自定义接口库、host/tiling、kernel 与配置。
- `_build_info.json`，记录目标 SoC、Python、主机架构和构建时的 PyTorch/torch_npu 版本。
- `ops_repo_opp_env.py` 与 `.pth` 启动配置，用于在导入 torch_npu 前设置包内 OPP 的搜索路径。

部署机器需要匹配的 NPU、系统架构、驱动/CANN 运行环境及构建时使用的 Python、PyTorch/torch_npu。wheel 文件名包含 SoC 的本地版本标识与 Python/平台标签；导入时还会检查记录的框架版本。**当前未采用 Stable ABI，不能将一个构建环境生成的 wheel 当作任意框架版本或芯片通用包。**

在只加载基础 CANN 环境的新终端中安装，路径替换为构建输出的准确文件名：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install --force-reinstall --no-deps "dist/ascend910_93/wheel/<实际文件名>.whl"

# 启动新 Python 进程后即可导入；无需单独安装 OPP 或 source vendor 环境。
python3 -c "from ops_repo import add; print('ops_repo loaded')"
# 仓库中的硬件验证示例：
python3 examples/check_torch_add.py --device 0
```

完整 wheel 默认自动发现包内 OPP，不需要设置 `OPS_REPO_OPP_PATH`。如果环境中残留另一份 `ops_repo` vendor 路径，会明确报冲突；请在只加载基础 CANN 的新终端中使用。若显式设置该变量，会选择外置 OPP，此时仍需按方式二在启动前加载 vendor 环境。旧的 Python/Notebook/worker 进程需要重启才能执行新安装的启动配置；只通过 `PYTHONPATH` 添加目录不会执行 `.pth`。自动配置仅处理本包 OPP，不安装 CANN 或框架，不选择 NPU 设备。

本机当前缺少完整 CANN 工具链与 torch/torch_npu，尚未产出或验证真实 NPU 完整 wheel。离线打包、启动配置与错误路径测试不能替代目标设备上的精度和运行验证。

## 方式二：分别构建 OPP 和 Python 包

### 构建 AscendC OPP

以下以 A3 为例；A2 换成 `ascend910b`，A5 换成 `ascend950`。`OPS_REPO_SOC` 在这些命令中用于 shell 传参，构建 CLI 仍以 `--soc` 为准。

```bash
export OPS_REPO_SOC=ascend910_93
bash build.sh --soc "$OPS_REPO_SOC" --ops ops_repo_add --jobs 8
python3 scripts/ops.py install --soc "$OPS_REPO_SOC"

export OPS_REPO_OPP_PATH="$PWD/.local/$OPS_REPO_SOC/opp"
source "$OPS_REPO_OPP_PATH/vendors/ops_repo/bin/set_env.bash"
```

请在启动 Python/vLLM 进程前加载 CANN 和 vendor 环境。安装目录按芯片隔离，但单个进程应选用与其设备匹配的一份 `ops_repo` OPP。相同芯片、相同 vendor 的新包会替换旧包；需要多个算子时，省略 `--ops` 构建该目标的全部已启用算子，避免用部分包覆盖完整包。

构建流程：向本机 `msopgen` 逐个传入接口定义 → 生成标准 OPP 工程 → 覆盖仓库自有源码 → 配置目标和 vendor → CMake 构建 binary/package → 保存 run 包和 SHA256。标准生成器使用 `-f tf -lan cpp`，脚本移除 framework 插件，只构建原生 ACLNN 自定义算子，不需要 TensorFlow。

只生成工程、暂不编译：

```bash
python3 scripts/ops.py generate --soc ascend910_93 --ops ops_repo_add
```

生成目录是 `build/<soc>/cann_project/`；**请始终修改 `ops/ascendc/` 下的源码**，生成成功会重新创建该目标的工程，失败则保留上次结果。旧版的 `operators/`、`build/cann_project/`、`.local/opp/` 不再是当前命令的工作目录。

### 安装 Python 包并调用

默认只安装 Python 层，不编译 C++ 扩展，也不安装 torch/torch_npu/Triton：

```bash
python3 -m pip install --no-build-isolation --no-deps .
```

包的安装与运行环境检查分开。`import ops_repo` 会自动加载 AscendC 运行时，因此仅安装纯 Python wheel 后，缺少 torch/torch_npu、原生扩展或 OPP 的导入仍会报错。使用前先完成上面的 OPP 构建安装，再在匹配的 torch/torch_npu 开发环境中编译绑定：

```bash
OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps .
```

```python
import torch
import torch_npu

torch.npu.set_device(0)
from ops_repo import add  # 导入自动完成运行时加载和 Fake 注册

with torch.inference_mode():
    x = torch.ones((1024,), device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    z = add(x, y, backend="ascendc")
```

也可直接 `from ops_repo.ops.ascendc import add`；同样会先完成包初始化。初始化封装在包内部，不需要用户调用 `initialize()`；该 API 保留为可重复调用的兼容接口。包不选择设备；外置 OPP 模式要求在进程启动前配置 CANN/vendor 环境，完整 wheel 自动配置包内 OPP。每个实际执行 worker 应在模型加载阶段、编译或图捕获之前完成导入。

接口当前仅用于推理，不包含反向实现。Triton 是同一包内的开发占位，不能绕过 AscendC 运行时独立使用；成功导入后调用 `backend="triton"` 仍明确报未实现，不会静默执行其他后端。FakeTensor 注册为图编译提供输出元数据；它本身不能证明 vLLM 图编译、NPU 图捕获或实际 kernel 已验证通过。

原生 ACLNN 调用不依赖 PyTorch：

```bash
cmake -S examples -B "build/$OPS_REPO_SOC/examples" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DOPS_REPO_SOC="$OPS_REPO_SOC" \
  -DOPS_REPO_OPP_PATH="$OPS_REPO_OPP_PATH"
cmake --build "build/$OPS_REPO_SOC/examples" --parallel 4
"./build/$OPS_REPO_SOC/examples/ops_repo_add" --device 0
```

## 新增自己的算子

```bash
python3 scripts/ops.py new fused_scale_add --category elementwise \
  --soc ascend910b ascend910_93 ascend950
```

这个命令创建 `ops/ascendc/elementwise/fused_scale_add/`。`--category` 默认 `custom`，支持 `gdn/chunk_fwd` 这样的多级分组。算子名必须在所有分组中唯一，`--ops` 仍使用算子名选择。新算子默认 `enabled=false`，host/kernel 占位源码带明确的编译错误，已有目录不会覆盖。只填写你计划实现的目标，不必每个算子都支持全部芯片。

1. 在 `operator.json` 固定数学接口、输入输出及属性，在 README 明确 shape、dtype、layout、广播、空输入、状态修改和精度要求。
2. 实现 `op_host/` 和 `op_kernel/`，将 `AddConfig` 与配置目标对应；核对每个目标的 API、数据类型及内存约束，必要时按架构分实现。
3. 实现完成后启用算子，运行 `check` 并分别构建。`soc` 表示配置目标；`hardware_validated` 仅登记已完成编译和上机验收的目标。
4. 在 `torch_custom/ops_repo/csrc/` 增加 ACLNN 绑定、dispatcher schema，在相邻的 `ops_repo/` Python 包中增加接口和 FakeTensor 实现；后续 Triton 实现放到 `ops/triton/triton_core/`，保持相同算子契约。`new` 只生成 AscendC 源码骨架，不会自动推导这些适配层。
5. 增加精度、尾块、异常、设备/stream、图编译测试，再修改 vllm-ascend 中选定模型的真实调用点。

各算子的源码会汇总到生成工程的 host/kernel 目录，文件名应带算子前缀，重名将报错。CMake 文件由本机 CANN 生成，无需每个算子自行编写。

## 验证与限制

```bash
python3 -m unittest discover -s tests -v
python3 scripts/ops.py check
```

离线测试覆盖工程工具、Python 接口、完整 wheel 的文件与 RECORD 校验、启动配置、外置 OPP 兼容和版本检查，不执行 NPU kernel。打包测试中的原生文件使用明确标记的合成产物，只验证归档与安装流程，不验证真实动态库链接。CPU PyTorch 环境中的元数据测试使用测试专用 schema 和隔离的运行时替身，检查 Meta、FakeTensor 与 Dynamo 追踪；它们不验证真实原生扩展。无 torch 环境会明确跳过元数据测试。

纯 Python wheel 可独立构建和安装，导入则要求完整运行环境；真实 msopgen 已生成 A2/A3/A5 工程。当前机器为 `Ascend910_9362`（A3），缺少完整 CANN Toolkit、AscendC 编译器和 CMake；**三个目标的 CANN 编译、C++ 扩展链接、NPU 精度/性能以及 vLLM 端到端验证均未完成**。

## 参考

- [FLA 后端与芯片组织，参考提交 54ae09b](https://github.com/flashserve/flash-linear-attention-npu/tree/54ae09b1243cd851d986c2624a57bbb5ee63dc27)。本仓独立实现，没有 FLA 运行时依赖。
- [Ascend/fbgemm-ascend 多目标构建](https://github.com/Ascend/fbgemm-ascend/blob/main/FbgemmAscend.cmake)。
- [CANN 工程生成与追加](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0021.html)、[编译与部署](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0024.html)。
- 生成的 CANN 工程遵循生成器自带许可证，保留在不提交的 `build/` 中；不复制 FLA 或 CANN 工程模板源码。
