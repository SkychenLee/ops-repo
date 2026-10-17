# 后端与交付边界

本仓以统一的 Python API 管理自己的算子，AscendC 实现通过自有 ACLNN 接口和 PyTorch 扩展运行；Triton 预留独立后端入口。它们共享算子语义，但编译、依赖和硬件支持需要分别验证。

计算源码位于 `ops/ascendc/<category>/<op>/` 和 `ops/triton/triton_core/`，Python 适配位于 `torch_custom/ops_repo/ops_repo/`，C++ 下发适配位于 `torch_custom/ops_repo/csrc/`。根 `setup.py` 统一打包，对外仍使用 `import ops_repo`，详见 [目录分层](layout.md)。

```python
from ops_repo import add  # 导入时自动加载 AscendC 运行时与 Fake 注册

output = add(x, y, backend="ascendc")
```

`import ops_repo` 或 `from ops_repo.ops.ascendc import add` 都会执行包内的 AscendC 初始化：确定包内或显式外置 OPP、检查完整 wheel 的构建兼容信息、加载 torch/torch_npu、原生扩展与 API 动态库并注册 Fake 实现。不需要额外调用 `initialize()`；它只作为可重复调用的兼容 API 保留。每个实际执行 worker 都应在模型加载阶段、编译或图捕获之前导入。包不选择 NPU 设备，也不改变调用方的设备选择策略。

| 后端 | 当前内容 | 执行路径 | 验证状态 |
| --- | --- | --- | --- |
| `ascendc` | `OpsRepoAdd` host/tiling/kernel、ACLNN 与 PyTorch 适配源码 | `add` → `torch.ops.ops_repo.add` → `aclnnOpsRepoAdd` → AscendC | 需要完整 CANN、匹配的 torch/torch_npu 和目标 NPU 实测 |
| `triton` | 明确报 `NotImplementedError` 的扩展位置 | 待实现 Triton Ascend kernel | 没有可执行 kernel，不提供自动回退 |

当前 `add` 只接受同一 NPU 上、相同 shape、非空、连续存储的 FP32 输入，返回新的同 shape FP32 张量。暂不支持广播、FP16/BF16、非连续输入或反向传播。接入模型之前必须核对这些条件；不能因为算子名为 add 就替换所有张量加法。

## 完整 wheel 与开发安装

分发时可从仓库根目录执行：

```bash
# 先配置 CANN，并在与部署一致的 Python、torch/torch_npu 环境中构建。
python3 scripts/build_wheel.py --soc ascend910_93 --ops ops_repo_add --jobs 8
```

完整 wheel 包含 Python 层、原生 `_C` 扩展和 `ops_repo/opp/vendors/ops_repo/` 中的 OPP。构建脚本将本次 `.run` 包安装到临时目录，使用干净源码副本编包，校验产物后输出到 `dist/<soc>/wheel/`。部署端在匹配的驱动/CANN、Python、torch/torch_npu 环境中，用脚本打印的准确路径执行 `python3 -m pip install --force-reinstall --no-deps <wheel路径>`，无需另装 OPP 或 source vendor 环境。构建机仍需完整 CANN 编译工具链；完整 wheel 尚待真实 NPU 编译、安装与运行验证。

wheel 的 `.pth` 启动配置使用纯标准库，在用户代码导入 torch_npu 前设置包内 vendor 路径，不导入框架或选择设备，也不修改 `LD_LIBRARY_PATH`。Python 根据安装后的实际路径加载自有 `libcust_opapi.so`，不执行 OPP 中可能含构建机路径的 `set_env.bash`。Notebook 或已有 worker 需重启；仅设置 `PYTHONPATH` 不会执行 `.pth`。若启动配置未执行，只有 torch_npu 尚未导入时才能在包初始化中补配，避免在 CANN 初始化后更改搜索路径。

完整 wheel 记录目标 SoC、Python 主次版本、主机架构及 torch/torch_npu 精确版本，加载原生扩展前检查兼容性。它使用普通 PyTorch C++ ABI，部署环境应匹配构建环境，包括 CANN 和系统动态库；当前没有跨框架版本或跨芯片通用的 Stable ABI 承诺。

开发时仍可将下面两种交付件分别安装：

1. **OPP 包**：`bash build.sh --soc <soc>` 编译，`python3 scripts/ops.py install --soc <soc>` 安装到 `.local/<soc>/opp`。其中保存 kernel、tiling、算子元数据和 `libcust_opapi.so`。
2. **Python 包/扩展**：普通 `pip install .` 只安装 Python 层；设置 `OPS_REPO_BUILD_TORCH=1` 才编译 PyTorch C++ 扩展。这种分开构建的 wheel 不携带 OPP，也不自动安装 CANN、PyTorch、torch_npu。纯 Python wheel 可以安装，但导入仍要求完整 AscendC 运行环境；Triton namespace 属于同一包，不是独立可用的后端。

```bash
# 先 source 目标环境的 CANN set_env.sh，安装匹配的 torch/torch_npu。
bash build.sh --soc ascend910b
python3 scripts/ops.py install --soc ascend910b
source .local/ascend910b/opp/vendors/ops_repo/bin/set_env.bash
export OPS_REPO_OPP_PATH="$PWD/.local/ascend910b/opp"

OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps .
python3 examples/check_torch_add.py --device 0
```

构建选项支持 `ascend910b`、`ascend910_93`、`ascend950`，对应 A2、A3、A5 的独立输出目录。选择某个 target 只决定生成和编译目标，不等于已在该硬件验证。请为每个目标使用对应 Toolkit/ops 包，分别验证精度、尾块、资源使用和性能。切换 SoC 时不要混用其他目标的 OPP；扩展也应在部署用的 Python、PyTorch、torch_npu、编译器 ABI 环境构建。

`OPS_REPO_OPP_PATH` 用于原生扩展构建以及显式选择外置 OPP，必须指向包含 `vendors/ops_repo` 的安装根，不能填成 vendor 子目录。显式设置时优先于 wheel 内嵌 OPP，并继续要求启动前 source vendor 的 `set_env.bash`，使 `ASCEND_CUSTOM_OPP_PATH` 包含本仓 vendor；外置模式只校验环境，不修改已经启动的 CANN 运行环境。未设置时，完整 wheel 自动使用包内 OPP；没有包内 OPP 的源码安装会提示配置该变量。Python 不根据 `OPS_REPO_SOC` 猜测安装位置。不要在同一进程中混用两份 `ops_repo` vendor；与其他不同 vendor 共存时保留其搜索路径。

扩展根据绝对路径从本仓 vendor 加载 `libcust_opapi.so`，并使用本地符号作用域定位 `aclnnOpsRepoAdd`。这可避免仅按同名动态库查找时误用 FLA 的库；算子 OpType 与 vendor 也使用本仓独立名称。下发适配使用 torch_npu 的设备守卫、当前流和 `OpCommand::RunOpApi`。这些接口仍需与部署的 torch_npu 版本匹配。

原生 ACLNN 示例仍可独立使用：

```bash
cmake -S examples -B build/ascend910b/examples \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DOPS_REPO_SOC=ascend910b
cmake --build build/ascend910b/examples --parallel 4
./build/ascend910b/examples/ops_repo_add --device 0
```

原生示例应在只启用本仓 custom OPP 的独立 shell 中验证：其 ELF 依赖按 `libcust_opapi.so` 的 SONAME 加载，其他 vendor 的 `LD_LIBRARY_PATH` 仍可能改变实际库选择；Python 扩展的绝对路径加载保护不适用于这个原生示例。

`OPS_REPO_OPP_PATH` 是 CMake 的安装根覆盖选项；未指定时根据 `OPS_REPO_SOC` 选取 `.local/<soc>/opp`。切换已配置过的 CMake target 时使用新的 build 目录，或者同时显式更新缓存中的 `OPS_REPO_OPP_PATH`。

## 新算子的接入顺序

先定义与实现算子的数学语义和支持范围，再增加后端。对于 AscendC，新增 `ops/ascendc/<category>/<name>/` 的 manifest、host、tiling、kernel，并通过仓库的构建工具生成和编译 OPP。`python3 scripts/ops.py new my_operator` 默认使用 `custom` 分类，也可传 `--category elementwise`。随后在 `torch_custom/ops_repo/` 为 PyTorch 添加自有 namespace 的 schema、NPU kernel 适配、Fake/Meta 输出推导及 Python 公共入口。新增 AscendC 源码目录不会自动生成可用的 Python 适配。

如果需要 Triton，在 `ops/triton/triton_core/` 增加真实的 Triton Ascend kernel，明确参数约束和启动配置，并用独立 CPU 参考结果验证。该目录安装为 `ops_repo.ops.triton.triton_core`；后端 API 留在 `torch_custom/ops_repo/ops_repo/ops/triton/`，不复制第二份 kernel 源码。不要让占位后端静默调用 `torch.add`，也不要将只有 Python 包可导入误认为 NPU kernel 已就绪。

包导入负责加载和注册；正常算子调用负责计算。首次导入放在模型编译/图捕获之外；当前流应在每次实际执行时取得，不能缓存成跨 worker 或跨线程共享的 stream 指针。接入路径见 [vllm-ascend 调用说明](vllm_ascend.md)。

## 参考的 FLA 架构

参考版本固定为 `flashserve/flash-linear-attention-npu@54ae09b1243cd851d986c2624a57bbb5ee63dc27`：其 [README](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/README.md) 展示 AscendC/Triton 算子布局，[Python 适配说明](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/torch_custom/fla_npu/README.md) 展示公共接口与 kernel 下发分层。本仓借鉴分层和调用习惯，使用独立 `ops_repo` namespace、`ops_repo` OPP vendor 与 `aclnnOpsRepoAdd` 名称，没有复制其 Stable ABI 适配体系，也不承诺普通 PyTorch 扩展跨版本 ABI 兼容。

OPP 的 vendor 目录与环境变量行为可查 [CANN 8.5 编译与部署说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0024.html)。
