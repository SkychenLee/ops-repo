# ops_repo 适配层

本目录集中维护 Ascend 算子的 Python 公共接口与 PyTorch C++ 绑定，对应 FLA 的 `torch_custom/fla_npu`。从仓库根目录执行构建和安装，打包配置统一位于根目录 `setup.py`。

```text
torch_custom/ops_repo/
├── csrc/torch_bindings.cpp      # torch.ops.ops_repo → 自有 ACLNN 接口
└── ops_repo/
    ├── _runtime.py             # 动态库定位、加载与初始化
    ├── _fake.py                # FakeTensor / Meta 输出推导
    ├── _validation.py          # 输入契约
    └── ops/
        ├── ascendc/            # AscendC Python 入口
        └── triton/             # Triton Python 入口，当前未实现
```

底层 AscendC 源码位于 `ops/ascendc/<category>/<name>/`。Triton 源码统一维护在 `ops/triton/triton_core/`，由打包配置映射为 `ops_repo.ops.triton.triton_core`，不用复制到本目录。

公开调用方式不受源码布局影响：

```python
from ops_repo import add
# 或使用明确的 AscendC 入口：
from ops_repo.ops.ascendc import add
```

根包导入自动加载 torch/torch_npu、检查 OPP、加载原生动态库并注册 Fake 实现；子模块导入同样先执行根包初始化。用户不需要单独调用初始化函数，`initialize()` 保留为可重复调用的兼容 API。包不选择设备；完整 wheel 在 Python 启动时自动配置包内 OPP，显式外置 OPP 模式仍要求启动前加载 vendor 环境。两种方式都需要基础 CANN 环境。vLLM 的每个实际执行 worker 应在模型加载阶段、编译或图捕获前导入使用算子的模块。

Triton 是同一包内的占位，不能作为不依赖 AscendC 运行时的独立后端；在完整环境中成功导入后，其执行入口仍明确抛出 `NotImplementedError`。

构建完整 wheel（包含原生适配与 OPP）：

```bash
# 在仓库根目录及匹配 CANN、torch/torch_npu 的构建环境执行。
python3 scripts/build_wheel.py --soc ascend910_93 --ops ops_repo_add --jobs 8
# 使用输出的准确路径，在匹配环境中安装：
python3 -m pip install --force-reinstall --no-deps "dist/ascend910_93/wheel/<实际文件名>.whl"
```

完整 wheel 通过 `ops_repo_opp_env.py` / `.pth` 提前配置内嵌 OPP；导入时检查构建记录中的框架版本并加载 `_C` 与自有接口库。它仍使用普通 PyTorch C++ ABI，不能跨任意框架版本、Python ABI 或芯片使用。显式 `OPS_REPO_OPP_PATH` 会选择外置 OPP，须自行配置对应 vendor 环境。完整 wheel 的实际 NPU 构建和执行仍待目标环境验证。

只安装 Python 骨架：

```bash
python3 -m pip install --no-build-isolation --no-deps .
```

安装不会检查运行环境，但 `import ops_repo` 会初始化 AscendC：纯 Python wheel 缺少原生扩展时不能单独完成导入，也不自动下载 torch/torch_npu 或 OPP。

在已配置完整 CANN、匹配的 torch/torch_npu、已安装自有 OPP 的环境中编译绑定：

```bash
OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps .
```

源码开发可为上述安装命令增加 `-e`，然后仍按 `ops_repo` 包名导入。不要在本目录直接运行根目录的打包命令。详细环境、目标选择和测试步骤见[根 README](../../README.md)、[后端说明](../../docs/backends.md)和 [C++ 适配说明](csrc/README.md)。
