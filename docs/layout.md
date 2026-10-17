# 源码布局与公共接口

本仓将计算源码放在 `ops/`，将框架适配放在 `torch_custom/ops_repo/`。开发者按实现职责修改对应目录；用户始终通过安装后的 `ops_repo` 包调用，不需要了解源码目录的层级。

| 职责 | 本仓源码位置 | 修改内容 |
| --- | --- | --- |
| AscendC 实现 | `ops/ascendc/<category>/<op>/` | manifest、算子定义、shape/tiling、NPU kernel |
| Triton 实现 | `ops/triton/triton_core/` | 未来的 Triton Ascend kernel；当前仅 namespace |
| Python 公共 API | `torch_custom/ops_repo/ops_repo/` | 后端选择、环境检查、导入时初始化、Fake/Meta 适配 |
| PyTorch NPU 适配 | `torch_custom/ops_repo/csrc/` | dispatcher 注册、参数转换、流与 workspace、ACLNN 调用 |
| OPP 构建工具 | `scripts/ops.py`、`build.sh` | 枚举算子、调用本机 msopgen、按 SoC 编译安装 |
| 完整 wheel | `scripts/build_wheel.py` | 串联 OPP 构建、临时安装与原生适配打包，输出 `dist/<soc>/wheel/` |
| Python 打包入口 | 根 `setup.py`、`pyproject.toml`、`MANIFEST.in` | 将适配和 kernel 源码打包为一个 `ops_repo` 包 |

目前提供的 AscendC 实现是 [`OpsRepoAdd`](../ops/ascendc/elementwise/ops_repo_add/README.md)，尚未完成 NPU 编译和上机验收。Triton 仍未实现 kernel，入口会返回明确错误。分类目录只用于组织源码，不会自动生成 Python 适配或把算子接入模型。

## 源码路径与 import 路径

```text
torch_custom/ops_repo/ops_repo/        -> import ops_repo
torch_custom/ops_repo/ops_repo/ops/   -> import ops_repo.ops
ops/triton/triton_core/               -> import ops_repo.ops.triton.triton_core
```

Triton core 的映射由根 `setup.py` 完成，源码只存在于 `ops/triton/triton_core/`。开发安装、构建 wheel 和构建原生扩展均从仓库根目录执行；`torch_custom/ops_repo/` 没有第二个独立打包入口。

```bash
# Python 层；不安装或编译 NPU 依赖
python3 -m pip install --no-build-isolation --no-deps .

# 已配置匹配 CANN、OPP、torch/torch_npu 后，开发安装原生扩展
OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps -e .
```

`import ops_repo` 自动加载 torch/torch_npu、检查 OPP、加载原生扩展与自定义 API 库并完成 Fake 注册；导入 `ops_repo` 的子模块也会先执行根包初始化。完整 wheel 包含已编译的原生扩展和 OPP，使用标准库启动 hook 自动配置包内 OPP。默认纯 Python wheel 可以安装，但不足以支持实际导入。Triton core 虽然只有源码 namespace，按安装后的包路径导入时同样需要 AscendC 运行环境。

用户只需 `from ops_repo import add` 或 `from ops_repo.ops.ascendc import add`。`initialize()` 保留为可重复调用的兼容 API，常规调用不需要它。vLLM 的每个实际执行 worker 应在模型加载阶段、编译/捕获之前导入；包不选择设备，基础 CANN 环境需在进程启动前配置。完整 wheel 自动配置包内 OPP；显式外置 OPP 仍需加载 vendor 环境。

## 与 FLA 的对照

参考版本为 `flashserve/flash-linear-attention-npu@54ae09b1243cd851d986c2624a57bbb5ee63dc27`。它把 AscendC 按计算模块组织，将 Triton core 与框架适配分开；这也是本仓采用分层目录的依据。

| FLA 的目录 | 本仓对应目录 |
| --- | --- |
| [`fla/ops/ascendc/`](https://github.com/flashserve/flash-linear-attention-npu/tree/54ae09b1243cd851d986c2624a57bbb5ee63dc27/fla/ops/ascendc) | `ops/ascendc/` |
| [`fla/ops/triton/triton_core/`](https://github.com/flashserve/flash-linear-attention-npu/tree/54ae09b1243cd851d986c2624a57bbb5ee63dc27/fla/ops/triton/triton_core) | `ops/triton/triton_core/` |
| [`torch_custom/fla_npu/`](https://github.com/flashserve/flash-linear-attention-npu/tree/54ae09b1243cd851d986c2624a57bbb5ee63dc27/torch_custom/fla_npu) | `torch_custom/ops_repo/` |

借鉴的是源码职责划分。本仓保留自己的 msopgen 构建工具、普通 PyTorch NPU 扩展、包名与 OPP vendor；没有复制 FLA 源码、Stable ABI 体系或其工作流规则。只有在出现实际共享代码、构建需求或持续集成任务时，才增加相应目录。构建与调用步骤见 [后端说明](backends.md) 和 [vllm-ascend 接入](vllm_ascend.md)。
