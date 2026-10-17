# 从 vllm-ascend 调用自有算子

本仓提供可安装的算子包和明确的 Python 调用入口。安装后，还需要在选定模型或 Ascend 算子实现中增加实际调用。当前没有指定目标模型，所以仓库不注册自动替换模型的插件。

源码按 [目录分层](layout.md) 组织：计算实现位于 `ops/ascendc/` 与 `ops/triton/triton_core/`，适配位于 `torch_custom/ops_repo/`，从仓库根统一打包。模型使用 `from ops_repo import add` 或 `from ops_repo.ops.ascendc import add`，无需导入源码目录路径或单独调用初始化 API。

## 已核对的上游接入方式

以下说明固定参考 `vllm-project/vllm-ascend@93f8d0fcacded2b3e1ce8824b54423694f7baac6`，不把它当作其他发行版的接口保证：

- [`ops/gdn.py`](https://github.com/vllm-project/vllm-ascend/blob/93f8d0fcacded2b3e1ce8824b54423694f7baac6/vllm_ascend/ops/gdn.py) 直接导入 `fla_npu.ops.ascendc` 中的接口，在其 Ascend GDN 实现中调用；prefill 的外部融合算子还通过专门的加载函数选择。
- [`patch/worker/patch_qwen3_5.py`](https://github.com/vllm-project/vllm-ascend/blob/93f8d0fcacded2b3e1ce8824b54423694f7baac6/vllm_ascend/patch/worker/patch_qwen3_5.py) 把具体模型的计算接到 Ascend 实现。这是已知模型的专用适配，不是所有第三方算子都会自动进入的通用入口。
- [`ops/register_custom_ops.py`](https://github.com/vllm-project/vllm-ascend/blob/93f8d0fcacded2b3e1ce8824b54423694f7baac6/vllm_ascend/ops/register_custom_ops.py) 使用 vLLM 的 `direct_register_custom_op` 注册部分内部算子及 fake 实现。本仓已经注册 `torch.ops.ops_repo.add`，模型可以通过本仓公共 API 调用，无需为了使用它再注册同名 vLLM op。

参考的 FLA 版本为 `54ae09b1243cd851d986c2624a57bbb5ee63dc27`；其 [适配层说明](https://github.com/flashserve/flash-linear-attention-npu/blob/54ae09b1243cd851d986c2624a57bbb5ee63dc27/torch_custom/fla_npu/README.md) 也明确区分公共 Python 接口和底层下发。

本仓目标调用链为：

```text
选定模型/Ascend 层的真实调用点
  -> ops_repo.add(x, y, backend="ascendc")
  -> torch.ops.ops_repo.add
  -> 本仓 PyTorch NPU 适配
  -> aclnnOpsRepoAdd
  -> 当前 SoC 的 AscendC kernel
```

## 在每个 worker 的模型加载阶段导入

先按 [后端说明](backends.md) 安装对应 SoC、Python 和框架版本的完整 wheel，或分别安装 OPP 与原生 PyTorch 扩展。在每个节点和 worker 使用的环境中准备匹配的 CANN。完整 wheel 在 Python 启动时自动配置内嵌 OPP；分开安装时，还需在启动 vLLM 前加载 vendor 环境。仅在 API server 主进程里 `import ops_repo` 不足以初始化分布式 worker。

`import ops_repo` 会在包内部完成 AscendC 初始化，包括 torch/torch_npu、OPP 与原生动态库加载以及 Fake 注册；从子模块导入同样会先初始化根包。只需让模型模块在每个实际执行 worker 的模型加载阶段导入接口，在模型编译/图捕获之前完成。API server 主进程的导入不替代各 worker 的导入。

```bash
# 以下仅用于外置 OPP 的开发安装，以 A2 为例；CANN 路径按实际环境修改。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source .local/ascend910b/opp/vendors/ops_repo/bin/set_env.bash
export OPS_REPO_OPP_PATH="$PWD/.local/ascend910b/opp"
```

`OPS_REPO_OPP_PATH` 指向安装根，`ASCEND_CUSTOM_OPP_PATH` 则由 vendor 脚本设置为包含 `vendors/ops_repo` 的路径。显式外置模式下包导入时会验证二者，不自动补环境。完整 wheel 默认不设置 `OPS_REPO_OPP_PATH`，由启动 hook 配置包内 OPP；切换安装方式后需要重启 worker，并避免残留另一份同 vendor 的路径。多个 OPP 共存时保留其他 vendor 的路径；本仓通过绝对路径定位自有 API 库，并使用独立算子名。

参考版本的 [`NPUWorker`](https://github.com/vllm-project/vllm-ascend/blob/93f8d0fcacded2b3e1ce8824b54423694f7baac6/vllm_ascend/worker/worker.py) 将设备初始化、模型加载与编译/warmup 分为不同阶段。将下面的导入放在使用算子的模型模块顶层，即可随该模块在 worker 的模型加载阶段完成注册：

```python
from ops_repo import add
```

[`examples/vllm_ascend_call.py`](../examples/vllm_ascend_call.py) 采用这种模块级导入方式，并提供一个简单模型层。无需仅为初始化创建 worker 子类或设置 `worker_cls`；设备仍由 vllm-ascend 的实际 worker 管理，本包不选择设备或在导入时取得执行 stream。`initialize()` 仍可作为兼容 API 重复调用，但模型集成不需要额外调用它。

对于独立脚本、Ray 或多节点部署，将使用算子的模型模块放进自己的可安装适配包；所有节点使用相同版本的包、扩展和 OPP，并确认每个 worker 在编译前已加载该模块。升级 vllm-ascend 后重新核对模型加载和编译的时序。

## 增加模型的实际调用点

[`OpsRepoAddLayer`](../examples/vllm_ascend_call.py) 展示最小调用方式：

```python
from ops_repo import add

def forward(self, x, y):
    return add(x, y, backend="ascendc")
```

把这样的调用放到已选定模型的 Ascend 实现中，并确认输入语义与输出用途完全一致。当前示例只支持 **FP32、相同 shape、非空、连续、同一 NPU、仅推理**；常见模型的 BF16/FP16 residual add 不能直接替换。扩展 dtype 需要同步修改 manifest、host/kernel、PyTorch schema/校验、Fake 实现和测试。

包导入只加载运行时和注册算子，不把 `OpsRepoAddLayer` 注入模型。没有上述实际调用点时，即使 vLLM 成功启动，也不能据此认为模型已使用自定义算子。确认调用可使用目标算子的 profiler 记录，并对比同一输入下的模型输出与基线。

## 验证顺序和限制

先执行独立硬件验证，再接模型：

```bash
python3 examples/check_torch_add.py --device 0
python3 examples/check_torch_add.py --device 0 --fake --compile
```

第一条比较真实 NPU 输出与 CPU 参考结果，覆盖长度 `1, 7, 8, 9, 1023, 1024, 1025, 4095, 4096, 4097, 65537`、标量和多维 shape，并检查输入没有被修改；还在非默认 stream 上连续提交 16 次调用，使用非零 storage offset 的连续 view，删除输入引用、保留输出及同流后续计算结果，在统一同步后校验，以覆盖流选择和异步生命周期相关路径。第二条额外检查 FakeTensor 元数据及 `torch.compile(fullgraph=True, backend="eager")` 的追踪路径。这里的 `eager` 编译后端只是 Dynamo 图追踪验证，不是 vllm-ascend 的 AscendCompiler、TorchAir lowering 或 ACL graph capture 验证。

模型集成先使用 eager 模式验证结果、批量与变长输入、多 worker、多 stream。接着在目标 vLLM/CANN/torch_npu 版本上验证编译与图捕获、重复 replay、workspace 生命周期、内存增长和性能。Fake/Meta 注册只解决输出元数据推导，不保证任意编译后端都能降低这个自定义 op；若后端要求额外 converter/lowering，需要针对它实现，不能用跳过错误或隐式 CPU 回退代替。

首次包导入会加载动态库并注册 fake 实现，必须发生在编译和 capture 之前；不要把首次导入放在 captured forward 内。当前交付环境没有完整的 PyTorch/CANN 开发环境，以上硬件、模型和图模式验证均需在目标机器执行，本仓不声明它们已经通过。
