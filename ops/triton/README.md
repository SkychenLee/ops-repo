# Triton 算子源码

`triton_core/` 保存未来的 Triton Ascend kernel，实现位置与 PyTorch/Python 适配分开。目前这里只有 namespace 占位，没有已实现算子。占位文件自身不导入 Triton，但通过安装后的包路径访问它，会先执行 `ops_repo` 根包初始化，因此需要 torch/torch_npu、原生扩展和完整 AscendC 运行环境。

| 用途 | 源码位置 | 安装后的路径 |
| --- | --- | --- |
| kernel 源码 | `ops/triton/triton_core/` | `ops_repo.ops.triton.triton_core` |
| 后端公共入口 | `torch_custom/ops_repo/ops_repo/ops/triton/` | `ops_repo.ops.triton` |

根 `setup.py` 直接将 kernel 源码目录映射到 Python 包；不在适配目录复制第二份 kernel，也不建立单独的嵌套安装工程。它不是独立可用的 Triton 后端；在完整环境中成功导入后，`initialize(backend="triton")` 和 `add(x, y, backend="triton")` 仍明确抛出 `NotImplementedError`。

新增真实实现时：

1. 在 `triton_core/` 下按算子组织 kernel 模块，记录支持的 SoC、dtype、shape、layout、尾块和数值误差要求。
2. 在适配层的 `ops/triton/` 中增加参数校验和调用入口，并明确其加载与注册时序；保留根包导入即完成 AscendC 初始化的公开语义。
3. 依赖安装按目标 CANN、Python、PyTorch 环境选择匹配的 Triton Ascend 版本，不在包导入时自动安装依赖。
4. 用独立 CPU 参考结果进行精度、边界和异常测试；图编译、捕获、多个 stream 和性能分别在目标硬件验证。

完成实现之前保留明确的未实现错误，不使用 CPU 或其他后端模拟成功。通用 API 和交付方式见 [后端说明](../../docs/backends.md)。
