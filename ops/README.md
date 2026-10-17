# 算子源码

本目录保存按后端组织的计算实现；Python 公共 API、PyTorch 注册和 ACLNN 下发适配位于 [`torch_custom/ops_repo/`](../torch_custom/ops_repo/)。

```text
ops/
├── ascendc/
│   └── elementwise/
│       └── ops_repo_add/
│           ├── operator.json
│           ├── op_host/
│           ├── op_kernel/
│           └── README.md
└── triton/
    ├── README.md
    └── triton_core/
        └── __init__.py
```

## AscendC

算子目录为 `ops/ascendc/<category>/<op>/`。`category` 用于组织算法类别，示例 `ops_repo_add` 位于 `elementwise`。单个算子维护自己的接口清单、host/tiling、kernel 和语义说明；其公开 Python 名称、输入输出和 dtype 仍须在适配层显式实现。

从仓库根目录创建算子：

```bash
python3 scripts/ops.py new my_operator
python3 scripts/ops.py new my_elementwise_op --category elementwise
```

未指定 `--category` 时使用 `custom`。生成的新算子默认禁用，开发者完成实现与验证后再启用。构建工具读取 `ops/ascendc` 下的算子清单，并调用本机 msopgen 生成工程；不要在 `build/` 的生成副本中维护源码。

## Triton

[`triton/triton_core/`](triton/triton_core/) 是单独维护的 Triton 计算源码位置，当前只有 namespace 占位，没有可执行 kernel。其安装后的 Python 路径由根 `setup.py` 映射为 `ops_repo.ops.triton.triton_core`，源码只保留这一份。按这个路径导入时仍会执行 `ops_repo` 的 AscendC 根包初始化，不能把占位目录当作独立可用后端。后端公共入口位于适配层，成功导入后调用会返回 `NotImplementedError`。开发方式见 [Triton 说明](triton/README.md)。

根目录的 `setup.py` 是唯一 Python 打包入口；完整源码与公开接口映射见 [目录分层](../docs/layout.md)。
