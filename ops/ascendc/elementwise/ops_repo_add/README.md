# OpsRepoAdd

独立的 AscendC 逐元素加法示例：`z = x + y`。配置了 A2（`ascend910b`）、A3（`ascend910_93`）和 A5（`ascend950`）三个编译目标，使用标准 AscendC 算子注册、tiling 和 kernel API，不依赖 FLA。必须安装支持所选芯片的 CANN Toolkit、ops 包和配套 msopgen；A5 的 API 支持依据为下方 CANN 9.1 文档。

`operator.json` 的 `soc` 记录已配置的目标，`hardware_validated` 只记录完成真实编译和上机验收的目标。当前后者为空：三个目标都没有完成硬件验证。`enabled=true` 表示允许纳入构建，不表示已验证可用于生产。

| 参数 | 方向 | 类型 / 格式 | 约束 |
| --- | --- | --- | --- |
| `x` | 输入 | FP32 / ND | 连续、非空 |
| `y` | 输入 | FP32 / ND | 连续，与 `x` 的 shape 完全相同 |
| `z` | 输出 | FP32 / ND | 连续，与输入 shape 相同，调用方分配独立存储 |

支持标量和任意正整数维度的张量；元素数量及其字节数需能用有符号 64 位整数表示。不支持广播、空张量、其他 dtype、非连续视图或自动求导。调用方必须保证连续存储；host 的 shape/descriptor 校验不能代替对调用方 stride 的检查。

`op_host/ops_repo_add.cpp` 完成 shape/dtype 推导、运行时校验、平台信息查询和 tiling。每个 tile 最多 4096 个元素，长度按 32 字节对齐，三个单缓冲队列合计不超过一半 UB。实际启动的 vector core 数量为 tile 数与平台 AIV 核数的较小值，按 tile 编号轮流分配给各核。

`op_host/ops_repo_add_tiling.h` 定义 `totalLength`、`tileLength`、`blockDim`，由 host 序列化，kernel 用 `GET_TILING_DATA` 读取。`op_kernel/ops_repo_add.cpp` 用 `TQue` 同步搬入、加法和搬出。尾块使用 `DataCopyPad`，GM 读写长度始终为有效元素数；补齐仅发生在 UB 内，不要求调用方为输入或输出额外分配尾部空间。算子不使用 GM workspace。

从仓库根目录使用统一构建入口生成 CANN 工程并编包，例如 `bash build.sh --soc ascend910_93 --ops ops_repo_add`。`build/<soc>/cann_project` 是生成产物，自有实现应始终修改本目录。三个目标分别保存在 `dist/<soc>`，`python3 scripts/ops.py install --soc <soc>` 安装到 `.local/<soc>/opp`。接口名称为 `OpsRepoAdd`，kernel 名称为 `ops_repo_add`，生成的 ACLNN 接口应为 `aclnnOpsRepoAddGetWorkspaceSize` 和 `aclnnOpsRepoAdd`。

当前实现尚未完成真实 CANN/NPU 编译、精度或性能验证。仓库的原生 ACLNN 示例已准备 12 种向量长度的 CPU 加法对照，包括 `1, 7, 8, 9, 4095, 4096, 4097, 1048581` 等边界和多 tile 场景。上机时还应补充标量、多维 shape、shape 不一致、空张量和非 FP32 输入的验收。性能参数属于保守的示例设置，未作性能承诺。

API 参考：CANN 9.1 的 [Add](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Add.md)、[DataCopyPad GM→UB](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md) 和 [DataCopyPad UB→GM](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.md) 均列出 Atlas A2、Atlas A3、Ascend 950PR/950DT 支持。此文档依据用于设置编译目标，不能代替本算子的编译、链接和运行验收。
