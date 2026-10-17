#include "ops_repo_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr uint32_t kAlignmentElements = 32 / sizeof(float);
constexpr uint32_t kMaxTileElements = 4096;
constexpr uint64_t kQueueCount = 3;  // One tile each for x, y and z.

ge::graphStatus Fail(const char* reason)
{
    std::fprintf(stderr, "OpsRepoAdd: %s\n", reason);
    return ge::GRAPH_FAILED;
}

bool SameShape(const gert::Shape& x, const gert::Shape& y)
{
    if (x.GetDimNum() != y.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < x.GetDimNum(); ++i) {
        if (x.GetDim(i) != y.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// A scalar has one element. Runtime dimensions must be positive and resolved.
bool ElementCount(const gert::Shape& shape, uint64_t& count)
{
    count = 1;
    const uint64_t maxElements =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / sizeof(float);
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || count > maxElements / static_cast<uint64_t>(dim)) {
            return false;
        }
        count *= static_cast<uint64_t>(dim);
    }
    return true;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return Fail("missing tiling context");
    }
    const auto* x = context->GetInputShape(0);
    const auto* y = context->GetInputShape(1);
    const auto* z = context->GetOutputShape(0);
    const auto* xDesc = context->GetInputDesc(0);
    const auto* yDesc = context->GetInputDesc(1);
    const auto* zDesc = context->GetOutputDesc(0);
    if (x == nullptr || y == nullptr || z == nullptr || xDesc == nullptr ||
        yDesc == nullptr || zDesc == nullptr) {
        return Fail("missing input/output shape or descriptor");
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT || yDesc->GetDataType() != ge::DT_FLOAT ||
        zDesc->GetDataType() != ge::DT_FLOAT) {
        return Fail("x, y and z must all have dtype float32");
    }
    if (!SameShape(x->GetStorageShape(), y->GetStorageShape()) ||
        !SameShape(x->GetStorageShape(), z->GetStorageShape())) {
        return Fail("x, y and z must have identical shapes; broadcasting is unsupported");
    }
    uint64_t totalLength = 0;
    if (!ElementCount(x->GetStorageShape(), totalLength)) {
        return Fail("runtime shape must be nonempty, fully resolved and fit in int64 bytes");
    }
    if (context->GetPlatformInfo() == nullptr) {
        return Fail("missing Ascend platform information");
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto coreCount = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (coreCount == 0) {
        return Fail("platform reports no vector cores");
    }

    // Keep half of UB available for compiler/runtime needs; cap each tile at
    // 16 KiB. All tile starts are 32-byte aligned, including across cores.
    const uint64_t capacity = (ubBytes / 2) / (kQueueCount * sizeof(float));
    const uint32_t tileLength = static_cast<uint32_t>(
        std::min<uint64_t>(capacity, kMaxTileElements) / kAlignmentElements *
        kAlignmentElements);
    if (tileLength == 0) {
        return Fail("platform reports insufficient UB memory");
    }
    const uint64_t tileCount = 1 + (totalLength - 1) / tileLength;
    const uint32_t blockDim =
        static_cast<uint32_t>(std::min<uint64_t>(tileCount, coreCount));

    OpsRepoAddTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_blockDim(blockDim);
    auto* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return Fail("tiling buffer is missing or too small");
    }
    auto* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return Fail("missing workspace size storage");
    }
    workspace[0] = 0;
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return Fail("missing shape inference context");
    }
    const auto* x = context->GetInputShape(0);
    const auto* y = context->GetInputShape(1);
    auto* z = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || z == nullptr) {
        return Fail("missing shape during inference");
    }
    if (!SameShape(*x, *y)) {
        return Fail("input shapes must match exactly");
    }
    for (size_t i = 0; i < x->GetDimNum(); ++i) {
        if (x->GetDim(i) == 0) {
            return Fail("empty tensors are unsupported");
        }
    }
    *z = *x;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr || context->GetInputDataType(0) != DT_FLOAT ||
        context->GetInputDataType(1) != DT_FLOAT) {
        return Fail("input dtypes must both be float32");
    }
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class OpsRepoAdd : public OpDef {
public:
    explicit OpsRepoAdd(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        // Configured build targets; registration does not establish hardware validation.
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(OpsRepoAdd);

}  // namespace ops
