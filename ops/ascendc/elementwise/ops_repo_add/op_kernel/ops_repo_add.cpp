#include "kernel_operator.h"

namespace {

constexpr uint32_t kAlignmentElements = 32 / sizeof(float);
constexpr int32_t kQueueDepth = 1;

class KernelOpsRepoAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength,
                               uint32_t tileLength, uint32_t blockDim)
    {
        totalLength_ = totalLength;
        tileLength_ = tileLength;
        blockDim_ = blockDim;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), totalLength);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), totalLength);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z), totalLength);
        pipe_.InitBuffer(xQueue_, kQueueDepth, tileLength * sizeof(float));
        pipe_.InitBuffer(yQueue_, kQueueDepth, tileLength * sizeof(float));
        pipe_.InitBuffer(zQueue_, kQueueDepth, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Tile t belongs to core t % blockDim. Thus every element is visited
        // exactly once, without overlapping stores between cores.
        const uint64_t step = static_cast<uint64_t>(blockDim_) * tileLength_;
        for (uint64_t offset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tileLength_;
             offset < totalLength_; offset += step) {
            const uint64_t remaining = totalLength_ - offset;
            const uint32_t count =
                remaining < tileLength_ ? static_cast<uint32_t>(remaining) : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        auto xLocal = xQueue_.AllocTensor<float>();
        auto yLocal = yQueue_.AllocTensor<float>();
        // blockLen is bytes, not elements. Pad only in UB: no rounded-up GM
        // read occurs, even for a one-element tensor or the final partial tile.
        const AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        const uint8_t rightPadding =
            static_cast<uint8_t>((kAlignmentElements - count % kAlignmentElements) %
                                 kAlignmentElements);
        const AscendC::DataCopyPadExtParams<float> pad{true, 0, rightPadding, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm_[offset], copy, pad);
        AscendC::DataCopyPad(yLocal, yGm_[offset], copy, pad);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = xQueue_.DeQue<float>();
        auto yLocal = yQueue_.DeQue<float>();
        auto zLocal = zQueue_.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, count);
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        auto zLocal = zQueue_.DeQue<float>();
        const AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        // The GM destination receives count elements, never the UB padding.
        AscendC::DataCopyPad(zGm_[offset], zLocal, copy);
        zQueue_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueueDepth> zQueue_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    AscendC::GlobalTensor<float> zGm_;
    uint64_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t blockDim_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void ops_repo_add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.tileLength == 0 || tilingData.blockDim == 0 ||
        AscendC::GetBlockIdx() >= tilingData.blockDim) {
        return;
    }
    KernelOpsRepoAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength, tilingData.blockDim);
    op.Process();
}
