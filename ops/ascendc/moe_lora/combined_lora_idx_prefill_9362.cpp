/*
 * Copyright (c) 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Combined LoRA gather index for the AllGather MoE backend — final phase of
 * _build_combined_lora_idx_allgather:
 *
 *   keys = dest.to(fp32) with inactive (-1) masked to n (aten, caller)
 *   inv  = at::argsort(keys)                                (aten, caller)
 *   THIS KERNEL: per row r, p = inv[r]:
 *                    token  = p / top_k
 *                    slot   = token < num_valid_tokens ? lora_indices[token] : -1
 *                    expert = topk_ids[p] - first_expert_idx   (0 without EP)
 *                    enabled = dest[p] >= 0 && slot >= 0
 *                              && adapter_enabled[slot] != 0
 *                              && 0 <= expert < num_experts
 *                    out[r] = enabled ? slot * num_experts + expert : -1
 *
 * Active dests are unique (npu_moe_init_routing_v2), so rows [0, available)
 * gather their exact pair; inactive pairs share the trailing key tie and
 * each yields -1 (enabled requires dest >= 0), so the tie order does not
 * affect the output and out is bit-identical to the torch chain
 * (tests/ut/lora/test_combined_idx_dedup.py).
 *
 * Constraints:
 *   - the kernel is launched after aten producers on the same stream (the
 *     add_lora z1/z2 pattern); an aten op consuming the output of a
 *     raw-launched kernel is not stream-ordered, so keys/argsort stay aten
 *   - GM outputs must be vectorized UB->GM DataCopy/DataCopyPad writes;
 *     GlobalTensor::SetValue GM stores are unreliable on this stack. GM
 *     scalar loads (GetValue) are fine (same pattern as add_lora_z1's index
 *     reads)
 *   - static shapes only; graph-capturable
 *
 * EDITING NOTE: parsed by the build's auto_gen tool — pointer parameters
 * MUST be written `__gm__ void* name` (star on the type, not the name).
 */

#include "kernel_operator.h"
#include "types.h"

namespace {

constexpr uint32_t CLI_PREFILL_BLOCK = 256;  // rows per unit (UB: 2KB inv + 2KB out)

template <typename DEST_T, typename TOPK_T>
class CombinedLoraIdxPrefill9362 {
public:
    __aicore__ inline CombinedLoraIdxPrefill9362(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *dest, __gm__ void *topk, __gm__ void *lora_indices,
                                __gm__ void *adapter_enabled, __gm__ void *inv, __gm__ void *out,
                                uint32_t units, uint32_t numPairs,
                                uint32_t topK, uint32_t numValidTokens, int64_t firstExpertIdx,
                                uint32_t numExperts)
    {
        units_ = units;
        numPairs_ = numPairs;
        topK_ = topK;
        numValidTokens_ = numValidTokens;
        firstExpertIdx_ = firstExpertIdx;
        numExperts_ = numExperts;
        destGm_.SetGlobalBuffer((__gm__ DEST_T *)dest);
        topkGm_.SetGlobalBuffer((__gm__ TOPK_T *)topk);
        loraGm_.SetGlobalBuffer((__gm__ int64_t *)lora_indices, numValidTokens);
        adapterGm_.SetGlobalBuffer((__gm__ int32_t *)adapter_enabled);
        invGm_.SetGlobalBuffer((__gm__ int64_t *)inv);
        outGm_.SetGlobalBuffer((__gm__ int64_t *)out);
        pipe_->InitBuffer(invBuf_, CLI_PREFILL_BLOCK * sizeof(int64_t));
        pipe_->InitBuffer(outBuf_, CLI_PREFILL_BLOCK * sizeof(int64_t));
        const uint32_t mapBytes = numValidTokens * sizeof(int64_t);
        const uint32_t mapAllocBytes = mapBytes == 0 ? 32u : (mapBytes + 31u) / 32u * 32u;
        pipe_->InitBuffer(loraBuf_, mapAllocBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        AscendC::LocalTensor<int64_t> lora = loraBuf_.Get<int64_t>();
        if (numValidTokens_ != 0) {
            const AscendC::DataCopyExtParams mapCopy{
                1, static_cast<uint32_t>(numValidTokens_ * sizeof(int64_t)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<int64_t> mapPad{false, 0, 0, 0};
            AscendC::DataCopyPad(lora, loraGm_, mapCopy, mapPad);
            const event_t mapReady = static_cast<event_t>(
                pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mapReady);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mapReady);
        }
        for (uint32_t u = blockIdx; u < units_; u += blockNum) {
            const uint32_t r0 = (uint32_t)u * CLI_PREFILL_BLOCK;
            const uint32_t len = numPairs_ - r0 < CLI_PREFILL_BLOCK ? numPairs_ - r0 : CLI_PREFILL_BLOCK;
            AscendC::LocalTensor<int64_t> inv = invBuf_.Get<int64_t>();
            AscendC::LocalTensor<int64_t> out = outBuf_.Get<int64_t>();
            if (len == CLI_PREFILL_BLOCK) {
                DataCopy(inv, invGm_[r0], CLI_PREFILL_BLOCK);
                AscendC::PipeBarrier<PIPE_MTE2>();
                const event_t invReady = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_S));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(invReady);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(invReady);
            }
            for (uint32_t i = 0; i < len; ++i) {
                const uint32_t p = (uint32_t)(len == CLI_PREFILL_BLOCK ? inv.GetValue(i)
                                                               : invGm_.GetValue(r0 + i));
                const int64_t d = (int64_t)destGm_.GetValue(p);
                int64_t value = -1;
                if (d >= 0) {
                    const uint32_t token = p / topK_;
                    if (token < numValidTokens_) {
                        const int64_t slot = lora.GetValue(token);
                        if (slot >= 0) {
                            const int64_t expert = (int64_t)topkGm_.GetValue(p) - firstExpertIdx_;
                            if (expert >= 0 && expert < (int64_t)numExperts_
                                && adapterGm_.GetValue((uint32_t)slot) != 0) {
                                value = slot * (int64_t)numExperts_ + expert;
                            }
                        }
                    }
                }
                out.SetValue(i, value);
            }
            AscendC::PipeBarrier<PIPE_V>();
            const event_t scalarDone = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarDone);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarDone);
            if (len == CLI_PREFILL_BLOCK) {
                DataCopy(outGm_[r0], out, CLI_PREFILL_BLOCK);
            } else {
                DataCopyPad(outGm_[r0], out, {1, (uint16_t)(len * sizeof(int64_t)), 0, 0});
            }
            const event_t outputDone = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(outputDone);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(outputDone);
            const event_t invReusable = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(invReusable);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(invReusable);
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> invBuf_, outBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECIN> loraBuf_;
    AscendC::GlobalTensor<DEST_T> destGm_;
    AscendC::GlobalTensor<TOPK_T> topkGm_;
    AscendC::GlobalTensor<int64_t> loraGm_;
    AscendC::GlobalTensor<int32_t> adapterGm_;
    AscendC::GlobalTensor<int64_t> invGm_;
    AscendC::GlobalTensor<int64_t> outGm_;
    uint32_t units_, numPairs_, topK_, numValidTokens_, numExperts_;
    int64_t firstExpertIdx_;
};

}  // namespace

#define COMBINED_LORA_IDX_PREFILL_DECLARE(DEST_T, TOPK_T, SUFFIX)                                  \
    extern "C" __global__ __aicore__ void ops_repo_combined_lora_idx_prefill_9362_##SUFFIX(                      \
        __gm__ void* dest, __gm__ void* topk, __gm__ void* lora_indices,                            \
        __gm__ void* adapter_enabled, __gm__ void* inv, __gm__ void* out, uint32_t units,           \
        uint32_t numPairs, uint32_t topK, uint32_t numValidTokens,           \
        int64_t firstExpertIdx, uint32_t numExperts)                                                \
    {                                                                                               \
        AscendC::TPipe pipe;                                                                        \
        CombinedLoraIdxPrefill9362<DEST_T, TOPK_T> op(&pipe);                                          \
        op.Init(dest, topk, lora_indices, adapter_enabled, inv, out, units, numPairs, \
                topK, numValidTokens, firstExpertIdx, numExperts);                                  \
        op.Process();                                                                               \
    }

COMBINED_LORA_IDX_PREFILL_DECLARE(int32_t, int64_t, i32_i64)

namespace ops_repo_lora {

// Cyclic tile ownership uses every available vector core on the current device.
extern void combined_lora_idx_prefill_9362_impl(void *stream, void *dest, void *topk,
                                               void *lora_indices, void *adapter_enabled,
                                               void *inv, void *out, uint32_t num_pairs,
                                               uint32_t top_k, uint32_t num_valid_tokens,
                                               int64_t first_expert_idx, uint32_t num_experts,
                                               uint32_t aiv_num)
{
    const uint32_t units = (num_pairs + CLI_PREFILL_BLOCK - 1) / CLI_PREFILL_BLOCK;
    if (units == 0 || aiv_num == 0) {
        return;
    }
    const uint32_t grid = units < aiv_num ? units : aiv_num;
    ops_repo_combined_lora_idx_prefill_9362_i32_i64<<<grid, nullptr, stream>>>(
        dest, topk, lora_indices, adapter_enabled, inv, out, units, num_pairs,
        top_k, num_valid_tokens, first_expert_idx, num_experts);
}
}  // namespace ops_repo_lora
