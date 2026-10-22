#pragma once

#include <acl/acl.h>
#include <cstdint>
#include <cstring>

namespace ops_repo_lora {
extern void combined_lora_idx_prefill_9362_impl(
    void *stream, void *dest, void *topk, void *lora_indices, void *adapter_enabled,
    void *inv, void *out, uint32_t num_pairs, uint32_t top_k, uint32_t num_valid_tokens,
    int64_t first_expert_idx, uint32_t num_experts, uint32_t aiv_num);

// A failed device query keeps the original route. No device-0 capability cache.
inline uint32_t CliPrefill9362AivNum(int64_t n, int64_t top_k,
                                    bool dest_is_int32, bool topk_is_int64)
{
    if (!dest_is_int32 || !topk_is_int64 || top_k != 6 || n < 49152 || n > 98304) {
        return 0;
    }
    const char *soc = aclrtGetSocName();
    if (soc == nullptr || std::strcmp(soc, "Ascend910_9362") != 0) {
        return 0;
    }
    int32_t device_id = 0;
    int64_t count = 0;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS ||
        aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &count) != ACL_SUCCESS ||
        count <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(count);
}
}  // namespace ops_repo_lora
