#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_ops_repo_add.h"

namespace {

void Check(int64_t status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed, status=" +
                                 std::to_string(status));
    }
}

void CheckCleanup(int64_t status, const char* operation, bool& failed) noexcept
{
    if (status != 0) {
        failed = true;
        std::cerr << operation << " failed during cleanup, status=" << status << '\n';
    }
}

// Initialization is separate from construction so partially initialized
// resources are also released when an ACL call throws.
class Session {
public:
    explicit Session(bool& cleanup_failed) : cleanup_failed_(cleanup_failed) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void Initialize(int32_t device)
    {
        device_ = device;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
        device_set_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    ~Session()
    {
        if (stream_ != nullptr) {
            CheckCleanup(aclrtDestroyStream(stream_), "aclrtDestroyStream", cleanup_failed_);
        }
        if (device_set_) {
            CheckCleanup(aclrtResetDevice(device_), "aclrtResetDevice", cleanup_failed_);
        }
        if (initialized_) {
            CheckCleanup(aclFinalize(), "aclFinalize", cleanup_failed_);
        }
    }

    aclrtStream stream() const { return stream_; }

private:
    bool& cleanup_failed_;
    aclrtStream stream_ = nullptr;
    int32_t device_ = 0;
    bool initialized_ = false;
    bool device_set_ = false;
};

struct CaseResources {
    CaseResources(aclrtStream stream, bool& cleanup_failed)
        : stream(stream), cleanup_failed(cleanup_failed) {}
    CaseResources(const CaseResources&) = delete;
    CaseResources& operator=(const CaseResources&) = delete;

    ~CaseResources()
    {
        // A failed launch may still have queued work. Drain it before releasing
        // tensor storage, workspace, or the executor.
        if (needs_sync) {
            CheckCleanup(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", cleanup_failed);
        }
        if (executor != nullptr) {
            CheckCleanup(aclDestroyAclOpExecutor(executor), "aclDestroyAclOpExecutor", cleanup_failed);
        }
        for (auto* tensor : tensors) {
            if (tensor != nullptr) {
                CheckCleanup(aclDestroyTensor(tensor), "aclDestroyTensor", cleanup_failed);
            }
        }
        if (workspace != nullptr) {
            CheckCleanup(aclrtFree(workspace), "aclrtFree(workspace)", cleanup_failed);
        }
        for (auto* buffer : buffers) {
            if (buffer != nullptr) {
                CheckCleanup(aclrtFree(buffer), "aclrtFree(tensor)", cleanup_failed);
            }
        }
    }

    aclrtStream stream;
    bool& cleanup_failed;
    std::array<void*, 3> buffers{};
    std::array<aclTensor*, 3> tensors{};
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    bool needs_sync = false;
};

void RunCase(int64_t length, aclrtStream stream, bool& cleanup_failed)
{
    if (length <= 0 || static_cast<uint64_t>(length) >
                           std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("The example requires a nonempty FP32 vector.");
    }
    const size_t count = static_cast<size_t>(length);
    const size_t bytes = count * sizeof(float);
    std::vector<float> x(count);
    std::vector<float> y(count);
    std::vector<float> output(count, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < count; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 127) - 63) * 0.25F;
        y[i] = static_cast<float>(static_cast<int>(i % 61) - 30) * 0.125F;
    }

    CaseResources resources(stream, cleanup_failed);
    const int64_t stride = 1;
    const std::array<const float*, 3> host_buffers{x.data(), y.data(), output.data()};
    for (size_t i = 0; i < resources.buffers.size(); ++i) {
        Check(aclrtMalloc(&resources.buffers[i], bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tensor)");
        Check(aclrtMemcpy(resources.buffers[i], bytes, host_buffers[i], bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
        resources.tensors[i] = aclCreateTensor(&length, 1, ACL_FLOAT, &stride, 0,
                                               ACL_FORMAT_ND, &length, 1, resources.buffers[i]);
        if (resources.tensors[i] == nullptr) {
            throw std::runtime_error("aclCreateTensor returned null");
        }
    }

    uint64_t workspace_size = 0;
    Check(aclnnOpsRepoAddGetWorkspaceSize(resources.tensors[0], resources.tensors[1],
                                        resources.tensors[2], &workspace_size,
                                        &resources.executor),
          "aclnnOpsRepoAddGetWorkspaceSize");
    if (resources.executor == nullptr) {
        throw std::runtime_error("aclnnOpsRepoAddGetWorkspaceSize returned a null executor");
    }
    // Explicit executor ownership also covers errors allocating workspace or
    // launching the kernel. These public APIs are available in CANN 8.x.
    Check(aclSetAclOpExecutorRepeatable(resources.executor), "aclSetAclOpExecutorRepeatable");
    if (workspace_size != 0) {
        Check(aclrtMalloc(&resources.workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(workspace)");
    }
    resources.needs_sync = true;
    Check(aclnnOpsRepoAdd(resources.workspace, workspace_size, resources.executor, stream),
          "aclnnOpsRepoAdd");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    resources.needs_sync = false;
    Check(aclrtMemcpy(output.data(), bytes, resources.buffers[2], bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");

    for (size_t i = 0; i < count; ++i) {
        const float expected = x[i] + y[i];
        // Inputs and their sums are exactly representable binary fractions.
        // NaN-initialized output also detects elements the kernel never writes.
        if (!std::isfinite(output[i]) || output[i] != expected) {
            throw std::runtime_error("Mismatch for length=" + std::to_string(length) +
                                     ", index=" + std::to_string(i) +
                                     ", expected=" + std::to_string(expected) +
                                     ", actual=" + std::to_string(output[i]));
        }
    }
    std::cout << "PASS length=" << length << '\n';
}

int32_t ParseDevice(const std::string& value)
{
    size_t parsed = 0;
    const long long device = std::stoll(value, &parsed);
    if (parsed != value.size() || device < 0 || device > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("--device must be a nonnegative 32-bit integer");
    }
    return static_cast<int32_t>(device);
}

}  // namespace

int main(int argc, char** argv)
{
    bool cleanup_failed = false;
    try {
        int32_t device = 0;
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [--device N]\n";
            return EXIT_SUCCESS;
        }
        if (argc == 3 && std::string(argv[1]) == "--device") {
            device = ParseDevice(argv[2]);
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: ops_repo_add [--device N]");
        }
        {
            Session session(cleanup_failed);
            session.Initialize(device);
            const std::array<int64_t, 12> lengths{
                1, 7, 8, 9, 1023, 1024, 1025, 4095, 4096, 4097, 65537, 1048581};
            for (int64_t length : lengths) {
                RunCase(length, session.stream(), cleanup_failed);
                if (cleanup_failed) {
                    throw std::runtime_error("Resource cleanup failed; stopping further cases");
                }
            }
        }
        if (cleanup_failed) {
            return EXIT_FAILURE;
        }
        std::cout << "All 12 FP32 cases passed on device " << device << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
