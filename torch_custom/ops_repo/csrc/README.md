# PyTorch / torch_npu adapter

`torch_bindings.cpp` registers `torch.ops.ops_repo.add` with a `PrivateUse1` kernel. It uses the installed torch/torch_npu C++ ABI; rebuild when changing that pair. It does not claim FLA's Stable ABI compatibility and does not copy FLA's runtime.

Build is optional: `OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps .`. `OPS_REPO_OPP_PATH` must name the installed OPP root containing `vendors/ops_repo`, and `ASCEND_HOME_PATH` must name a complete CANN toolkit. Source the CANN environment and the OPP vendor's `set_env.bash` before starting either the build or runtime Python process. The generated ACLNN header is used as the native signature contract; custom symbols are resolved from the exact vendor library with a local `dlopen` handle, not through a global library name.

Installing the Python-only package is supported, but importing `ops_repo` requires the native extension and the complete AscendC runtime. The root package imports torch/torch_npu, validates the OPP environment, loads the native libraries, and registers the Fake implementation automatically; importing a submodule executes that same root initialization. No explicit `initialize()` call is required. That API remains idempotent for compatibility. Package initialization does not select a device or repair the environment; every execution worker should import its model module before compilation or graph capture.

Run packaging commands from the repository root; its `setup.py` is the only package build entry. The Python source lives under `torch_custom/ops_repo/ops_repo` so a repository checkout does not shadow an installed native wheel. The C++ adapter lives alongside it in `torch_custom/ops_repo/csrc`. Editable development is supported by setuptools: use the same native build flag with `python3 -m pip install --no-build-isolation --no-deps -e .`; rebuild after C++ changes. Alternatively, `OPS_REPO_BUILD_TORCH=1 python3 setup.py build_ext --inplace` places the library in `torch_custom/ops_repo/ops_repo`, and `PYTHONPATH="$PWD/torch_custom/ops_repo"` selects that development copy. Native/editable compilation still needs validation on the target CANN machine.

Kernel sources are separate from the adapter: AscendC operators live under `ops/ascendc/`, while `ops/triton/triton_core/` is mapped by the root setup to `ops_repo.ops.triton.triton_core` without copying its sources. Install the package (normally or editable) to import that mapped core namespace; a bare `PYTHONPATH` entry alone does not apply setuptools package mappings. The public Triton backend remains explicitly unimplemented.

The adapter uses `NpuExtension`, `NPUGuard`, the current `NPUStream`, and `OpCommand::RunOpApi`. The handler retains the inputs, output, workspace, ACL tensor descriptors, and an explicitly owned executor until the queued host launch finishes. Tensor storage is recorded on the launch stream and released through the NPU caching allocator. Each call uses a separate executor; no cross-call executor cache or synchronization is introduced. ND format is checked in C++, including callers that bypass the Python wrapper.

Official interfaces consulted:

- [NpuExtension build helper](https://github.com/Ascend/pytorch/blob/master/torch_npu/utils/cpp_extension.py)
- [Official extension using RunOpApi](https://github.com/Ascend/pytorch/blob/master/test/cpp_extensions/extension.cpp)
- [OpCommand task-queue implementation](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/OpCommand.cpp)
- [NPU guards](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGuard.h) and [NPU streams](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.h)
- [NPU format helpers](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUFormat.h) and [stream-aware allocator](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUCachingAllocator.h)

The native adapter has not been compiled or executed on an NPU in the development environment. Fake/Meta registration enables metadata propagation; it does not establish compatibility with a specific TorchAir converter, NPU graph capture mode, or vLLM release. Those require hardware integration tests, including repeated calls, nondefault streams, multi-device guards, asynchronous lifetime, and graph replay.
