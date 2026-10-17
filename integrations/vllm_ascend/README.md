# vLLM Ascend misha integration

`misha-ops-repo.patch` adapts vLLM Ascend's LoRA call sites to the standalone
`ops_repo` public API. Its source baseline is the `misha` branch at commit
`72b7912616e0e296e64f0ca0003afcc2ef7ce6bc`. The patch includes the new backend
selector and its regression tests, and excludes pre-existing untracked files.
The migration scope explicitly excludes operators, helper functions, and build
configuration from other branches. Existing experimental files remain untouched;
repairing their cross-branch dependencies is outside this migration.

Apply the patch to a checkout of that baseline after installing an `ops_repo`
wheel built with LoRA support in the worker's Python environment:

```bash
OPS_REPO_CHECKOUT=/path/to/ops-repo
cd /path/to/vllm-ascend
git apply --check "$OPS_REPO_CHECKOUT/integrations/vllm_ascend/misha-ops-repo.patch"
git apply "$OPS_REPO_CHECKOUT/integrations/vllm_ascend/misha-ops-repo.patch"
```

The implementation container's checkout is
`ds-ltc-v25-ep:/vllm-workspace/vllm-ascend`; the patch has already been applied
there. Do not apply it a second time. No vLLM Ascend C++ rebuild is needed for
these Python changes.

The selector resolves the backend on the first LoRA operator call. An installed
`ops_repo` takes priority; if the package is absent, the original
`torch.ops._C_ascend` implementation remains available. An installed package
that fails to load or lacks any required public operator raises an explicit
`ImportError`, retaining the original load error as its cause. It does not
silently fall back. Restart workers after installing or changing the library.
Importing the MoE execution modules without using LoRA does not load `ops_repo`.
No environment variable is added.

The ten required public operators are `bgmv_shrink`, `bgmv_expand`,
`sgmv_shrink`, `sgmv_expand`, `build_combined_lora_idx`, `add_lora`,
`add_lora_shrink`, `add_lora_expand`, `add_lora_expand_delta`, and
`add_lora_swiglu_quant`. Their existing arguments, return values, mutation
semantics, and selection of decode/prefill and fully sharded paths are retained.
The standalone implementations register under `torch.ops.ops_repo`.

Run the CPU and fake-tensor checks from the patched vLLM Ascend checkout:

```bash
python -m pytest -q tests/ut/lora/test_ops_backend.py tests/ut/lora/test_quant_moe.py
python -m ruff check vllm_ascend/lora/ops_backend.py vllm_ascend/lora/lora_ops.py tests/ut/lora/test_ops_backend.py
python -m ruff format --check vllm_ascend/lora/ops_backend.py vllm_ascend/lora/lora_ops.py tests/ut/lora/test_ops_backend.py
git diff --check
```

The backend tests cover delayed loading, missing and broken installations,
the complete API requirement, argument forwarding, fake tensors, and cold and
warm `torch.compile(fullgraph=True)` calls for both backends. The compilation
test uses a CPU custom operator with a Meta implementation and launches no NPU
work. It blocks NPU lazy initialization and suppresses Dynamo's accelerator
stream/device probing only within the CPU compilation test. NPU numerical and
performance tests are not part of this validation.

The installed `0.1.0+ascend910.93` wheel was also checked through the actual
selector and vLLM wrappers with FakeTensor inputs. A single full graph contained
`torch.ops.ops_repo.bgmv_shrink` and `torch.ops.ops_repo.bgmv_expand`, and NPU
initialization remained disabled. The local reproduction script and output are
in `build/migration/vllm-selector-fake-compile.py` and its `.log` file.

Two pre-existing environment and test limitations were identified:

- The container's existing `vllm_ascend_C` shared library fails to load with
  `undefined symbol: vllm_ascend_recover_ub_bytes`. Tests that explicitly import
  it, such as `test_combined_idx_dedup.py`, fail during collection. This patch
  does not rebuild or repair that unrelated shared library. The missing helper
  is referenced by an untracked experimental source file and belongs to another
  branch; neither that helper nor its build configuration is included here.
- `test_lora.py::test_moe_lora_apply_uses_adapter_enabled` and
  `test_lora.py::test_moe_lora_apply_propagates_fully_sharded_metadata` have mock
  return values that cause `Tensor += Mock`. Both failures also occur with the
  original `misha` HEAD implementation of `fused_moe.py`.
