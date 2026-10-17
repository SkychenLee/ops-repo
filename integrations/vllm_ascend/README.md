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
Each worker logs the selected package path once, when it first resolves
`ops_repo`; there is no per-token logging.

For the `ds_16_v25.sh` scenario, the patch also recognizes
`additional_config.moe_lora_ep_backend`. The default `"auto"` retains the
original device-based selection. Explicit `"allgather"` selects the existing
EP-aware AllGather path for LoRA on A2/A3/A5; no-LoRA and non-EP selection is
unchanged. Other values are rejected at configuration time. This makes the
script's explicit setting effective on A3, where the old selector otherwise
used AlltoAll. It uses only the tracked misha dispatcher and local-expert
mapping, without experimental files.

The script's `VLLM_ASCEND_DISABLE_PIN_MEMORY=1` is now honored by the Ascend
platform. Set it before launching vLLM: upstream caches this policy during
module import, and the LoRA checkpoint loader then avoids pinned host
allocations. The default remains `0` (pinning enabled). This prevents the
observed `aclrtMallocHostWithCfg` host-memory allocation failure when loading
the large adapter on eight workers. Five CPU regression cases include fresh
processes that exercise the actual upstream LoRA loader with both settings.

The ten required public operators are `bgmv_shrink`, `bgmv_expand`,
`sgmv_shrink`, `sgmv_expand`, `build_combined_lora_idx`, `add_lora`,
`add_lora_shrink`, `add_lora_expand`, `add_lora_expand_delta`, and
`add_lora_swiglu_quant`. Their existing arguments, return values, mutation
semantics, and selection of decode/prefill and fully sharded paths are retained.
The standalone implementations register under `torch.ops.ops_repo`.

Run the CPU and fake-tensor checks from the patched vLLM Ascend checkout:

```bash
python -m pytest -q \
  tests/ut/lora/test_pin_memory_policy.py \
  tests/ut/lora/test_ep_backend_config.py \
  tests/ut/test_ascend_forward_context.py \
  tests/ut/lora/test_ops_backend.py tests/ut/lora/test_quant_moe.py
python -m ruff check vllm_ascend/lora/ops_backend.py vllm_ascend/lora/lora_ops.py tests/ut/lora/test_ops_backend.py
python -m ruff format --check vllm_ascend/lora/ops_backend.py vllm_ascend/lora/lora_ops.py tests/ut/lora/test_ops_backend.py
git diff --check
```

The backend tests cover delayed loading, missing and broken installations,
the complete API requirement, argument forwarding, fake tensors, and cold and
warm `torch.compile(fullgraph=True)` calls for both backends. The compilation
test uses a CPU custom operator with a Meta implementation and launches no NPU
work. It blocks NPU lazy initialization and suppresses Dynamo's accelerator
stream/device probing only within the CPU compilation test. These CPU checks
are separate from the NPU validation recorded in [the migration report](../../docs/moe_lora.md).
The five files above passed all 82 cases, including the original communication
selector tests and the new EP configuration and pin-memory policy tests.

The installed `0.1.0+ascend910.93` wheel was also checked through the actual
selector and vLLM wrappers with FakeTensor inputs. A single full graph contained
`torch.ops.ops_repo.bgmv_shrink` and `torch.ops.ops_repo.bgmv_expand`, and NPU
initialization remained disabled. The local reproduction script and output are
in `build/migration/vllm-selector-fake-compile.py` and its `.log` file.

The pre-existing tests
`test_lora.py::test_moe_lora_apply_uses_adapter_enabled` and
`test_lora.py::test_moe_lora_apply_propagates_fully_sharded_metadata` have mock
return values that cause `Tensor += Mock`. Both failures also occur with the
original `misha` HEAD implementation of `fused_moe.py`.

## Validated v1 TP8 + EP launch

The active scenario is `/mnt/share/litianchen/v25_ep/ds_16_v25.sh` in
`ds-ltc-v25-ep`, using vLLM `0.25.1` at
`752a3a504485790a2e8491cacbb35c137339ad34`. It uses model runner v1, TP8,
expert parallelism with `moe_lora_ep_backend="allgather"`, rank-16 LoRA,
`max_loras=3`, `max_num_seqs=8`, DSpark with five speculative tokens, and
`FULL_DECODE_ONLY`. Fully sharded LoRA is disabled. The model is
`/mnt/weight/DeepSeek-V4-Flash-0731-w8a8` on physical devices 8–15, port 8888.

The script's adapter path `/mnt/share/remote_share/litianchen/ds_lora_16`
does not exist in this environment. The validation launch copy uses
`/mnt/share/litianchen/ds_lora_16`; all other workload settings are retained.
The original script is unchanged. To prepare the same launch copy:

```bash
OPS_REPO_CHECKOUT=/mnt/share/litianchen/ops-repo
mkdir -p "$OPS_REPO_CHECKOUT/build/migration/model-validation"
sed 's|/mnt/share/remote_share/litianchen/ds_lora_16|/mnt/share/litianchen/ds_lora_16|g' \
  /mnt/share/litianchen/v25_ep/ds_16_v25.sh \
  > "$OPS_REPO_CHECKOUT/build/migration/model-validation/ds_16_v25.local.sh"
# Run inside the prepared container when devices 8–15 and port 8888 are available.
docker exec -it ds-ltc-v25-ep bash \
  "$OPS_REPO_CHECKOUT/build/migration/model-validation/ds_16_v25.local.sh"
```

After the server is ready, the standard-library HTTP example checks both
models, short/long prompts, two-turn conversations, and mixed concurrent
base/LoRA requests:

```bash
python3 examples/check_vllm_lora_http.py \
  --base-url http://127.0.0.1:8888 --timeout 600 \
  --concurrency 8 --rounds 3 --max-tokens 128 \
  --output build/migration/model-validation/http-smoke.json
```

Validation captured 14 full-decode graphs. All eight workers mapped both
`ops_repo/_lora_C*.so` and `ops_repo/libops_repo_moe_lora_kernels.so`.
All 32 HTTP requests passed (16 for `ds`, 16 for `lora1`), each finished with
`stop`, and the successful server log contains no ERROR lines. The longest
prompt was about 1949 tokens; this is an integration smoke test, not a 128K
stress test or a quality/performance benchmark. No profiler trace was collected.

Local evidence is in `build/migration/model-validation/v1-ep-20261017-113246/`:
`launch.sh`, `original-ds_16_v25.sh`, `server.log`,
`backend-library-maps.json`, `http-smoke.json`, and `http-smoke.log`.
The current workers imported the backend before its INFO logger was corrected;
their library selection is evidenced by the process maps. Subsequent restarts
use the final patch's visible one-time selection log. These local runtime
artifacts are excluded from Git; the HTTP example and NPU tests are committed.

## Clean native rebuild performed in the implementation container

The implementation container previously failed to import `vllm_ascend_C` with
`undefined symbol: vllm_ascend_recover_ub_bytes`. Its untracked
`csrc/moe_lora_single_adapter_indices.cpp` also referenced
`vllm_ascend_recover_vector_cores`. The top-level CMake file uses
`file(GLOB ... csrc/*.cpp)`, so rebuilding the working directory would include
that experimental source even though neither helper exists in the committed
`misha` baseline. Both helpers belong to another branch and remain outside this
migration.

This container has now been repaired by rebuilding the original native extension
from a clean archive of `72b7912616e0e296e64f0ca0003afcc2ef7ce6bc` and replacing
only its binding and two copies of the direct-kernel library. No experimental
source was deleted or changed; fingerprints of all ten untracked C++ source and
header files remained unchanged. The Python integration changes and installed
custom OPP were retained. Other servers with a working native extension do not
need this rebuild to apply the Python integration patch.

The following commands reproduce the build in `ds-ltc-v25-ep`. Paths identify
the implementation environment; substitute local checkout and toolkit paths on
another server. Use the matching worker Python environment and restart workers
after replacing native libraries.

```bash
docker exec -it ds-ltc-v25-ep bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -euo pipefail
VLLM_CHECKOUT=/vllm-workspace/vllm-ascend
OPS_REPO_CHECKOUT=/mnt/share/litianchen/ops-repo
mkdir -p "$OPS_REPO_CHECKOUT/build/migration"
NATIVE_REBUILD_DIR=$(mktemp -d "$OPS_REPO_CHECKOUT/build/migration/native-rebuild-72b791261-XXXXXXXX")
mkdir -p "$NATIVE_REBUILD_DIR"/{source,native-build,stage}

git -C "$VLLM_CHECKOUT" archive 72b7912616e0e296e64f0ca0003afcc2ef7ce6bc \
  | tar -xf - -C "$NATIVE_REBUILD_DIR/source"
# git archive does not include a submodule's contents. Export the revision
# recorded by this baseline, rather than copying its working directory.
git -C "$VLLM_CHECKOUT/csrc/third_party/catlass" archive \
  41bf90da655bba3c66d0acd7e00abe33960ecfd6 \
  | tar -xf - -C "$NATIVE_REBUILD_DIR/source/csrc/third_party/catlass"

cmake -S "$NATIVE_REBUILD_DIR/source" -B "$NATIVE_REBUILD_DIR/native-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSOC_VERSION=ascend910_9391 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
  -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)" \
  -DTORCH_NPU_PATH="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch_npu").origin).parent)')" \
  -DCMAKE_INSTALL_PREFIX="$NATIVE_REBUILD_DIR/stage" \
  2>&1 | tee "$NATIVE_REBUILD_DIR/configure.log"
cmake --build "$NATIVE_REBUILD_DIR/native-build" \
  --target vllm_ascend_C --parallel 8 \
  2>&1 | tee "$NATIVE_REBUILD_DIR/build.log"
```

Only the top-level `vllm_ascend_C` target and its direct-kernel dependency are
built. This does not invoke `setup.py`, `build_aclnn.sh`, device discovery, or a
custom OPP rebuild. Stage and validate the new libraries before backing up and
replacing the three existing binaries:

```bash
python - "$NATIVE_REBUILD_DIR" "$VLLM_CHECKOUT" <<'PY'
from pathlib import Path
import hashlib, json, shutil, subprocess, sys

run, checkout = map(Path, sys.argv[1:])
build, stage, package = run / "native-build", run / "stage", checkout / "vllm_ascend"
new_extensions = list(build.glob("vllm_ascend_C*.so"))
old_extensions = list(package.glob("vllm_ascend_C*.so"))
assert len(new_extensions) == len(old_extensions) == 1
new_extension = new_extensions[0]
new_kernel = build / "lib/libvllm_ascend_kernels.so"
for source in (new_extension, new_kernel):
    shutil.copy2(source, stage / source.name)

check = '''
import importlib, sys, torch, torch_npu
def forbid_init(*args, **kwargs):
    raise AssertionError("NPU initialization is forbidden in this load check")
assert not torch.npu.is_initialized()
torch.npu._lazy_init = forbid_init
torch_npu._C._npu_init = forbid_init
if sys.argv[1] == "installed":
    importlib.import_module("vllm_ascend.vllm_ascend_C")
else:
    torch.ops.load_library(sys.argv[1])
names = ("bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
         "build_combined_lora_idx", "add_lora", "add_lora_shrink",
         "add_lora_expand", "add_lora_expand_delta", "add_lora_swiglu_quant")
for name in names:
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::" + name, "PrivateUse1"), name
import ops_repo
assert all(hasattr(torch.ops.ops_repo, name) for name in names)
assert not torch.npu.is_initialized()
print("Native import, ten legacy operators, and ops_repo coexistence passed; NPU not initialized")
'''
subprocess.run([sys.executable, "-c", check, str(stage / new_extension.name)], check=True)
backup = run / "original-binaries-backup"
backup.mkdir()
pairs = [(old_extensions[0], stage / new_extension.name),
         (package / new_kernel.name, stage / new_kernel.name),
         (package / "lib" / new_kernel.name, stage / new_kernel.name)]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = []
for target, source in pairs:
    saved = backup / target.relative_to(package)
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, saved)
    assert digest(saved) == digest(target)
    receipt.append({"target": str(target), "backup": str(saved),
                    "old_sha256": digest(saved), "new_sha256": digest(source)})
(run / "replacement-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
try:
    for target, source in pairs:
        temporary = target.with_name(target.name + ".native-rebuild.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    subprocess.run([sys.executable, "-c", check, "installed"], check=True)
except BaseException:
    for target, _ in pairs:
        temporary = target.with_name(target.name + ".native-rollback.tmp")
        shutil.copy2(backup / target.relative_to(package), temporary)
        temporary.replace(target)
    raise
print("Original binaries backed up to", backup)
PY
```

The completed repair used
`/mnt/share/litianchen/ops-repo/build/migration/native-rebuild-72b791261-twwqwgho/`.
Its `original-binaries-backup/` contains the original binding and both kernel
libraries; `replacement-receipt.json` records their old/new SHA256 hashes.
`build-commands.json`, `build.log`, `staged-load-check.log`, and
`installed-load-check.log` retain the exact build arguments and validation
results. These local build artifacts are intentionally excluded from Git.

The installed extension now imports successfully, has all ten legacy operators
registered for `PrivateUse1`, and coexists with the installed `ops_repo` library.
Neither rebuilt binary references the missing `recover_*` helpers. Validation
kept `torch.npu.is_initialized()` false and executed no NPU operators. This load
check does not establish model accuracy or performance, and the two baseline
mock-test failures above are separate issues.

A subsequent isolated numerical comparison used physical NPU 9 only
(`ASCEND_RT_VISIBLE_DEVICES=9`, logical `npu:0`) after the native rebuild.
All 16 output comparisons between `torch.ops._C_ascend` and `torch.ops.ops_repo`
were bitwise equal, with maximum absolute difference zero: FP16/BF16 fused
rank-16 LoRA, rank-2 shrink, rank-16 expand, dense GMM fallback, MoE GMM
shrink/expand/delta, FP16/BF16 SwiGLU activation/INT8/scale, and INT32/INT64
routing metadata. The script asserted that exactly one device was visible and
has exited. Its local script, log, and JSON results are
`build/migration/legacy-vs-ops-repo-npu9.{py,log,json}`. This checks migration
equivalence on those inputs; it does not measure performance or replace full
model validation.
