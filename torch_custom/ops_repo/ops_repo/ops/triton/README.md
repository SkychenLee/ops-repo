# Triton backend development layout

This namespace is reserved. Once the package has imported successfully, `add()` and `initialize()` raise `NotImplementedError`; neither imports Triton nor falls back to a different backend. Importing this namespace first executes the root `ops_repo` initialization, which requires the AscendC runtime and native extension. This placeholder is not an independently usable Triton backend.

The device-source namespace lives in `ops/triton/triton_core/`, mapped by the repository-root setup to `ops_repo.ops.triton.triton_core` in the installed package. It has one source location; this adapter directory owns backend routing. Install normally or editable to use the mapped namespace.

When implementing an Ascend Triton version, add `_add_kernel.py` under the core directory with the `@triton.jit` device function and a separate `add.py` launcher. Load them through package-managed backend initialization, before tracing or capture, after checking the installed `triton-ascend` and CANN versions. A proposed implementation should include:

```python
# Design sketch only: not an executable or registered kernel.
# @triton.jit
# def _add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
#     offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
#     valid = offsets < N
#     ...  # masked loads, FP32 addition, masked store
#
# def add(x, y):
#     ...  # shared shape/dtype/device/contiguity checks
#     ...  # allocate output, launch ceil_div(N, BLOCK) programs
```

Validate tail masks, scalar inputs, nonempty equal shapes, nondefault streams, Fake/Meta behavior, graph execution, and accuracy on the intended NPU before replacing the placeholder. Add the corresponding registration and tests explicitly; an unavailable Triton kernel must never silently run the AscendC implementation.
