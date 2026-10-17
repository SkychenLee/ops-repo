"""Metadata checks shared by the Python entry point and Fake/Meta kernel."""


def validate_add_inputs(x, y, *, require_npu=True):
    # Validation uses the PyTorch dependency already loaded by package import.
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("ops_repo.add expects two torch.Tensor inputs")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise ValueError("ops_repo.add requires dense strided tensors")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("ops_repo.add supports only float32 inputs")
    if x.device != y.device:
        raise ValueError("ops_repo.add inputs must be on the same device")
    if require_npu and x.device.type not in ("npu", "privateuseone", "meta"):
        raise ValueError("ops_repo.add requires NPU inputs; CPU execution is not implemented")
    if x.dim() != y.dim():
        raise ValueError("ops_repo.add input shapes must match; broadcasting is unsupported")
    # torch._check preserves symbolic shape guards instead of concretizing SymInt.
    for left, right in zip(x.shape, y.shape):
        torch._check(left == right)
    torch._check(x.numel() > 0)
    torch._check(x.numel() <= (2**63 - 1) // 4)
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("ops_repo.add requires contiguous inputs; call contiguous() explicitly")
    if x.requires_grad or y.requires_grad:
        raise RuntimeError("ops_repo.add is inference-only; pass tensors with requires_grad=False")
