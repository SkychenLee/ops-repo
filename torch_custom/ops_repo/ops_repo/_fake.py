"""Registration helpers imported only after the native dispatcher schema exists."""

from ._validation import validate_add_inputs

_registered = False


def _fake_add(x, y):
    # Meta/Fake execution describes metadata only and never touches an NPU.
    # Device equality is checked, but CPU FakeTensors are useful for CPU-only
    # schema tests; a real CPU dispatcher kernel is deliberately not registered.
    validate_add_inputs(x, y, require_npu=False)
    return x.new_empty(x.shape)


def register_fake():
    global _registered
    if _registered:
        return
    import torch

    if not hasattr(torch.library, "register_fake"):
        raise RuntimeError("ops_repo requires PyTorch >= 2.4 with torch.library.register_fake")
    torch.library.register_fake("ops_repo::add")(_fake_add)
    _registered = True
