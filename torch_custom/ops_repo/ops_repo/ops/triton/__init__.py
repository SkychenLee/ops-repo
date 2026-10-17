"""Reserved Triton backend. No kernel is implemented or loaded."""


def initialize():
    raise NotImplementedError("ops_repo Triton backend is a development placeholder; no kernel is implemented")


def add(x, y):
    raise NotImplementedError("ops_repo.ops.triton.add has no kernel yet; select backend='ascendc'")


__all__ = ["add", "initialize"]
