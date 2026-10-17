"""Backend routing after the package's AscendC import-time initialization."""

from .ops.ascendc import add as _ascendc_add
from ._runtime import initialize_package as _initialize_ascendc


def initialize(backend="ascendc"):
    """Idempotent compatibility API; AscendC is already initialized by import."""
    if backend == "ascendc":
        return _initialize_ascendc()
    if backend == "triton":
        from .ops.triton import initialize as initialize_triton

        return initialize_triton()
    raise ValueError("backend must be 'ascendc' or 'triton', got %r" % (backend,))


def add(x, y, backend="ascendc"):
    """Return x + y using the selected implementation; no implicit fallback."""
    if backend == "ascendc":
        return _ascendc_add(x, y)
    if backend == "triton":
        from .ops.triton import add as triton_add

        return triton_add(x, y)
    raise ValueError("backend must be 'ascendc' or 'triton', got %r" % (backend,))
