"""AscendC implementation initialized by the root package before this import."""

from ..._runtime import call_add as add
from ..._runtime import initialize_ascendc as initialize

__all__ = ["add", "initialize"]
