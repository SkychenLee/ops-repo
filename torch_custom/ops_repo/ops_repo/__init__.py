"""Ascend operator bindings, initialized when the package is imported.

Configure CANN before starting Python. Complete wheels configure their embedded
OPP at interpreter startup; an explicit external OPP needs its vendor environment.
The standalone MoE/LoRA component uses direct kernels and needs no custom OPP.
Import failures are reported immediately; import does not select an NPU device.
"""

from ._runtime import initialize_package as _initialize_ascendc
from ._runtime import package_version as _package_version

_initialize_ascendc()

from ._api import add, initialize
from .moe_lora import (
    add_lora, add_lora_expand, add_lora_expand_delta, add_lora_shrink,
    add_lora_swiglu_quant, bgmv_expand, bgmv_shrink, build_combined_lora_idx,
    sgmv_expand, sgmv_shrink,
)

del _initialize_ascendc

__version__ = _package_version()
del _package_version
__all__ = ["add", "initialize", "add_lora", "add_lora_expand", "add_lora_expand_delta",
           "add_lora_shrink", "add_lora_swiglu_quant", "bgmv_expand", "bgmv_shrink",
           "build_combined_lora_idx", "sgmv_expand", "sgmv_shrink"]
