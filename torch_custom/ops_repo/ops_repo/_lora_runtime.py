"""Load the independent direct-launch component without an OPP or vLLM dependency."""

import importlib
import json
from pathlib import Path
import platform
import sys
import threading

OP_NAMES = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "build_combined_lora_idx", "add_lora", "add_lora_shrink", "add_lora_expand",
    "add_lora_expand_delta", "add_lora_swiglu_quant",
)
_lock = threading.Lock()
_ready = False
_loaded = False


def is_installed():
    package = Path(__file__).resolve().parent
    return (package / "_lora_build_info.json").exists() or bool(list(package.glob("_lora_C*.so")))


def initialize():
    global _ready, _loaded
    if _ready:
        return
    with _lock:
        if _ready:
            return
        package = Path(__file__).resolve().parent
        if not is_installed():
            raise RuntimeError(
                "ops_repo MoE LoRA component is not built. Build a wheel with "
                "OPS_REPO_BUILD_LORA=1 OPS_REPO_SOC=ascend910_93 (or ascend910b) "
                "in a matching CANN/torch/torch_npu environment."
            )
        try:
            info = json.loads((package / "_lora_build_info.json").read_text())
            if (info["schema_version"] != 1 or info["component"] != "moe_lora"
                    or info["soc"] not in ("ascend910b", "ascend910_93")):
                raise ValueError("unsupported component or target")
            expected = {"python": "%d.%d" % sys.version_info[:2], "machine": platform.machine()}
            for name, actual in expected.items():
                if info[name] != actual:
                    raise ValueError("built for %s=%s, running %s" % (name, info[name], actual))
            torch = importlib.import_module("torch")
            torch_npu = importlib.import_module("torch_npu")
            for name, module in (("torch", torch), ("torch_npu", torch_npu)):
                if info[name] != module.__version__:
                    raise ValueError("built for %s=%s, running %s" % (name, info[name], module.__version__))
            candidates = list(package.glob("_lora_C*.so"))
            if len(candidates) != 1 or not (package / "libops_repo_moe_lora_kernels.so").is_file():
                raise ValueError("expected one native extension and its direct kernel library")
            if not _loaded:
                torch.ops.load_library(str(candidates[0]))
                _loaded = True
            for name in OP_NAMES:
                if not hasattr(torch.ops.ops_repo, name):
                    raise ValueError("native extension is missing " + name)
        except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise RuntimeError("Cannot load ops_repo MoE LoRA component: %s. Rebuild/reinstall "
                               "the wheel for this environment." % exc) from exc
        _ready = True


def call(name, *args, **kwargs):
    initialize()
    import torch

    return getattr(torch.ops.ops_repo, name)(*args, **kwargs)
