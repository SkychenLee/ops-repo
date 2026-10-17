"""Initialize the PyTorch extension and packaged or explicit custom OPP."""

import importlib
import json
import os
from pathlib import Path
import platform
import sys
import threading

_lock = threading.Lock()
_ready = False
_torch = None
_library_loaded = False


def _op_api_library():
    root = os.environ.get("OPS_REPO_OPP_PATH")
    if not root:
        package = Path(__file__).resolve().parent
        if ((package / "opp").exists() or (package / "opp").is_symlink()
                or (package / "_build_info.json").exists()
                or (package / "_build_info.json").is_symlink()):
            try:
                from ops_repo_opp_env import configure
            except ImportError as exc:
                raise RuntimeError(
                    "The complete ops_repo wheel is missing its startup helper. "
                    "Reinstall the complete wheel to restore ops_repo_opp_env.py and its .pth hook."
                ) from exc
            return configure(package)
        raise RuntimeError(
            "Set OPS_REPO_OPP_PATH to the installed OPP root (containing vendors/ops_repo) "
            "and source its vendor set_env.bash before starting Python."
        )
    vendor = (Path(root).expanduser().resolve() / "vendors" / "ops_repo")
    configured = {
        Path(part).expanduser().resolve()
        for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if part
    }
    for path in configured:
        other = path / "vendors" / "ops_repo"
        if path.name == "ops_repo" and path.parent.name == "vendors":
            other = path
        elif path.name in ("lib", "lib64") and path.parent.name == "op_api":
            candidate = path.parent.parent
            if candidate.name == "ops_repo" and candidate.parent.name == "vendors":
                other = candidate
        if other.is_dir() and other.resolve() != vendor.resolve():
            raise RuntimeError(
                "Conflicting ops_repo OPP in ASCEND_CUSTOM_OPP_PATH: %s. Remove the old "
                "ops_repo paths and start a new Python process with the selected vendor environment."
                % other
            )
    if vendor not in configured:
        raise RuntimeError(
            "ASCEND_CUSTOM_OPP_PATH must contain %s. Source its set_env.bash before "
            "starting Python; initialize() does not repair an already initialized CANN runtime."
            % vendor
        )
    matches = {
        path.resolve()
        for folder in ("lib", "lib64")
        for path in [vendor / "op_api" / folder / "libcust_opapi.so"]
        if path.is_file()
    }
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one installed ops_repo libcust_opapi.so under %s/op_api/lib[64]; "
            "found %d. Build and install the matching OPP first." % (vendor, len(matches))
        )
    return next(iter(matches))


def _build_information():
    """Validate complete-wheel identity before importing any framework."""
    package = Path(__file__).resolve().parent
    path = package / "_build_info.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        path = path.resolve(strict=True)
        path.relative_to(package)
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Damaged ops_repo wheel: invalid or escaping _build_info.json") from exc
    fields = ("soc", "python", "machine", "torch", "torch_npu")
    if (not isinstance(info, dict) or type(info.get("schema_version")) is not int
            or info["schema_version"] != 1
            or any(not isinstance(info.get(key), str) or not info[key] for key in fields)
            or info["soc"] not in ("ascend910b", "ascend910_93", "ascend950")
            or ("version" in info and (not isinstance(info["version"], str) or not info["version"]))):
        raise RuntimeError("Damaged ops_repo wheel: invalid build metadata schema")
    actual = {"python": "%d.%d" % sys.version_info[:2], "machine": platform.machine()}
    for key, current in actual.items():
        if info[key] != current:
            raise RuntimeError(
                "Incompatible ops_repo wheel: built for %s=%s, current %s=%s. "
                "Install or build a wheel for this environment."
                % (key, info[key], key, current)
            )
    return info


def _check_framework_versions(info, torch, torch_npu):
    if info is None:
        return
    for name, module in (("torch", torch), ("torch_npu", torch_npu)):
        actual = str(getattr(module, "__version__", "<unknown>"))
        if actual != info[name]:
            raise RuntimeError(
                "Incompatible ops_repo wheel: built for %s=%s, current %s=%s. "
                "This extension requires the exact torch/torch_npu build versions; "
                "install or rebuild the matching wheel."
                % (name, info[name], name, actual)
            )


def package_version():
    info = _build_information()
    if info is None:
        lora_info = Path(__file__).resolve().parent / "_lora_build_info.json"
        if lora_info.is_file():
            return json.loads(lora_info.read_text())["version"]
        return "0.1.0"
    return info.get("version", "0.1.0+" + info["soc"].replace("_", "."))


def _extension_library():
    package = Path(__file__).resolve().parent
    candidates = sorted(package.glob("_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            "ops_repo's optional PyTorch extension is not built (or multiple copies exist). "
            "With matching torch, torch_npu, CANN and an installed OPP, run "
            "OPS_REPO_BUILD_TORCH=1 python3 -m pip install --no-build-isolation --no-deps . "
            "For an editable development install, also set OPS_REPO_BUILD_TORCH=1 when installing -e ."
        )
    candidate = candidates[0]
    if (package / "_build_info.json").exists():
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(package)
            if not candidate.is_file():
                raise ValueError("extension is not a regular file")
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Damaged ops_repo wheel: _C extension is missing or escapes the package") from exc
    return candidate


def initialize_ascendc():
    """Complete initialization once; retain completed loading steps on failure."""
    global _ready, _torch, _library_loaded
    if _ready:
        return
    with _lock:
        if _ready:
            return
        info = _build_information()
        # Configure embedded OPP before torch, whose plugin discovery can import
        # torch_npu. Explicit external OPPs still require startup shell setup.
        op_api = _op_api_library()
        extension = _extension_library()
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Importing ops_repo requires a matching torch and torch_npu environment; "
                "ops_repo does not install them automatically."
            ) from exc
        if not hasattr(torch.library, "register_fake"):
            raise RuntimeError("ops_repo requires PyTorch >= 2.4 (tested API target: 2.6)")
        try:
            torch_npu = importlib.import_module("torch_npu")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Cannot import torch_npu. Install the torch_npu version matching torch "
                "and source the matching CANN environment before starting Python."
            ) from exc
        _check_framework_versions(info, torch, torch_npu)
        try:
            if not _library_loaded:
                torch.ops.load_library(str(extension))
                _library_loaded = True
            torch.ops.ops_repo._initialize(str(op_api))
            from ._fake import register_fake

            register_fake()
        except (OSError, RuntimeError, AttributeError) as exc:
            raise RuntimeError(
                "Could not initialize ops_repo AscendC bindings. Rebuild the extension against "
                "this exact torch/torch_npu environment and verify the selected OPP. Cause: %s" % exc
            ) from exc
        _torch = torch
        _ready = True


def initialize_package():
    """Initialize components present in this wheel; retain the Add-only contract."""
    from . import _lora_runtime

    package = Path(__file__).resolve().parent
    has_lora = _lora_runtime.is_installed()
    if not has_lora or list(package.glob("_C*.so")):
        initialize_ascendc()
    if has_lora:
        _lora_runtime.initialize()


def call_add(x, y):
    if not _ready:
        raise RuntimeError("ops_repo Add component initialization must complete before calling add(). Install a wheel built with "
                           "OPS_REPO_BUILD_TORCH=1 and its matching OPP; a LoRA-only wheel does "
                           "not include OpsRepoAdd.")
    from ._validation import validate_add_inputs

    validate_add_inputs(x, y)
    return _torch.ops.ops_repo.add(x, y)
