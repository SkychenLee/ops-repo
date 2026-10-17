"""Configure a complete wheel's OPP before torch_npu initializes CANN.

The accompanying .pth calls configure() during Python's site initialization.
This module uses only the standard library and never imports ops_repo or torch.
"""

import json
import os
from pathlib import Path
import sys


def _inside(path, package):
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(package)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Damaged ops_repo wheel: path is missing or escapes the package: %s" % path) from exc
    return resolved


def _other_vendor(path):
    """Recognize vendor paths and OPP roots accepted by CANN."""
    if path.name == "ops_repo" and path.parent.name == "vendors":
        return path
    if path.name in ("lib", "lib64") and path.parent.name == "op_api":
        vendor = path.parent.parent
        if vendor.name == "ops_repo" and vendor.parent.name == "vendors":
            return vendor
    vendor = path / "vendors" / "ops_repo"
    if vendor.is_dir():
        return vendor.resolve()
    return None


def configure(package_dir=None):
    """Return the embedded op_api path, or None for an external/source install.

    package_dir is supplied by the package runtime when site processing was
    skipped. External OPP overrides must be configured by the caller's shell.
    """
    if os.environ.get("OPS_REPO_OPP_PATH"):
        return None
    package = (Path(package_dir) if package_dir is not None else
               Path(__file__).resolve().parent / "ops_repo").resolve()
    metadata = package / "_build_info.json"
    opp = package / "opp"
    if not (metadata.exists() or metadata.is_symlink() or opp.exists() or opp.is_symlink()):
        return None

    # A partial payload must never silently switch to a system-wide OPP.
    metadata = _inside(metadata, package)
    try:
        info = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Damaged ops_repo wheel: invalid _build_info.json") from exc
    fields = ("soc", "python", "machine", "torch", "torch_npu")
    if (not isinstance(info, dict) or type(info.get("schema_version")) is not int
            or info["schema_version"] != 1
            or any(not isinstance(info.get(key), str) or not info[key] for key in fields)):
        raise RuntimeError("Damaged ops_repo wheel: invalid build metadata schema")
    extensions = list(package.glob("_C*.so"))
    if len(extensions) != 1 or not _inside(extensions[0], package).is_file():
        raise RuntimeError("Damaged ops_repo wheel: expected exactly one packaged _C extension")
    _inside(opp, package)
    vendor = _inside(opp / "vendors" / "ops_repo", package)
    if not vendor.is_dir():
        raise RuntimeError("Damaged ops_repo wheel: packaged OPP vendor is not a directory")
    # Check links in the payload, including kernel/tiling links that CANN opens.
    for path in vendor.rglob("*"):
        if path.is_symlink():
            _inside(path, package)
    candidates = []
    for folder in ("lib", "lib64"):
        path = vendor / "op_api" / folder / "libcust_opapi.so"
        if path.exists() or path.is_symlink():
            resolved = _inside(path, package)
            if not resolved.is_file():
                raise RuntimeError("Damaged ops_repo wheel: custom op_api is not a file: %s" % path)
            candidates.append(resolved)
    if len(candidates) != 1:
        raise RuntimeError("Damaged ops_repo wheel: expected exactly one packaged libcust_opapi.so; found %d"
                           % len(candidates))

    parts = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    configured = {Path(part).expanduser().resolve() for part in parts}
    for path in configured:
        other = _other_vendor(path)
        if other is not None and other != vendor:
            raise RuntimeError(
                "Conflicting ops_repo OPP in ASCEND_CUSTOM_OPP_PATH: %s; this wheel uses %s. "
                "Remove the old ops_repo paths and start a new Python process. Other vendors can remain."
                % (other, vendor)
            )
    if vendor not in configured:
        if sys.modules.get("torch_npu") is not None:
            raise RuntimeError(
                "Cannot configure the embedded ops_repo OPP after torch_npu has loaded. "
                "Restart Python with site/.pth processing enabled (ops_repo_opp_env.pth), "
                "or call ops_repo_opp_env.configure() before importing torch or torch_npu."
            )
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([str(vendor), *parts])
    return candidates[0]
