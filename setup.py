"""Pure Python, local native adapter, or an explicitly bundled OPP wheel."""

import json
import os
from pathlib import Path
import platform
import shutil
import sys

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from wheel_support import build_info, copy_vendor_tree, remove_built_payload, validate_vendor_tree
from lora_build import LoraExtension, lora_build_command

extensions = []
commands = {}
metadata = None
vendor_payload = {}


def _find_parent(paths, filename, label):
    for directory in paths:
        if (directory / filename).is_file():
            return directory
    raise RuntimeError("Missing %s (%s); searched %s" % (label, filename, ", ".join(map(str, paths))))


build_torch = os.environ.get("OPS_REPO_BUILD_TORCH", "0")
if build_torch not in ("0", "1"):
    raise RuntimeError("OPS_REPO_BUILD_TORCH must be 0 or 1")
build_lora = os.environ.get("OPS_REPO_BUILD_LORA", "0")
if build_lora not in ("0", "1"):
    raise RuntimeError("OPS_REPO_BUILD_LORA must be 0 or 1")
if build_lora == "1":
    if os.environ.get("OPS_REPO_SOC") not in ("ascend910b", "ascend910_93"):
        raise RuntimeError("MoE LoRA requires OPS_REPO_SOC=ascend910b or ascend910_93")
    if any(command in sys.argv for command in ("develop", "editable_wheel")):
        raise RuntimeError("MoE LoRA native payload requires a regular wheel installation")
bundle_opp = os.environ.get("OPS_REPO_BUNDLE_OPP", "0")
if bundle_opp not in ("0", "1"):
    raise RuntimeError("OPS_REPO_BUNDLE_OPP must be 0 or 1")
if bundle_opp == "1":
    if build_torch != "1":
        raise RuntimeError("OPS_REPO_BUNDLE_OPP=1 requires OPS_REPO_BUILD_TORCH=1")
    if any(command in sys.argv for command in ("develop", "editable_wheel")):
        raise RuntimeError("Bundled OPP does not support editable installation; use a regular wheel")
    if not os.environ.get("OPS_REPO_OPP_PATH"):
        raise RuntimeError("OPS_REPO_BUNDLE_OPP=1 requires OPS_REPO_OPP_PATH")
    soc = os.environ.get("OPS_REPO_SOC")
    vendor_payload = validate_vendor_tree(os.environ["OPS_REPO_OPP_PATH"], soc)

if build_torch == "1":
    # Default packaging does not import or install any compute framework.
    try:
        import torch
        import torch_npu
        from torch.utils.cpp_extension import BuildExtension
        from torch_npu.utils.cpp_extension import NpuExtension
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "OPS_REPO_BUILD_TORCH=1 requires a preinstalled matching torch/torch_npu pair. "
            "Source CANN first and use --no-build-isolation --no-deps."
        ) from exc

    toolkit_value = os.environ.get("ASCEND_HOME_PATH")
    opp_value = os.environ.get("OPS_REPO_OPP_PATH")
    if not toolkit_value or not opp_value:
        raise RuntimeError("Native build requires ASCEND_HOME_PATH and OPS_REPO_OPP_PATH")
    toolkit = Path(toolkit_value).expanduser().resolve()
    vendor = Path(opp_value).expanduser().resolve() / "vendors" / "ops_repo"
    host_arch = platform.machine() + "-linux"
    includes = [toolkit / "include", toolkit / "include" / "nnopbase",
                toolkit / host_arch / "include", toolkit / host_arch / "include" / "nnopbase",
                toolkit / "runtime" / "include"]
    libraries = [toolkit / "lib64", toolkit / host_arch / "lib64", toolkit / "runtime" / "lib64"]
    acl_include = _find_parent(includes, "acl/acl.h", "CANN ACL headers")
    aclnn_include = _find_parent(includes, "aclnn/acl_meta.h", "CANN ACLNN headers")
    custom_include = _find_parent(
        [vendor / "op_api" / "include", vendor / "op_api" / "include" / "aclnnop"],
        "aclnn_ops_repo_add.h", "installed OpsRepoAdd ACLNN header")
    acl_lib = _find_parent(libraries, "libascendcl.so", "CANN runtime library")
    base_name = "nnopbase" if any((p / "libnnopbase.so").is_file() for p in libraries) else "opapi"
    base_lib = _find_parent(libraries, "lib%s.so" % base_name, "CANN ACLNN base library")
    _find_parent([vendor / "op_api" / "lib", vendor / "op_api" / "lib64"],
                 "libcust_opapi.so", "installed OpsRepoAdd library")
    extensions = [NpuExtension(
        name="ops_repo._C",
        sources=["torch_custom/ops_repo/csrc/torch_bindings.cpp"],
        include_dirs=list(dict.fromkeys(map(str, [acl_include, aclnn_include, custom_include]))),
        library_dirs=list(dict.fromkeys(map(str, [acl_lib, base_lib]))),
        libraries=["ascendcl", base_name, "dl"],
        extra_compile_args=["-O2", "-std=c++17"],
    )]
    commands["build_ext"] = BuildExtension.with_options(use_ninja=False)
    if bundle_opp == "1":
        metadata = build_info(soc, torch.__version__, torch_npu.__version__)

if build_lora == "1":
    extensions.append(LoraExtension())
    commands["build_ext"] = lora_build_command(commands.get("build_ext", build_ext))


class BuildPython(build_py):
    def run(self):
        remove_built_payload(self.build_lib)
        super().run()
        if metadata is not None:
            package = Path(self.build_lib) / "ops_repo"
            copy_vendor_tree(opp_value, package / "opp" / "vendors" / "ops_repo", soc)
            (package / "_build_info.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            for name in ("ops_repo_opp_env.py", "ops_repo_opp_env.pth"):
                shutil.copy2(ROOT / "torch_custom" / "ops_repo" / name, Path(self.build_lib) / name)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        if metadata is not None:
            base = Path(self.build_lib)
            outputs += [str(base / name) for name in (
                "ops_repo/_build_info.json", "ops_repo_opp_env.py", "ops_repo_opp_env.pth")]
            outputs += [str(base / "ops_repo/opp/vendors/ops_repo" / name) for name in vendor_payload]
        return outputs


commands["build_py"] = BuildPython

setup(
    name="ops-repo",
    version="0.1.0" + ("+" + os.environ["OPS_REPO_SOC"].replace("_", ".")
                        if metadata is not None or build_lora == "1" else ""),
    description="Independent Ascend operator library with optional PyTorch bindings",
    python_requires=">=3.9",
    package_dir={
        "": "torch_custom/ops_repo",
        "ops_repo.ops.triton.triton_core": "ops/triton/triton_core",
    },
    packages=(
        find_packages("torch_custom/ops_repo", include=["ops_repo", "ops_repo.*"])
        + ["ops_repo.ops.triton.triton_core"]
        + ["ops_repo.ops.triton.triton_core." + name
           for name in find_packages("ops/triton/triton_core")]
    ),
    package_data={"ops_repo.ops.triton": ["README.md"]},
    include_package_data=False,
    install_requires=[],
    ext_modules=extensions,
    cmdclass=commands,
    zip_safe=False,
)
