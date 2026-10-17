"""Setuptools build support for the standalone AscendC MoE/LoRA component."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension
from setuptools.command.build_ext import build_ext

from wheel_support import build_info


class LoraExtension(Extension):
    def __init__(self):
        super().__init__("ops_repo._lora_C", sources=[])


def build_lora(command, extension):
    import torch
    import torch_npu

    soc = os.environ.get("OPS_REPO_SOC")
    targets = {"ascend910b": "ascend910b1", "ascend910_93": "ascend910_9391"}
    if soc not in targets:
        raise RuntimeError("MoE LoRA requires OPS_REPO_SOC=ascend910b or ascend910_93 (A2/A3)")
    soc_version = os.environ.get("OPS_REPO_SOC_VERSION", targets[soc]).lower()
    if not soc_version.startswith(soc):
        raise RuntimeError("OPS_REPO_SOC_VERSION must belong to OPS_REPO_SOC")
    toolkit = os.environ.get("ASCEND_HOME_PATH")
    if not toolkit:
        raise RuntimeError("Source the CANN set_env.sh before building MoE LoRA")
    root = Path(__file__).resolve().parents[1]
    output = Path(command.get_ext_fullpath(extension.name)).resolve().parent
    build = Path(command.build_temp).resolve() / ("moe_lora_" + soc_version)
    jobs = int(os.environ.get("MAX_JOBS", "4"))
    if jobs < 1:
        raise RuntimeError("MAX_JOBS must be positive")
    # AscendC ExternalProject stubs can survive signature/toolchain changes.
    # This directory is owned exclusively by this build component.
    if build.is_symlink():
        raise RuntimeError("MoE LoRA build directory must not be a symbolic link")
    if build.exists():
        shutil.rmtree(build)
    subprocess.run([
        "cmake", "-S", str(root / "torch_custom/ops_repo/csrc"), "-B", str(build),
        "-DSOC_VERSION=" + soc_version, "-DASCEND_CANN_PACKAGE_PATH=" + toolkit,
        "-DPython3_EXECUTABLE=" + sys.executable,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(output),
    ], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    subprocess.run(["cmake", "--install", str(build), "--component", "ops_repo_lora"], check=True)
    # Use setuptools' extension suffix so wheel/editable installers track it.
    native = output / "_lora_C.so"
    destination = Path(command.get_ext_fullpath(extension.name)).resolve()
    if native != destination:
        native.replace(destination)
    info = build_info(soc, torch.__version__, torch_npu.__version__)
    info.update({"component": "moe_lora", "soc_version": soc_version,
                 "source_commit": "72b7912616e0e296e64f0ca0003afcc2ef7ce6bc",
                 "version": "0.1.0+" + soc.replace("_", ".")})
    (output / "_lora_build_info.json").write_text(json.dumps(info, indent=2) + "\n")


def lora_build_command(base=build_ext):
    class BuildNative(base):
        def build_extension(self, extension):
            if isinstance(extension, LoraExtension):
                build_lora(self, extension)
            else:
                super().build_extension(extension)

        def get_outputs(self):
            outputs = super().get_outputs()
            for extension in self.extensions:
                if isinstance(extension, LoraExtension):
                    package = Path(self.get_ext_fullpath(extension.name)).parent
                    outputs += [str(package / name) for name in (
                        "libops_repo_moe_lora_kernels.so", "_lora_build_info.json")]
            return outputs

    return BuildNative
