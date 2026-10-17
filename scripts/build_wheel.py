#!/usr/bin/env python3
"""Build an OPP package, stage it in isolation, and create a complete native wheel."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import ops
from wheel_support import (build_info, clean_source_copy, receipt_package,
                           validate_vendor_tree, validate_wheel)


def preflight(args):
    """Fail before compiling if the wheel's Python build environment is incomplete."""
    failures = []
    for name in ("pip", "setuptools", "wheel"):
        try:
            version = importlib.metadata.version(name)
            if name == "setuptools" and int(version.split(".")[0]) < 61:
                failures.append("setuptools>=61 is required (found %s)" % version)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            failures.append("Missing build dependency: %s" % name)
    info = None
    try:
        import torch
        import torch_npu
        from torch.utils.cpp_extension import BuildExtension
        from torch_npu.utils.cpp_extension import NpuExtension
        info = build_info(args.soc, torch.__version__, torch_npu.__version__)
    except (ImportError, OSError, AttributeError, RuntimeError) as exc:
        failures.append("A working, matching torch/torch_npu build environment is required: %s" % exc)
    for check in ops.environment_checks(args.cann_path, args.msopgen):
        if not check["ok"]:
            failures.append("%s: %s" % (check["name"], check["detail"]))
    if failures:
        raise RuntimeError("Wheel build environment is incomplete:\n  - " + "\n  - ".join(failures)
                           + "\nInstall the prerequisites and source CANN before retrying; no dependencies were installed.")
    return info


def build(args):
    soc, _ = ops.resolve_target(args)
    if args.jobs < 1:
        raise RuntimeError("--jobs must be greater than zero")
    selected = ops.select_operators(ops.load_operators(), args.ops, soc)
    names = [op["name"] for op in selected]
    if "ops_repo_add" not in names:
        raise RuntimeError("The current torch adapter requires ops_repo_add; --ops must include it")
    info = preflight(args)
    ops.build(args)
    dist = ops.managed_directory(ops.ROOT, "dist", soc)
    package = receipt_package(dist, soc, names)
    # Installing into an empty directory prevents a previous OPP installation
    # from contributing stale kernels, headers or libraries to this wheel.
    with tempfile.TemporaryDirectory(prefix="ops-repo-wheel-") as temporary:
        stage = Path(temporary)
        opp = stage / "opp"
        opp.mkdir()
        ops.run(["bash", package, "--install-path=%s" % opp])
        payload = validate_vendor_tree(opp, soc)
        source = stage / "source"
        clean_source_copy(ops.ROOT, source)
        wheel_directory = stage / "wheel"
        wheel_directory.mkdir()
        env = dict(os.environ)
        env.update({"OPS_REPO_BUILD_TORCH": "1", "OPS_REPO_BUNDLE_OPP": "1",
                    "OPS_REPO_OPP_PATH": str(opp), "OPS_REPO_SOC": soc,
                    "ASCEND_HOME_PATH": str(ops.cann_root(args.cann_path)),
                    "MAX_JOBS": str(args.jobs)})
        ops.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
                 "--no-index", "--wheel-dir", wheel_directory, "."], cwd=source, env=env)
        wheels = list(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected one wheel from this build, found %s" % len(wheels))
        validate_wheel(wheels[0], info, payload)
        output = ops.managed_directory(ops.ROOT, "dist", soc, "wheel", create=True)
        destination = output / wheels[0].name
        if destination.is_symlink():
            raise RuntimeError("Wheel output must not be a symbolic link: %s" % destination)
        # Commit only the fully validated archive, avoiding a partial replacement.
        with tempfile.NamedTemporaryFile(prefix=".wheel-", dir=output, delete=False) as stream:
            staged_output = Path(stream.name)
        try:
            shutil.copy2(wheels[0], staged_output)
            staged_output.replace(destination)
        finally:
            staged_output.unlink(missing_ok=True)
    print("Complete wheel: %s" % destination)
    print("Install in a matching CANN, Python, torch/torch_npu environment:")
    print(shlex.join([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(destination)]))
    print("Archive validation passed; run examples/check_torch_add.py on the target NPU to validate execution.")
    return destination


def parser():
    result = argparse.ArgumentParser(description="构建内嵌 OPP 和 PyTorch 适配层的完整 wheel")
    result.add_argument("--soc", required=True, choices=ops.SOC_TARGETS)
    result.add_argument("--ops", help="逗号分隔的已启用算子，必须包含 ops_repo_add；默认该芯片全部启用算子")
    result.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    result.add_argument("--cann-path", help="已安装的 CANN Toolkit 根目录")
    result.add_argument("--msopgen", help="与 Toolkit 匹配的 msopgen 脚本")
    result.add_argument("--soc-version", help="属于 --soc 的生成器型号，与 scripts/ops.py build 相同")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        build(args)
        return 0
    except (ops.RepoError, RuntimeError, OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
