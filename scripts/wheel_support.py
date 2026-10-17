"""Offline validation and staging helpers for the bundled binary wheel."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import zipfile


SOCS = ("ascend910b", "ascend910_93", "ascend950")
BUILD_INFO_FIELDS = {"schema_version", "soc", "python", "machine", "torch", "torch_npu"}


def build_info(soc, torch_version, npu_version):
    if soc not in SOCS:
        raise RuntimeError("OPS_REPO_SOC must explicitly select " + ", ".join(SOCS))
    if not torch_version or not npu_version:
        raise RuntimeError("torch and torch_npu must expose their exact __version__")
    return {"schema_version": 1, "soc": soc,
            "python": "%s.%s" % sys.version_info[:2], "machine": platform.machine(),
            "torch": str(torch_version), "torch_npu": str(npu_version)}


def vendor_files(opp):
    """Enumerate all files, dereferencing only links confined to this vendor tree."""
    vendor = Path(opp).expanduser().resolve() / "vendors" / "ops_repo"
    if not vendor.is_dir():
        raise RuntimeError("Missing OPP vendor directory: %s" % vendor)
    boundary = vendor.resolve()
    files = {}

    def visit(directory, relative, ancestors):
        try:
            real = directory.resolve(strict=True)
            real.relative_to(boundary)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("OPP link escapes its vendor tree or is broken: %s" % directory) from exc
        if real in ancestors:
            raise RuntimeError("OPP directory link cycle: %s" % directory)
        for path in sorted(directory.iterdir()):
            name = relative / path.name
            try:
                target = path.resolve(strict=True)
                target.relative_to(boundary)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("OPP link escapes its vendor tree or is broken: %s" % path) from exc
            if path.is_dir():
                visit(path, name, ancestors | {real})
            elif path.is_file():
                files[name.as_posix()] = target
            else:
                raise RuntimeError("OPP contains a non-regular file: %s" % path)

    visit(vendor, Path(), set())
    return files


def validate_vendor_tree(opp, soc):
    if soc not in SOCS:
        raise RuntimeError("Unsupported OPP target: %s" % soc)
    files = vendor_files(opp)

    def nonempty(name):
        return name in files and files[name].stat().st_size > 0

    def elf(name):
        if not nonempty(name):
            return False
        with files[name].open("rb") as stream:
            return stream.read(4) == b"\x7fELF"

    api = [name for name in ("op_api/lib/libcust_opapi.so", "op_api/lib64/libcust_opapi.so")
           if name in files]
    if len(api) != 1 or not elf(api[0]):
        raise RuntimeError("OPP needs exactly one nonempty ELF libcust_opapi.so in op_api/lib or lib64")
    if not any(nonempty(name) for name in ("op_api/include/aclnn_ops_repo_add.h",
                                          "op_api/include/aclnnop/aclnn_ops_repo_add.h")):
        raise RuntimeError("OPP is missing the generated aclnn_ops_repo_add.h header")
    kernels = [name for name in files if name.startswith("op_impl/ai_core/tbe/kernel/%s/" % soc)
               and (name.endswith(".o") or name.endswith(".bin"))]
    if not any(elf(name) if name.endswith(".o") else nonempty(name) for name in kernels):
        raise RuntimeError("OPP has no compiled kernel binary for %s" % soc)
    tiling = [name for name in files if name.startswith("op_impl/ai_core/tbe/op_tiling/")
              and ".so" in PurePosixPath(name).name]
    if not any(elf(name) for name in tiling):
        raise RuntimeError("OPP has no compiled op_tiling ELF library")
    return files


def copy_vendor_tree(opp, destination, soc):
    files = validate_vendor_tree(opp, soc)
    destination = Path(destination)
    for name, source in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return files


def remove_built_payload(build_lib):
    """A reused setuptools build directory must not turn a pure wheel into a native one."""
    build_lib = Path(build_lib)
    package = build_lib / "ops_repo"
    paths = [package / "opp", package / "_build_info.json",
             build_lib / "ops_repo_opp_env.py", build_lib / "ops_repo_opp_env.pth"]
    paths += list(package.glob("_C*.so")) + list(package.glob("_C*.pyd"))
    paths += list(package.glob("_lora_C*.so"))
    paths += [package / "libops_repo_moe_lora_kernels.so", package / "_lora_build_info.json"]
    paths += [package / "include" / "ops_repo_moe_lora_kernels",
              package / "lib" / "libops_repo_moe_lora_kernels.so"]
    paths += list((build_lib / "__pycache__").glob("ops_repo_opp_env.*"))
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def clean_source_copy(source, destination):
    """Copy only wheel inputs, excluding generated binaries and previous build state."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def ignored(directory, names):
        result = []
        for name in names:
            path = Path(directory) / name
            if (name in ("__pycache__", "build", "dist", ".git", ".local", "opp", "_build_info.json", "_lora_build_info.json")
                    or name.endswith((".egg-info", ".pyc", ".pyo", ".so", ".o", ".pyd"))):
                result.append(name)
            elif path.is_symlink():
                raise RuntimeError("Source input must not be a symbolic link: %s" % path)
        return result

    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "build.sh",
                 "scripts", "ops", "torch_custom"):
        path = source / name
        if path.is_symlink():
            raise RuntimeError("Source input must not be a symbolic link: %s" % path)
        if path.is_dir():
            shutil.copytree(path, destination / name, ignore=ignored)
        elif path.is_file():
            shutil.copy2(path, destination / name)
        else:
            raise RuntimeError("Missing source input: %s" % path)


def receipt_package(dist, soc, operators):
    dist = Path(dist)
    receipt_path = dist / "build.json"
    if receipt_path.is_symlink():
        raise RuntimeError("Build receipt must not be a symbolic link")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict) or receipt.get("vendor") != "ops_repo"
            or receipt.get("soc") != soc or receipt.get("operators") != list(operators)):
        raise RuntimeError("Build receipt vendor, SoC or operators do not match this build")
    name = receipt.get("package")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".run"):
        raise RuntimeError("Invalid package path in build receipt")
    package = dist / name
    if (package.is_symlink() or not package.is_file()
            or hashlib.sha256(package.read_bytes()).hexdigest() != receipt.get("sha256")):
        raise RuntimeError("OPP package is missing or differs from the build receipt SHA-256")
    return package


def validate_wheel(wheel, expected_info, expected_files):
    """Check the archive and every RECORD hash before publishing the build output."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
            raise RuntimeError("Wheel contains duplicate or unsafe archive paths")
        actual = json.loads(archive.read("ops_repo/_build_info.json"))
        if actual != expected_info:
            raise RuntimeError("Wheel build metadata does not match this build")
        required = {"ops_repo_opp_env.py", "ops_repo_opp_env.pth"}
        if not required.issubset(names):
            raise RuntimeError("Wheel is missing its interpreter startup hook")
        if archive.read("ops_repo_opp_env.pth").strip() != b"import ops_repo_opp_env; ops_repo_opp_env.configure()":
            raise RuntimeError("Wheel has an unexpected interpreter startup hook")
        native = [name for name in names if name.startswith("ops_repo/_C")
                  and "/" not in name[len("ops_repo/"):] and name.endswith(".so")]
        if len(native) != 1 or not archive.read(native[0]).startswith(b"\x7fELF"):
            raise RuntimeError("Wheel must contain exactly one compiled ops_repo._C ELF extension")
        prefix = "ops_repo/opp/vendors/ops_repo/"
        bundled = {name[len(prefix):] for name in names if name.startswith(prefix)}
        if bundled != set(expected_files):
            raise RuntimeError("Wheel OPP payload differs from the validated staging tree")
        for name, source in expected_files.items():
            if hashlib.sha256(archive.read(prefix + name)).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise RuntimeError("Wheel OPP file changed during packaging: %s" % name)
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError("Wheel must contain exactly one RECORD")
        record_name = records[0]
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        if len(rows) != len(names) or {row[0] for row in rows} != set(names):
            raise RuntimeError("Wheel RECORD does not cover every archive member exactly once")
        for name, digest, size in rows:
            if name == record_name:
                if digest or size:
                    raise RuntimeError("RECORD must not hash itself")
                continue
            data = archive.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if digest != expected or size != str(len(data)):
                raise RuntimeError("Wheel RECORD mismatch: %s" % name)
        dist_info = record_name.removesuffix("RECORD")
        metadata = archive.read(dist_info + "WHEEL").decode()
        python_tag = "cp" + expected_info["python"].replace(".", "")
        if "Root-Is-Purelib: false" not in metadata or not any(
                line.startswith("Tag: %s-%s-" % (python_tag, python_tag)) and not line.endswith("-any")
                for line in metadata.splitlines()):
            raise RuntimeError("Bundled wheel must carry a native platform/ABI tag")
        expected_version = "0.1.0+" + expected_info["soc"].replace("_", ".")
        if "Version: " + expected_version not in archive.read(dist_info + "METADATA").decode().splitlines():
            raise RuntimeError("Wheel distribution version does not identify the target SoC")
