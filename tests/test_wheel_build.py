"""Offline packaging tests; all ELF fixtures are synthetic and cannot run on an NPU."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_wheel
import wheel_support as support


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)


def make_opp(root, soc="ascend910_93"):
    opp = root / "opp"
    vendor = opp / "vendors/ops_repo"
    files = {
        "op_api/lib/libcust_opapi.so": b"\x7fELFsynthetic opapi fixture",
        "op_api/include/aclnnop/aclnn_ops_repo_add.h": "// generated header fixture\n",
        "op_impl/ai_core/tbe/kernel/%s/ops_repo_add/add.o" % soc: b"\x7fELFsynthetic kernel fixture",
        "op_impl/ai_core/tbe/kernel/config/%s/ops_repo_add.json" % soc: "{}\n",
        "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so": b"\x7fELFsynthetic tiling fixture",
        "op_proto/lib/linux/aarch64/libcust_opsproto.so": b"\x7fELFsynthetic proto fixture",
        "bin/set_env.bash": "# An installer may write machine-specific paths here.\n",
    }
    for name, content in files.items():
        write(vendor / name, content)
    (vendor / "op_impl/ai_core/tbe/op_tiling/liboptiling.so").symlink_to(
        "lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    return opp


class WheelHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ops wheel fixture ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.opp = make_opp(self.root)
        self.vendor = self.opp / "vendors/ops_repo"

    def test_copy_preserves_whole_vendor_and_resolves_internal_links(self):
        files = support.validate_vendor_tree(self.opp, "ascend910_93")
        target = self.root / "copy"
        support.copy_vendor_tree(self.opp, target, "ascend910_93")
        self.assertEqual(set(files), {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()})
        self.assertFalse(any(p.is_symlink() for p in target.rglob("*")))
        self.assertEqual((target / "op_impl/ai_core/tbe/op_tiling/liboptiling.so").read_bytes(),
                         b"\x7fELFsynthetic tiling fixture")

    def test_incomplete_or_wrong_soc_vendor_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            support.validate_vendor_tree(self.opp, "ascend950")
        header = self.vendor / "op_api/include/aclnnop/aclnn_ops_repo_add.h"
        header.write_text("")
        with self.assertRaisesRegex(RuntimeError, "header"):
            support.validate_vendor_tree(self.opp, "ascend910_93")
        header.write_text("header")
        (self.vendor / "op_api/lib/libcust_opapi.so").write_text("placeholder")
        with self.assertRaisesRegex(RuntimeError, "ELF"):
            support.validate_vendor_tree(self.opp, "ascend910_93")

    def test_symlink_escape_and_directory_loop_are_rejected(self):
        outside = self.root / "secret"
        outside.write_text("must not be copied")
        link = self.vendor / "escape"
        link.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            support.vendor_files(self.opp)
        link.unlink()
        link.symlink_to(self.vendor, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "cycle"):
            support.vendor_files(self.opp)

    def test_receipt_must_match_package_soc_and_operator_selection(self):
        package = self.root / "ops.run"
        package.write_bytes(b"synthetic installer fixture")
        receipt = {"vendor": "ops_repo", "soc": "ascend910_93", "operators": ["ops_repo_add"],
                   "package": package.name, "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
        path = self.root / "build.json"
        path.write_text(json.dumps(receipt))
        self.assertEqual(support.receipt_package(self.root, "ascend910_93", ["ops_repo_add"]), package)
        for key, value in (("package", "../ops.run"), ("soc", "ascend950"),
                           ("operators", ["other"]), ("sha256", "changed")):
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(receipt, **{key: value})))
                with self.assertRaises(RuntimeError):
                    support.receipt_package(self.root, "ascend910_93", ["ops_repo_add"])

    def test_cli_rejects_missing_add_before_any_build(self):
        args = argparse.Namespace(soc="ascend910_93", soc_version=None, jobs=1,
                                  ops="other", cann_path=None, msopgen=None)
        other = {"name": "other", "enabled": True, "soc": ["ascend910_93"]}
        with patch.object(build_wheel.ops, "load_operators", return_value=[other]), \
                patch.object(build_wheel, "preflight") as preflight, \
                patch.object(build_wheel.ops, "build") as build:
            with self.assertRaisesRegex(RuntimeError, "requires ops_repo_add"):
                build_wheel.build(args)
            preflight.assert_not_called()
            build.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("setuptools") and importlib.util.find_spec("wheel"),
                     "setuptools and wheel are needed for archive integration tests")
class WheelArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ops wheel archive fixture ")
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        support.clean_source_copy(ROOT, cls.source)
        cls.opp = make_opp(cls.root)
        cls.info = support.build_info("ascend910_93", "2.7.1+fixture", "2.7.1.post1+fixture")
        toolkit = cls.root / "toolkit"
        for name in ("include/acl/acl.h", "include/aclnn/acl_meta.h", "lib64/libascendcl.so", "lib64/libnnopbase.so"):
            write(toolkit / name, "fixture")
        # Replace framework builders, never compile/load production Ascend code.
        # setuptools still builds the real wheel archive, tags and RECORD.
        frameworks = cls.root / "framework fixtures"
        write(frameworks / "torch/__init__.py", '__version__ = "2.7.1+fixture"\n')
        write(frameworks / "torch/utils/__init__.py", "")
        write(frameworks / "torch/utils/cpp_extension.py", textwrap.dedent('''\
            from pathlib import Path
            from setuptools.command.build_ext import build_ext
            class BuildExtension(build_ext):
                @classmethod
                def with_options(cls, **kwargs):
                    return cls
                def build_extension(self, ext):
                    output = Path(self.get_ext_fullpath(ext.name))
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"\\x7fELFsynthetic adapter fixture, not executable")
        '''))
        write(frameworks / "torch_npu/__init__.py", '__version__ = "2.7.1.post1+fixture"\n')
        write(frameworks / "torch_npu/utils/__init__.py", "")
        write(frameworks / "torch_npu/utils/cpp_extension.py", textwrap.dedent('''\
            from setuptools import Extension
            def NpuExtension(name, **kwargs):
                return Extension(name, sources=[])
        '''))
        cls.env = dict(os.environ, PYTHONPATH=str(frameworks), OPS_REPO_BUILD_TORCH="1",
                       OPS_REPO_BUNDLE_OPP="1", OPS_REPO_SOC="ascend910_93",
                       OPS_REPO_OPP_PATH=str(cls.opp), ASCEND_HOME_PATH=str(toolkit))
        cls.bundle = cls.run_build("bundled", cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def run_build(cls, directory, env):
        output = cls.root / directory
        result = subprocess.run([sys.executable, "setup.py", "build", "--build-lib", "shared-build",
                                 "bdist_wheel", "--dist-dir", str(output)],
                                cwd=cls.source, env=env, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError("Expected one fixture wheel")
        return wheels[0]

    def test_bundled_archive_contains_native_tag_complete_payload_and_record(self):
        support.validate_wheel(self.bundle, self.info, support.validate_vendor_tree(self.opp, "ascend910_93"))
        self.assertIn("0.1.0+ascend910.93", self.bundle.name)
        self.assertNotIn("none-any", self.bundle.name)
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(json.loads(archive.read("ops_repo/_build_info.json")), self.info)

    def test_record_corruption_is_rejected(self):
        corrupted = self.root / "corrupt.whl"
        with zipfile.ZipFile(self.bundle) as source, zipfile.ZipFile(corrupted, "w") as target:
            for name in source.namelist():
                data = source.read(name)
                if name == "ops_repo_opp_env.py":
                    data += b"\n# modified after RECORD generation\n"
                target.writestr(name, data)
        with self.assertRaisesRegex(RuntimeError, "RECORD mismatch"):
            support.validate_wheel(corrupted, self.info, support.validate_vendor_tree(self.opp, "ascend910_93"))

    def test_pth_runs_on_fresh_interpreter_without_importing_frameworks(self):
        installed = self.root / "installed site"
        env = dict(os.environ)
        for name in ("OPS_REPO_OPP_PATH", "OPS_REPO_SOC", "ASCEND_CUSTOM_OPP_PATH"):
            env.pop(name, None)
        install = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                                  "--target", str(installed), str(self.bundle)],
                                 env=env, capture_output=True, text=True)
        self.assertEqual(install.returncode, 0, install.stdout + install.stderr)
        code = """\
import json, os, site, sys
site.addsitedir(sys.argv[1])
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert 'ops_repo' not in sys.modules
print(json.dumps({'opp': os.environ.get('ASCEND_CUSTOM_OPP_PATH')}))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code, str(installed)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["opp"], str(installed / "ops_repo/opp/vendors/ops_repo"))

    def test_plain_rebuild_drops_all_bundled_payload_from_reused_build_directory(self):
        env = dict(self.env, OPS_REPO_BUILD_TORCH="0", OPS_REPO_BUNDLE_OPP="0")
        plain = self.run_build("plain", env)
        self.assertIn("0.1.0-py3-none-any", plain.name)
        with zipfile.ZipFile(plain) as archive:
            names = archive.namelist()
            self.assertTrue(any(name == "ops_repo/__init__.py" for name in names))
            self.assertFalse(any(name.startswith("ops_repo/opp/") or name.startswith("ops_repo/_C")
                                 or name.startswith("ops_repo_opp_env") or name == "ops_repo/_build_info.json"
                                 for name in names))

    def test_bundle_requires_explicit_native_opt_in_and_rejects_editable(self):
        for command, env, expected in (
                ("--name", dict(self.env, OPS_REPO_BUILD_TORCH="0"), "requires OPS_REPO_BUILD_TORCH=1"),
                ("editable_wheel", self.env, "does not support editable")):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "setup.py", command], cwd=self.source,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_complete_cli_pipeline_uses_isolated_install_and_publishes_validated_wheel(self):
        installed_prefix = self.root / "installer-prefix.txt"

        def fake_cann_build(args):
            dist = self.source / "dist" / args.soc
            dist.mkdir(parents=True, exist_ok=True)
            installer = dist / "synthetic-opp.run"
            # Exercise the real bash installer invocation with a synthetic OPP.
            installer.write_text(
                '#!/bin/bash\nset -eu\ninstall_path="${1#--install-path=}"\n'
                + 'cp -a ' + shlex.quote(str(self.opp / "vendors")) + ' "$install_path/vendors"\n'
                + 'printf "%s" "$install_path" > ' + shlex.quote(str(installed_prefix)) + '\n')
            receipt = {"vendor": "ops_repo", "soc": args.soc, "operators": ["ops_repo_add"],
                       "package": installer.name, "sha256": hashlib.sha256(installer.read_bytes()).hexdigest()}
            (dist / "build.json").write_text(json.dumps(receipt))

        args = argparse.Namespace(soc="ascend910_93", soc_version=None, jobs=1, ops="ops_repo_add",
                                  cann_path=self.env["ASCEND_HOME_PATH"], msopgen=None)
        output = io.StringIO()
        with patch.object(build_wheel, "preflight", return_value=self.info), \
                patch.object(build_wheel.ops, "build", side_effect=fake_cann_build), \
                patch.object(build_wheel.ops, "ROOT", self.source), \
                patch.dict(os.environ, self.env), redirect_stdout(output):
            wheel = build_wheel.build(args)
        self.assertEqual(wheel.parent, self.source / "dist/ascend910_93/wheel")
        self.assertTrue(wheel.is_file())
        support.validate_wheel(wheel, self.info, support.validate_vendor_tree(self.opp, "ascend910_93"))
        staging_opp = Path(installed_prefix.read_text())
        self.assertTrue(staging_opp.parent.name.startswith("ops-repo-wheel-"))
        self.assertFalse(staging_opp.exists(), "temporary OPP staging should be removed")
        self.assertIn("--force-reinstall", output.getvalue())


if __name__ == "__main__":
    unittest.main()
