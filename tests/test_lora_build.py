"""Offline build/package regressions; synthetic ELF fixtures never execute on NPU."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lora_build
import wheel_support


KERNEL_NAMES = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "combined_lora_idx", "add_lora_fused", "add_lora_swiglu_quant",
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("OPS_REPO_")}


class LoraBuildPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ops lora build policy ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "package" / "_lora_C.fixture.so"
        self.command = SimpleNamespace(
            build_temp=str(self.root / "build"),
            get_ext_fullpath=lambda name: str(self.output),
        )
        self.extension = lora_build.LoraExtension()
        self.env = {"OPS_REPO_SOC": "ascend910_93", "ASCEND_HOME_PATH": str(self.root / "cann")}
        self.frameworks = {"torch": SimpleNamespace(__version__="2.10.0+fixture"),
                           "torch_npu": SimpleNamespace(__version__="2.10.0.post4+fixture")}

    def test_unsupported_soc_and_cross_family_override_fail_before_cmake(self):
        cases = (
            ({"OPS_REPO_SOC": "ascend950"}, "A2/A3"),
            ({"OPS_REPO_SOC": "ascend910b", "OPS_REPO_SOC_VERSION": "ascend910_9391"}, "belong"),
            ({"OPS_REPO_SOC_VERSION": "ascend910b1"}, "belong"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides), \
                    patch.dict(os.environ, dict(self.env, **overrides), clear=True), \
                    patch.dict(sys.modules, self.frameworks), \
                    patch.object(lora_build.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, expected):
                    lora_build.build_lora(self.command, self.extension)
                run.assert_not_called()

    def test_concrete_soc_override_reaches_build_and_receipt(self):
        calls = []

        def fake_cmake(args, **kwargs):
            calls.append(args)
            if "--install" in args:
                write(self.output.parent / "_lora_C.so", b"\x7fELFfixture")
                write(self.output.parent / "libops_repo_moe_lora_kernels.so", b"\x7fELFfixture")

        with patch.dict(os.environ, dict(self.env, OPS_REPO_SOC_VERSION="Ascend910_9381"), clear=True), \
                patch.dict(sys.modules, self.frameworks), \
                patch.object(lora_build.subprocess, "run", side_effect=fake_cmake):
            lora_build.build_lora(self.command, self.extension)
        self.assertTrue(any("-DSOC_VERSION=ascend910_9381" in args for args in calls))
        info = json.loads((self.output.parent / "_lora_build_info.json").read_text())
        self.assertEqual(info["soc"], "ascend910_93")
        self.assertEqual(info["soc_version"], "ascend910_9381")
        self.assertTrue(self.output.is_file())
        self.assertFalse((self.output.parent / "_lora_C.so").exists())

    def test_cleanup_removes_lora_and_cann_install_artifacts_only(self):
        package = self.root / "ops_repo"
        stale = (
            "_lora_C.fixture.so", "_lora_build_info.json", "libops_repo_moe_lora_kernels.so",
            "include/ops_repo_moe_lora_kernels/aclrtlaunch_generated.h",
            "lib/libops_repo_moe_lora_kernels.so",
        )
        keep = ("moe_lora.py", "include/unrelated/header.h", "lib/libunrelated.so")
        for name in stale + keep:
            write(package / name, "fixture")
        wheel_support.remove_built_payload(self.root)
        for name in stale:
            self.assertFalse((package / name).exists(), name)
        self.assertFalse((package / "include/ops_repo_moe_lora_kernels").exists())
        for name in keep:
            self.assertTrue((package / name).is_file(), name)


@unittest.skipUnless(importlib.util.find_spec("setuptools") and importlib.util.find_spec("wheel"),
                     "setuptools and wheel are required for archive tests")
class LoraArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="ops lora archive ")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.source = cls.root / "source"
        wheel_support.clean_source_copy(ROOT, cls.source)
        fixtures = cls.root / "fixtures"
        write(fixtures / "torch/__init__.py", '__version__ = "2.10.0+fixture"\n')
        write(fixtures / "torch_npu/__init__.py", '__version__ = "2.10.0.post4+fixture"\n')
        # Model CANN's default extra install payload. Omitting --component leaks
        # launcher headers and a duplicate lib/ tree into the real wheel archive.
        cmake = fixtures / "bin/cmake"
        write(cmake, "#!" + sys.executable + "\n" + textwrap.dedent('''\
            import json
            from pathlib import Path
            import sys
            args = sys.argv[1:]
            if "-B" in args:
                build = Path(args[args.index("-B") + 1])
                build.mkdir(parents=True, exist_ok=True)
                output = next(arg.split("=", 1)[1] for arg in args
                              if arg.startswith("-DCMAKE_INSTALL_PREFIX="))
                (build / "fixture.json").write_text(json.dumps({"output": output}))
            elif "--install" in args:
                build = Path(args[args.index("--install") + 1])
                output = Path(json.loads((build / "fixture.json").read_text())["output"])
                output.mkdir(parents=True, exist_ok=True)
                for name in ("_lora_C.so", "libops_repo_moe_lora_kernels.so"):
                    (output / name).write_bytes(b"\\x7fELFsynthetic fixture, not executable")
                if "--component" not in args or args[args.index("--component") + 1] != "ops_repo_lora":
                    headers = output / "include/ops_repo_moe_lora_kernels"
                    headers.mkdir(parents=True, exist_ok=True)
                    (headers / "aclrtlaunch_fixture.h").write_text("generated header")
                    (output / "lib").mkdir(exist_ok=True)
                    (output / "lib/libops_repo_moe_lora_kernels.so").write_bytes(b"\\x7fELFduplicate")
        '''))
        cmake.chmod(0o755)
        cls.env = clean_environment()
        cls.env.update({"PYTHONPATH": str(fixtures), "PATH": str(cmake.parent) + os.pathsep + os.environ["PATH"],
                        "OPS_REPO_BUILD_LORA": "1", "OPS_REPO_SOC": "ascend910_93",
                        "ASCEND_HOME_PATH": str(cls.root / "synthetic cann"), "MAX_JOBS": "1"})
        cls.native = cls.build_wheel("native", cls.env)

    @classmethod
    def run_setup(cls, args, env):
        result = subprocess.run([sys.executable, "setup.py", *args], cwd=cls.source,
                                env=env, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    @classmethod
    def build_wheel(cls, name, env):
        output = cls.root / name
        cls.run_setup(["build", "--build-lib", "shared-build", "bdist_wheel",
                       "--dist-dir", str(output)], env)
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError("Expected exactly one wheel")
        return wheels[0]

    def test_lora_wheel_contains_only_the_two_owned_native_libraries(self):
        with zipfile.ZipFile(self.native) as archive:
            names = archive.namelist()
            native = [name for name in names if name.endswith(".so")]
            self.assertEqual(len(native), 2, native)
            self.assertIn("ops_repo/libops_repo_moe_lora_kernels.so", native)
            self.assertEqual(len([name for name in native if name.startswith("ops_repo/_lora_C.")]), 1)
            self.assertFalse(any(name.startswith(("ops_repo/include/", "ops_repo/lib/", "ops_repo/opp/"))
                                 for name in names))
            self.assertNotIn("ops_repo_opp_env.pth", names)
            self.assertNotIn("ops_repo/_build_info.json", names)
            info = json.loads(archive.read("ops_repo/_lora_build_info.json"))
            self.assertEqual(info["soc_version"], "ascend910_9391")
            self.assertEqual(info["torch_npu"], "2.10.0.post4+fixture")
            wheel_info = next(name for name in names if name.endswith(".dist-info/WHEEL"))
            self.assertIn("Root-Is-Purelib: false", archive.read(wheel_info).decode())
        self.assertNotIn("none-any", self.native.name)

    def test_plain_rebuild_drops_native_payload_and_cann_default_install_remnants(self):
        package = self.source / "shared-build/ops_repo"
        write(package / "include/ops_repo_moe_lora_kernels/old_launcher.h", "stale")
        write(package / "lib/libops_repo_moe_lora_kernels.so", b"\x7fELFstale")
        pure = self.build_wheel("pure", dict(self.env, OPS_REPO_BUILD_LORA="0"))
        with zipfile.ZipFile(pure) as archive:
            names = archive.namelist()
            self.assertIn("ops_repo/moe_lora.py", names)
            self.assertFalse(any(name.endswith(".so") or "_lora_build_info" in name
                                 or name.startswith(("ops_repo/include/", "ops_repo/lib/"))
                                 for name in names))
        self.assertIn("py3-none-any", pure.name)

    def test_setup_rejects_a5_and_editable_before_framework_build(self):
        cases = (
            (["--name"], dict(self.env, OPS_REPO_SOC="ascend950"), "ascend910b or ascend910_93"),
            (["editable_wheel"], self.env, "requires a regular wheel"),
            (["develop"], self.env, "requires a regular wheel"),
        )
        for args, env, expected in cases:
            with self.subTest(args=args, soc=env["OPS_REPO_SOC"]):
                result = subprocess.run([sys.executable, "setup.py", *args], cwd=self.source,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_sdist_carries_all_native_build_inputs(self):
        output = self.root / "sdist"
        self.run_setup(["sdist", "--dist-dir", str(output)], dict(self.env, OPS_REPO_BUILD_LORA="0"))
        archives = list(output.glob("*.tar.gz"))
        self.assertEqual(len(archives), 1)
        with tarfile.open(archives[0], "r:gz") as archive:
            names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
        required = {"ops/ascendc/moe_lora/" + name + ".cpp" for name in KERNEL_NAMES}
        required.update({
            "ops/ascendc/moe_lora/types.h", "ops/ascendc/moe_lora/provenance.json",
            "torch_custom/ops_repo/csrc/CMakeLists.txt", "scripts/lora_build.py",
            "torch_custom/ops_repo/csrc/moe_lora/ops.h",
            "torch_custom/ops_repo/csrc/moe_lora/torch_binding.cpp",
            "torch_custom/ops_repo/csrc/moe_lora/torch_binding_meta.cpp",
            "torch_custom/ops_repo/csrc/moe_lora/aclnn_torch_adapter/op_api_common.h",
            "torch_custom/ops_repo/csrc/moe_lora/aclnn_torch_adapter/NPUBridge.h",
            "torch_custom/ops_repo/csrc/moe_lora/aclnn_torch_adapter/NPUBridge.cpp",
            "torch_custom/ops_repo/csrc/moe_lora/aclnn_torch_adapter/NPUStorageImpl.h",
            "torch_custom/ops_repo/csrc/moe_lora/aclnn_torch_adapter/NPUStorageImpl.cpp",
        })
        self.assertTrue(required.issubset(names), sorted(required - names))
        self.assertFalse(any(name.endswith(".so") for name in names))


if __name__ == "__main__":
    unittest.main()
