"""Exercise wheel startup in real interpreters, with fake framework/native I/O.

The .pth, startup helper, package initialization and metadata checks are real.
These tests do not load native binaries or claim NPU execution coverage.
"""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "torch_custom" / "ops_repo"


class WheelRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "site-packages"
        self.site.mkdir()
        self.package = self.site / "ops_repo"
        shutil.copytree(SOURCE / "ops_repo", self.package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename in ("ops_repo_opp_env.py", "ops_repo_opp_env.pth"):
            shutil.copy2(SOURCE / filename, self.site / filename)
        self.vendor = self.package / "opp" / "vendors" / "ops_repo"
        self.api = self.vendor / "op_api" / "lib" / "libcust_opapi.so"
        self.api.parent.mkdir(parents=True)
        self.api.write_bytes(b"path fixture; never loaded")
        (self.package / "_C.fixture.so").write_bytes(b"path fixture; never loaded")
        self.info = {
            "schema_version": 1, "soc": "ascend910_93",
            "python": "%d.%d" % sys.version_info[:2], "machine": platform.machine(),
            "torch": "2.6.0+fixture", "torch_npu": "2.6.0.post1+fixture",
        }
        self._write_info()
        (self.site / "torch_npu.py").write_text('''
import os
from pathlib import Path
expected = os.environ.get("EXPECTED_VENDOR")
if expected:
    assert Path(expected) in {Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p}
__version__ = os.environ.get("FIXTURE_TORCH_NPU", "2.6.0.post1+fixture")
''', encoding="utf-8")
        (self.site / "torch.py").write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
# Model torch's backend plugin discovery: OPP must be ready even here.
import torch_npu
__version__ = os.environ.get("FIXTURE_TORCH", "2.6.0+fixture")
library = SimpleNamespace(register_fake=lambda name: lambda function: function)
def load_library(path):
    os.environ["NATIVE_LOADED"] = str(path)
def initialize(path):
    os.environ["OPAPI_INITIALIZED"] = str(path)
ops = SimpleNamespace(load_library=load_library, ops_repo=SimpleNamespace(_initialize=initialize))
''', encoding="utf-8")

    def _write_info(self):
        (self.package / "_build_info.json").write_text(json.dumps(self.info), encoding="utf-8")

    def _run(self, code, environment=None):
        env = {key: value for key, value in os.environ.items()
               if key not in ("OPS_REPO_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH",
                              "NATIVE_LOADED", "OPAPI_INITIALIZED", "FIXTURE_TORCH", "FIXTURE_TORCH_NPU")}
        env.update({"EXPECTED_VENDOR": str(self.vendor), "LD_LIBRARY_PATH": "/unchanged/cann/libraries"})
        if environment:
            env.update(environment)
        preamble = "import os, sys, site\nfrom pathlib import Path\nwheel_site = %r\nvendor = %r\n" % (
            str(self.site), str(self.vendor))
        result = subprocess.run([sys.executable, "-S", "-c", preamble + code], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_pth_configures_before_framework_import_and_loads_embedded_api(self):
        self._run('''
site.addsitedir(wheel_site)
assert "ops_repo" not in sys.modules and "torch" not in sys.modules and "torch_npu" not in sys.modules
assert os.environ["LD_LIBRARY_PATH"] == "/unchanged/cann/libraries"
import torch_npu
import ops_repo
assert ops_repo.__version__ == "0.1.0+ascend910.93"
assert os.environ["OPAPI_INITIALIZED"] == str(Path(vendor) / "op_api/lib/libcust_opapi.so")
assert os.environ["NATIVE_LOADED"] == str(Path(wheel_site) / "ops_repo/_C.fixture.so")
''')

    def test_runtime_configures_before_torch_plugin_when_pth_not_processed(self):
        self._run('''
sys.path.insert(0, wheel_site)
import ops_repo
assert "torch_npu" in sys.modules
assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == vendor
assert os.environ["LD_LIBRARY_PATH"] == "/unchanged/cann/libraries"
assert "NATIVE_LOADED" in os.environ
''')

    def test_hook_is_idempotent_and_preserves_other_vendors(self):
        self._run('''
site.addsitedir(wheel_site)
before = dict(os.environ)
import ops_repo_opp_env
ops_repo_opp_env.configure()
ops_repo_opp_env.configure()
assert dict(os.environ) == before
assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [vendor, "/other/opp/vendors/other_vendor"]
assert "torch_npu" not in sys.modules and "ops_repo" not in sys.modules
''', {"ASCEND_CUSTOM_OPP_PATH": "/other/opp/vendors/other_vendor"})

    def test_explicit_external_opp_wins_and_hook_does_not_modify_environment(self):
        external = self.root / "external"
        vendor = external / "vendors" / "ops_repo"
        api = vendor / "op_api" / "lib" / "libcust_opapi.so"
        api.parent.mkdir(parents=True)
        api.write_bytes(b"external fixture")
        self._run('''
before = dict(os.environ)
site.addsitedir(wheel_site)
assert dict(os.environ) == before
import ops_repo
assert os.environ["OPAPI_INITIALIZED"] == str(Path(os.environ["EXPECTED_VENDOR"]) / "op_api/lib/libcust_opapi.so")
''', {"OPS_REPO_OPP_PATH": str(external), "ASCEND_CUSTOM_OPP_PATH": str(vendor),
      "EXPECTED_VENDOR": str(vendor)})

    def test_external_override_still_requires_shell_configuration(self):
        self._run('''
site.addsitedir(wheel_site)
assert "ASCEND_CUSTOM_OPP_PATH" not in os.environ
try:
    import ops_repo
except RuntimeError as exc:
    assert "before starting Python" in str(exc), str(exc)
else:
    raise AssertionError("unconfigured external OPP was accepted")
assert "torch" not in sys.modules and "NATIVE_LOADED" not in os.environ
''', {"OPS_REPO_OPP_PATH": str(self.root / "external")})

    def test_external_override_does_not_allow_escaping_packaged_extension(self):
        external = self.root / "external"
        vendor = external / "vendors" / "ops_repo"
        api = vendor / "op_api" / "lib" / "libcust_opapi.so"
        api.parent.mkdir(parents=True)
        api.write_bytes(b"external fixture")
        outside = self.root / "outside.so"
        outside.write_bytes(b"outside extension")
        extension = self.package / "_C.fixture.so"
        extension.unlink()
        extension.symlink_to(outside)
        self._run('''
site.addsitedir(wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "_C extension is missing or escapes the package" in str(exc), str(exc)
else:
    raise AssertionError("escaping extension was accepted")
assert "torch" not in sys.modules and "NATIVE_LOADED" not in os.environ
''', {"OPS_REPO_OPP_PATH": str(external), "ASCEND_CUSTOM_OPP_PATH": str(vendor),
      "EXPECTED_VENDOR": str(vendor)})

    def test_unprocessed_hook_after_torch_npu_import_requires_restart(self):
        self._run('''
sys.path.insert(0, wheel_site)
import torch_npu
try:
    import ops_repo
except RuntimeError as exc:
    assert "after torch_npu" in str(exc) and "Restart Python" in str(exc), str(exc)
else:
    raise AssertionError("late configuration was accepted")
assert "ASCEND_CUSTOM_OPP_PATH" not in os.environ
assert "torch" not in sys.modules and "NATIVE_LOADED" not in os.environ
''', {"EXPECTED_VENDOR": ""})

    def test_conflicting_vendor_and_opp_root_are_rejected_without_changes(self):
        old_root = self.root / "old" / "opp"
        old_vendor = old_root / "vendors" / "ops_repo"
        old_vendor.mkdir(parents=True)
        for path in (old_root, old_vendor, old_vendor / "op_api" / "lib"):
            with self.subTest(path=path):
                self._run('''
sys.path.insert(0, wheel_site)
before = dict(os.environ)
import ops_repo_opp_env
try:
    ops_repo_opp_env.configure()
except RuntimeError as exc:
    assert "Conflicting ops_repo OPP" in str(exc), str(exc)
else:
    raise AssertionError("conflicting vendor was accepted")
assert dict(os.environ) == before
assert "torch" not in sys.modules
''', {"ASCEND_CUSTOM_OPP_PATH": str(path) + os.pathsep + "/other/opp/vendors/unrelated"})

    def test_partial_payloads_are_rejected_before_framework_import(self):
        targets = (self.api, self.package / "_C.fixture.so", self.package / "_build_info.json")
        for target in targets:
            with self.subTest(target=target):
                saved = target.read_bytes()
                target.unlink()
                try:
                    self._run('''
sys.path.insert(0, wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "Damaged ops_repo wheel" in str(exc), str(exc)
else:
    raise AssertionError("partial wheel was accepted")
assert "torch" not in sys.modules and "ASCEND_CUSTOM_OPP_PATH" not in os.environ
''')
                finally:
                    target.write_bytes(saved)

    def test_ambiguous_embedded_opapi_is_rejected(self):
        other = self.vendor / "op_api" / "lib64" / "libcust_opapi.so"
        other.parent.mkdir()
        other.write_bytes(b"second fixture")
        self._run('''
sys.path.insert(0, wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "exactly one packaged libcust_opapi.so; found 2" in str(exc), str(exc)
else:
    raise AssertionError("ambiguous opapi was accepted")
assert "torch" not in sys.modules
''')

    def test_escaping_library_or_kernel_symlink_is_rejected(self):
        outside = self.root / "outside.so"
        outside.write_bytes(b"outside package")
        kernel = self.vendor / "op_impl" / "kernel.so"
        kernel.parent.mkdir()
        for target in (self.api, kernel):
            with self.subTest(target=target):
                saved = target.read_bytes() if target.exists() else None
                target.unlink(missing_ok=True)
                target.symlink_to(outside)
                try:
                    self._run('''
sys.path.insert(0, wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "escapes the package" in str(exc), str(exc)
else:
    raise AssertionError("escaping payload was accepted")
assert "torch" not in sys.modules and "NATIVE_LOADED" not in os.environ
''')
                finally:
                    target.unlink()
                    if saved is not None:
                        target.write_bytes(saved)

    def test_python_and_architecture_mismatch_fail_before_framework_import(self):
        for key, wrong in (("python", "0.0"), ("machine", "wrong-machine")):
            with self.subTest(key=key):
                saved = self.info[key]
                self.info[key] = wrong
                self._write_info()
                try:
                    self._run('''
sys.path.insert(0, wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "Incompatible ops_repo wheel" in str(exc), str(exc)
else:
    raise AssertionError("incompatible wheel was accepted")
assert "torch" not in sys.modules and "NATIVE_LOADED" not in os.environ
''')
                finally:
                    self.info[key] = saved
                    self._write_info()

    def test_exact_framework_versions_are_checked_before_native_loading(self):
        for variable, wrong, dependency in (("FIXTURE_TORCH", "2.6.0+other", "torch"),
                                            ("FIXTURE_TORCH_NPU", "2.6.0.post1+other", "torch_npu")):
            with self.subTest(dependency=dependency):
                self._run('''
site.addsitedir(wheel_site)
try:
    import ops_repo
except RuntimeError as exc:
    assert "Incompatible ops_repo wheel" in str(exc) and "exact torch/torch_npu" in str(exc), str(exc)
else:
    raise AssertionError("incompatible framework was accepted")
assert "NATIVE_LOADED" not in os.environ and "OPAPI_INITIALIZED" not in os.environ
''', {variable: wrong})


if __name__ == "__main__":
    unittest.main()
