"""CPU-only package/metadata tests; these do not validate an Ascend kernel."""

import importlib
import importlib.util
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "torch_custom" / "ops_repo"
sys.path.insert(0, str(SOURCE))


@contextmanager
def _fresh_runtime():
    """Preload the real runtime so tests can mock native I/O before root import."""
    previous = {name: module for name, module in sys.modules.copy().items()
                if name == "ops_repo" or name.startswith("ops_repo.")}
    for name in previous:
        del sys.modules[name]
    specification = importlib.util.spec_from_file_location(
        "ops_repo._runtime", SOURCE / "ops_repo" / "_runtime.py")
    runtime = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = runtime
    try:
        specification.loader.exec_module(runtime)
        yield runtime
    finally:
        for name in list(sys.modules):
            if name == "ops_repo" or name.startswith("ops_repo."):
                del sys.modules[name]
        sys.modules.update(previous)


@contextmanager
def _module_overrides(replacements):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _native_paths(runtime):
    with mock.patch.object(runtime, "_op_api_library", return_value=Path("/opp/libcust_opapi.so")), \
         mock.patch.object(runtime, "_extension_library", return_value=Path("/package/_C.so")):
        yield


@contextmanager
def _mocked_package(first_import="import ops_repo"):
    """Execute production package initialization with test-only native stand-ins."""
    torch = types.ModuleType("torch")
    torch.library = types.SimpleNamespace(register_fake=mock.Mock(return_value=mock.Mock()))
    torch.ops = types.SimpleNamespace(
        load_library=mock.Mock(), ops_repo=types.SimpleNamespace(_initialize=mock.Mock()))
    with _fresh_runtime() as runtime, \
         _module_overrides({"torch": torch, "torch_npu": types.ModuleType("torch_npu")}):
        with _native_paths(runtime):
            exec(first_import, {})
        yield sys.modules["ops_repo"], runtime, torch


class PackageTests(unittest.TestCase):
    def setUp(self):
        context = _mocked_package()
        self.package, self.runtime, self.torch = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_all_public_import_forms_initialize_before_returning(self):
        forms = ("import ops_repo", "from ops_repo import add",
                 "from ops_repo.ops.ascendc import add")
        for statement in forms:
            with self.subTest(statement=statement):
                code = """
import sys
from test_python_api import _mocked_package
with _mocked_package(first_import=%r) as (package, runtime, torch):
    assert runtime._ready
    assert runtime._torch is torch
    assert 'ops_repo.ops.ascendc' in sys.modules
    torch.ops.load_library.assert_called_once_with('/package/_C.so')
    torch.ops.ops_repo._initialize.assert_called_once_with('/opp/libcust_opapi.so')
    torch.library.register_fake.assert_called_once_with('ops_repo::add')
    assert sys.modules['ops_repo._fake']._registered
    import ops_repo
    from ops_repo.ops.ascendc import initialize
    package.initialize()
    initialize()
    torch.ops.load_library.assert_called_once()
    torch.ops.ops_repo._initialize.assert_called_once()
    torch.library.register_fake.assert_called_once()
    assert 'triton' not in sys.modules
    assert 'fla_npu' not in sys.modules
""" % statement
                environment = dict(os.environ, PYTHONPATH=str(ROOT / "tests"))
                result = subprocess.run([sys.executable, "-c", code], env=environment,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_triton_is_explicitly_unimplemented(self):
        from ops_repo.ops import triton

        for call in (lambda: self.package.initialize("triton"),
                     lambda: triton.add(None, None)):
            with self.assertRaises(NotImplementedError):
                call()

    def test_import_failure_is_visible_and_dependency_retry_succeeds(self):
        with _fresh_runtime() as runtime:
            with _native_paths(runtime), _module_overrides({"torch": None}):
                with self.assertRaisesRegex(RuntimeError, "Importing ops_repo requires"):
                    __import__("ops_repo")
            self.assertNotIn("ops_repo", sys.modules)
            self.assertFalse(runtime._ready)
            self.assertFalse(runtime._library_loaded)
            with _native_paths(runtime):
                __import__("ops_repo")
            self.assertTrue(runtime._ready)

    def test_import_requires_startup_opp_environment(self):
        with _fresh_runtime() as runtime, mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPS_REPO_OPP_PATH"):
                __import__("ops_repo")
            self.assertEqual(dict(os.environ), {})
            self.assertNotIn("ops_repo", sys.modules)
            self.assertFalse(runtime._ready)
            self.assertFalse(runtime._library_loaded)

    def test_missing_torch_npu_fails_before_loading_and_can_retry(self):
        with _fresh_runtime() as runtime, _native_paths(runtime):
            self.torch.ops.load_library.reset_mock()
            with _module_overrides({"torch_npu": None}):
                with self.assertRaisesRegex(RuntimeError, "Cannot import torch_npu"):
                    __import__("ops_repo")
            self.assertNotIn("ops_repo", sys.modules)
            self.assertFalse(runtime._ready)
            self.torch.ops.load_library.assert_not_called()
            __import__("ops_repo")
            self.assertTrue(runtime._ready)
            self.torch.ops.load_library.assert_called_once()

    def test_failed_loading_or_registration_can_retry_import(self):
        for stage in ("extension", "native", "fake"):
            with self.subTest(stage=stage), _fresh_runtime() as runtime, _native_paths(runtime):
                self.torch.ops.load_library.reset_mock(side_effect=True)
                self.torch.ops.ops_repo._initialize.reset_mock(side_effect=True)
                self.torch.library.register_fake.reset_mock()
                decorator = self.torch.library.register_fake.return_value
                decorator.reset_mock(side_effect=True)
                failing_call = {"extension": self.torch.ops.load_library,
                                "native": self.torch.ops.ops_repo._initialize,
                                "fake": decorator}[stage]
                failing_call.side_effect = [RuntimeError("simulated %s failure" % stage), None]
                with self.assertRaisesRegex(RuntimeError, "simulated %s failure" % stage):
                    __import__("ops_repo")
                self.assertNotIn("ops_repo", sys.modules)
                self.assertFalse(runtime._ready)
                self.assertIsNone(runtime._torch)
                self.assertEqual(runtime._library_loaded, stage != "extension")
                if stage == "fake":
                    self.assertFalse(sys.modules["ops_repo._fake"]._registered)
                package = __import__("ops_repo")
                self.assertTrue(runtime._ready)
                self.assertTrue(sys.modules["ops_repo._fake"]._registered)
                self.assertEqual(self.torch.ops.load_library.call_count,
                                 2 if stage == "extension" else 1)
                native_calls = 1 if stage == "extension" else 2
                self.assertEqual(self.torch.ops.ops_repo._initialize.call_count, native_calls)
                package.initialize()
                self.assertEqual(self.torch.ops.ops_repo._initialize.call_count, native_calls)

    def test_add_does_not_attempt_late_initialization(self):
        with mock.patch.object(self.runtime, "_ready", False), \
             mock.patch.object(self.runtime, "initialize_ascendc") as initialize:
            with self.assertRaisesRegex(RuntimeError, "initialization must complete"):
                self.package.add(None, None)
            initialize.assert_not_called()

    def test_unknown_backend_is_rejected_without_initialization(self):
        from ops_repo import add, initialize

        for function, arguments in ((add, (None, None)), (initialize, ())):
            with self.assertRaisesRegex(ValueError, "backend"):
                function(*arguments, backend="typo")

    def test_missing_native_extension_has_build_instructions(self):
        from ops_repo import _runtime

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(_runtime, "__file__", str(Path(temp) / "_runtime.py")):
                with self.assertRaisesRegex(RuntimeError, "OPS_REPO_BUILD_TORCH=1"):
                    _runtime._extension_library()

    def test_opp_resolution_requires_explicit_matching_vendor(self):
        from ops_repo import _runtime

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vendor = root / "vendors" / "ops_repo"
            api = vendor / "op_api" / "lib" / "libcust_opapi.so"
            api.parent.mkdir(parents=True)
            api.write_bytes(b"path-discovery fixture, never loaded")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OPS_REPO_OPP_PATH"):
                    _runtime._op_api_library()
            environment = {"OPS_REPO_OPP_PATH": str(root), "ASCEND_CUSTOM_OPP_PATH": str(vendor)}
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(_runtime._op_api_library(), api)
                self.assertEqual(dict(os.environ), environment)
            with mock.patch.dict(os.environ, {"OPS_REPO_OPP_PATH": str(root)}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "before starting Python"):
                    _runtime._op_api_library()

    def test_ambiguous_vendor_libraries_are_rejected(self):
        from ops_repo import _runtime

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vendor = root / "vendors" / "ops_repo"
            for folder in ("lib", "lib64"):
                api = vendor / "op_api" / folder / "libcust_opapi.so"
                api.parent.mkdir(parents=True)
                api.write_bytes(b"not loaded")
            with mock.patch.dict(os.environ, {"OPS_REPO_OPP_PATH": str(root),
                                             "ASCEND_CUSTOM_OPP_PATH": str(vendor)}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "found 2"):
                    _runtime._op_api_library()

    def test_initialize_is_idempotent(self):
        from ops_repo import _fake, _runtime

        torch = types.SimpleNamespace(
            library=types.SimpleNamespace(register_fake=mock.Mock()),
            ops=types.SimpleNamespace(load_library=mock.Mock(),
                                      ops_repo=types.SimpleNamespace(_initialize=mock.Mock())),
        )
        def import_dependency(name):
            return torch if name == "torch" else types.SimpleNamespace()

        with mock.patch.object(_runtime, "_ready", False), \
             mock.patch.object(_runtime, "_torch", None), \
             mock.patch.object(_runtime, "_library_loaded", False), \
             mock.patch.object(_runtime, "_op_api_library", return_value=Path("/opp/libcust_opapi.so")), \
             mock.patch.object(_runtime, "_extension_library", return_value=Path("/package/_C.so")), \
             mock.patch.object(_runtime.importlib, "import_module", side_effect=import_dependency) as imports, \
             mock.patch.object(_fake, "register_fake") as register:
            _runtime.initialize_ascendc()
            _runtime.initialize_ascendc()
            self.assertEqual(imports.call_count, 2)
            torch.ops.load_library.assert_called_once_with("/package/_C.so")
            torch.ops.ops_repo._initialize.assert_called_once_with("/opp/libcust_opapi.so")
            register.assert_called_once()


class TorchMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
        except ImportError as exc:
            raise unittest.SkipTest("optional CPU torch is not installed") from exc
        if not hasattr(cls.torch.library, "register_fake"):
            raise unittest.SkipTest("torch >= 2.4 is required for Fake/Meta tests")
        # This is an explicit test-only schema. No C++/NPU/CPU execution kernel
        # is installed. Its lifetime is retained for the entire test process.
        cls.schema = cls.torch.library.Library("ops_repo", "DEF")
        cls.schema.define("add(Tensor x, Tensor y) -> Tensor")
        context = _fresh_runtime()
        cls.runtime = context.__enter__()
        cls.addClassCleanup(context.__exit__, None, None, None)
        # Run the real package __init__ and real Fake registration. Only native
        # library I/O and torch_npu are mocked; there is no production bypass.
        with _native_paths(cls.runtime), \
             _module_overrides({"torch_npu": types.ModuleType("torch_npu")}), \
             mock.patch.object(cls.torch.ops, "load_library"), \
             mock.patch.object(cls.torch.ops.ops_repo, "_initialize", create=True):
            cls.package = __import__("ops_repo")
        assert cls.runtime._ready
        assert cls.runtime._torch is cls.torch

    def test_meta_and_scalar_metadata(self):
        torch = self.torch
        for shape in ((), (1,), (3, 5)):
            x = torch.empty(shape, device="meta")
            result = torch.ops.ops_repo.add(x, x)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "meta")
            self.assertTrue(result.is_contiguous())

    def test_fake_tensor_metadata_without_native_backend(self):
        torch = self.torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode():
            x = torch.empty(3, 5)
            result = torch.ops.ops_repo.add(x, x)
        self.assertIsInstance(result, FakeTensor)
        self.assertEqual(result.shape, (3, 5))

    def test_actual_cpu_execution_is_not_registered(self):
        torch = self.torch
        self.assertFalse(torch._C._dispatch_has_kernel_for_dispatch_key("ops_repo::add", "CPU"))
        with self.assertRaises(NotImplementedError):
            torch.ops.ops_repo.add(torch.ones(2), torch.ones(2))

    def test_validation_rejects_unsupported_inputs(self):
        torch = self.torch
        from ops_repo._validation import validate_add_inputs

        good = torch.empty(2, 3)
        invalid = [
            (None, good, TypeError),
            (good.double(), good.double(), TypeError),
            (good, torch.empty(2, 4), RuntimeError),
            (torch.empty(0), torch.empty(0), RuntimeError),
            (good.t(), good.t(), ValueError),
            (good.requires_grad_(), good, RuntimeError),
        ]
        for x, y, exception in invalid:
            with self.subTest(exception=exception, shape=getattr(x, "shape", None)):
                with self.assertRaises(exception):
                    validate_add_inputs(x, y, require_npu=False)
        with self.assertRaisesRegex(ValueError, "NPU inputs"):
            validate_add_inputs(torch.ones(2), torch.ones(2))

    def test_public_api_fullgraph_dynamic_meta(self):
        torch = self.torch

        # Import already completed real Python/Fake initialization. This checks
        # public tracing, not the mocked native loading or NPU execution.
        compiled = torch.compile(self.package.add, backend="eager", fullgraph=True, dynamic=True)
        for shape in ((3, 5), (7, 11)):
            x = torch.empty(shape, device="meta")
            result = compiled(x, x)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
