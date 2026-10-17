# SPDX-License-Identifier: Apache-2.0
"""Native dispatcher/Meta/Fake checks, with no NPU allocation or initialization.

These tests use the real installed _lora_C extension and C++ Meta kernels. A
source-only checkout skips this suite. To require the compiled component:
  OPS_REPO_REQUIRE_LORA_TESTS=1 python -m unittest discover -s tests -p test_moe_lora.py -v

Hardware numerics live in tests/npu and require a separate explicit opt-in.
"""

from contextlib import ExitStack
import importlib.util
import json
import os
import platform
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
import unittest

OP_NAMES = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "build_combined_lora_idx", "add_lora", "add_lora_shrink", "add_lora_expand",
    "add_lora_expand_delta", "add_lora_swiglu_quant",
)
MUTATING = {
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "add_lora", "add_lora_shrink", "add_lora_expand",
}


class MoeLoraRuntimeTests(unittest.TestCase):
    """Exercise the real loader with explicit fake native I/O, without torch."""

    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "torch_custom" / "ops_repo"
                  / "ops_repo" / "_lora_runtime.py")
        spec = importlib.util.spec_from_file_location("_ops_repo_lora_runtime_test", source)
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.package = Path(temporary.name)
        self.runtime.__file__ = str(self.package / "_lora_runtime.py")
        self.info = {
            "schema_version": 1, "component": "moe_lora", "soc": "ascend910_93",
            "python": "%d.%d" % sys.version_info[:2], "machine": platform.machine(),
            "torch": "2.7.1+fixture", "torch_npu": "2.7.1.post1+fixture",
        }
        self.metadata = self.package / "_lora_build_info.json"
        self.metadata.write_text(json.dumps(self.info))
        self.extension = self.package / "_lora_C.fixture.so"
        self.extension.write_bytes(b"synthetic loader fixture: never dlopened")
        (self.package / "libops_repo_moe_lora_kernels.so").write_bytes(
            b"synthetic kernel fixture: never executed")
        self.dispatch = SimpleNamespace(**{name: mock.Mock(name=name) for name in OP_NAMES})
        self.torch = SimpleNamespace(__version__=self.info["torch"], ops=SimpleNamespace(
            load_library=mock.Mock(), ops_repo=self.dispatch))
        self.torch_npu = SimpleNamespace(__version__=self.info["torch_npu"])
        self.importer_patch = mock.patch.object(self.runtime.importlib, "import_module",
            side_effect=lambda name: {"torch": self.torch, "torch_npu": self.torch_npu}[name])
        self.importer = self.importer_patch.start()
        self.addCleanup(self.importer_patch.stop)

    def test_load_once_without_opp_and_call_public_dispatcher(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.runtime.initialize()
            self.runtime.initialize()
        self.assertTrue(self.runtime._ready)
        self.torch.ops.load_library.assert_called_once_with(str(self.extension))
        self.assertEqual([call.args[0] for call in self.importer.call_args_list],
                         ["torch", "torch_npu"])
        # call imports torch normally; replace only that local dependency.
        with mock.patch.dict(sys.modules, {"torch": self.torch}):
            result = self.runtime.call("add_lora", "sentinel")
        self.dispatch.add_lora.assert_called_once_with("sentinel")
        self.assertIs(result, self.dispatch.add_lora.return_value)

    def test_missing_component_has_actionable_build_error(self):
        self.metadata.unlink()
        self.extension.unlink()
        self.assertFalse(self.runtime.is_installed())
        with self.assertRaisesRegex(RuntimeError, "OPS_REPO_BUILD_LORA=1"):
            self.runtime.initialize()
        self.importer.assert_not_called()
        self.torch.ops.load_library.assert_not_called()

    def test_framework_abi_mismatch_rejected_before_library_load(self):
        for name in ("torch", "torch_npu"):
            with self.subTest(framework=name):
                module = getattr(self, name)
                original = module.__version__
                module.__version__ = "incompatible.fixture"
                with self.assertRaisesRegex(RuntimeError, "built for " + name):
                    self.runtime.initialize()
                module.__version__ = original
                self.torch.ops.load_library.assert_not_called()
                self.assertFalse(self.runtime._ready)

    def test_wrong_host_and_target_rejected_before_importing_framework(self):
        for field, value in (("python", "0.0"), ("machine", "not_this_machine"),
                             ("soc", "ascend950"), ("component", "other")):
            with self.subTest(field=field):
                self.metadata.write_text(json.dumps(dict(self.info, **{field: value})))
                with self.assertRaisesRegex(RuntimeError, "Cannot load ops_repo MoE LoRA"):
                    self.runtime.initialize()
                self.importer.assert_not_called()
                self.torch.ops.load_library.assert_not_called()

    def test_incomplete_or_ambiguous_native_payload_rejected(self):
        extra = self.package / "_lora_C.extra.so"
        extra.write_bytes(b"synthetic duplicate")
        with self.assertRaisesRegex(RuntimeError, "expected one native extension"):
            self.runtime.initialize()
        extra.unlink()
        (self.package / "libops_repo_moe_lora_kernels.so").unlink()
        with self.assertRaisesRegex(RuntimeError, "expected one native extension"):
            self.runtime.initialize()
        self.torch.ops.load_library.assert_not_called()

    def test_partial_registration_retry_does_not_reload_native_library(self):
        del self.dispatch.add_lora
        with self.assertRaisesRegex(RuntimeError, "missing add_lora"):
            self.runtime.initialize()
        self.assertTrue(self.runtime._loaded)
        self.assertFalse(self.runtime._ready)
        self.dispatch.add_lora = mock.Mock()
        self.runtime.initialize()
        self.torch.ops.load_library.assert_called_once_with(str(self.extension))
        self.assertTrue(self.runtime._ready)


class MoeLoraNativeMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = os.environ.get("OPS_REPO_REQUIRE_LORA_TESTS") == "1"
        if importlib.util.find_spec("torch") is None:
            if required:
                raise RuntimeError("native metadata tests require the compiled torch/ops_repo environment")
            raise unittest.SkipTest("torch and compiled ops_repo MoE LoRA component are not installed")
        spec = importlib.util.find_spec("ops_repo")
        package = Path(spec.origin).parent if spec and spec.origin else None
        if package is None or not list(package.glob("_lora_C*.so")):
            if required:
                raise RuntimeError("OPS_REPO_REQUIRE_LORA_TESTS=1 but _lora_C is not installed")
            raise unittest.SkipTest("compiled ops_repo MoE LoRA component is not installed")
        import torch
        cls.torch = torch
        # Querying this flag does not initialize the runtime or select a device.
        cls.npu_was_initialized = bool(getattr(getattr(torch, "npu", None),
                                              "is_initialized", lambda: False)())
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "_lazy_init"):
            guard = mock.patch.object(npu, "_lazy_init", side_effect=RuntimeError(
                "NPU initialization is forbidden in offline Meta/Fake tests"))
            guard.start()
            cls.addClassCleanup(guard.stop)
        import ops_repo
        cls.package = ops_repo
        cls.ops = torch.ops.ops_repo

    @classmethod
    def tearDownClass(cls):
        initialized = bool(getattr(getattr(cls.torch, "npu", None),
                                   "is_initialized", lambda: False)())
        if initialized != cls.npu_was_initialized:
            raise AssertionError("Meta/Fake tests must not initialize the NPU runtime")

    def _inputs(self, device="meta", rows=7):
        torch = self.torch
        x = torch.empty(rows, 512, device=device, dtype=torch.bfloat16)
        y = torch.empty(rows, 256, device=device, dtype=torch.bfloat16)
        a = torch.empty(3, 1, 16, 512, device=device, dtype=torch.bfloat16)
        b = torch.empty(3, 1, 128, 16, device=device, dtype=torch.bfloat16)
        idx = torch.empty(rows, device=device, dtype=torch.long)
        seq = torch.empty(3, device=device, dtype=torch.long)
        z = torch.empty(rows, 16, device=device, dtype=torch.float32)
        # Real host scalar tensors are safe here: Meta must never read their data.
        flag = torch.empty((), device=device, dtype=torch.bool)
        return x, y, a, b, idx, seq, z, flag

    def _call_mutating_ops(self, device="meta"):
        x, y, a, b, idx, seq, z, flag = self._inputs(device)
        self.assertIsNone(self.ops.bgmv_shrink(x, a, idx, z, 0.5))
        self.assertIs(self.ops.bgmv_expand(z, b, idx, y, 0, 128), y)
        self.assertIsNone(self.ops.sgmv_shrink(x, a, idx, seq, z, 0.5))
        self.assertIs(self.ops.sgmv_expand(z, b, idx, seq, y, 0, 128), y)
        self.assertIsNone(self.ops.add_lora_shrink(
            [z], x, [a], idx, seq, idx, 0.5, flag, flag))
        self.assertIsNone(self.ops.add_lora_expand(
            y, [z], [b], idx, seq, idx, [128], 0, True, flag, flag))
        self.assertIsNone(self.ops.add_lora(
            y, x, [a], [b], idx, seq, idx, [128], 0, 0.5, True, flag, flag, flag))

    def test_native_schemas_mutate_only_outputs(self):
        for name in OP_NAMES:
            with self.subTest(op=name):
                op = getattr(self.ops, name).default
                schema = op._schema
                writes = {arg.name for arg in schema.arguments
                          if arg.alias_info is not None and arg.alias_info.is_write}
                self.assertEqual(writes, {"y"} if name in MUTATING else set())
                if name in ("bgmv_expand", "sgmv_expand"):
                    y_arg = next(arg for arg in schema.arguments if arg.name == "y")
                    self.assertEqual(schema.returns[0].alias_info.after_set,
                                     y_arg.alias_info.after_set)
                elif name in MUTATING:
                    self.assertEqual(len(schema.returns), 0)
                self.assertTrue(self.torch._C._dispatch_has_kernel_for_dispatch_key(
                    "ops_repo::" + name, "Meta"))
                self.assertTrue(self.torch._C._dispatch_has_kernel_for_dispatch_key(
                    "ops_repo::" + name, "PrivateUse1"))
                self.assertFalse(self.torch._C._dispatch_has_kernel_for_dispatch_key(
                    "ops_repo::" + name, "CPU"))
                self.assertTrue(callable(getattr(self.package, name)))

    def test_meta_mutation_and_alias_contracts(self):
        self._call_mutating_ops()

    def test_swiglu_meta_shapes_and_dtypes(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (0, 1, 7, 125):
                with self.subTest(dtype=dtype, rows=rows):
                    gate = torch.empty(rows, 1024, device="meta", dtype=dtype)
                    act, quant, scale = self.package.add_lora_swiglu_quant(gate, None, 2.0)
                    self.assertEqual((act.shape, quant.shape, scale.shape),
                                     ((rows, 512), (rows, 512), (rows,)))
                    self.assertEqual((act.dtype, quant.dtype, scale.dtype),
                                     (dtype, torch.int8, torch.float32))
                    self.assertTrue(all(t.device.type == "meta" for t in (act, quant, scale)))
                    self.assertIsNot(act, gate)

    def test_routing_and_delta_meta_shapes(self):
        torch = self.torch
        for rows in (0, 1, 11):
            with self.subTest(rows=rows):
                dest = torch.empty(rows, 6, device="meta", dtype=torch.int32)
                indices = torch.empty(rows, device="meta", dtype=torch.long)
                enabled = torch.empty(4, device="meta", dtype=torch.int32)
                combined = self.package.build_combined_lora_idx(dest, dest, indices, enabled, 8, 4, 6)
                self.assertEqual(combined.shape, (rows * 6,))
                self.assertEqual(combined.dtype, torch.long)
                _, gate, _, b, _, seq, z, _ = self._inputs(rows=rows)
                scratch = [torch.empty(rows, 3 * 16, device="meta", dtype=torch.float32),
                           torch.empty(rows, 3 * 16, device="meta", dtype=torch.float32),
                           torch.empty(2, rows, 16, device="meta", dtype=torch.float32)]
                delta = self.package.add_lora_expand_delta(gate, scratch, [b, b], seq)
                self.assertEqual(delta.shape, gate.shape)
                self.assertEqual(delta.dtype, gate.dtype)
                self.assertIsNot(delta, gate)

    def test_fake_tensor_all_native_operators(self):
        torch = self.torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode():
            self._call_mutating_ops("cpu")
            gate = torch.empty(7, 1024, dtype=torch.bfloat16)
            outputs = self.package.add_lora_swiglu_quant(gate, torch.empty_like(gate), 0.0)
            dest = torch.empty(7, 6, dtype=torch.int32)
            idx = torch.empty(7, dtype=torch.long)
            enabled = torch.empty(4, dtype=torch.int32)
            combined = self.package.build_combined_lora_idx(dest, dest, idx, enabled, 0, 4, 6)
            b = torch.empty(3, 1, 512, 16, dtype=torch.bfloat16)
            scratch = [torch.empty(7, 48), torch.empty(1, 7, 16)]
            delta = self.package.add_lora_expand_delta(gate[:, :512], scratch, [b], idx)
            self.assertTrue(all(isinstance(t, FakeTensor) for t in (*outputs, combined, delta)))
            self.assertEqual(outputs[0].shape, (7, 512))
            self.assertEqual(combined.shape, (42,))
            self.assertEqual(delta.shape, (7, 512))

    def test_public_api_dynamic_fullgraph_meta(self):
        torch = self.torch

        def function(gate, dest, idx, enabled):
            act, quant, scale = self.package.add_lora_swiglu_quant(gate, None, 0.75)
            combined = self.package.build_combined_lora_idx(dest, dest, idx, enabled, 0, 4, 6)
            return act, quant, scale, combined

        # Newer Dynamo versions inspect the ambient accelerator stream even
        # for an all-Meta graph. Suppress device discovery in this CPU-only
        # test; all operator schemas and C++ Meta implementations remain real.
        # The class-level lazy-init guard still fails any accidental NPU access.
        with ExitStack() as stack:
            if hasattr(torch, "accelerator"):
                stack.enter_context(mock.patch.object(torch.accelerator, "is_available", return_value=False))
            compiled = torch.compile(function, backend="eager", fullgraph=True, dynamic=True)
            for rows in (3, 11):
                outputs = compiled(
                    torch.empty(rows, 1024, device="meta", dtype=torch.bfloat16),
                    torch.empty(rows, 6, device="meta", dtype=torch.int32),
                    torch.empty(rows, device="meta", dtype=torch.long),
                    torch.empty(4, device="meta", dtype=torch.int32))
                self.assertEqual([tuple(t.shape) for t in outputs],
                                 [(rows, 512), (rows, 512), (rows,), (rows * 6,)])

    def test_real_cpu_execution_has_no_silent_fallback(self):
        torch = self.torch
        with self.assertRaises(NotImplementedError):
            self.package.add_lora_swiglu_quant(torch.empty(1, 128, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
