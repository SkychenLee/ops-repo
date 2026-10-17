"""Offline tests for repository operations; no CANN installation or NPU needed."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_ops", REPOSITORY / "scripts/ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


# A real subprocess exercises argument quoting, incremental generation, staging,
# and errors without depending on an installed version of the CANN generator.
FAKE_MSOPGEN = textwrap.dedent('''\
    import argparse
    import json
    from pathlib import Path
    import re
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["gen"])
    parser.add_argument("-i", required=True)
    parser.add_argument("-f", choices=["tf"], required=True)
    parser.add_argument("-c", required=True)
    parser.add_argument("-lan", choices=["cpp"], required=True)
    parser.add_argument("-out", required=True)
    parser.add_argument("-m")
    args = parser.parse_args()
    project = Path(args.out)
    if project.exists() != (args.m == "1"):
        sys.exit("incremental generation must pass -m 1")
    schema, = json.loads(Path(args.i).read_text())
    if schema["op"] == "FailOp":
        sys.exit("intentional generation failure")
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", schema["op"])
    name = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", first).lower()
    project.mkdir(exist_ok=True)
    for component in ("op_host", "op_kernel", "framework"):
        folder = project / component
        folder.mkdir(exist_ok=True)
        (folder / "CMakeLists.txt").write_text("# generated build configuration\\n")
        (folder / (name + ".cpp")).write_text("GENERATED PLACEHOLDER\\n")
    (project / "op_host" / (name + "_tiling.h")).write_text("GENERATED TILING\\n")
    (project / "CMakeLists.txt").write_text("# generated root configuration\\n")
    if not (project / "CMakePresets.json").exists():
        (project / "CMakePresets.json").write_text(json.dumps({
            "version": 1, "configurePresets": [{
                "name": "default", "cacheVariables": {
                    "CMAKE_INSTALL_PREFIX": {"type": "PATH", "value": "${sourceDir}/build_out"}
                }
            }]
        }))
    with (project / "calls.jsonl").open("a") as stream:
        stream.write(json.dumps(vars(args)) + "\\n")
''')


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ops repo tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generator = self.root / "fake msopgen.py"
        self.generator.write_text(FAKE_MSOPGEN, encoding="utf-8")
        self.output = io.StringIO()
        self.stdout_redirect = redirect_stdout(self.output)
        self.stdout_redirect.__enter__()
        self.addCleanup(self.stdout_redirect.__exit__, None, None, None)

    def add_operator(self, name, enabled=True, soc=None, category="custom"):
        ops.new_operator(argparse.Namespace(name=name, category=category), root=self.root)
        directory = self.root / "ops" / "ascendc" / category / name
        metadata = json.loads((directory / "operator.json").read_text())
        metadata["enabled"] = enabled
        if soc is not None:
            metadata["soc"] = soc
        ops.write_json(directory / "operator.json", metadata)
        for component in ("op_host", "op_kernel"):
            (directory / component / (name + ".cpp")).write_text(
                f"// Handwritten {name} {component}\n", encoding="utf-8"
            )
        return directory

    def arguments(self, selection=None, soc="ascend910b", soc_version=None):
        return argparse.Namespace(
            ops=selection, cann_path=str(self.root / "absent-toolkit"),
            msopgen=str(self.generator), soc=soc, soc_version=soc_version, jobs=2,
        )

    def old_build(self, soc="ascend910b"):
        project = self.root / "build" / soc / "cann_project"
        project.mkdir(parents=True)
        sentinel = project / "keep.txt"
        sentinel.write_text("previous successful project\n")
        return sentinel

    def assert_old_build_retained(self, sentinel):
        self.assertEqual(sentinel.read_text(), "previous successful project\n")
        self.assertEqual(list((self.root / "build").rglob(".generate-*")), [])

    def test_new_operator_is_disabled_unimplemented_and_never_overwrites(self):
        args = argparse.Namespace(name="scale_add")
        ops.new_operator(args, root=self.root)
        directory = self.root / "ops/ascendc/custom/scale_add"
        before = {str(p.relative_to(directory)): p.read_bytes()
                  for p in directory.rglob("*") if p.is_file()}
        operator, = ops.load_operators(self.root)
        self.assertFalse(operator["enabled"])
        self.assertEqual(operator["schema"]["op"], "ScaleAdd")
        for component in ("op_host", "op_kernel"):
            self.assertIn("#error", (directory / component / "scale_add.cpp").read_text())
        with self.assertRaises(ops.RepoError):
            ops.new_operator(args, root=self.root)
        self.assertEqual(before, {str(p.relative_to(directory)): p.read_bytes()
                                 for p in directory.rglob("*") if p.is_file()})
        with self.assertRaises(ops.RepoError):
            ops.select_operators([operator])

    def test_new_rejects_paths_and_ambiguous_names_without_creating_files(self):
        for name in ("../escape", "/absolute", "Upper", "a__b", "a_", "", "1add", "a_2"):
            with self.subTest(name=name), self.assertRaises(ops.RepoError):
                ops.new_operator(argparse.Namespace(name=name), root=self.root)
        self.assertFalse((self.root / "ops").exists())

    def test_invalid_metadata_is_rejected_offline(self):
        directory = self.add_operator("test_add")
        path = directory / "operator.json"
        valid = json.loads(path.read_text())
        mutations = (
            lambda x: x.update(name="another_name"),
            lambda x: x.update(enabled="true"),
            lambda x: x.update(soc=[]),
            lambda x: x.update(soc=["unknown-chip"]),
            lambda x: x.update(soc=["ascend910b", "ascend910b"]),
            lambda x: x.update(hardware_validated=["ascend950"]),
            lambda x: x.update(hardware_validated="true"),
            lambda x: x.update(schema=[]),
            lambda x: x["schema"].update(op="OtherName"),
            lambda x: x["schema"].update(input_desc=[]),
            lambda x: x["schema"]["input_desc"][0].update(type=["float", "float16"]),
            lambda x: x["schema"]["input_desc"][0].update(param_type="unknown"),
            lambda x: x["schema"]["input_desc"][1].update(name="x"),
        )
        for mutate in mutations:
            invalid = json.loads(json.dumps(valid))
            mutate(invalid)
            ops.write_json(path, invalid)
            with self.subTest(metadata=invalid), self.assertRaises(ops.RepoError):
                ops.load_operators(self.root)
        path.write_text("[]")
        with self.assertRaises(ops.RepoError):
            ops.load_operators(self.root)
        path.write_text("{broken")
        with self.assertRaises(ops.RepoError):
            ops.load_operators(self.root)

    def test_missing_source_is_rejected(self):
        directory = self.add_operator("test_add")
        (directory / "op_kernel/test_add.cpp").unlink()
        with self.assertRaises(ops.RepoError):
            ops.load_operators(self.root)

    def test_selection_respects_enabled_state_soc_and_explicit_order(self):
        self.add_operator("first_op")
        self.add_operator("second_op")
        self.add_operator("unfinished_op", enabled=False)
        operators = ops.load_operators(self.root)
        self.assertEqual([o["name"] for o in ops.select_operators(operators)], ["first_op", "second_op"])
        self.assertEqual([o["name"] for o in ops.select_operators(operators, " second_op,first_op ")],
                         ["second_op", "first_op"])
        for selection in ("missing", "unfinished_op", "first_op,first_op", "first_op,", ",", " ", ""):
            with self.subTest(selection=selection), self.assertRaises(ops.RepoError):
                ops.select_operators(operators, selection)
        self.add_operator("wrong_chip", soc=["ascend950"])
        with self.assertRaises(ops.RepoError):
            ops.select_operators(ops.load_operators(self.root), "wrong_chip")

    def test_multi_operator_generation_preserves_sources_and_removes_placeholders(self):
        first = self.add_operator("first_op")
        second = self.add_operator("second_op")
        self.add_operator("unfinished_op", enabled=False)
        header = first / "op_host/first_op_tiling.h"
        header.write_text("// Handwritten tiling layout\n")
        nested = second / "op_kernel/include/second_op_extra.h"
        nested.parent.mkdir()
        nested.write_text("// Handwritten helper\n")
        sentinel = self.old_build()
        project = ops.generate(self.arguments(), root=self.root)
        self.assertFalse(sentinel.exists())
        self.assertFalse((project / "framework").exists())
        for directory in (first, second):
            for component in ("op_host", "op_kernel"):
                for source in (directory / component).rglob("*"):
                    if source.is_file():
                        target = project / component / source.relative_to(directory / component)
                        self.assertEqual(target.read_bytes(), source.read_bytes())
                self.assertTrue((project / component / "CMakeLists.txt").is_file())
        self.assertFalse((project / "op_host/second_op_tiling.h").exists())
        self.assertFalse((project / "op_kernel/unfinished_op.cpp").exists())
        calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([c["m"] for c in calls], [None, "1"])
        self.assertEqual([c["c"] for c in calls], ["ai_core-Ascend910B1"] * 2)
        receipt = json.loads((project / "opsrepo-build.json").read_text())
        self.assertEqual(receipt["operators"], ["first_op", "second_op"])
        presets = json.loads((project / "CMakePresets.json").read_text())
        cache = presets["configurePresets"][0]["cacheVariables"]
        self.assertEqual(cache["ASCEND_COMPUTE_UNIT"]["value"], "ascend910b")
        self.assertEqual(cache["vendor_name"]["value"], "ops_repo")
        self.assertIs(cache["ENABLE_CROSS_COMPILE"]["value"], False)
        self.assertIs(cache["ENABLE_SOURCE_PACKAGE"]["value"], False)

    def test_generator_failure_keeps_previous_project(self):
        self.add_operator("first_op")
        self.add_operator("fail_op")
        sentinel = self.old_build()
        with self.assertRaises(subprocess.CalledProcessError):
            ops.generate(self.arguments("first_op,fail_op"), root=self.root)
        self.assert_old_build_retained(sentinel)

    def test_same_filename_conflict_keeps_previous_project(self):
        for name in ("first_op", "second_op"):
            folder = self.add_operator(name)
            (folder / "op_host/shared.h").write_text(name)
        sentinel = self.old_build()
        with self.assertRaises(ops.RepoError):
            ops.generate(self.arguments(), root=self.root)
        self.assert_old_build_retained(sentinel)

    def test_file_directory_conflict_keeps_previous_project(self):
        first = self.add_operator("first_op")
        second = self.add_operator("second_op")
        (first / "op_host/common").write_text("a file")
        (second / "op_host/common").mkdir()
        (second / "op_host/common/second_op.h").write_text("a nested file")
        sentinel = self.old_build()
        with self.assertRaises((ops.RepoError, OSError)):
            ops.generate(self.arguments(), root=self.root)
        self.assert_old_build_retained(sentinel)

    def test_operator_cannot_override_generated_cmake(self):
        directory = self.add_operator("test_add")
        (directory / "op_host/CMakeLists.txt").write_text("unwanted override")
        sentinel = self.old_build()
        with self.assertRaises(ops.RepoError):
            ops.generate(self.arguments(), root=self.root)
        self.assert_old_build_retained(sentinel)

    def test_linked_source_file_is_rejected_without_touching_target(self):
        directory = self.add_operator("test_add")
        external = self.root / "external.h"
        external.write_text("external contents")
        (directory / "op_host/external.h").symlink_to(external)
        sentinel = self.old_build()
        with self.assertRaises(ops.RepoError):
            ops.generate(self.arguments(), root=self.root)
        self.assertEqual(external.read_text(), "external contents")
        self.assert_old_build_retained(sentinel)

    def test_linked_source_directory_is_rejected(self):
        directory = self.add_operator("test_add")
        external = self.root / "external_headers"
        external.mkdir()
        (external / "external.h").write_text("external contents")
        (directory / "op_host/linked_headers").symlink_to(external, target_is_directory=True)
        sentinel = self.old_build()
        with self.assertRaises(ops.RepoError):
            ops.generate(self.arguments(), root=self.root)
        self.assert_old_build_retained(sentinel)

    def test_build_directory_symlink_is_rejected(self):
        self.add_operator("test_add")
        external = self.root / "external_build"
        external.mkdir()
        (self.root / "build").symlink_to(external, target_is_directory=True)
        with self.assertRaises(ops.RepoError):
            ops.generate(self.arguments(), root=self.root)
        self.assertEqual(list(external.iterdir()), [])

    def test_cli_new_and_check_work_without_environment_setup(self):
        script = self.root / "scripts/ops.py"
        script.parent.mkdir()
        shutil.copy2(REPOSITORY / "scripts/ops.py", script)
        environment = dict(os.environ)
        environment.pop("ASCEND_HOME_PATH", None)
        environment.pop("ASCEND_TOOLKIT_HOME", None)
        for arguments in (["new", "offline_op"], ["check"], ["list"]):
            completed = subprocess.run(
                [sys.executable, str(script), *arguments], cwd=self.root,
                env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("offline_op", completed.stdout)
        completed = subprocess.run(
            [sys.executable, str(script), "new", "../escape"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("ERROR:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_doctor_reports_missing_tools_instead_of_throwing(self):
        with patch.object(ops, "cann_root", return_value=None), \
                patch.object(ops.shutil, "which", return_value=None):
            checks = ops.environment_checks(msopgen=str(self.root / "missing-msopgen"))
        failures = {check["name"] for check in checks if not check["ok"]}
        self.assertTrue({"CANN Toolkit headers", "AscendC compiler", "msopgen", "cmake", "g++", "make"} <= failures)

    def test_build_uses_native_cmake_paths_and_records_the_current_package(self):
        self.add_operator("test_add")
        project = ops.generate(self.arguments(), root=self.root)
        toolkit = self.root / "fake toolkit"
        package_content = b"fake OPP package for bookkeeping test only\n"
        commands = []

        def fake_cmake(command, **kwargs):
            command = [str(part) for part in command]
            commands.append((command, kwargs))
            if command[1] == "--build":
                # CPack may leave identical copies in more than one subdirectory.
                for relative in ("ops_repo.run", "_CPack_Packages/staging/ops_repo.run"):
                    package = project / "build_out" / relative
                    package.parent.mkdir(parents=True, exist_ok=True)
                    package.write_bytes(package_content)

        with patch.object(ops, "ROOT", self.root), \
                patch.object(ops, "generate", return_value=project), \
                patch.object(ops, "environment_checks", return_value=[]), \
                patch.object(ops, "cann_root", return_value=toolkit), \
                patch.object(ops, "run", side_effect=fake_cmake):
            ops.build(self.arguments())
        configure, configure_kwargs = commands[0]
        self.assertEqual(configure[:5], ["cmake", "-S", str(project), "-B", str(project / "build_out")])
        self.assertIn("-DENABLE_CROSS_COMPILE=OFF", configure)
        self.assertIn("-DASCEND_COMPUTE_UNIT=ascend910b", configure)
        self.assertIn(f"-DCMAKE_INSTALL_PREFIX={project / 'build_out'}", configure)
        self.assertIn(f"-DASCEND_CANN_PACKAGE_PATH={toolkit}", configure)
        self.assertEqual(configure_kwargs["env"]["ASCEND_HOME_PATH"], str(toolkit))
        self.assertEqual(commands[1][0], ["cmake", "--build", str(project / "build_out"),
                                       "--target", "binary", "package", "--parallel", "2"])
        dist = self.root / "dist/ascend910b"
        self.assertEqual((dist / "ops_repo.run").read_bytes(), package_content)
        receipt = json.loads((dist / "build.json").read_text())
        self.assertEqual(receipt["package"], "ops_repo.run")
        self.assertEqual(receipt["sha256"], hashlib.sha256(package_content).hexdigest())
        self.assertEqual(receipt["operators"], ["test_add"])

    def test_ambiguous_build_packages_do_not_replace_previous_distribution(self):
        self.add_operator("test_add")
        project = ops.generate(self.arguments(), root=self.root)
        for subfolder, content in (("a", b"first"), ("b", b"different")):
            package = project / "build_out" / subfolder / "ops_repo.run"
            package.parent.mkdir(parents=True)
            package.write_bytes(content)
        receipt = self.root / "dist/ascend910b/build.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"previous": true}\n')
        with patch.object(ops, "ROOT", self.root), \
                patch.object(ops, "generate", return_value=project), \
                patch.object(ops, "environment_checks", return_value=[]), \
                patch.object(ops, "cann_root", return_value=self.root / "toolkit"), \
                patch.object(ops, "run"), self.assertRaises(ops.RepoError):
            ops.build(self.arguments())
        self.assertEqual(receipt.read_text(), '{"previous": true}\n')
        self.assertEqual(list(receipt.parent.glob("*.run")), [])

    def test_new_multi_target_operator_stays_disabled_and_unvalidated(self):
        args = ops.parser().parse_args([
            "new", "multi_add", "--soc", "ascend910b", "ascend910_93", "ascend950",
        ])
        ops.new_operator(args, root=self.root)
        operator, = ops.load_operators(self.root)
        self.assertEqual(operator["soc"], list(ops.SOC_TARGETS))
        self.assertFalse(operator["enabled"])
        self.assertEqual(operator["hardware_validated"], [])
        for soc in ops.SOC_TARGETS:
            with self.subTest(soc=soc), self.assertRaises(ops.RepoError):
                ops.select_operators([operator], soc=soc)

    def test_target_defaults_and_model_mismatches(self):
        expected = {
            "ascend910b": "Ascend910B1",
            "ascend910_93": "Ascend910_93",
            "ascend950": "Ascend950",
        }
        self.assertEqual(ops.resolve_target(ops.parser().parse_args(["generate"])),
                         ("ascend910b", "Ascend910B1"))
        for soc, version in expected.items():
            with self.subTest(soc=soc):
                args = ops.parser().parse_args(["generate", "--soc", soc])
                self.assertEqual(ops.resolve_target(args), (soc, version))
                self.assertEqual(ops.resolve_target(self.arguments(soc=soc, soc_version=version)), (soc, version))
        for soc, version in (("ascend950", "Ascend910B1"), ("ascend910b", "Ascend950"),
                             ("ascend910_93", "Ascend910B1"), ("ascend910b", "Ascend910B")):
            with self.subTest(soc=soc, version=version), self.assertRaises(ops.RepoError):
                ops.resolve_target(self.arguments(soc=soc, soc_version=version))
        with self.assertRaises(ops.RepoError):
            ops.resolve_target(self.arguments(soc="../escape"))

    def test_default_selection_filters_by_target_but_explicit_mismatch_errors(self):
        self.add_operator("a_two_only", soc=["ascend910b"])
        self.add_operator("a_three_only", soc=["ascend910_93"])
        self.add_operator("shared_op", soc=list(ops.SOC_TARGETS))
        operators = ops.load_operators(self.root)
        self.assertEqual([o["name"] for o in ops.select_operators(operators, soc="ascend910b")],
                         ["a_two_only", "shared_op"])
        self.assertEqual([o["name"] for o in ops.select_operators(operators, soc="ascend910_93")],
                         ["a_three_only", "shared_op"])
        self.assertEqual([o["name"] for o in ops.select_operators(operators, soc="ascend950")], ["shared_op"])
        with self.assertRaises(ops.RepoError):
            ops.select_operators(operators, "a_two_only", soc="ascend950")

    def test_generate_isolates_targets_and_uses_their_compute_units(self):
        self.add_operator("shared_op", soc=list(ops.SOC_TARGETS))
        generated = {}
        for soc in ops.SOC_TARGETS:
            with self.subTest(soc=soc):
                project = ops.generate(self.arguments(soc=soc), root=self.root)
                self.assertEqual(project, self.root / "build" / soc / "cann_project")
                receipt = json.loads((project / "opsrepo-build.json").read_text())
                self.assertEqual(receipt["soc"], soc)
                self.assertEqual(receipt["hardware_validated"], {"shared_op": False})
                call, = (project / "calls.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(call)["c"], "ai_core-" + ops.SOC_TARGETS[soc]["default_version"])
                presets = json.loads((project / "CMakePresets.json").read_text())
                self.assertEqual(presets["configurePresets"][0]["cacheVariables"]["ASCEND_COMPUTE_UNIT"]["value"], soc)
                generated[project] = (project / "opsrepo-build.json").read_bytes()
        for project, receipt in generated.items():
            self.assertEqual((project / "opsrepo-build.json").read_bytes(), receipt)
        self.add_operator("fail_op", soc=list(ops.SOC_TARGETS))
        with self.assertRaises(subprocess.CalledProcessError):
            ops.generate(self.arguments("fail_op", soc="ascend950"), root=self.root)
        for project, receipt in generated.items():
            self.assertEqual((project / "opsrepo-build.json").read_bytes(), receipt)

    def test_build_and_install_keep_same_named_packages_separate_for_each_target(self):
        self.add_operator("shared_op", soc=list(ops.SOC_TARGETS))
        installs = []
        for soc in ops.SOC_TARGETS:
            project = ops.generate(self.arguments(soc=soc), root=self.root)

            def fake_cmake(command, **kwargs):
                if command[1] == "--build":
                    package = project / "build_out/ops_repo.run"
                    package.parent.mkdir(parents=True)
                    package.write_text(soc)

            with patch.object(ops, "ROOT", self.root), \
                    patch.object(ops, "generate", return_value=project), \
                    patch.object(ops, "environment_checks", return_value=[]), \
                    patch.object(ops, "cann_root", return_value=self.root / "toolkit"), \
                    patch.object(ops, "run", side_effect=fake_cmake):
                ops.build(self.arguments(soc=soc))
            with patch.object(ops, "ROOT", self.root), patch.object(ops, "run") as run:
                ops.install(argparse.Namespace(soc=soc))
                installs.append(run.call_args.args[0])
        for soc, command in zip(ops.SOC_TARGETS, installs):
            package = self.root / "dist" / soc / "ops_repo.run"
            self.assertEqual(package.read_text(), soc)
            self.assertEqual(command, ["bash", package,
                                      f"--install-path={self.root / '.local' / soc / 'opp'}"])
            receipt = json.loads((package.parent / "build.json").read_text())
            self.assertEqual(receipt["soc"], soc)
            self.assertEqual(receipt["sha256"], hashlib.sha256(soc.encode()).hexdigest())

    def test_install_rejects_wrong_target_vendor_and_modified_package(self):
        dist = self.root / "dist/ascend950"
        dist.mkdir(parents=True)
        package = dist / "ops_repo.run"
        package.write_bytes(b"original")
        valid = {"vendor": "ops_repo", "soc": "ascend950", "package": package.name,
                 "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
        invalid_receipts = [dict(valid, soc="ascend910b"), dict(valid, vendor="other_vendor"),
                            dict(valid, package="../escape.run"), dict(valid, sha256="wrong")]
        with patch.object(ops, "ROOT", self.root), patch.object(ops, "run") as run:
            for receipt in invalid_receipts:
                ops.write_json(dist / "build.json", receipt)
                with self.subTest(receipt=receipt), self.assertRaises(ops.RepoError):
                    ops.install(argparse.Namespace(soc="ascend950"))
            run.assert_not_called()
        self.assertFalse((self.root / ".local").exists())

    def test_target_subdirectory_symlinks_are_rejected(self):
        external = self.root / "external"
        external.mkdir()
        for folder in ("build", "dist", ".local"):
            parent = self.root / folder
            parent.mkdir()
            (parent / "ascend950").symlink_to(external, target_is_directory=True)
            with self.subTest(folder=folder), self.assertRaises(ops.RepoError):
                ops.managed_directory(self.root, folder, "ascend950", "nested", create=True)
        self.assertEqual(list(external.iterdir()), [])

    def test_grouped_discovery_is_recursive_and_generation_uses_all_groups(self):
        self.add_operator("second_op", category="elementwise")
        self.add_operator("first_op", category="gdn/chunk/preprocess")
        operators = ops.load_operators(self.root)
        self.assertEqual([op["name"] for op in operators], ["first_op", "second_op"])
        self.assertEqual([op["category"] for op in operators], ["gdn/chunk/preprocess", "elementwise"])
        project = ops.generate(self.arguments(), root=self.root)
        receipt = json.loads((project / "opsrepo-build.json").read_text())
        self.assertEqual(receipt["operators"], ["first_op", "second_op"])
        for name in ("first_op", "second_op"):
            self.assertTrue((project / "op_host" / (name + ".cpp")).is_file())
            self.assertTrue((project / "op_kernel" / (name + ".cpp")).is_file())

    def test_operator_names_are_unique_across_categories(self):
        original = self.add_operator("shared_name", category="elementwise")
        original_metadata = (original / "operator.json").read_bytes()
        with self.assertRaisesRegex(ops.RepoError, "不同分组不能重名"):
            ops.new_operator(argparse.Namespace(name="shared_name", category="gdn/internal"), root=self.root)
        self.assertFalse((self.root / "ops/ascendc/gdn").exists())
        self.assertEqual((original / "operator.json").read_bytes(), original_metadata)
        duplicate = self.root / "ops/ascendc/gdn/internal/shared_name"
        shutil.copytree(original, duplicate)
        with self.assertRaisesRegex(ops.RepoError, "所有分组中唯一"):
            ops.load_operators(self.root)

    def test_legacy_and_other_backend_manifests_are_not_discovered(self):
        directory = self.add_operator("legacy_op")
        legacy = self.root / "operators/legacy_op"
        legacy.parent.mkdir()
        shutil.move(str(directory), legacy)
        shutil.copytree(legacy, self.root / "ops/triton/elementwise/triton_op")
        with self.assertRaisesRegex(ops.RepoError, "ops/ascendc"):
            ops.load_operators(self.root)
        self.add_operator("current_op", category="elementwise")
        self.assertEqual([op["name"] for op in ops.load_operators(self.root)], ["current_op"])

    def test_invalid_category_cannot_escape_or_create_directories(self):
        for category in ("", ".", "..", "../outside", "a/../b", "/absolute", "a//b", "a/", "./a",
                         "a/./b", "Upper", "a-b", "a\\b", "a__b", None):
            with self.subTest(category=category), self.assertRaises(ops.RepoError):
                ops.new_operator(argparse.Namespace(name="test_add", category=category), root=self.root)
        self.assertFalse((self.root / "ops").exists())

    def test_cli_accepts_nested_category_and_records_no_hardware_validation(self):
        args = ops.parser().parse_args([
            "new", "nested_op", "--category", "gdn/internal_ops", "--soc", "ascend910_93",
        ])
        ops.new_operator(args, root=self.root)
        directory = self.root / "ops/ascendc/gdn/internal_ops/nested_op"
        operator, = ops.load_operators(self.root)
        self.assertEqual(operator["directory"], directory)
        self.assertEqual(operator["category"], "gdn/internal_ops")
        self.assertEqual(operator["soc"], ["ascend910_93"])
        self.assertEqual(operator["hardware_validated"], [])
        self.assertFalse(operator["enabled"])
        readme = (directory / "README.md").read_text()
        self.assertIn("ops/ascendc/elementwise/ops_repo_add/", readme)
        self.assertNotIn("../ops_repo_add/", readme)

    def test_source_roots_and_intermediate_category_links_are_rejected(self):
        external = self.root / "external_sources"
        external.mkdir()
        for index, relative in enumerate(("ops", "ops/ascendc", "ops/ascendc/gdn", "ops/ascendc/gdn/internal")):
            repository = self.root / f"case_{index}"
            link = repository / relative
            link.parent.mkdir(parents=True)
            link.symlink_to(external, target_is_directory=True)
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(ops.RepoError, "符号链接"):
                    ops.load_operators(repository)
                with self.assertRaisesRegex(ops.RepoError, "符号链接"):
                    ops.new_operator(argparse.Namespace(name="test_add", category="gdn/internal"), root=repository)
        self.assertEqual(list(external.iterdir()), [])

    def test_existing_manifests_require_valid_group_and_matching_operator_directory(self):
        directory = self.add_operator("test_add")
        invalid_group = self.root / "ops/ascendc/bad-group/test_add"
        shutil.copytree(directory, invalid_group)
        with self.assertRaisesRegex(ops.RepoError, "category"):
            ops.load_operators(self.root)
        shutil.rmtree(invalid_group.parent)
        misplaced = self.root / "ops/ascendc/test_add"
        shutil.move(str(directory), misplaced)
        with self.assertRaisesRegex(ops.RepoError, "<category>"):
            ops.load_operators(self.root)


if __name__ == "__main__":
    unittest.main()
