#!/usr/bin/env python3
"""Manage an independent AscendC repository using the installed CANN generator.

This CLI intentionally uses only the Python standard library. Generated CANN
files stay in build/; handwritten operator sources stay in ops/ascendc/.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = "ops_repo"
DEFAULT_SOC = "ascend910b"
SOC_TARGETS = {
    "ascend910b": {
        "label": "A2", "default_version": "Ascend910B1",
        "versions": ("Ascend910B1", "Ascend910B2", "Ascend910B2C", "Ascend910B3", "Ascend910B4", "Ascend910B4-1"),
    },
    "ascend910_93": {
        "label": "A3", "default_version": "Ascend910_93",
        "versions": ("Ascend910_93", "Ascend910_9391", "Ascend910_9392", "Ascend910_9381", "Ascend910_9382", "Ascend910_9372", "Ascend910_9362"),
    },
    "ascend950": {
        "label": "A5", "default_version": "Ascend950",
        "versions": ("Ascend950", "Ascend950PR", "Ascend950DT"),
    },
}
NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
OP_TYPE = re.compile(r"[A-Z][A-Za-z0-9]*\Z")


class RepoError(Exception):
    pass


def validate_soc(soc):
    if soc not in SOC_TARGETS:
        raise RepoError(f"未知目标芯片: {soc}；请选择 {', '.join(SOC_TARGETS)}")
    return soc


def resolve_target(args):
    soc = validate_soc(getattr(args, "soc", DEFAULT_SOC))
    config = SOC_TARGETS[soc]
    version = getattr(args, "soc_version", None) or config["default_version"]
    if version not in config["versions"]:
        raise RepoError(f"{version} 不属于 {soc}；可选型号: {', '.join(config['versions'])}")
    return soc, version


def managed_directory(root, *parts, create=False):
    """Keep generated artifacts within the repository, including all parents."""
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise RepoError(f"仓库目录不能是符号链接: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RepoError(f"无法读取 JSON {path}: {exc}") from exc


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snake_case(value):
    # Match msopgen's CamelCase-to-file-name convention, including acronyms.
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def category_parts(category):
    if not isinstance(category, str):
        raise RepoError("category 必须是 snake_case 分组名，可用 / 分隔多级分组")
    parts = category.split("/")
    if not parts or any(not NAME.fullmatch(part) for part in parts):
        raise RepoError("category 必须是 snake_case 分组名，可用 / 分隔；不允许绝对路径、空段、. 或 ..")
    return parts


def operator_manifests(root):
    source_root = managed_directory(root, "ops", "ascendc")
    # Check every entry, including directory links which rglob does not follow.
    # A linked category must fail explicitly instead of silently hiding operators.
    entries = sorted(source_root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise RepoError(f"AscendC 算子源码树不能包含符号链接: {path}")
    return source_root, [path for path in entries if path.name == "operator.json" and path.is_file()]


def load_operators(root=ROOT, allow_empty=False):
    operators = []
    seen_names = {}
    seen_types = set()
    source_root, manifests = operator_manifests(root)
    for path in manifests:
        relative = path.parent.relative_to(source_root)
        if len(relative.parts) < 2:
            raise RepoError(f"{path}: 算子必须放在 ops/ascendc/<category>/<name>/ 中")
        category = "/".join(relative.parts[:-1])
        category_parts(category)
        item = read_json(path)
        if not isinstance(item, dict):
            raise RepoError(f"{path}: 顶层必须是对象")
        name = item.get("name", "")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name != path.parent.name:
            raise RepoError(f"{path}: name 必须是与目录名一致的 snake_case 名称")
        if name in seen_names:
            raise RepoError(f"算子名必须在所有分组中唯一: {name} 同时位于 {seen_names[name]} 和 {path.parent}")
        seen_names[name] = path.parent
        if type(item.get("enabled")) is not bool:
            raise RepoError(f"{path}: enabled 必须显式设置为 true 或 false")
        if not isinstance(item.get("soc"), list) or not item["soc"] or any(
            not isinstance(s, str) for s in item["soc"]
        ):
            raise RepoError(f"{path}: soc 必须为非空字符串列表")
        if len(set(item["soc"])) != len(item["soc"]) or any(s not in SOC_TARGETS for s in item["soc"]):
            raise RepoError(f"{path}: soc 不能重复且必须来自 {', '.join(SOC_TARGETS)}")
        validated = item.get("hardware_validated", [])
        if (not isinstance(validated, list) or any(not isinstance(s, str) or s not in item["soc"] for s in validated)
                or len(set(validated)) != len(validated)):
            raise RepoError(f"{path}: hardware_validated 必须为 soc 的无重复子集，只登记已完成上机验收的目标")
        item["hardware_validated"] = validated
        schema = item.get("schema")
        if not isinstance(schema, dict):
            raise RepoError(f"{path}: 缺少 schema 对象")
        op_type = schema.get("op", "")
        if not isinstance(op_type, str) or not OP_TYPE.fullmatch(op_type):
            raise RepoError(f"{path}: schema.op 必须是 PascalCase 名称")
        if snake_case(op_type) != name:
            raise RepoError(f"{path}: schema.op 与 name 的文件命名不一致")
        if op_type in seen_types:
            raise RepoError(f"重复的算子类型: {op_type}")
        seen_types.add(op_type)
        for field in ("input_desc", "output_desc"):
            descs = schema.get(field)
            if not isinstance(descs, list) or not descs:
                raise RepoError(f"{path}: {field} 必须为非空列表")
            names = set()
            for desc in descs:
                if not isinstance(desc, dict) or not isinstance(desc.get("name"), str):
                    raise RepoError(f"{path}: {field} 含无效参数")
                if not NAME.fullmatch(desc["name"]) or desc["name"] in names:
                    raise RepoError(f"{path}: 参数名无效或重复: {desc['name']}")
                names.add(desc["name"])
                types, formats = desc.get("type"), desc.get("format")
                if (not isinstance(types, list) or not types or
                        not isinstance(formats, list) or len(types) != len(formats) or
                        any(not isinstance(v, str) for v in types + formats)):
                    raise RepoError(f"{path}: {desc['name']} 的 type/format 列表必须等长非空")
                if desc.get("param_type") not in ("required", "optional", "dynamic"):
                    raise RepoError(f"{path}: {desc['name']} 的 param_type 无效")
        for relative in (f"op_host/{name}.cpp", f"op_kernel/{name}.cpp"):
            if not (path.parent / relative).is_file():
                raise RepoError(f"缺少算子实现: {path.parent / relative}")
        for component in ("op_host", "op_kernel"):
            source = path.parent / component
            if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
                raise RepoError(f"算子源码目录不能包含符号链接: {source}")
        item["directory"] = path.parent
        item["category"] = category
        operators.append(item)
    if not operators and not allow_empty:
        raise RepoError(f"{source_root} 中没有算子；只扫描 ops/ascendc/<category>/<name>/operator.json")
    return sorted(operators, key=lambda op: op["name"])


def select_operators(operators, requested=None, soc=DEFAULT_SOC):
    validate_soc(soc)
    by_name = {op["name"]: op for op in operators}
    names = [n.strip() for n in requested.split(",")] if requested is not None else [
        op["name"] for op in operators if op["enabled"] and soc in op["soc"]
    ]
    if not names or any(not name for name in names):
        raise RepoError(f"没有为 {soc} 启用的算子，或 --ops 列表为空")
    if len(set(names)) != len(names):
        raise RepoError("--ops 中不能重复指定算子")
    result = []
    for name in names:
        if name not in by_name:
            raise RepoError(f"未知算子: {name}")
        op = by_name[name]
        if not op["enabled"]:
            raise RepoError(f"{name} 尚未启用；实现并检查后将 operator.json 的 enabled 改为 true")
        if soc not in op["soc"]:
            raise RepoError(f"{name} 未配置目标 {soc}")
        result.append(op)
    return result


def cann_candidates(explicit=None):
    if explicit:
        return [Path(explicit).expanduser().resolve()]
    values = [os.environ.get(key) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")]
    values += ["/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann"]
    return list(dict.fromkeys(Path(v).expanduser().resolve() for v in values if v))


def cann_root(explicit=None):
    for path in cann_candidates(explicit):
        if any((path / relative).is_file() for relative in (
            "include/acl/acl.h", "aarch64-linux/include/acl/acl.h", "x86_64-linux/include/acl/acl.h"
        )):
            return path
    return None


def find_msopgen(explicit=None, cann=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        binary = shutil.which("msopgen")
        if binary:
            candidates.append(Path(binary))
        for prefix in ([cann] if cann else cann_candidates()) + [Path("/usr/local/Ascend")]:
            candidates += [prefix / "python/site-packages/bin/msopgen", prefix / "bin/msopgen"]
    for path in candidates:
        if path.is_file():
            env = dict(os.environ)
            # MindStudio may be installed outside the active Python site-packages.
            if (path.parent.parent / "msopgen").is_dir():
                extra = str(path.parent.parent)
                env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            return path.resolve(), env
    raise RepoError("找不到 msopgen；source 匹配版本的 CANN set_env.sh，或传 --msopgen /path/to/msopgen")


def environment_checks(explicit=None, msopgen=None):
    root = cann_root(explicit)
    checks = [{"name": "Python >= 3.9", "ok": sys.version_info >= (3, 9), "detail": sys.version.split()[0]}]
    for binary in ("cmake", "g++", "make"):
        path = shutil.which(binary)
        checks.append({"name": binary, "ok": bool(path), "detail": path or "未安装/不在 PATH"})
    checks.append({"name": "CANN Toolkit headers", "ok": root is not None, "detail": str(root) if root else "未找到 acl/acl.h；驱动不等于 Toolkit"})
    compiler = None
    if root:
        compiler = next((p for p in [
            root / "compiler/ccec_compiler/bin/bisheng", root / "compiler/ccec_compiler/bin/ccec",
            root / "aarch64-linux/compiler/ccec_compiler/bin/bisheng",
            root / "x86_64-linux/compiler/ccec_compiler/bin/bisheng",
        ] if p.is_file()), None)
    checks.append({"name": "AscendC compiler", "ok": compiler is not None, "detail": str(compiler) if compiler else "未找到 CANN 编译器"})
    try:
        binary, env = find_msopgen(msopgen, root)
        result = subprocess.run([sys.executable, str(binary), "gen", "--help"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        checks.append({"name": "msopgen", "ok": result.returncode == 0, "detail": str(binary) if result.returncode == 0 else result.stdout.strip()})
    except (RepoError, OSError, subprocess.TimeoutExpired) as exc:
        checks.append({"name": "msopgen", "ok": False, "detail": str(exc)})
    return checks


def run(command, **kwargs):
    print("+ " + shlex.join(str(p) for p in command), flush=True)
    subprocess.run([str(p) for p in command], check=True, **kwargs)


def patch_presets(project, cann=None, soc=DEFAULT_SOC):
    validate_soc(soc)
    path = project / "CMakePresets.json"
    if not path.is_file():
        raise RepoError("本机 msopgen 未生成标准 CMakePresets.json；请使用与 CANN Toolkit 匹配的生成器")
    data = read_json(path)
    preset = next((p for p in data.get("configurePresets", []) if p.get("name") == "default"), None)
    if preset is None:
        raise RepoError("msopgen 工程缺少 default CMake preset")
    values = {
        "vendor_name": VENDOR, "ASCEND_COMPUTE_UNIT": soc,
        "ASCEND_PYTHON_EXECUTABLE": sys.executable,
        "ENABLE_TEST": False, "ENABLE_CROSS_COMPILE": False,
        "ENABLE_SOURCE_PACKAGE": False, "ENABLE_BINARY_PACKAGE": True,
    }
    if cann:
        values["ASCEND_CANN_PACKAGE_PATH"] = str(cann)
    cache = preset.setdefault("cacheVariables", {})
    for key, value in values.items():
        cache[key] = {"type": "BOOL" if type(value) is bool else "STRING", "value": value}
    write_json(path, data)
    return cache


def generate(args, root=ROOT):
    soc, soc_version = resolve_target(args)
    operators = select_operators(load_operators(root), args.ops, soc)
    cann = cann_root(args.cann_path)
    binary, env = find_msopgen(args.msopgen, cann)
    build_dir = managed_directory(root, "build", soc, create=True)
    # Stage separately: a failing generator must not destroy the previous build.
    with tempfile.TemporaryDirectory(prefix=".generate-", dir=build_dir) as temporary:
        staging = Path(temporary)
        project = staging / "cann_project"
        for index, op in enumerate(operators):
            spec = staging / f"{op['name']}.json"
            write_json(spec, [op["schema"]])
            command = [sys.executable, binary, "gen", "-i", spec, "-f", "tf", "-c", f"ai_core-{soc_version}", "-lan", "cpp", "-out", project]
            if index:
                command += ["-m", "1"]
            run(command, env=env, stdin=subprocess.DEVNULL)
        # The TF generator provides the standard OPP project. We only build ACLNN;
        # the framework plugin is deliberately excluded, with no TensorFlow dependency.
        shutil.rmtree(project / "framework", ignore_errors=True)
        for component in ("op_host", "op_kernel"):
            destination = project / component
            if not (destination / "CMakeLists.txt").is_file():
                raise RepoError(f"msopgen 工程缺少 {component}/CMakeLists.txt")
            for path in destination.iterdir():
                if path.is_file() and path.suffix in (".cpp", ".cc", ".h", ".hpp"):
                    path.unlink()
            for op in operators:
                source = op["directory"] / component
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        if path.name == "CMakeLists.txt" or path.is_symlink():
                            raise RepoError(f"算子目录不能覆盖构建文件或包含链接: {path}")
                        target = destination / path.relative_to(source)
                        if target.exists():
                            raise RepoError(f"多个算子文件重名: {component}/{path.relative_to(source)}；请加算子名前缀")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
        patch_presets(project, cann, soc)
        write_json(project / "opsrepo-build.json", {
            "vendor": VENDOR, "soc": soc, "soc_version": soc_version,
            "operators": [op["name"] for op in operators], "msopgen": str(binary),
            "hardware_validated": {op["name"]: soc in op["hardware_validated"] for op in operators},
        })
        destination = build_dir / "cann_project"
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise RepoError(f"{destination} 不能是符号链接")
            shutil.rmtree(destination)
        shutil.move(str(project), destination)
    print(f"已生成工程: {destination}")
    return destination


def build(args):
    soc, _ = resolve_target(args)
    checks = environment_checks(args.cann_path, args.msopgen)
    failures = [c for c in checks if not c["ok"]]
    if failures:
        raise RepoError("编译环境未就绪:\n" + "\n".join(f"  - {c['name']}: {c['detail']}" for c in failures))
    cann = cann_root(args.cann_path)
    project = generate(args)
    cache = patch_presets(project, cann, soc)
    native_build = project / "build_out"
    env = dict(os.environ)
    env["ASCEND_HOME_PATH"] = str(cann)
    env["ASCEND_CANN_PACKAGE_PATH"] = str(cann)
    options = []
    for name, entry in cache.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if type(value) is bool:
            value = "ON" if value else "OFF"
        value = str(value).replace("${sourceDir}", str(project))
        options.append(f"-D{name}={value}")
    run(["cmake", "-S", project, "-B", native_build, *options], env=env)
    run(["cmake", "--build", native_build, "--target", "binary", "package", "--parallel", str(args.jobs)], env=env)
    packages = {}
    for path in native_build.rglob("*.run"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        packages[(path.name, digest)] = path
    if len(packages) != 1:
        raise RepoError(f"期望本轮构建产生一个 OPP run 包，实际找到 {len(packages)} 个不同产物；请检查 {native_build}")
    (filename, digest), package = next(iter(packages.items()))
    dist = managed_directory(ROOT, "dist", soc, create=True)
    if (dist / filename).is_symlink() or (dist / "build.json").is_symlink():
        raise RepoError("构建产物和记录不能是符号链接")
    shutil.copy2(package, dist / filename)
    receipt = read_json(project / "opsrepo-build.json")
    receipt.update({"package": filename, "sha256": digest})
    write_json(dist / "build.json", receipt)
    print(f"构建完成: {dist / filename}\n安装到仓库本地目录: python3 scripts/ops.py install --soc {soc}")


def install(args):
    soc, _ = resolve_target(args)
    dist = managed_directory(ROOT, "dist", soc)
    receipt = read_json(dist / "build.json")
    if not isinstance(receipt, dict) or receipt.get("soc") != soc or receipt.get("vendor") != VENDOR:
        raise RepoError(f"构建记录的目标芯片或 vendor 不匹配，期望 {soc}/{VENDOR}")
    filename = receipt.get("package")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".run"):
        raise RepoError("构建记录中的 package 路径无效")
    package = dist / filename
    if package.is_symlink() or not package.is_file() or hashlib.sha256(package.read_bytes()).hexdigest() != receipt.get("sha256"):
        raise RepoError("安装包缺失或与本轮构建记录不一致，请重新构建")
    prefix = managed_directory(ROOT, ".local", soc, "opp", create=True)
    run(["bash", package, f"--install-path={prefix}"])
    print(f"在当前 shell 中执行:\nsource {shlex.quote(str(prefix / 'vendors' / VENDOR / 'bin/set_env.bash'))}")


def new_operator(args, root=ROOT):
    name = args.name
    if not NAME.fullmatch(name):
        raise RepoError("算子名必须是小写 snake_case，例如 fused_scale_add")
    op_type = "".join(part.capitalize() for part in name.split("_"))
    if snake_case(op_type) != name:
        raise RepoError("该名称无法无歧义地转换为 msopgen 文件名，请使用英文单词加下划线")
    targets = getattr(args, "soc", [DEFAULT_SOC])
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
        raise RepoError("新算子的 --soc 必须为无重复目标列表")
    for soc in targets:
        validate_soc(soc)
    groups = category_parts(getattr(args, "category", "custom"))
    for op in load_operators(root, allow_empty=True):
        if op["name"] == name:
            raise RepoError(f"算子名已存在于 {op['directory']}，不同分组不能重名；不会覆盖")
    parent = managed_directory(root, "ops", "ascendc", *groups)
    destination = parent / name
    if destination.exists() or destination.is_symlink():
        raise RepoError(f"目录已存在，不会覆盖: {destination}")
    destination.mkdir(parents=True)
    desc = lambda n: {"name": n, "param_type": "required", "format": ["ND"], "type": ["float"]}
    write_json(destination / "operator.json", {
        "name": name, "enabled": False, "soc": targets, "hardware_validated": [],
        "description": "TODO: 定义算子语义和支持范围；soc 是配置目标，不代表已上机验证",
        "schema": {"op": op_type, "input_desc": [desc("x"), desc("y")], "output_desc": [desc("z")], "attr": []},
    })
    for component in ("op_host", "op_kernel"):
        folder = destination / component
        folder.mkdir()
        (folder / f"{name}.cpp").write_text(
            f'// {op_type}: implement {component} after fixing the API contract.\n'
            f'#error "{name}: implement {component} before enabling this operator"\n', encoding="utf-8")
    (destination / "README.md").write_text(
        f"# {op_type}\n\n状态：未实现，默认不参与构建。\n\n"
        "先在 operator.json 固定输入、输出和属性，填写数学语义、shape、dtype、layout、"
        "广播规则、空输入与异常行为、SoC 和性能目标。soc 记录配置目标，"
        "hardware_validated 只在对应芯片完成编译和上机验收后登记。\n\n"
        "参考仓库根 README.md 的开发流程与 ops/ascendc/elementwise/ops_repo_add/ 示例"
        "实现 host/tiling/kernel，使用独立 CPU 实现定义预期结果。"
        "完成实现并检查后设置 enabled=true；新增原生调用精度/边界测试，再执行构建。\n",
        encoding="utf-8")
    print(f"已创建 {destination}（enabled=false，含明确的未实现编译错误，不会默默执行错误算法）")


def parser():
    result = argparse.ArgumentParser(description="A2/A3/A5 AscendC 自定义算子仓库工具")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出算子及启用状态")
    commands.add_parser("check", help="离线检查算子清单与源码组织")
    doctor = commands.add_parser("doctor", help="检查 CANN 编译环境，不要求 torch")
    doctor.add_argument("--cann-path")
    doctor.add_argument("--msopgen")
    doctor.add_argument("--json", action="store_true")
    new = commands.add_parser("new", help="创建默认禁用的算子骨架，不覆盖现有文件")
    new.add_argument("name")
    new.add_argument("--category", default="custom", help="源码分组，默认 custom；支持 gdn/preprocess 这类多级 snake_case 路径")
    new.add_argument("--soc", nargs="+", choices=SOC_TARGETS, default=[DEFAULT_SOC], help="配置一个或多个目标；默认 ascend910b，骨架仍保持禁用")
    for command in ("generate", "build"):
        sub = commands.add_parser(command, help="生成 CANN 工程" if command == "generate" else "生成并编译指定芯片的 OPP run 包")
        sub.add_argument("--soc", choices=SOC_TARGETS, default=DEFAULT_SOC, help="目标 compute unit；默认 ascend910b")
        sub.add_argument("--ops", help="逗号分隔的已启用算子；默认该目标的全部已启用算子")
        sub.add_argument("--cann-path", help="已安装的 CANN Toolkit 根目录")
        sub.add_argument("--msopgen", help="与 Toolkit 匹配的 msopgen 脚本")
        sub.add_argument("--soc-version", help="可选生成器型号，必须属于 --soc；默认 A2=Ascend910B1、A3=Ascend910_93、A5=Ascend950")
        if command == "build":
            sub.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    installer = commands.add_parser("install", help="校验并安装该目标最近一次 run 包到 .local/<soc>/opp")
    installer.add_argument("--soc", choices=SOC_TARGETS, default=DEFAULT_SOC)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("list", "check"):
            operators = load_operators()
            for op in operators:
                state = "enabled" if op["enabled"] else "disabled"
                validated = ','.join(op['hardware_validated']) or 'none'
                print(f"{op['name']:24} {op['schema']['op']:24} {state:8} category={op['category']} configured={','.join(op['soc'])} hardware_validated={validated}")
            if args.command == "check":
                print("清单与目录检查通过（不代表 NPU 编译或运行通过）")
        elif args.command == "doctor":
            checks = environment_checks(args.cann_path, args.msopgen)
            if args.json:
                print(json.dumps(checks, ensure_ascii=False, indent=2))
            else:
                for item in checks:
                    print(f"{'OK' if item['ok'] else 'MISSING':7} {item['name']}: {item['detail']}")
            return 0 if all(item["ok"] for item in checks) else 1
        elif args.command == "new":
            new_operator(args)
        elif args.command == "generate":
            generate(args)
        elif args.command == "build":
            if args.jobs < 1:
                raise RepoError("--jobs 必须大于 0")
            build(args)
        elif args.command == "install":
            install(args)
        return 0
    except (RepoError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
