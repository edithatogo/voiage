"""Source identity and dependency-boundary checks for local Spack overlays."""

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / "packaging/spack-overlay"
EXPECTED_BUILTIN_LICENSES = {
    "expat": ("Expat", "MIT"),
    "openssl": ("Openssl", "Apache-2.0"),
    "python": ("Python", "0BSD"),
    "xsimd": ("Xsimd", "BSD-3-Clause"),
}


def test_overlay_recipe_manifest_binds_every_package() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    recipes = sorted((ROOT / "packages").glob("*/package.py"))
    assert manifest["recipe_sha256"] == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in recipes
    }
    assert manifest["patch_sha256"] == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "packages").rglob("*.patch"))
    }
    for relative_path, expected_sha256 in manifest["support_sha256"].items():
        support_path = ROOT / relative_path
        assert support_path.is_file()
        assert hashlib.sha256(support_path.read_bytes()).hexdigest() == expected_sha256
    catalogue_support = {
        path
        for path in manifest["support_sha256"]
        if path.startswith("catalogue-license-sources/")
    }
    assert catalogue_support == {
        f"catalogue-license-sources/{path.name}"
        for path in (ROOT / "catalogue-license-sources").iterdir()
        if path.is_file()
    }
    assert "packages/py-pyarrow/for_aarch64.patch" in manifest["patch_sha256"]
    for path in recipes:
        ast.parse(path.read_text())
    assert not manifest["native_builds_executed"]
    assert not manifest["upstream_submitted"]


def test_solver_receipt_binds_inspectable_logs_and_package_manifest() -> None:
    receipt = json.loads((ROOT / "solver-receipt.json").read_text())
    forbidden_prefixes = (
        "/Users/",
        "/Volumes/",
        "/private/",
        "/tmp/",  # noqa: S108 - forbidden evidence text, not a filesystem operation
    )

    def assert_no_private_prefix(path: Path) -> None:
        text = path.read_text()
        for private_prefix in forbidden_prefixes:
            assert private_prefix not in text

    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
    )
    for result in receipt["results"]:
        path = (ROOT / result["log_path"]).resolve()
        assert path.is_relative_to((ROOT / "solver-logs").resolve())
        assert result["log_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        text = path.read_text()
        assert_no_private_prefix(path)
        if result["exit_code"] == 0:
            assert result["status"] == "concretized"
            assert result["spec"] in text
            assert "Error: failed to concretize" not in text
            if result["spec"] == "py-pyarrow@25.0.0":
                arrow = next(
                    line for line in text.splitlines() if "^arrow@25.0.0" in line
                )
                for feature in ["python", "csv", "dataset", "filesystem", "parquet"]:
                    assert f"+{feature}" in arrow
        else:
            assert result["status"] == "failed"
            assert "Error: failed to concretize" in text
        concretize_log = (ROOT / result["concretize_log_path"]).resolve()
        assert concretize_log.is_relative_to((ROOT / "solver-logs").resolve())
        assert (
            result["concretize_log_sha256"]
            == hashlib.sha256(concretize_log.read_bytes()).hexdigest()
        )
        assert_no_private_prefix(concretize_log)
    lock = (ROOT / receipt["concrete_lock"]["path"]).resolve()
    dag = (ROOT / receipt["concrete_dag"]["path"]).resolve()
    assert lock.is_relative_to((ROOT / "solver-logs").resolve())
    assert dag.is_relative_to((ROOT / "solver-logs").resolve())
    assert (
        receipt["concrete_lock"]["sha256"]
        == hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    assert (
        receipt["concrete_dag"]["sha256"]
        == hashlib.sha256(dag.read_bytes()).hexdigest()
    )
    assert_no_private_prefix(lock)
    assert_no_private_prefix(dag)
    nodes = json.loads(dag.read_text())["spec"]["nodes"]
    assert len(nodes) == receipt["concrete_dag"]["nodes"] == 154
    assert {tuple(node["arch"].values()) for node in nodes} == {
        ("linux", "ubuntu24.04", "aarch64")
    }
    versions = {node["name"]: node["version"] for node in nodes}
    assert versions["py-voiage"] == "2.2.0"
    assert {
        name: versions[name] for name in receipt["concrete_dag"]["security_floor"]
    } == receipt["concrete_dag"]["security_floor"]
    assert {
        name: versions[name] for name in receipt["concrete_dag"]["arrow_patch"]
    } == receipt["concrete_dag"]["arrow_patch"]
    duplicate_counts = {
        name: count
        for name, count in Counter(node["name"] for node in nodes).items()
        if count > 1
    }
    assert duplicate_counts == receipt["concrete_dag"]["duplicated_build_tools"]
    concretization = receipt["concretization"]
    assert concretization["mode"] == "fresh_isolated"
    assert concretization["overlay_manifest_sha256"] == receipt["manifest_sha256"]
    assert (
        concretization["xsimd_recipe_sha256"]
        == hashlib.sha256((ROOT / "packages/xsimd/package.py").read_bytes()).hexdigest()
    )
    for forbidden_reuse in (
        "previous_lock_reused",
        "previous_dag_reused",
        "native_store_reused",
        "source_install_executed",
    ):
        assert concretization[forbidden_reuse] is False
    assert not receipt["builds_executed"]


def test_pre_xsimd_fix_solver_evidence_remains_historical() -> None:
    history = ROOT / "history/pre-xsimd-license-fix-12c57216"
    current = json.loads((ROOT / "solver-receipt.json").read_text())
    historical = json.loads((history / "solver-receipt.json").read_text())
    assert (ROOT / current["prior_failure_evidence"]).resolve() == (
        history / "solver-receipt.json"
    ).resolve()
    assert (
        historical["manifest_sha256"]
        == hashlib.sha256((history / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        historical["concrete_dag"]["sha256"]
        == hashlib.sha256(
            (history / historical["concrete_dag"]["path"]).read_bytes()
        ).hexdigest()
    )
    for result in historical["results"]:
        assert (
            result["log_sha256"]
            == hashlib.sha256((history / result["log_path"]).read_bytes()).hexdigest()
        )
    assert historical["concrete_dag"]["sha256"] != current["concrete_dag"]["sha256"]


def test_overlay_versions_match_verified_source_digests() -> None:
    audit = json.loads((ROOT / "source-audit.json").read_text())
    audit += json.loads((ROOT / "transitive-source-audit.json").read_text())
    for entry in audit:
        source = entry.get("sources", [entry])[0]
        assert source.get("download_hash_verified", source.get("hash_verified"))
        recipe = ROOT / "packages" / f"py-{entry['name']}" / "package.py"
        versions = {
            node.args[0].value: keyword.value.value
            for node in ast.walk(ast.parse(recipe.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "version"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            for keyword in node.keywords
            if keyword.arg == "sha256" and isinstance(keyword.value, ast.Constant)
        }
        assert versions[entry["version"]] == source["sha256"]
    arrow = json.loads((ROOT / "arrow-source-audit.json").read_text())
    assert arrow["sha256"] in (ROOT / "packages/arrow/package.py").read_text()
    assert arrow["sha512_verified_against"].endswith(".sha512")
    assert arrow["signature_verified"] is False


def test_native_dependency_requirements_are_not_downgraded() -> None:
    core = (ROOT / "packages/py-pydantic-core/package.py").read_text()
    assert 'depends_on("rust@1.88:1"' in core
    assert 'depends_on("py-maturin@1.10:1"' in core
    runtime = (ROOT / "packages/py-polars-runtime-32/package.py").read_text()
    assert 'depends_on("rust@nightly-2026-04-01"' in runtime
    polars = (ROOT / "packages/py-polars/package.py").read_text()
    assert 'depends_on("py-polars-runtime-32@1.42.1"' in polars
    assert '@when("@:1.29 ~nightly")' in polars
    arrow = (ROOT / "packages/py-pyarrow/package.py").read_text()
    assert 'depends_on("py-libcst@1.8.6:"' in arrow
    assert 'depends_on("py-scikit-build-core"' in arrow
    assert '"arrow@25.0.0+python+csv+dataset+filesystem+parquet"' in arrow


def _builtin_package_bases(tree: ast.Module) -> set[str]:
    """Return aliases that import package classes from Spack's builtin repo."""
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("spack_repo.builtin.packages.")
        for alias in node.names
    }


def _direct_unconditional_licenses(tree: ast.Module, class_name: str) -> list[str]:
    """Extract unconditional licence directives from one package class."""
    package_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    licenses: list[str] = []
    for statement in package_class.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id == "license"
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            continue
        licenses.append(call.args[0].value)
    return licenses


def _pinned_catalogue_licenses() -> dict[str, str]:
    """Read licence directives from hash-bound sources at the pinned commit."""
    source_root = ROOT / "catalogue-license-sources"
    evidence = json.loads((source_root / "evidence.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert evidence["catalogue_commit"] == manifest["catalogue_commit"]
    assert evidence["catalogue_repository"] == manifest["catalogue_repository"]
    assert set(evidence["sources"]) == set(EXPECTED_BUILTIN_LICENSES)

    licenses: dict[str, str] = {}
    for package_name, (
        class_name,
        expected_license,
    ) in EXPECTED_BUILTIN_LICENSES.items():
        source_record = evidence["sources"][package_name]
        assert source_record["upstream_path"] == (
            f"repos/spack_repo/builtin/packages/{package_name}/package.py"
        )
        donor_source = source_root / source_record["path"]
        assert donor_source.parent == source_root
        assert (
            hashlib.sha256(donor_source.read_bytes()).hexdigest()
            == (source_record["sha256"])
        )
        directives = _direct_unconditional_licenses(
            ast.parse(donor_source.read_text()), class_name
        )
        assert directives == [expected_license]
        licenses[package_name] = directives[0]
    return licenses


def test_pinned_catalogue_license_sources_are_bound() -> None:
    """Reject changed, removed or unexpectedly valued builtin directives."""
    assert _pinned_catalogue_licenses() == {
        package: license_value
        for package, (_, license_value) in EXPECTED_BUILTIN_LICENSES.items()
    }


def test_spack_directives_are_explicit_without_redeclaring_inherited_licenses() -> None:
    for recipe in (ROOT / "packages").glob("*/package.py"):
        tree = ast.parse(recipe.read_text())
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "spack.package"
            for alias in node.names
        }
        assert "*" not in names
        assert "depends_on" in names
        builtin_bases = _builtin_package_bases(tree)
        subclasses_builtin_package = any(
            isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id in builtin_bases
                for base in node.bases
            )
            for node in tree.body
        )
        if subclasses_builtin_package:
            # Thin version overlays inherit the builtin package's licence directives.
            # Repeating an unconditional licence raises when Spack's post-install
            # SBOM hook evaluates the lazy descriptor.
            assert "license" not in names
            assert not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "license"
                for node in ast.walk(tree)
            )
        else:
            # Python also defines a builtin named license; this is Spack's directive.
            assert "license" in names
            package_classes = [
                node for node in tree.body if isinstance(node, ast.ClassDef)
            ]
            assert len(package_classes) == 1
            assert any(
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "license"
                for statement in package_classes[0].body
            ), f"{recipe} must declare at least one class-level licence"


@pytest.mark.skipif(shutil.which("spack") is None, reason="Spack is not installed")
def test_installed_spack_evaluates_every_inherited_overlay_license(
    tmp_path: Path,
) -> None:
    """Exercise overlays with hash-bound catalogue licence directives."""
    probe = tmp_path / "probe.py"
    pinned_licenses = _pinned_catalogue_licenses()
    recipes = []
    for recipe in sorted((ROOT / "packages").glob("*/package.py")):
        tree = ast.parse(recipe.read_text())
        builtin_bases = _builtin_package_bases(tree)
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id in builtin_bases
                for base in node.bases
            )
        ]
        if classes:
            assert len(classes) == 1
            donor = next(
                node
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("spack_repo.builtin.packages.")
            )
            assert len(donor.names) == 1
            expected_class, _ = EXPECTED_BUILTIN_LICENSES[recipe.parent.name]
            assert donor.names[0].name == expected_class
            recipes.append(
                (
                    recipe,
                    classes[0].name,
                    donor.module,
                    donor.names[0].name,
                    pinned_licenses[recipe.parent.name],
                )
            )
    spack_executable = shutil.which("spack") or "spack"
    spack_version = subprocess.run(
        [spack_executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert (
        spack_version
        == json.loads((ROOT / "solver-receipt.json").read_text())["spack_version"]
    )
    spack_root = subprocess.run(
        [spack_executable, "location", "-r"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    probe.write_text(
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "import types\n"
        "from spack.directives import OverlappingLicenseError, _execute_license\n"
        "from spack.package import Package, license\n"
        "from spack.spec import Spec\n"
        f"recipes = {[(str(path), *fields) for path, *fields in recipes]!r}\n"
        "results = {}\n"
        "duplicate_errors = {}\n"
        "for index, (path, class_name, donor_module, donor_name, expected) in enumerate(recipes):\n"
        "    parts = donor_module.split('.')\n"
        "    for length in range(1, len(parts) + 1):\n"
        "        module_name = '.'.join(parts[:length])\n"
        "        if module_name not in sys.modules:\n"
        "            package_module = types.ModuleType(module_name)\n"
        "            package_module.__path__ = []\n"
        "            sys.modules[module_name] = package_module\n"
        "    donor = sys.modules[donor_module]\n"
        "    namespace = {'Package': Package, 'license': license}\n"
        "    exec(f'class {donor_name}(Package):\\n    license({expected!r})', namespace)\n"
        "    setattr(donor, donor_name, namespace[donor_name])\n"
        "    spec = importlib.util.spec_from_file_location(f'voiage_overlay_{index}', path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    overlay_class = getattr(module, class_name)\n"
        "    package = overlay_class(Spec(class_name.lower()))\n"
        "    results[class_name] = list(package.licenses.values())\n"
        "    duplicate_value = 'MIT' if expected != 'MIT' else 'Apache-2.0'\n"
        "    overlay_class.name = class_name.lower()\n"
        "    try:\n"
        "        _execute_license(overlay_class, duplicate_value, None)\n"
        "    except OverlappingLicenseError as error:\n"
        "        duplicate_errors[class_name] = str(error)\n"
        "    else:\n"
        "        raise AssertionError(f'duplicate licence accepted for {class_name}')\n"
        "print(json.dumps({'licenses': results, 'duplicate_errors': duplicate_errors}, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(spack_root) / "lib/spack"),
                    str(Path(spack_root) / "lib/spack/_vendoring"),
                ]
            ),
        },
    )
    assert completed.returncode == 0, completed.stderr
    probe_result = json.loads(completed.stdout)
    licenses = probe_result["licenses"]
    assert set(licenses) == {entry[1] for entry in recipes}
    expected_by_class = {entry[1]: entry[4] for entry in recipes}
    assert licenses == {name: [value] for name, value in expected_by_class.items()}
    assert set(probe_result["duplicate_errors"]) == set(expected_by_class)
    for message in probe_result["duplicate_errors"].values():
        assert "license" in message.lower()


def test_upstream_notices_remain_with_derived_recipes() -> None:
    for name in ["COPYRIGHT", "NOTICE", "LICENSE-APACHE", "LICENSE-MIT"]:
        assert (ROOT / name).read_text().strip()
    for name in ["py-pydantic", "py-polars", "py-pyarrow", "arrow"]:
        assert (
            "Copyright Spack Project Developers"
            in (ROOT / "packages" / name / "package.py").read_text()
        )
