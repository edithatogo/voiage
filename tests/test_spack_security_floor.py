"""Security source and inherited build-system contracts for the Spack overlay."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "packaging/spack-overlay"


def test_security_sources_match_official_archive_digests() -> None:
    audit = json.loads((ROOT / "security-source-audit.json").read_text())
    expected = {"python": "3.12.14", "expat": "2.8.3", "openssl": "3.6.4"}
    assert {
        entry["package"]: entry["version"] for entry in audit["sources"]
    } == expected
    for entry in audit["sources"]:
        assert entry["archive_downloaded"]
        assert entry["published_digest_verified"]
        assert entry["bytes"] > 100000
        tree = ast.parse(
            (ROOT / "packages" / entry["package"] / "package.py").read_text()
        )
        declarations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "version"
        ]
        assert len(declarations) == 1
        declaration = declarations[0]
        assert declaration.args[0].value == entry["version"]
        assert {kw.arg: kw.value.value for kw in declaration.keywords}[
            "sha256"
        ] == entry["sha256"]
    assert not audit["native_builds_executed"]


def test_security_overlays_preserve_catalogue_methods_and_variants() -> None:
    for package, name in [
        ("python", "Python"),
        ("expat", "Expat"),
        ("openssl", "Openssl"),
    ]:
        tree = ast.parse((ROOT / "packages" / package / "package.py").read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        donor = next(
            node
            for node in imports
            if node.module == f"spack_repo.builtin.packages.{package}.package"
        )
        assert [(alias.name, alias.asname) for alias in donor.names] == [
            (name, f"Builtin{name}")
        ]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert len(classes) == 1
        assert [base.id for base in classes[0].bases] == [f"Builtin{name}"]
        # A local method, variant, phase assignment, or custom builder would replace
        # the upstream dispatch verified by the actual Spack probe.
        assert all(isinstance(node, ast.Expr) for node in classes[0].body)
        calls = [
            node.value for node in classes[0].body if isinstance(node.value, ast.Call)
        ]
        assert {node.func.id for node in calls} <= {"license", "version", "depends_on"}


def test_optional_parser_and_tls_floors_do_not_force_variants() -> None:
    tree = ast.parse((ROOT / "packages/python/package.py").read_text())
    dependencies = {
        node.args[0].value: {kw.arg: kw.value.value for kw in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "depends_on"
    }
    assert dependencies["openssl@3.6.4:"] == {"when": "@3.12.14: +ssl"}
    assert dependencies["expat@2.8.3:"] == {"when": "@3.12.14: +pyexpat"}
    recipe = (ROOT / "packages/py-voiage/package.py").read_text()
    assert 'depends_on("python@3.12.14:3.14", type=("build", "run"))' in recipe


def test_current_security_graph_and_historical_evidence_are_separate() -> None:
    receipt = json.loads((ROOT / "solver-receipt.json").read_text())
    dag = ROOT / receipt["concrete_dag"]["path"]
    assert (
        hashlib.sha256(dag.read_bytes()).hexdigest()
        == receipt["concrete_dag"]["sha256"]
    )
    nodes = json.loads(dag.read_text())["spec"]["nodes"]
    versions = {node["name"]: node["version"] for node in nodes}
    assert versions["python"] == "3.12.14"
    assert versions["expat"] == "2.8.3"
    assert versions["openssl"] == "3.6.4"
    history = ROOT / "history/pre-security-floor-b3f53c2b"
    old = json.loads((history / "solver-receipt.json").read_text())
    assert (
        hashlib.sha256((history / "manifest.json").read_bytes()).hexdigest()
        == old["manifest_sha256"]
    )
    assert (
        old["manifest_sha256"]
        == "0c1be98e5df3e661782bc901fd5dd404af8db0901ee98b9a70ed379c5f72b78a"
    )
    assert (
        hashlib.sha256((history / old["concrete_dag"]["path"]).read_bytes()).hexdigest()
        == old["concrete_dag"]["sha256"]
    )


def test_actual_spack_dispatch_evidence_retains_install_hooks() -> None:
    audit = json.loads((ROOT / "builder-dispatch-audit.json").read_text())
    for name, digest in audit["recipe_sha256"].items():
        assert (
            hashlib.sha256(
                (ROOT / "packages" / name / "package.py").read_bytes()
            ).hexdigest()
            == digest
        )
    methods = audit["method_identity"]
    assert methods["expat"]["AutotoolsBuilder"].endswith(
        "builtin.packages.expat.package.AutotoolsBuilder"
    )
    assert methods["expat"]["CMakeBuilder"].endswith(
        "builtin.packages.expat.package.CMakeBuilder"
    )
    callbacks = audit["concrete_phase_callbacks"]
    assert {"link_system_certs", "copy_mozilla_certs"} <= set(
        callbacks["openssl"]["phases"]["install"]["after"]
    )
    assert {"filter_compilers", "symlink", "install_python_gdb", "import_tests"} <= set(
        callbacks["python"]["phases"]["install"]["after"]
    )
    assert not audit["native_build_executed"]
