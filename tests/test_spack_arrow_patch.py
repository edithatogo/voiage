"""Verify the selected Arrow source patch and its explicit build closure."""

import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "packaging/spack-overlay"


def test_arrow_patch_sources_match_declared_recipe_versions() -> None:
    audit = json.loads((ROOT / "arrow-patch-source-audit.json").read_text())
    for name, source in audit["sources"].items():
        package = {"pyarrow": "py-pyarrow", "libcst": "py-libcst"}.get(name, name)
        tree = ast.parse((ROOT / "packages" / package / "package.py").read_text())
        versions = {
            node.args[0].value: keyword.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "version"
            for keyword in node.keywords
            if keyword.arg == "sha256"
        }
        assert versions[source["version"]] == source["sha256"]
        assert source["downloaded_digest_verified"]
    assert audit["sources"]["arrow"]["version"] == "25.0.1"
    assert audit["sources"]["pyarrow"]["version"] == "25.0.1"
    assert audit["sources"]["xsimd"]["version"] == "14.2.0"
    assert audit["upstream_fix"]["removed_sve128_unpack"]
    assert not audit["native_build_executed"]


def test_selected_graph_preserves_backend_edges_and_stable_xsimd() -> None:
    path = ROOT / "solver-logs/voiage.json"
    nodes = json.loads(path.read_text())["spec"]["nodes"]
    by_hash = {node["hash"]: node for node in nodes}
    by_name = {node["name"]: node for node in nodes}
    assert by_name["arrow"]["version"] == by_name["py-pyarrow"]["version"] == "25.0.1"
    assert by_name["xsimd"]["version"] == "14.2.0"
    audit = json.loads((ROOT / "arrow-patch-source-audit.json").read_text())
    assert (
        audit["qualified_dag_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for package, requirements in audit["concrete_build_requirements"].items():
        edges = {edge["name"]: edge for edge in by_name[package]["dependencies"]}
        assert len(requirements) == (5 if package == "py-pyarrow" else 4)
        for requirement in requirements:
            edge = edges[requirement["package"]]
            assert "build" in edge["parameters"]["deptypes"]
            assert by_hash[edge["hash"]]["version"] == requirement["version"]
    rust = next(
        edge for edge in by_name["py-libcst"]["dependencies"] if edge["name"] == "rust"
    )
    assert rust["parameters"]["deptypes"] == ["build"]
    assert by_hash[rust["hash"]]["version"] == "1.96.0"
    dated = {("rust", "nightly-2026-04-01"), ("rust-bootstrap", "beta-2026-03-05")}
    for node in nodes:
        assert (node["name"], node["version"]) in dated or re.fullmatch(
            r"\d+(?:[.-]\d+)*(?:p\d+|\.post\d+)?", node["version"]
        )


def test_historical_arrow_graph_is_not_rewritten_as_current() -> None:
    history = ROOT / "history/pre-arrow-patch-38b573b5"
    receipt = json.loads((history / "solver-receipt.json").read_text())
    assert (
        receipt["manifest_sha256"]
        == "6e5143ea9a0546d52f62988cc975ee2ed5fb17c9904a2564038dac261c9885ca"
    )
    assert (
        hashlib.sha256((history / "manifest.json").read_bytes()).hexdigest()
        == receipt["manifest_sha256"]
    )
    dag = history / receipt["concrete_dag"]["path"]
    assert (
        hashlib.sha256(dag.read_bytes()).hexdigest()
        == receipt["concrete_dag"]["sha256"]
    )
    versions = {
        node["name"]: node["version"]
        for node in json.loads(dag.read_text())["spec"]["nodes"]
    }
    assert versions["arrow"] == versions["py-pyarrow"] == "25.0.0"


def test_backend_proof_does_not_claim_a_native_source_build() -> None:
    audit = json.loads((ROOT / "arrow-patch-source-audit.json").read_text())
    assert {"--no-deps", "--no-index", "--no-build-isolation"} <= set(
        audit["builder_probe"]["pip_arguments"]
    )
    assert audit["builder_probe"]["arrow_patch_matches_only_declared_filter"]
    assert audit["builder_probe"]["xsimd_cmake_args_inherited"]
    probe = audit["backend_import_probe"]
    assert probe["backend_imported"]
    assert probe["libcst_parser_import_smoke"]
    assert not probe["native_source_build"]
    assert not probe["installed_arrow"]
    assert "wheels" in probe["provider_installation"]


def test_libcst_cargo_and_source_licenses_are_bound() -> None:
    audit = json.loads((ROOT / "arrow-patch-source-audit.json").read_text())
    closure = json.loads((ROOT / audit["libcst_cargo_source_closure"]).read_text())
    assert (
        closure["cargo_lock_sha256"] == audit["source_members"]["libcst_lock"]["sha256"]
    )
    assert closure["libcst_source_sha256"] == audit["sources"]["libcst"]["sha256"]
    assert len(closure["crate_archives"]) == closure["registry_archive_count"] == 95
    assert (
        len({(entry["name"], entry["version"]) for entry in closure["crate_archives"]})
        == 95
    )
    assert closure["offline_metadata_package_count"] == 97
    assert closure["archive_bytes_independently_verified"]
    assert not closure["native_source_build_executed"]
    recipe = (ROOT / "packages/py-libcst/package.py").read_text()
    assert 'license("MIT AND PSF-2.0 AND Apache-2.0", when="@1.8.6:")' in recipe
    assert 'depends_on("rust@1.70:1", when="@1.8.6:", type="build")' in recipe
