"""Validate source-bound provider contracts for the foss 2023a EasyBuild foundation."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1] / "packaging/easybuild-2023a-overlay"


class _Names(ast.NodeTransformer):
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.values:
            return ast.parse(repr(self.values[node.id]), mode="eval").body
        return node


def _recipe(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {
        "SOURCE_TGZ": "source.tgz",
        "SOURCE_TAR_GZ": "source.tar.gz",
        "SOURCELOWER_TAR_GZ": "source.tar.gz",
        "PYPI_SOURCE": "https://pypi.org/source/",
        "SYSTEM": {"name": "system", "version": ""},
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(
                    _Names(values).visit(copy.deepcopy(node.value))
                )
            except ValueError:
                # Native recipes have computed sanity paths and components.
                # Contract fields must remain statically inspectable.
                assert not any(
                    isinstance(t, ast.Name)
                    and t.id
                    in {
                        "name",
                        "version",
                        "toolchain",
                        "dependencies",
                        "builddependencies",
                        "exts_list",
                    }
                    for t in node.targets
                )
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def test_manifest_binds_every_owned_source_recipe_patch_and_receipt() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert not manifest["native_scientific_build"]
    assert not manifest["full_voiage_graph"]


def test_all_python_runtime_and_build_edges_use_current_patch() -> None:
    recipes = [_recipe(p) for p in (ROOT / "2023a").glob("*.eb")]
    assert len(recipes) == 20
    for recipe in recipes:
        assert recipe["toolchain"]["version"] in {"12.3.0", "2023a", ""}
        if recipe["toolchain"]["version"] == "":
            assert recipe["name"] == "OpenSSL"
        for dep in recipe.get("dependencies", []) + recipe.get("builddependencies", []):
            if dep[0] == "Python":
                assert dep[1] == "3.12.14"
        if recipe["name"] == "Python":
            assert recipe["version"] == "3.12.14"
    ninja = next(r for r in recipes if r["name"] == "Ninja")
    assert ("Python", "3.12.14") in ninja["builddependencies"]


def test_source_backends_and_runtime_requirements_follow_provider_order() -> None:
    sources = json.loads((ROOT / "source-manifest.json").read_text())["sources"]
    sources += json.loads((ROOT / "scientific-consumer-sources.json").read_text())[
        "sources"
    ]
    by = {(canonicalize_name(s["name"]), s["version"]): s for s in sources}
    recipes = {_recipe(p)["name"]: _recipe(p) for p in (ROOT / "2023a").glob("*.eb")}
    environment = {
        "python_version": "3.12",
        "python_full_version": "3.12.14",
        "extra": "",
    }
    for recipe in recipes.values():
        available = {}
        for dep in recipe.get("dependencies", []) + recipe.get("builddependencies", []):
            name, version = dep[:2]
            available[canonicalize_name(name)] = version
            for ext_name, ext_version, _ in recipes.get(name, {}).get("exts_list", []):
                available[canonicalize_name(ext_name)] = ext_version
        exts = recipe.get("exts_list", [])
        if not exts and (canonicalize_name(recipe["name"]), recipe["version"]) in by:
            exts = [(recipe["name"], recipe["version"], {})]
        for name, version, options in exts:
            source = by[canonicalize_name(name), version]
            if options:
                assert options["checksums"] == [source["sha256"]]
            requirements = list(source.get("build_system", {}).get("requires", []))
            requirements += source.get("requires_dist") or []
            for text in requirements:
                req = Requirement(text)
                if req.marker and not req.marker.evaluate(environment):
                    continue
                assert canonicalize_name(req.name) in available, (
                    recipe["name"],
                    name,
                    text,
                )
                assert req.specifier.contains(available[canonicalize_name(req.name)])
            available[canonicalize_name(name)] = version
    hatch = recipes["hatchling"]["exts_list"]
    names = [x[0] for x in hatch]
    assert (
        names.index("calver")
        < names.index("trove-classifiers")
        < names.index("hatchling")
    )
    assert not any(".whl" in str(x) for x in hatch)


def test_scientific_and_pybind_test_constraints_remain_strict() -> None:
    recipes = {_recipe(p)["name"]: _recipe(p) for p in (ROOT / "2023a").glob("*.eb")}
    science = recipes["SciPy-bundle"]
    options = {n: o for n, _, o in science["exts_list"]}
    for name in ["numpy", "scipy"]:
        assert options[name]["ignore_test_result"] is False
    assert options["scipy"]["enable_slow_tests"] is True
    assert ("Cython", "3.0.10") in science["builddependencies"]
    assert ("Meson", "1.5.2") in science["builddependencies"]
    assert ("pybind11", "2.13.6") in science["builddependencies"]
    pybind = recipes["pybind11"]
    for dep in [
        ("CMake", "3.26.3"),
        ("Ninja", "1.11.1"),
        ("Catch2", "2.13.9"),
        ("hypothesis", "6.103.1"),
    ]:
        assert dep in pybind["builddependencies"]
    assert ("Boost", "1.82.0") in pybind["dependencies"]
    assert not any(d[0] == "Python-bundle-PyPI" for d in pybind["builddependencies"])


def test_retained_robot_and_consumer_evidence_do_not_claim_native_build() -> None:
    log = (ROOT / "evidence/scientific-robot.log").read_text()
    consumer_log = (ROOT / "evidence/scientific-consumers-robot.log").read_text()
    for path in (ROOT / "2023a").glob("*.eb"):
        assert path.name in log or path.name in consumer_log
    edges = json.loads((ROOT / "evidence/native-bootstrap-edges.json").read_text())
    assert len(edges["updated_python_build_only_edges"]) == 4
    assert all(
        e["python_version"] == "3.12.14"
        for e in edges["updated_python_build_only_edges"]
    )
    python_modules = {
        line.split("(module: ", 1)[1].rstrip(")")
        for line in log.splitlines()
        if "(module: Python/" in line
    }
    assert python_modules == {"Python/3.12.14-GCCcore-12.3.0"}
    assert "Python/3.11" not in log
    assert "Python/3.12.3-" not in log
    assert "13.3.0" not in log
    assert "(module: OpenSSL/1.1)" not in log
    smoke = json.loads((ROOT / "evidence/backend-source-smoke.json").read_text())
    assert smoke["pip_check"] == "PASS"
    assert len(smoke["source_builds"]) == 9
    assert smoke["native_scientific_build"] is False
    assert all(x["download_hash_verified"] for x in smoke["source_builds"])


def test_python_patch_refresh_preserves_changes_and_requires_zero_fuzz() -> None:
    original = ROOT / "history/Python-3.11.5-custom-ctypes.patch"
    refreshed = ROOT / "2023a/Python-3.12.14-custom-ctypes.patch"

    def changed_lines(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text().splitlines()
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]

    assert changed_lines(original) == changed_lines(refreshed)
    receipt = json.loads((ROOT / "evidence/strict-patch-checks.json").read_text())
    assert len(receipt["checks"]) == 3
    for check in receipt["checks"]:
        assert check["exit_code"] == 0
        assert "--fuzz=0" in check["command"]
        assert (
            hashlib.sha256((ROOT / check["patch"]).read_bytes()).hexdigest()
            == check["sha256"]
        )
    assert receipt["native_build_executed"] is False


def test_native_interpreter_overrides_preserve_all_other_catalogue_bytes() -> None:
    for name in [
        "ICU-73.2-GCCcore-12.3.0.eb",
        "BLIS-0.9.0-GCC-12.3.0.eb",
        "OpenBLAS-0.3.23-GCC-12.3.0.eb",
        "FlexiBLAS-3.3.1-GCC-12.3.0.eb",
    ]:
        original = (ROOT / "catalogue-reference" / name).read_text()
        current = (ROOT / "2023a" / name).read_text()
        assert current == original.replace(
            "('Python', '3.11.3')", "('Python', '3.12.14')"
        )
    original = (ROOT / "catalogue-reference/OpenSSL-3.eb").read_text()
    current = (ROOT / "2023a/OpenSSL-3.eb").read_text()
    assert current == original.replace("'3.5.7'", "'3.5.8'").replace(
        "a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8",
        "a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2",
    )


def test_every_selected_source_hash_is_bound_by_an_active_recipe() -> None:
    source_hashes = {
        source["sha256"]
        for source in json.loads((ROOT / "source-manifest.json").read_text())["sources"]
    }
    source_hashes |= {
        source["sha256"]
        for source in json.loads(
            (ROOT / "scientific-consumer-sources.json").read_text()
        )["sources"]
    }
    recipe_text = "\n".join(path.read_text() for path in (ROOT / "2023a").glob("*.eb"))
    assert len(source_hashes) == 71
    for source_hash in source_hashes:
        assert source_hash in recipe_text
    patch_hashes = {
        hashlib.sha256(patch.read_bytes()).hexdigest()
        for patch in (ROOT / "2023a").glob("*.patch")
    }
    recipe_hashes = set(re.findall(r'["\']([0-9a-f]{64})["\']', recipe_text))
    assert recipe_hashes == source_hashes | patch_hashes


def test_openssl_consumers_change_only_the_wrapper_dependency() -> None:
    for name in [
        "cURL-8.0.1-GCCcore-12.3.0.eb",
        "libarchive-3.6.2-GCCcore-12.3.0.eb",
        "CMake-3.26.3-GCCcore-12.3.0.eb",
    ]:
        original = (ROOT / "catalogue-reference" / name).read_text()
        current = (ROOT / "2023a" / name).read_text()
        assert current == original.replace(
            "('OpenSSL', '1.1', '', SYSTEM)", "('OpenSSL', '3', '', SYSTEM)"
        )
