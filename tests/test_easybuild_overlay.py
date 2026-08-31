"""Validate source-bound provider contracts for the partial EasyBuild backport."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1] / "packaging/easybuild-overlay"


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
        "SOURCELOWER_TAR_GZ": "source.tar.gz",
        "PYPI_SOURCE": "https://pypi.org/source/",
        "SYSTEM": {"name": "system", "version": ""},
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            value = ast.literal_eval(_Names(values).visit(copy.deepcopy(node.value)))
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def test_manifest_binds_all_recipe_patch_license_and_evidence_bytes() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert set(manifest["files"]) == {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert "LICENSE.easybuild" in manifest["files"]
    assert len(list((ROOT / "2024a").glob("*.patch"))) == 2
    assert manifest["full_voiage_ready"] is False
    assert manifest["native_python_or_scientific_builds_executed"] is False


def test_provider_map_matches_actual_extensions_and_verified_source_hashes() -> None:
    providers = json.loads((ROOT / "providers.json").read_text())["providers"]
    sources = json.loads((ROOT / "source-manifest.json").read_text())["sources"]
    by = {(canonicalize_name(s["name"]), s["version"]): s for s in sources}
    actual = {}
    for path in sorted((ROOT / "2024a").glob("*.eb")):
        recipe = _recipe(path)
        assert recipe["toolchain"]["version"] in {"2024a", "13.3.0"}
        for name, version, options in recipe.get("exts_list", []):
            normalized = canonicalize_name(name)
            assert normalized not in actual
            actual[normalized] = {
                "version": version,
                "recipe": str(path.relative_to(ROOT)),
            }
            source = by[normalized, version]
            assert source["download_hash_verified"]
            assert source["bytes"] > 0
            assert options["checksums"] == [source["sha256"]]
    assert providers == actual


def test_source_build_requirements_are_available_in_extension_order() -> None:
    sources = json.loads((ROOT / "source-manifest.json").read_text())["sources"]
    by = {(canonicalize_name(s["name"]), s["version"]): s for s in sources}
    recipes = {
        _recipe(path)["name"]: _recipe(path) for path in (ROOT / "2024a").glob("*.eb")
    }
    module_recipes = dict(recipes)
    module_recipes["hatchling"] = _recipe(
        ROOT / "catalogue-reference/hatchling-1.24.2-GCCcore-13.3.0.eb"
    )
    environment = {
        "python_version": "3.12",
        "python_full_version": "3.12.3",
        "extra": "",
    }
    for recipe in recipes.values():
        available = {}
        for dep in recipe.get("dependencies", []) + recipe.get("builddependencies", []):
            name, version = dep[:2]
            available[canonicalize_name(name)] = version
            if name in module_recipes:
                for ext_name, ext_version, _ in module_recipes[name].get(
                    "exts_list", []
                ):
                    available[canonicalize_name(ext_name)] = ext_version
        extensions = recipe.get("exts_list", [])
        if (
            not extensions
            and (canonicalize_name(recipe["name"]), recipe["version"]) in by
        ):
            extensions = [(recipe["name"], recipe["version"], {})]
        for name, version, _ in extensions:
            source = by[canonicalize_name(name), version]
            requirements = source.get("build_system", {}).get("requires", [])
            requirements += source.get("requires_dist") or []
            for text in requirements:
                requirement = Requirement(text)
                if requirement.marker and not requirement.marker.evaluate(environment):
                    continue
                dependency = canonicalize_name(requirement.name)
                assert dependency in available, (recipe["name"], name, text)
                assert requirement.specifier.contains(available[dependency]), (
                    name,
                    text,
                )
            available[canonicalize_name(name)] = version
    assert recipes["Python"]["exts_list"][6][1] == "7.1.0"  # dateutil requires <8


def test_scientific_versions_stay_within_immutable_release_requirements() -> None:
    providers = json.loads((ROOT / "providers.json").read_text())["providers"]
    for requirement in ["numpy>=2.2.6,<3", "scipy>=1.16.3,<1.17", "pandas>=1.3,<3"]:
        bound = Requirement(requirement)
        assert bound.specifier.contains(providers[bound.name]["version"])
    science = _recipe(next((ROOT / "2024a").glob("SciPy-bundle*.eb")))
    assert ("Meson", "1.5.2") in science["builddependencies"]
    assert science["pip_no_index"]
    assert science["download_dep_fail"]
    assert science["sanity_pip_check"]
    assert science["pip_ignore_installed"] is True
    hypothesis = _recipe(ROOT / "2024a/hypothesis-6.103.1-GCCcore-13.3.0.eb")
    assert hypothesis["dependencies"] == [
        ("Python", "3.12.3"),
        ("Voiage-Python-support", "2.2.0"),
    ]
    assert hypothesis["pip_ignore_installed"] is True
    options = {name: opts for name, _, opts in science["exts_list"]}
    assert options["numpy"]["ignore_test_result"] is False
    assert options["scipy"]["ignore_test_result"] is False
    assert options["scipy"]["enable_slow_tests"] is True
    assert "RUSTC_BOOTSTRAP" not in str(science)


def test_robot_and_source_smoke_are_distinct_from_native_build_evidence() -> None:
    log = (ROOT / "evidence/scientific-robot.log").read_text()
    assert "Dry run: printing build status" in log
    assert "SciPy-bundle/2026.09-gfbf-2024a-voiage-2.2.0" in log
    for path in (ROOT / "2024a").glob("*.eb"):
        assert path.name in log
    for prefix in ["/Users/", "/Volumes/", "/var/folders/"]:
        assert prefix not in log
    smoke = json.loads((ROOT / "evidence/support-source-smoke.json").read_text())
    assert smoke["pip_check"] == smoke["typer_click_help"] == "PASS"
    assert smoke["native_scientific_builds"] is False
    installed = {
        canonicalize_name(name): version
        for name, version in smoke["installed_versions"].items()
    }
    for entry in smoke["source_builds"]:
        assert installed[canonicalize_name(entry["name"])] == entry["version"]
    providers = json.loads((ROOT / "providers.json").read_text())
    assert providers["foss_2023a_prepared"] is False
    assert providers["whole_voiage_graph_resolved"] is False
    assert {"polars", "pyarrow", "pydantic", "jsonschema"}.issubset(
        providers["deferred_voiage_dependencies"]
    )


def test_scipy_developer_test_cli_has_explicit_source_built_helpers() -> None:
    smoke = json.loads((ROOT / "evidence/scipy-test-cli-smoke.json").read_text())
    providers = json.loads((ROOT / "providers.json").read_text())["providers"]
    sources = json.loads((ROOT / "source-manifest.json").read_text())["sources"]
    scipy = next(source for source in sources if source["name"] == "scipy")
    assert smoke["scipy_source_sha256"] == scipy["sha256"]
    assert [item["args"] for item in smoke["results"]] == [
        ["--help"],
        ["--no-build", "test", "--help"],
    ]
    assert all(item["exit_code"] == 0 for item in smoke["results"])
    for source in smoke["added_source_builds"]:
        assert (
            providers[canonicalize_name(source["name"])]["version"] == source["version"]
        )
    assert smoke["native_builds_executed"] is False
    assert smoke["pip_check"] == "PASS"
