"""Validate source-bound provider contracts for the partial EasyBuild backport."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

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
        "SOURCE_TAR_GZ": "source.tar.gz",
        "SOURCELOWER_TAR_GZ": "source.tar.gz",
        "PYPI_SOURCE": "https://pypi.org/source/",
        "SYSTEM": {"name": "system", "version": ""},
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id in {"sanity_check_paths", "components"}
                for target in node.targets
            ):
                continue
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
    assert len(list((ROOT / "2024a").glob("*.patch"))) == 12
    assert manifest["full_voiage_ready"] is False
    assert manifest["native_python_or_scientific_builds_executed"] is False


def test_provider_map_matches_actual_extensions_and_verified_source_hashes() -> None:
    providers = json.loads((ROOT / "providers.json").read_text())["providers"]
    sources = json.loads((ROOT / "source-manifest-python31214.json").read_text())[
        "sources"
    ]
    sources += json.loads((ROOT / "scientific-consumer-sources.json").read_text())[
        "sources"
    ]
    by = {(canonicalize_name(s["name"]), s["version"]): s for s in sources}
    actual = {}
    for path in sorted((ROOT / "2024a").glob("*.eb")):
        recipe = _recipe(path)
        assert recipe["toolchain"]["version"] in {"2024a", "13.3.0", ""}
        if recipe["name"] == "Voiage-scientific-build-support":
            continue
        for name, version, options in recipe.get("exts_list", []):
            normalized = canonicalize_name(name)
            if normalized in actual:
                assert actual[normalized]["version"] == version
                actual[normalized]["recipes"].append(str(path.relative_to(ROOT)))
            else:
                actual[normalized] = {
                    "version": version,
                    "recipes": [str(path.relative_to(ROOT))],
                }
            source = by[normalized, version]
            assert source["download_hash_verified"]
            assert source["bytes"] > 0
            assert options["checksums"] == [source["sha256"]]
    assert providers == actual
    assert json.loads((ROOT / "providers.json").read_text())[
        "build_only_provider_overrides"
    ] == {
        "setuptools": {
            "version": "84.0.0",
            "recipes": [
                "2024a/Voiage-scientific-build-support-2.2.0-GCCcore-13.3.0.eb"
            ],
            "consumer": "2024a/Voiage-scientific-consumers-2.2.0-gfbf-2024a.eb",
        }
    }


def test_source_build_requirements_are_available_in_extension_order() -> None:
    sources = json.loads((ROOT / "source-manifest-python31214.json").read_text())[
        "sources"
    ]
    sources += json.loads((ROOT / "scientific-consumer-sources.json").read_text())[
        "sources"
    ]
    by = {(canonicalize_name(s["name"]), s["version"]): s for s in sources}
    recipes = {
        _recipe(path)["name"]: _recipe(path) for path in (ROOT / "2024a").glob("*.eb")
    }
    module_recipes = dict(recipes)
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
            if name in module_recipes:
                for ext_name, ext_version, _ in module_recipes[name].get(
                    "exts_list", []
                ):
                    available[canonicalize_name(ext_name)] = ext_version
        extensions = recipe.get("exts_list", [])
        if (
            not extensions
            and (
                recipe.get("easyblock") == "PythonPackage"
                or recipe["name"] == "pybind11"
            )
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
        ("Python", "3.12.14"),
        ("Voiage-Python-support", "2.2.0"),
    ]
    assert hypothesis["pip_ignore_installed"] is True
    options = {name: opts for name, _, opts in science["exts_list"]}
    assert options["numpy"]["ignore_test_result"] is False
    assert options["scipy"]["ignore_test_result"] is False
    assert options["scipy"]["enable_slow_tests"] is True
    assert "RUSTC_BOOTSTRAP" not in str(science)


def test_robot_and_source_smoke_are_distinct_from_native_build_evidence() -> None:
    log = (ROOT / "evidence/scientific-robot-python31214.log").read_text()
    consumer_log = (ROOT / "evidence/scientific-consumers-robot.log").read_text()
    assert "Dry run: printing build status" in log
    assert "SciPy-bundle/2026.09-gfbf-2024a-voiage-2.2.0" in log
    for path in (ROOT / "2024a").glob("*.eb"):
        assert path.name in log or path.name in consumer_log
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
    assert providers["deferred_voiage_dependencies"] == []
    assert providers["external_provider_overlays"] == {
        "pydantic": "packaging/easybuild-2024a-rust-overlay",
        "jsonschema": "packaging/easybuild-2024a-rust-overlay",
        "polars": "packaging/easybuild-2024a-polars-overlay",
        "pyarrow": "packaging/easybuild-2024a-arrow-overlay",
    }


def test_scipy_developer_test_cli_has_explicit_source_built_helpers() -> None:
    smoke = json.loads((ROOT / "evidence/scipy-test-cli-smoke.json").read_text())
    providers = json.loads((ROOT / "providers.json").read_text())["providers"]
    sources = json.loads((ROOT / "source-manifest-python31214.json").read_text())[
        "sources"
    ]
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


def _assert_current_python_closure(log: str) -> None:
    versions = re.findall(r"\(module: Python/([^ )]+)\)", log)
    assert versions == ["3.12.14-GCCcore-13.3.0"]


def test_robot_closure_rejects_a_second_older_build_only_python() -> None:
    log = (ROOT / "evidence/scientific-robot-python31214.log").read_text()
    _assert_current_python_closure(log)
    with pytest.raises(AssertionError):
        _assert_current_python_closure(
            log + "\n * [ ] old.eb (module: Python/3.12.3-GCCcore-13.3.0)"
        )
    for path in (ROOT / "2024a").glob("*.eb"):
        recipe = _recipe(path)
        for dependency in recipe.get("dependencies", []) + recipe.get(
            "builddependencies", []
        ):
            if dependency[0] == "Python":
                assert dependency[1] == "3.12.14"


def test_historical_receipts_and_numerical_test_controls_remain_unchanged() -> None:
    historical = json.loads((ROOT / "history/manifest-0b43545c.json").read_text())
    for path in ["source-manifest.json", "evidence/scientific-robot.log"]:
        assert (
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            == historical["files"][path]
        )
    assert hashlib.sha256((ROOT / "source-manifest.json").read_bytes()).hexdigest() == (
        "bfee658562db323e1af1c371017de5542121812254c52d7212e90ee4fc8596de"
    )
    for name in [
        "BLIS-1.0-GCC-13.3.0.eb",
        "OpenBLAS-0.3.27-GCC-13.3.0.eb",
        "ICU-75.1-GCCcore-13.3.0.eb",
        "FlexiBLAS-3.4.4-GCC-13.3.0.eb",
    ]:
        original = (ROOT / "catalogue-reference" / name).read_text()
        assert (ROOT / "2024a" / name).read_text() == original.replace(
            "('Python', '3.12.3')", "('Python', '3.12.14')"
        )


def test_ctypes_refresh_changes_context_only_and_security_sources_are_bound() -> None:
    original = ROOT / "2024a/Python-3.11.5-custom-ctypes.patch"
    refreshed = ROOT / "2024a/Python-3.12.14-custom-ctypes.patch"

    def edited_lines(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text().splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]

    assert edited_lines(original) == edited_lines(refreshed)
    receipt = json.loads((ROOT / "evidence/python31214-patch-refresh.json").read_text())
    assert (
        receipt["original_patch_sha256"]
        == hashlib.sha256(original.read_bytes()).hexdigest()
    )
    assert (
        receipt["derived_patch_sha256"]
        == hashlib.sha256(refreshed.read_bytes()).hexdigest()
    )
    assert receipt["added_removed_lines_identical"] is True
    assert receipt["patched_files_identical"] is True
    sources = json.loads((ROOT / "source-manifest-python31214.json").read_text())[
        "sources"
    ]
    python = next(source for source in sources if source["name"] == "Python")
    assert python["version"] == "3.12.14"
    assert (
        python["sha256"]
        == "6c6df908d2c3fd24e6d76869e92542abd0f33aec9dfc18df8875f89660286d43"
    )
    python_recipe = _recipe(ROOT / "2024a/Python-3.12.14-GCCcore-13.3.0.eb")
    assert python_recipe["checksums"][0] == {python["filename"]: python["sha256"]}
    assert python_recipe["patch_ctypes_ld_library_path"] == refreshed.name
    openssl = next(source for source in sources if source["name"] == "OpenSSL")
    assert openssl["version"] == "3.5.8"
    wrapper = (ROOT / "2024a/OpenSSL-3.eb").read_text()
    original_wrapper = (ROOT / "catalogue-reference/OpenSSL-3.eb").read_text()
    assert wrapper == original_wrapper.replace("'3.5.7'", "'3.5.8'").replace(
        "a8c0d28a529ca480f9f36cf5792e2cd21984552a3c8e4aa11a24aa31aeac98e8",
        openssl["sha256"],
    )


def _validate_pyarrow_provider_integration(providers: dict[str, object]) -> None:
    assert providers["deferred_voiage_dependencies"] == []
    assert "pyarrow" not in providers["providers"]
    overlays = providers["external_provider_overlays"]
    evidence = providers["external_provider_evidence"]
    assert isinstance(overlays, dict)
    assert isinstance(evidence, dict)
    assert overlays["pyarrow"] == "packaging/easybuild-2024a-arrow-overlay"
    assert evidence == {
        "pyarrow": {
            "version": "25.0.1",
            "manifest": "packaging/easybuild-2024a-arrow-overlay/manifest.json",
            "manifest_sha256": "de736380795902122bd0a25a1f05c7d07589f5e5b874bb2b539e9a20d5db7a66",
            "native_build_executed": False,
            "full_voiage_ready": False,
            "consumer_recipe": "packaging/easybuild/voiage-2.2.0-foss-2024a.eb",
            "consumer_recipe_sha256": "372e30dee9f40874514ac09c89c61d9cf3677f80edabda0cebb49955540b365f",
            "consumer_dependency": "Arrow/25.0.1",
        }
    }
    binding = evidence["pyarrow"]
    assert isinstance(binding, dict)
    manifest_path = ROOT.parents[1] / binding["manifest"]
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == binding["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["arrow_version"] == binding["version"]
    assert manifest["native_arrow_build_executed"] is False
    assert manifest["full_voiage_ready"] is False
    consumer_path = ROOT.parents[1] / binding["consumer_recipe"]
    assert (
        hashlib.sha256(consumer_path.read_bytes()).hexdigest()
        == binding["consumer_recipe_sha256"]
    )
    consumer = _recipe(consumer_path)
    assert ("Arrow", "25.0.1") in consumer["dependencies"]
    assert not any(name == "PyArrow" for name, *_ in consumer["dependencies"])
    assert binding["consumer_dependency"] == "Arrow/25.0.1"


def test_pyarrow_external_provider_is_bound_and_fail_closed() -> None:
    providers = json.loads((ROOT / "providers.json").read_text())
    _validate_pyarrow_provider_integration(providers)


@pytest.mark.parametrize(
    "mutation",
    [
        "deferred",
        "path",
        "digest",
        "native",
        "full",
        "version",
        "consumer_digest",
        "consumer_dependency",
    ],
)
def test_pyarrow_external_provider_rejects_mutated_evidence(mutation: str) -> None:
    providers = json.loads((ROOT / "providers.json").read_text())
    mutated = copy.deepcopy(providers)
    binding = mutated["external_provider_evidence"]["pyarrow"]
    if mutation == "deferred":
        mutated["deferred_voiage_dependencies"] = ["pyarrow"]
    elif mutation == "path":
        mutated["external_provider_overlays"]["pyarrow"] = "packaging/missing"
    elif mutation == "digest":
        binding["manifest_sha256"] = "0" * 64
    elif mutation == "native":
        binding["native_build_executed"] = True
    elif mutation == "full":
        binding["full_voiage_ready"] = True
    elif mutation == "consumer_digest":
        binding["consumer_recipe_sha256"] = "0" * 64
    elif mutation == "consumer_dependency":
        binding["consumer_dependency"] = "PyArrow/25.0.0"
    else:
        binding["version"] = "25.0.0"
    with pytest.raises((AssertionError, FileNotFoundError)):
        _validate_pyarrow_provider_integration(mutated)
