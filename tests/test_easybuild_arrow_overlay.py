"""Check Arrow's actual source requirements against the isolated HPC providers."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
import tomllib
from typing import Any

from packaging.requirements import Requirement
from packaging.version import Version
import pytest

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "packaging/easybuild-2024a-arrow-overlay"
SOURCE = OVERLAY / "evidence/source"


def _recipe(name: str) -> dict[str, Any]:
    path = next((OVERLAY / "2024a").glob(f"{name}-*.eb"), None)
    assert path is not None, f"missing EasyBuild recipe for {name}"
    values: dict[str, Any] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.List):
                # The Rust provider uses EasyBuild's system-toolchain constant.
                value = ast.parse(
                    ast.unparse(value).replace(
                        "SYSTEM", "{'name': 'system', 'version': ''}"
                    ),
                    mode="eval",
                ).body
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            values[target.id] = ast.literal_eval(value)
    return values


def _check_arrow(arrow: dict[str, Any]) -> None:
    assert arrow["version"] == "25.0.1"
    assert arrow["toolchain"] == {"name": "gfbf", "version": "2024a"}
    assert arrow["checksums"] == [
        "43d5de0a581f43cf63a2c06b4dcf13b9ff6fcd800f023324596e5781093bc500"
    ]
    flags = dict(
        flag.removeprefix("-D").split("=", 1)
        for flag in shlex.split(arrow["configopts"])
    )
    assert flags["ARROW_DEPENDENCY_SOURCE"] == "SYSTEM"
    assert flags["CMAKE_CXX_STANDARD"] == "20"
    for feature in (
        "ACERO",
        "COMPUTE",
        "DATASET",
        "PARQUET",
        "CSV",
        "JSON",
        "FILESYSTEM",
        "IPC",
    ):
        assert flags[f"ARROW_{feature}"] == "ON"
    for feature in (
        "MIMALLOC",
        "JEMALLOC",
        "ORC",
        "S3",
        "GCS",
        "AZURE",
        "HDFS",
        "FLIGHT",
        "GANDIVA",
        "SUBSTRAIT",
    ):
        assert flags[f"ARROW_{feature}"] == "OFF"
    assert flags["FETCHCONTENT_FULLY_DISCONNECTED"] == "ON"
    deps = {
        item[0]: item[1] for item in arrow["builddependencies"] + arrow["dependencies"]
    }
    assert deps["Python"] == "3.12.14"
    assert deps["SciPy-bundle"] == "2026.09"
    assert "Python-bundle-PyPI" not in deps
    for name in (
        "Thrift",
        "xsimd",
        "RE2",
        "utf8proc",
        "RapidJSON",
        "libcst",
        "Voiage-Arrow-build-support",
    ):
        assert name in deps
    assert Version(deps["CMake"]) >= Version("3.25")
    required = re.search(
        r'resolve_dependency\(xsimd.*?REQUIRED_VERSION\s+"([^"]+)"',
        (SOURCE / "arrow-cpp-cmake_modules-ThirdpartyToolchain.cmake").read_text(),
        re.DOTALL,
    )
    assert required
    assert Version(deps["xsimd"]) >= Version(required[1])
    extension = arrow["exts_list"][0]
    assert extension[:2] == ("pyarrow", "25.0.1")
    options = extension[2]
    assert options["nosource"] is True
    assert options["pip_no_index"] is True
    assert options["download_dep_fail"] is True
    assert options["sanity_pip_check"] is True
    assert "sed " not in options["preinstallopts"]
    assert "numpy" not in options["preinstallopts"]
    assert "PYARROW_WITH_ACERO=1" in options["preinstallopts"]


def test_arrow_features_dependency_controls_and_actual_native_bounds() -> None:
    _check_arrow(_recipe("Arrow"))


@pytest.mark.parametrize(
    "feature", ["DATASET", "ACERO", "PARQUET", "CSV", "FILESYSTEM", "IPC"]
)
def test_required_interchange_feature_removal_is_rejected(feature: str) -> None:
    recipe = copy.deepcopy(_recipe("Arrow"))
    recipe["configopts"] = recipe["configopts"].replace(
        f"ARROW_{feature}=ON", f"ARROW_{feature}=OFF"
    )
    with pytest.raises(AssertionError):
        _check_arrow(recipe)


def test_automatic_downloads_old_xsimd_and_missing_backend_are_rejected() -> None:
    recipe = _recipe("Arrow")
    changed = copy.deepcopy(recipe)
    changed["configopts"] = changed["configopts"].replace(
        "ARROW_DEPENDENCY_SOURCE=SYSTEM", "ARROW_DEPENDENCY_SOURCE=AUTO"
    )
    with pytest.raises(AssertionError):
        _check_arrow(changed)
    changed = copy.deepcopy(recipe)
    changed["builddependencies"] = [
        item if item[0] != "xsimd" else ("xsimd", "8.1.0")
        for item in changed["builddependencies"]
    ]
    with pytest.raises(AssertionError):
        _check_arrow(changed)
    changed = copy.deepcopy(recipe)
    changed["builddependencies"] = [
        item for item in changed["builddependencies"] if item[0] != "libcst"
    ]
    with pytest.raises(AssertionError):
        _check_arrow(changed)


def test_pyarrow_source_requirements_have_compatible_separate_build_providers() -> None:
    project = tomllib.loads((SOURCE / "arrow-python-pyproject.toml").read_text())
    assert project["build-system"]["build-backend"] == "_build_backend"
    support = _recipe("Voiage-Arrow-build-support")
    available = {
        name.replace("_", "-").lower(): version
        for name, version, _ in support["exts_list"]
    }
    available.update(
        {
            "numpy": "2.2.6",
            "cython": _recipe("Cython")["version"],
            "libcst": _recipe("libcst")["version"],
        }
    )
    for text in project["build-system"]["requires"]:
        requirement = Requirement(text)
        assert requirement.specifier.contains(
            available[requirement.name.replace("_", "-").lower()]
        )
    assert available["trove-classifiers"] == "2026.6.1.19"
    assert support["pip_ignore_installed"] is True
    assert support["pip_no_index"] is True
    assert support["sanity_pip_check"] is True
    assert ("Voiage-Arrow-build-support", "25.0.1") in _recipe("Arrow")[
        "builddependencies"
    ]
    assert not any(
        dep[0] == "Voiage-Arrow-build-support"
        for dep in _recipe("Arrow")["dependencies"]
    )
    foundation = (
        ROOT / "packaging/easybuild-overlay/2024a/Python-3.12.14-GCCcore-13.3.0.eb"
    ).read_text()
    assert '"setuptools_scm", "7.1.0"' in foundation or '"7.1.0"' in foundation


def test_libcst_locked_registry_closure_matches_all_recipe_checksums() -> None:
    lock = tomllib.loads((SOURCE / "libcst-native-Cargo.lock").read_text())
    packages = sorted(
        (p for p in lock["package"] if "source" in p),
        key=lambda p: (p["name"], p["version"]),
    )
    assert len(packages) == 95
    assert all(
        p["source"] == "registry+https://github.com/rust-lang/crates.io-index"
        for p in packages
    )
    recipe = _recipe("libcst")
    assert recipe["easyblock"] == "CargoPythonPackage"
    assert recipe["offline"] is True
    assert recipe["crates"] == [(p["name"], p["version"]) for p in packages]
    assert recipe["checksums"][1:] == [p["checksum"] for p in packages]
    assert ("Rust", "1.96.0") in recipe["builddependencies"]
    assert _recipe("Rust")["channel"] == "stable"
    assert "rust.download-rustc=false" in _recipe("Rust")["configopts"]
    assert "build.vendor=true" in _recipe("Rust")["configopts"]


def test_archived_source_members_and_manifest_are_hash_bound() -> None:
    manifest = json.loads((OVERLAY / "manifest.json").read_text())
    assert manifest["native_arrow_build_executed"] is False
    assert manifest["full_voiage_ready"] is False
    assert set(manifest["files"]) == {
        str(p.relative_to(OVERLAY))
        for p in OVERLAY.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for name, digest in manifest["files"].items():
        path = (OVERLAY / name).resolve()
        assert path.is_relative_to(OVERLAY.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    members = json.loads((OVERLAY / "evidence/source-members.json").read_text())
    for name, member in members.items():
        assert (
            hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest()
            == member["sha256"]
        )
    sources = json.loads((OVERLAY / "source-manifest.json").read_text())["sources"]
    assert len(sources) == 116
    assert all(s["download_hash_verified"] and s["bytes"] > 0 for s in sources)


def test_installed_sanity_command_checks_values_and_all_promised_surfaces() -> None:
    command = shlex.split(_recipe("Arrow")["sanity_check_commands"][0])
    assert command[:3] == ["python", "-s", "-c"]
    ast.parse(command[3])
    for term in (
        "assert_array_equal",
        "pq.read_table",
        "csv.read_csv",
        "ipc.open_stream",
        "ds.dataset",
        "fs.LocalFileSystem",
        "acero.Declaration",
    ):
        assert term in command[3]
    assert "257" in command[3]


def test_source_build_backend_requirements_are_satisfied_in_extension_order() -> None:
    available = {
        "setuptools": "70.0.0",
        "wheel": "0.43.0",
        "packaging": "24.0",
        "calver": "2022.6.26",
        "hatchling": "1.24.2",
        "hatch-vcs": "0.4.0",
        "cython": "3.1.8",
        "setuptools-scm": "7.1.0",
        "flit-core": "3.12.0",
    }
    environment = {
        "python_version": "3.12",
        "python_full_version": "3.12.14",
        "extra": "",
    }
    records = json.loads((OVERLAY / "source-manifest.json").read_text())["sources"]
    sources = {s["name"].lower().replace("_", "-"): s for s in records}
    for name, version, _ in _recipe("Voiage-Arrow-build-support")["exts_list"]:
        key = name.lower().replace("_", "-")
        snapshot = SOURCE / f"{name}-pyproject.toml"
        requirements = []
        if snapshot.exists():
            requirements += tomllib.loads(snapshot.read_text())["build-system"][
                "requires"
            ]
        if key in sources:
            requirements += sources[key].get("requires_dist") or []
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = requirement.name.lower().replace("_", "-")
            assert dependency in available, (name, text)
            assert requirement.specifier.contains(available[dependency]), (name, text)
        available[key] = version
    assert available["setuptools-scm"] == "9.2.2"
