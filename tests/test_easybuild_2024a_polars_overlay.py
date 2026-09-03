"""Contracts for the source-bound EasyBuild 2024a Polars provider."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1] / "packaging/easybuild-2024a-polars-overlay"
SPACK_RUST_AUDIT = (
    Path(__file__).resolve().parents[1]
    / "packaging/spack-overlay/rust-source-audit.json"
)


class _Names(ast.NodeTransformer):
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.values:
            return ast.parse(repr(self.values[node.id]), mode="eval").body
        return node


def _recipe_path(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {
        "SOURCE_TAR_GZ": "source.tar.gz",
        "SYSTEM": {"name": "system", "version": ""},
    }
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        value = ast.literal_eval(_Names(values).visit(copy.deepcopy(node.value)))
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _recipe(name: str) -> dict[str, Any]:
    return _recipe_path(ROOT / "2024a" / name)


def test_manifest_binds_every_retained_byte_and_fail_closed_flags() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["catalogue_commit"] == "58e8b5a48767cbed1bf5669675d9638580d7259f"
    expected = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == expected
    for relative, digest in manifest["files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["robot_dry_run"] == "PASS"
    assert manifest["native_builds_executed"] is False
    assert manifest["full_voiage_graph"] is False
    assert manifest["upstream_submitted"] is False


def test_sources_and_cargo_lock_are_complete_and_exact() -> None:
    sources = json.loads((ROOT / "source-manifest.json").read_text())
    assert {(x["name"], x["version"], x["sha256"]) for x in sources["sources"]} == {
        (
            "polars",
            "1.42.1",
            "2fe94f3059334650bd850ae19a9c165dcd5d9cb12cd95ea04de2201662e70e8a",
        ),
        (
            "polars-runtime-32",
            "1.42.1",
            "4d4809e1c1b9a6611f6944f27b24abea902b5159e6b6fa262fd716e947af5afd",
        ),
    }
    assert all(
        x["download_hash_verified"] and x["bytes"] > 0 for x in sources["sources"]
    )
    compiler = sources["rust_compiler"]
    spack = json.loads(SPACK_RUST_AUDIT.read_text())["compiler"]
    for key in (
        "version",
        "reported_version",
        "source_commit",
        "url",
        "sha256",
        "bytes",
    ):
        assert compiler[key] == spack[key]
    assert compiler == {
        "version": "nightly-2026-04-01",
        "reported_version": "1.96.0-nightly (48cc71ee8 2026-03-31)",
        "source_commit": "48cc71ee88cd0f11217eced958b9930970da998b",
        "url": "https://static.rust-lang.org/dist/2026-04-01/rustc-nightly-src.tar.xz",
        "sha256": "9de9d7a01a3bd2ec3581fec866453061036914f6df153d0b25d5fdc54a28f035",
        "bytes": 240051588,
        "source_archive_verified_in": "packaging/spack-overlay/rust-source-audit.json",
    }
    assert sources["native_build_executed"] is False

    cargo = json.loads((ROOT / "cargo-source-audit.json").read_text())
    lock_path = ROOT / "evidence/polars-runtime-32-Cargo.lock"
    lock_bytes = lock_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    assert cargo["source"] == "polars-runtime-32-1.42.1/Cargo.lock"
    assert cargo["cargo_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()

    locked_registry = {
        (package["name"], package["version"], package["checksum"])
        for package in lock["package"]
        if package.get("source", "").startswith("registry+")
    }
    audited_registry = {
        (package["name"], package["version"], package["sha256"])
        for package in cargo["registry_packages"]
    }
    assert cargo["registry_count"] == len(cargo["registry_packages"]) == 530
    assert audited_registry == locked_registry
    assert all(
        package["filename"] == f"{package['name']}-{package['version']}.crate"
        and package["lock_checksum_verified"] is True
        for package in cargo["registry_packages"]
    )

    locked_git = {
        (package["name"], package["source"].rsplit("#", 1)[1])
        for package in lock["package"]
        if package.get("source", "").startswith("git+")
    }
    audited_git = {
        (package["name"], package["commit"]) for package in cargo["git_sources"]
    }
    assert cargo["git_count"] == len(cargo["git_sources"]) == 3
    assert audited_git == locked_git
    assert {package["name"]: package["url"] for package in cargo["git_sources"]} == {
        "color-backtrace": "https://github.com/orlp/color-backtrace/archive/bb62ccf1e9eb1f6b7af5f16acff1fd7151a876dd.tar.gz",
        "object_store": "https://github.com/kdn36/arrow-rs-object-store/archive/f50a6e5c564b2b5933eca15cd20ff9b5614374a1.tar.gz",
        "tikv-jemalloc-sys": "https://github.com/pola-rs/jemallocator/archive/0d683dfb157097e2075d5e0eaf25f71f514a7552.tar.gz",
    }
    assert all(
        package["archive_downloaded"] is True for package in cargo["git_sources"]
    )
    assert cargo["all_registry_lock_checksums_verified"] is True
    assert cargo["all_git_commits_archived"] is True
    assert all(
        x["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", x["sha256"])
        for x in cargo["registry_packages"] + cargo["git_sources"]
    )
    assert cargo["offline_metadata_executed"] is False
    assert cargo["native_build_executed"] is False

    runtime = _recipe("polars-runtime-32-1.42.1-GCCcore-13.3.0.eb")
    assert runtime["easyblock"] == "CargoPythonPackage"
    assert set(runtime["crates"]) == {
        (name, version) for name, version, _checksum in locked_registry
    }
    declared_checksums = {
        filename: digest
        for entry in runtime["checksums"]
        for filename, digest in entry.items()
    }
    expected_crate_checksums = [
        {f"{package['name']}-{package['version']}.tar.gz": package["sha256"]}
        for package in cargo["registry_packages"]
    ]
    assert runtime["checksums"][-len(locked_registry) :] == expected_crate_checksums
    assert {
        (
            package["name"],
            package["version"],
            declared_checksums[f"{package['name']}-{package['version']}.tar.gz"],
        )
        for package in cargo["registry_packages"]
    } == locked_registry
    declared_sources = {
        source["filename"]: source
        for source in runtime["sources"]
        if isinstance(source, dict)
    }
    for package in cargo["git_sources"]:
        source = declared_sources[package["filename"]]
        assert source["download_filename"] == f"{package['commit']}.tar.gz"
        assert declared_checksums[package["filename"]] == package["sha256"]
    patch = (
        ROOT / "2024a/polars-runtime-32-1.42.1_offline-git-sources.patch"
    ).read_text()
    assert "git =" in patch
    for path in (
        "../color-backtrace-bb62ccf1e9eb1f6b7af5f16acff1fd7151a876dd",
        "../arrow-rs-object-store-f50a6e5c564b2b5933eca15cd20ff9b5614374a1",
        "../jemallocator-0d683dfb157097e2075d5e0eaf25f71f514a7552/jemalloc-sys",
    ):
        assert f'path = "{path}"' in patch
    assert runtime["offline"] is True


def test_recipes_preserve_stable_and_dated_nightly_compiler_roles() -> None:
    nightly = _recipe("Rust-nightly-2026-04-01-GCCcore-13.3.0.eb")
    runtime = _recipe("polars-runtime-32-1.42.1-GCCcore-13.3.0.eb")
    polars = _recipe("polars-1.42.1-GCCcore-13.3.0.eb")
    assert nightly["easyblock"] == "EB_Rust"
    assert nightly["name"] == "Rust-nightly"
    assert nightly["version"] == "2026-04-01"
    assert nightly["channel"] == "nightly"
    assert (
        nightly["checksums"][0]
        == "9de9d7a01a3bd2ec3581fec866453061036914f6df153d0b25d5fdc54a28f035"
    )
    assert "rust.download-rustc=false" in nightly["configopts"]
    assert runtime["builddependencies"] == [
        ("Rust-nightly", "2026-04-01"),
        ("maturin", "1.13.1"),
    ]
    assert "prebuildopts" not in runtime
    assert "EBROOTRUSTMINNIGHTLY/bin/rustc" in runtime["preinstallopts"]
    assert "EBROOTRUSTMINNIGHTLY/bin/cargo" in runtime["preinstallopts"]
    assert ("Rust", "1.96.0") not in runtime["builddependencies"]
    assert polars["dependencies"] == [
        ("Python", "3.12.14"),
        ("polars-runtime-32", "1.42.1"),
    ]
    for recipe in (runtime, polars):
        assert recipe["pip_no_index"] is True
        assert recipe["download_dep_fail"] is True
        assert recipe["sanity_pip_check"] is True


def test_root_consumer_uses_the_recipe_declared_lowercase_module_name() -> None:
    root_recipe = _recipe_path(
        ROOT.parents[1] / "packaging/easybuild/voiage-2.2.0-foss-2024a.eb"
    )
    polars_recipe = _recipe("polars-1.42.1-GCCcore-13.3.0.eb")
    assert polars_recipe["name"] == "polars"
    assert ("polars", "1.42.1") in root_recipe["dependencies"]
    assert not any(name == "Polars" for name, *_ in root_recipe["dependencies"])


def test_provider_roles_and_robot_evidence_match_recipes() -> None:
    providers = json.loads((ROOT / "providers.json").read_text())
    assert providers["runtime_providers"] == {
        "polars": "1.42.1",
        "polars-runtime-32": "1.42.1",
    }
    assert providers["compiler_roles"] == {
        "polars-runtime-32": "Rust-nightly/2026-04-01-GCCcore-13.3.0",
        "maturin": "Rust/1.96.0-GCCcore-13.3.0",
    }
    assert providers["native_builds_executed"] is False
    assert providers["full_voiage_graph"] is False
    assert providers["upstream_submitted"] is False
    log = (ROOT / "evidence/robot.log").read_text()
    for module in (
        "Rust/1.96.0-GCCcore-13.3.0",
        "Rust-nightly/2026-04-01-GCCcore-13.3.0",
        "maturin/1.13.1-GCCcore-13.3.0",
        "polars-runtime-32/1.42.1-GCCcore-13.3.0",
        "polars/1.42.1-GCCcore-13.3.0",
    ):
        assert f"module: {module}" in log
    assert "Dry run: printing build status" in log
    for prefix in ("/Users/", "/Volumes/", "/var/folders/"):
        assert prefix not in log


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_builds_executed", True),
        ("full_voiage_graph", True),
        ("upstream_submitted", True),
    ],
)
def test_authority_flags_cannot_be_promoted_by_mutation(
    field: str, value: bool
) -> None:
    providers = json.loads((ROOT / "providers.json").read_text())
    providers[field] = value
    assert providers != json.loads((ROOT / "providers.json").read_text())
