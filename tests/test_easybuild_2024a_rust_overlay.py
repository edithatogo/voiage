"""Contracts for source-bound stable Rust and validation package candidates."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "packaging/easybuild-2024a-rust-overlay"


def read_json(path: str) -> dict | list:
    return json.loads((OVERLAY / path).read_text())


def recipes() -> dict[str, dict]:
    result = {}
    for path in (OVERLAY / "2024a").glob("*.eb"):
        values = {}
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            # Reject executable expressions while accepting the two EasyBuild constants.
            for child in ast.walk(node.value):
                if isinstance(child, ast.Name):
                    assert child.id in {"SYSTEM", "PYPI_SOURCE"}
            text = (
                ast.unparse(node.value)
                .replace("SYSTEM", "'SYSTEM'")
                .replace("PYPI_SOURCE", "'PYPI_SOURCE'")
            )
            values[node.targets[0].id] = ast.literal_eval(text)
        result[values["name"]] = values
    return result


def test_shared_source_qualification_is_exact_and_does_not_reuse_robot_or_native_claims() -> (
    None
):
    reuse = read_json("source-evidence-reuse.json")
    source = ROOT / reuse["source_overlay"]
    assert reuse["source_generation"] == "2023a"
    assert reuse["consumer_generation"] == "2024a"
    assert reuse["native_builds_executed"] is False
    assert "Robot and native EasyBuild outcomes are not reused" in reuse["reuse_scope"]
    assert "evidence/robot.log" not in reuse["files"]
    for name, digest in reuse["files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() == digest


def test_manifest_binds_every_artifact() -> None:
    manifest = read_json("manifest.json")
    actual = {
        str(path.relative_to(OVERLAY))
        for path in OVERLAY.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert actual == set(manifest["files"])
    for name, digest in manifest["files"].items():
        path = OVERLAY / name
        assert path.resolve().is_relative_to(OVERLAY.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["native_builds_executed"] is False
    assert manifest["full_voiage_graph"] is False


def test_build_tools_do_not_leak_into_runtime_dependencies() -> None:
    data = recipes()
    assert len(data) == 8
    tools = {"Rust", "maturin", "setuptools-rust", "Voiage-Rust-build-support"}
    for recipe in data.values():
        assert recipe["toolchain"] == {"name": "GCCcore", "version": "13.3.0"}
        assert not tools.intersection(dep[0] for dep in recipe["dependencies"])
        for dependency in recipe["dependencies"] + recipe["builddependencies"]:
            if dependency[0] == "Python":
                assert dependency[1] == "3.12.14"
    assert ("pydantic-core", "2.46.4") in data["pydantic"]["dependencies"]
    for name in ["maturin", "pydantic-core", "rpds-py"]:
        assert ("Rust", "1.96.0") in data[name]["builddependencies"]
        assert data[name]["offline"] is True
        assert data[name]["easyblock"] == "CargoPythonPackage"
    assert data["maturin"]["preinstallopts"] == "MATURIN_NO_INSTALL_RUST=1"
    assert all("modulename" not in recipe for recipe in data.values())


def test_actual_source_backend_and_runtime_requirements_have_providers() -> None:
    data = recipes()
    sources = {
        row["name"].replace("_", "-").lower(): row
        for row in read_json("python-source-audit.json")
    }
    baseline = {
        "setuptools": "70.0.0",
        "setuptools-scm": "7.1.0",
        "flit-core": "3.12.0",
        "wheel": "0.43.0",
        "packaging": "24.0",
    }
    support = {"typing-extensions": "4.16.0"}
    modules = {name: {name: recipe["version"]} for name, recipe in data.items()}
    for name, recipe in data.items():
        modules[name].update({ext[0]: ext[1] for ext in recipe.get("exts_list", [])})
    for recipe in data.values():
        available = baseline | support
        for dep in recipe["dependencies"] + recipe["builddependencies"]:
            available.update(modules.get(dep[0], {}))
        for name, version, _ in recipe.get("exts_list", []):
            row = sources[name]
            for requirement in (row.get("build_system") or {}).get("requires", []):
                req = Requirement(requirement)
                if req.marker and not req.marker.evaluate(
                    {"python_version": "3.12", "extra": ""}
                ):
                    continue
                key = req.name.replace("_", "-").lower()
                assert key in available, (name, key)
                assert Version(available[key]) in req.specifier, (
                    name,
                    requirement,
                    available[key],
                )
            for requirement in row.get("source_requires_dist", []):
                req = Requirement(requirement)
                if req.marker and not req.marker.evaluate(
                    {
                        "python_version": "3.12",
                        "extra": "",
                        "sys_platform": "linux",
                        "platform_system": "Linux",
                    }
                ):
                    continue
                key = req.name.replace("_", "-").lower()
                assert key in available, (name, key)
                assert Version(available[key]) in req.specifier, (name, requirement)
            available[name] = version


def test_cargo_recipes_bind_all_locked_registry_sources() -> None:
    audit = read_json("cargo-source-audit.json")
    sources = {(row["name"], row["version"]): row for row in audit["sources"]}
    assert len(sources) == 551
    data = recipes()
    python_sources = {row["name"]: row for row in read_json("python-source-audit.json")}
    for name, packages in audit["graphs"].items():
        lock_bytes = (OVERLAY / "evidence" / f"{name}-Cargo.lock").read_bytes()
        assert (
            hashlib.sha256(lock_bytes).hexdigest()
            == python_sources[name]["Cargo.lock_sha256"]
        )
        locked = tomllib.loads(lock_bytes.decode())["package"]
        assert [row for row in locked if "source" in row] == packages
        recipe = data[name]
        checksums = {
            name: digest for row in recipe["checksums"] for name, digest in row.items()
        }
        assert set(recipe["crates"]) == {
            (row["name"], row["version"]) for row in packages
        }
        for package in packages:
            key = package["name"], package["version"]
            source = sources[key]
            assert (
                source["sha256"] == package["checksum"] == checksums[source["filename"]]
            )
            assert source["archive_verified"] is True
            if source["rust_version"]:
                assert Version(source["rust_version"]) <= Version("1.96.0")


def test_rust_bootstrap_exception_and_source_configuration_are_explicit() -> None:
    rust = recipes()["Rust"]
    assert rust["channel"] == "stable"
    assert "rust.download-rustc=false" in rust["configopts"]
    assert "build.vendor=true" in rust["configopts"]
    audit = read_json("dispatch-audit.json")
    bootstrap = audit["bootstrap_exception"]
    assert bootstrap["compiler_version"] == "1.95.0"
    assert bootstrap["compiler_date"] == "2026-04-16"
    assert bootstrap["prebuilt_bootstrap_required"] is True
    stage0 = (OVERLAY / "evidence/rust-1.96-stage0.txt").read_bytes()
    assert hashlib.sha256(stage0).hexdigest() == bootstrap["stage0_source_sha256"]
    stage0_values = dict(
        line.split("=", 1)
        for line in stage0.decode().splitlines()
        if "=" in line and not line.startswith("#")
    )
    for component in bootstrap["archives"]:
        path = component["url"].removeprefix("https://static.rust-lang.org/")
        assert stage0_values[path] == component["sha256_from_verified_source"]
    assert len(bootstrap["archives"]) == 6
    assert all(row["archive_downloaded"] is False for row in bootstrap["archives"])
    assert audit["rust_source_config"]["download_ci_llvm"] is False
    assert audit["patch"]["strict_zero_fuzz_dry_run_exit_code"] == 0
    assert audit["patch"]["changed_lines_identical"] is True

    def changes(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text().splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]

    assert changes(
        OVERLAY / "2024a/Rust-1.96_sysroot-fix-interpreter.patch"
    ) == changes(OVERLAY / "history/Rust-1.70_sysroot-fix-interpreter.patch")


def test_observed_robot_offline_and_backend_scope() -> None:
    log = (OVERLAY / "evidence/robot.log").read_text()
    assert set(re.findall(r"\(module: Python/([^)]*)\)", log)) == {
        "3.12.14-GCCcore-13.3.0"
    }
    assert set(re.findall(r"\(module: Rust/([^)]*)\)", log)) == {
        "1.96.0-GCCcore-13.3.0"
    }
    assert "nightly" not in log
    offline = read_json("evidence/offline-metadata-receipt.json")
    assert [row["exit_code"] for row in offline["results"]] == [0, 0, 0]
    assert "metadata only" in offline["scope"]
    backend = read_json("evidence/backend-source-receipt.json")
    assert len(backend["results"]) == 11
    assert all(row["exit_code"] == 0 for row in backend["results"])
    rows = [
        json.loads(line)
        for line in (OVERLAY / "evidence/backend-module.log").read_text().splitlines()
    ]
    assert [row["setuptools"] for row in rows] == ["70.0.0", "84.0.0", "70.0.0"]
    assert [row["hatchling"] for row in rows] == ["1.24.2", "1.29.0", "1.24.2"]


def test_cargo_python_backends_and_retained_dags() -> None:
    import gzip

    sources = {row["name"]: row for row in read_json("python-source-audit.json")}
    available = {
        "setuptools": "84.0.0",
        "setuptools-rust": "1.12.0",
        "maturin": "1.13.1",
    }
    for name in ["maturin", "pydantic-core", "rpds-py"]:
        for text in sources[name]["build_system"]["requires"]:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate(
                {"python_version": "3.12", "extra": ""}
            ):
                continue
            assert Version(available[req.name]) in req.specifier
    for row in read_json("evidence/offline-dag-bindings.json"):
        path = OVERLAY / "evidence" / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        dag = json.loads(gzip.decompress(path.read_bytes()))
        assert len(dag["packages"]) == row["packages"]
        assert dag["resolve"]["nodes"]
    imports = read_json("evidence/backend-import.json")
    assert imports[0] == imports[2]
    for name in ["setuptools_file", "hatchling_file"]:
        assert "/backend-prefix/" in imports[1][name]
        assert "/backend-smoke-env/" in imports[0][name]
    config = read_json("evidence/rust-configure.json")
    assert config["exit_code"] == 0
    assert config["generated_config"]["rust"]["download-rustc"] is False
    assert config["generated_config"]["llvm"]["download-ci-llvm"] is False
    assert config["generated_config"]["build"]["vendor"] is True


def test_provider_inventory_separates_runtime_and_build_only_edges() -> None:
    providers = read_json("providers.json")
    assert providers["generation"] == "2024a"
    assert providers["python"] == "3.12.14"
    assert providers["rust"] == "1.96.0"
    assert providers["runtime_providers"] == {
        "pydantic-core": "2.46.4",
        "annotated-types": "0.7.0",
        "typing-inspection": "0.4.2",
        "pydantic": "2.13.4",
        "rpds-py": "2026.6.3",
        "attrs": "23.2.0",
        "referencing": "0.37.0",
        "jsonschema-specifications": "2025.9.1",
        "jsonschema": "4.26.0",
    }
    assert providers["build_only_providers"] == {
        "Rust": "1.96.0",
        "setuptools": "84.0.0",
        "packaging": "26.3",
        "hatchling": "1.29.0",
        "setuptools-rust": "1.12.0",
        "maturin": "1.13.1",
    }
    assert providers["native_builds_executed"] is False
    assert providers["full_voiage_graph"] is False


def test_robot_uses_only_the_2024a_generation_and_current_provider_versions() -> None:
    log = (OVERLAY / "evidence/robot.log").read_text()
    assert "GCCcore-12.3.0" not in log
    assert "gfbf-2023a" not in log
    for module in [
        "Python/3.12.14-GCCcore-13.3.0",
        "Rust/1.96.0-GCCcore-13.3.0",
        "pydantic-core/2.46.4-GCCcore-13.3.0",
        "pydantic/2.13.4-GCCcore-13.3.0",
        "rpds-py/2026.6.3-GCCcore-13.3.0",
        "jsonschema/4.26.0-GCCcore-13.3.0",
    ]:
        assert f"(module: {module})" in log
    assert "/Volumes/" not in log
    assert "/Users/" not in log
    assert "/var/folders/" not in log
