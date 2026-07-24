from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

import pytest

from scripts.compose_polyglot_sbom import (
    CYCLONEDX_SCHEMA,
    SbomError,
    compose_sbom,
    validate_sbom,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, content: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def python_sbom(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "python.cdx.json",
        {
            "$schema": CYCLONEDX_SCHEMA,
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "version": 1,
            "metadata": {
                "component": {
                    "bom-ref": "root-component",
                    "name": "voiage",
                    "type": "library",
                    "version": "0.0.0",
                },
                "properties": [{"name": "cdx:reproducible", "value": "true"}],
            },
            "components": [
                {
                    "bom-ref": "pkg:pypi/numpy@2.3.1",
                    "name": "numpy",
                    "purl": "pkg:pypi/numpy@2.3.1",
                    "type": "library",
                    "version": "2.3.1",
                }
            ],
            "dependencies": [
                {
                    "ref": "root-component",
                    "dependsOn": ["pkg:pypi/numpy@2.3.1"],
                },
                {"ref": "pkg:pypi/numpy@2.3.1"},
            ],
        },
    )


@pytest.fixture
def cargo_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "rust"
    _write(
        workspace / "Cargo.toml",
        """\
[workspace]
members = ["crates/voiage-core", "crates/voiage-ffi"]
resolver = "2"

[workspace.package]
version = "1.2.3"
license = "Apache-2.0"
repository = "https://github.com/edithatogo/voiage"
""",
    )
    _write(
        workspace / "crates/voiage-core/Cargo.toml",
        """\
[package]
name = "voiage-core"
version.workspace = true
license.workspace = true
repository.workspace = true
description = "Core fixture"

[dependencies]
serde = "1"
""",
    )
    _write(
        workspace / "crates/voiage-ffi/Cargo.toml",
        """\
[package]
name = "voiage-ffi"
version.workspace = true
license.workspace = true
repository.workspace = true
description = "FFI fixture"

[dependencies]
voiage-core = { path = "../voiage-core" }
""",
    )
    _write(
        workspace / "Cargo.lock",
        """\
version = 4

[[package]]
name = "serde"
version = "1.0.219"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "1111111111111111111111111111111111111111111111111111111111111111"

[[package]]
name = "voiage-core"
version = "1.2.3"
dependencies = ["serde"]

[[package]]
name = "voiage-ffi"
version = "1.2.3"
dependencies = ["voiage-core"]
""",
    )
    return workspace


@pytest.fixture
def r_description(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "DESCRIPTION",
        """\
Package: voiageR
Type: Package
Title: Fixture
Version: 1.2.3
Description: A test binding.
License: Apache License (>= 2.0)
Imports:
    jsonlite (>= 1.8)
Suggests:
    testthat (>= 3.0.0)
""",
    )


@pytest.fixture
def julia_project(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "Project.toml",
        """\
name = "Voiage"
uuid = "8c7ad020-41fd-4d68-9c3f-5d4e11c48f1f"
version = "1.2.3"

[deps]
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
JSON = "0.21"
julia = "1.10"
""",
    )


def _compose(
    *,
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
) -> dict[str, Any]:
    return compose_sbom(
        python_sbom=python_sbom,
        cargo_workspace=cargo_workspace,
        r_description=r_description,
        julia_project=julia_project,
        source_version="1.2.3",
        source_commit=SOURCE_COMMIT,
        source_tag="v1.2.3",
    )


def _properties(component: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in component.get("properties", [])
        if isinstance(item, dict)
    }


def test_composes_all_ecosystems_with_explicit_resolution_scope(
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
) -> None:
    document = _compose(
        python_sbom=python_sbom,
        cargo_workspace=cargo_workspace,
        r_description=r_description,
        julia_project=julia_project,
    )

    validate_sbom(
        document,
        expected_commit=SOURCE_COMMIT,
        expected_tag="v1.2.3",
        expected_version="1.2.3",
    )
    metadata = document["metadata"]
    metadata_properties = _properties(metadata)
    assert metadata_properties["voiage:sbom:scope"] == "mixed-language-release"
    assert metadata_properties["voiage:sbom:ecosystems"] == "python,cargo,r,julia"

    components = {
        component["bom-ref"]: component for component in document["components"]
    }
    assert {
        "pkg:pypi/numpy@2.3.1",
        "pkg:cargo/serde@1.0.219",
        "pkg:cargo/voiage-core@1.2.3",
        "pkg:cargo/voiage-ffi@1.2.3",
        "pkg:cran/voiageR@1.2.3",
        "pkg:cran/jsonlite",
        "pkg:cran/testthat",
        ("pkg:julia/Voiage@1.2.3?uuid=8c7ad020-41fd-4d68-9c3f-5d4e11c48f1f"),
        "pkg:julia/JSON?uuid=682c06a0-de6a-54ab-a142-c8b1cf79cde6",
        "pkg:julia/Test?uuid=8dfed614-e22c-5e08-85e1-65c5234f0b40",
    } <= components.keys()
    assert (
        _properties(components["pkg:cargo/serde@1.0.219"])[
            "voiage:inventory:resolution"
        ]
        == "resolved-lock"
    )
    assert (
        _properties(components["pkg:cran/jsonlite"])["voiage:inventory:resolution"]
        == "declared-unresolved"
    )
    assert components["pkg:cran/testthat"]["scope"] == "optional"
    assert (
        _properties(
            components["pkg:julia/JSON?uuid=682c06a0-de6a-54ab-a142-c8b1cf79cde6"]
        )["voiage:inventory:version-constraint"]
        == "0.21"
    )

    graph = {
        entry["ref"]: entry.get("dependsOn", []) for entry in document["dependencies"]
    }
    assert "pkg:cargo/voiage-ffi@1.2.3" in graph["pkg:cran/voiageR@1.2.3"]
    assert (
        "pkg:cargo/voiage-ffi@1.2.3"
        in graph["pkg:julia/Voiage@1.2.3?uuid=8c7ad020-41fd-4d68-9c3f-5d4e11c48f1f"]
    )
    assert "pkg:cran/voiageR@1.2.3" in graph["root-component"]


def test_composition_is_byte_for_byte_deterministic(
    tmp_path: Path,
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
) -> None:
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/compose_polyglot_sbom.py"),
        "compose",
        "--python-sbom",
        str(python_sbom),
        "--cargo-workspace",
        str(cargo_workspace),
        "--r-description",
        str(r_description),
        "--julia-project",
        str(julia_project),
        "--source-version",
        "1.2.3",
        "--source-commit",
        SOURCE_COMMIT,
        "--source-tag",
        "v1.2.3",
    ]
    for output in outputs:
        subprocess.run(
            [*command, "--output", str(output)],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/compose_polyglot_sbom.py"),
            "validate",
            str(outputs[0]),
            "--expected-version",
            "1.2.3",
            "--expected-commit",
            SOURCE_COMMIT,
            "--expected-tag",
            "v1.2.3",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in result.stdout


def test_validator_rejects_dangling_dependency_reference(
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
) -> None:
    document = _compose(
        python_sbom=python_sbom,
        cargo_workspace=cargo_workspace,
        r_description=r_description,
        julia_project=julia_project,
    )
    broken = deepcopy(document)
    broken["dependencies"][0].setdefault("dependsOn", []).append("missing-component")

    with pytest.raises(SbomError, match="unknown ref: missing-component"):
        validate_sbom(broken)


def test_release_composition_rejects_binding_version_drift(
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
) -> None:
    r_description.write_text(
        r_description.read_text(encoding="utf-8").replace(
            "Version: 1.2.3",
            "Version: 1.2.2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SbomError, match="voiageR=1.2.2"):
        _compose(
            python_sbom=python_sbom,
            cargo_workspace=cargo_workspace,
            r_description=r_description,
            julia_project=julia_project,
        )


def test_real_manifests_produce_complete_mixed_language_inventory(
    python_sbom: Path,
) -> None:
    document = compose_sbom(
        python_sbom=python_sbom,
        cargo_workspace=REPOSITORY_ROOT / "rust",
        r_description=REPOSITORY_ROOT / "r-package/voiageR/DESCRIPTION",
        julia_project=REPOSITORY_ROOT / "bindings/julia/Project.toml",
        source_version="0+test",
        source_commit=SOURCE_COMMIT,
        source_tag="",
    )

    components = document["components"]
    ecosystems = {
        _properties(component).get("voiage:ecosystem") for component in components
    }
    assert ecosystems == {"python", "cargo", "r", "julia"}
    assert sum(
        _properties(component).get("voiage:ecosystem") == "cargo"
        for component in components
    ) == len(
        tomllib.loads(
            (REPOSITORY_ROOT / "rust/Cargo.lock").read_text(encoding="utf-8")
        )["package"]
    )


def test_workflow_composes_and_schema_validates_the_retained_sbom() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/sbom.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/compose_polyglot_sbom.py compose" in workflow
    assert "scripts/compose_polyglot_sbom.py validate" in workflow
    assert '"${cyclonedx_cli}" validate' in workflow
    assert "--input-version v1_6" in workflow
    assert ".assurance/python.sbom.cdx.json" in workflow
    assert ".assurance/sbom.cdx.json" in workflow
    assert "repos/${cyclonedx_repository}/releases/tags/" in workflow
    assert 'test "${asset_url}" = "${expected_url}"' in workflow
    assert 'test "${asset_digest}" = "sha256:${CYCLONEDX_CLI_SHA256}"' in workflow
    assert ".assurance/cyclonedx-cli-source.json" in workflow


def test_tag_sbom_is_independent_of_software_heritage_and_release_assets() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/sbom.yml").read_text(
        encoding="utf-8"
    )

    assert 'test -n "${swhid}"' not in workflow
    assert "release_evidence_ready=${release_evidence_ready}" in workflow
    assert "if: steps.source.outputs.audit_published_release == 'true'" in workflow
    assert "if: steps.source.outputs.release_evidence_ready == 'true'" in workflow
    assert "partial_missing_software_heritage_snapshot" in workflow
    assert ".assurance/release-evidence-status.json" in workflow


def test_release_target_commitish_is_metadata_not_commit_identity() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/sbom.yml").read_text(
        encoding="utf-8"
    )

    assert (
        'test "$(jq -r .targetCommitish .assurance/github-release.json)" '
        '= "${SOURCE_COMMIT}"'
    ) not in workflow
    assert 'git rev-parse --verify "refs/tags/${SOURCE_TAG}^{commit}"' in workflow
    assert "targetCommitish (recorded metadata)" in workflow
