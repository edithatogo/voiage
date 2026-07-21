"""Executable checks for retained binding build and lifecycle gates."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _workflow_text() -> str:
    return "\n".join(
        (
            (ROOT / ".github" / "workflows" / "bindings-ci.yml").read_text(
                encoding="utf-8"
            ),
            (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        )
    )


def test_each_retained_binding_has_a_build_test_and_package_gate() -> None:
    matrix = json.loads(
        (ROOT / "specs" / "v1" / "binding-matrix.json").read_text(encoding="utf-8")
    )
    workflow = _workflow_text()
    required = {
        "python": ("coverage", "python"),
        "rust": ("cargo test", "cargo clippy"),
        "julia": ("cargo build --release --locked --package voiage-ffi", "Pkg.test()"),
        "r": (
            "cargo build --release --locked --package voiage-ffi",
            "R CMD build",
            "rcmdcheck",
        ),
    }
    for binding in matrix["bindings"]:
        binding_id = binding["id"]
        assert binding_id in required
        assert all(command in workflow for command in required[binding_id])


def test_abi_lifecycle_and_error_contracts_are_in_the_rust_gate() -> None:
    workflow = _workflow_text()
    assert "cargo test --workspace --all-features --locked" in workflow
    assert (ROOT / "rust/crates/voiage-ffi/tests/lifecycle.rs").exists()
    assert (ROOT / "rust/crates/voiage-ffi/tests/error_transport.rs").exists()
    header = (ROOT / "rust/crates/voiage-ffi/include/voiage_v1.h").read_text(
        encoding="utf-8"
    )
    assert "voiage_v1_abi_version" in header
    assert "voiage_v1_handle_create" in header
    assert "voiage_v1_error_message" in header
    assert "voiage_v1_evpi" in header


def test_retained_bindings_declare_supported_runtime_versions_and_ci_probes() -> None:
    matrix = json.loads(
        (ROOT / "specs" / "v1" / "binding-matrix.json").read_text(encoding="utf-8")
    )
    workflow = _workflow_text()
    workflow_requirements = {
        "python": ("python-version",),
        "r": ("setup-r",),
        "julia": ('version: "1.11"',),
        "rust": ("rust-msrv", 'toolchain: "1.85"'),
    }
    for binding in matrix["bindings"]:
        versions = binding.get("supported_runtime_versions")
        assert versions
        assert all(isinstance(version, str) for version in versions)
        assert binding["id"] in workflow_requirements
        assert all(
            requirement in workflow
            for requirement in workflow_requirements[binding["id"]]
        )
