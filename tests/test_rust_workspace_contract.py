"""Contract tests for the production Rust workspace architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = ROOT / "rust"
WORKSPACE_MANIFEST = RUST_ROOT / "Cargo.toml"

EXPECTED_MEMBERS = {
    "crates/voiage-diagnostics",
    "crates/voiage-domain",
    "crates/voiage-ffi",
    "crates/voiage-numerics",
    "crates/voiage-python",
    "crates/voiage-serialization",
    "crates/voiage-test-support",
    "crates/voiage-wasm",
}
CORE_CRATES = {
    "voiage-diagnostics",
    "voiage-domain",
    "voiage-numerics",
    "voiage-serialization",
}
ADAPTER_CRATES = {"voiage-ffi", "voiage-python", "voiage-wasm"}
ALLOWED_INTERNAL_DEPENDENCIES = {
    "voiage-domain": set(),
    "voiage-diagnostics": {"voiage-domain"},
    "voiage-numerics": {"voiage-diagnostics", "voiage-domain"},
    "voiage-serialization": {"voiage-diagnostics", "voiage-domain"},
    "voiage-test-support": CORE_CRATES,
    "voiage-ffi": {
        "voiage-diagnostics",
        "voiage-domain",
        "voiage-serialization",
    },
    "voiage-python": {
        "voiage-diagnostics",
        "voiage-domain",
        "voiage-numerics",
        "voiage-serialization",
    },
    "voiage-wasm": CORE_CRATES,
}
BINDING_SPECIFIC_DEPENDENCIES = {
    "cbindgen",
    "js-sys",
    "libc",
    "napi",
    "pyo3",
    "wasm-bindgen",
    "web-sys",
}


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _dependency_names(manifest: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for table_name in ("dependencies", "build-dependencies", "dev-dependencies"):
        names.update(manifest.get(table_name, {}))
    for target in manifest.get("target", {}).values():
        for table_name in ("dependencies", "build-dependencies", "dev-dependencies"):
            names.update(target.get(table_name, {}))
    return names


def _crate_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for member in EXPECTED_MEMBERS:
        manifest_path = RUST_ROOT / member / "Cargo.toml"
        assert manifest_path.is_file(), f"workspace member is missing: {member}"
        manifest = _load_toml(manifest_path)
        manifests[manifest["package"]["name"]] = manifest
    return manifests


def test_workspace_declares_the_production_crate_layout() -> None:
    manifest = _load_toml(WORKSPACE_MANIFEST)

    assert "workspace" in manifest, "Rust root must be a Cargo workspace"
    assert set(manifest["workspace"]["members"]) == EXPECTED_MEMBERS
    assert manifest["workspace"]["resolver"] == "2"


def test_workspace_centralizes_msrv_and_lint_policy() -> None:
    manifest = _load_toml(WORKSPACE_MANIFEST)
    workspace = manifest["workspace"]

    rust_version = workspace["package"]["rust-version"]
    assert isinstance(rust_version, str)
    assert rust_version.strip()

    lints = workspace["lints"]
    assert lints["rust"]["unsafe_code"] in {"deny", "forbid"}
    assert lints["rust"]["missing_docs"] in {"warn", "deny", "forbid"}
    assert lints["clippy"]["all"] in {"warn", "deny", "forbid"}

    for member in EXPECTED_MEMBERS:
        crate = _load_toml(RUST_ROOT / member / "Cargo.toml")
        assert crate["package"]["rust-version"]["workspace"] is True
        if member == "crates/voiage-ffi":
            assert crate["lints"]["rust"]["unsafe_code"] == "deny"
            assert crate["lints"]["rust"]["unsafe_op_in_unsafe_fn"] == "deny"
        else:
            assert crate["lints"]["workspace"] is True


def test_internal_dependencies_follow_the_architecture_direction() -> None:
    manifests = _crate_manifests()
    assert set(manifests) == set(ALLOWED_INTERNAL_DEPENDENCIES)

    for crate_name, manifest in manifests.items():
        internal = _dependency_names(manifest) & set(manifests)
        forbidden = internal - ALLOWED_INTERNAL_DEPENDENCIES[crate_name]
        assert not forbidden, (
            f"{crate_name} has forbidden inward dependency/dependencies: "
            f"{sorted(forbidden)}"
        )


def test_ffi_python_and_wasm_are_leaf_adapters() -> None:
    manifests = _crate_manifests()

    for crate_name, manifest in manifests.items():
        if crate_name in ADAPTER_CRATES:
            continue
        adapter_dependencies = _dependency_names(manifest) & ADAPTER_CRATES
        assert not adapter_dependencies, (
            f"{crate_name} must not depend on leaf adapter(s): "
            f"{sorted(adapter_dependencies)}"
        )


def test_ffi_infrastructure_does_not_depend_on_numerics_before_phase_5() -> None:
    manifests = _crate_manifests()

    assert "voiage-numerics" not in _dependency_names(manifests["voiage-ffi"]), (
        "voiage-ffi must remain infrastructure-only until Phase 5 kernels are "
        "ready to cross the C ABI"
    )


def test_domain_and_numerics_have_no_binding_specific_dependencies() -> None:
    manifests = _crate_manifests()

    for crate_name in ("voiage-domain", "voiage-numerics"):
        forbidden = (
            _dependency_names(manifests[crate_name]) & BINDING_SPECIFIC_DEPENDENCIES
        )
        assert not forbidden, (
            f"{crate_name} contains binding-specific dependencies: {sorted(forbidden)}"
        )


def test_pyo3_is_isolated_to_the_python_leaf_adapter() -> None:
    manifests = _crate_manifests()

    for crate_name, manifest in manifests.items():
        has_pyo3 = "pyo3" in _dependency_names(manifest)
        assert has_pyo3 is (crate_name == "voiage-python"), (
            f"pyo3 dependency boundary violated by {crate_name}"
        )
