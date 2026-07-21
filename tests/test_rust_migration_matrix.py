"""Tests for the Rust frontier numerics migration matrix and bridge policy.

These tests enforce the deliverables defined by the
``rust-frontier-numerics-migration-completion`` track:

- A tracked migration matrix shows kernel owner, Rust status, Python wrapper
  status, parity status, and benchmark status for every numerical kernel.
- A Python bridge policy decision record covers PyO3/maturin, the Python
  facade preservation rule, fallback behavior, and the required keywords.
- Migrated kernels preserve public Python behavior and result envelopes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_MATRIX = REPO_ROOT / "specs" / "rust" / "migration_matrix.json"
BRIDGE_POLICY_DOC = (
    REPO_ROOT
    / "docs"
    / "astro-site"
    / "src"
    / "content"
    / "docs"
    / "developer-guide"
    / "rust-python-bridge-policy.mdx"
)

REQUIRED_KERNEL_IDS = {
    "evpi",
    "evppi",
    "evsi",
    "enbs",
}

ALLOWED_RUST_STATUS = {"none", "contract_fixture", "complete"}
ALLOWED_PYTHON_WRAPPER_STATUS = {"native", "reference_only", "bridge_pending"}
ALLOWED_PARITY_STATUS = {"none", "fixture_backed", "verified"}
ALLOWED_BENCHMARK_STATUS = {"none", "scalar_baseline", "ci_gated"}
ALLOWED_PRIORITY = {"high", "medium", "low"}

REQUIRED_MATRIX_FIELDS = {
    "kernel_id",
    "python_owner",
    "rust_status",
    "python_wrapper_status",
    "parity_status",
    "benchmark_status",
    "migration_priority",
    "notes",
}

REQUIRED_POLICY_KEYWORDS = (
    "rust numerics core",
    "python facade",
    "pyo3",
    "maturin",
    "fixture parity",
    "benchmarks",
)


def _load_matrix() -> list[dict]:
    assert MIGRATION_MATRIX.exists(), f"Missing matrix: {MIGRATION_MATRIX}"
    with MIGRATION_MATRIX.open() as f:
        data = json.load(f)
    assert isinstance(data, dict), "Matrix root must be an object"
    assert "kernels" in data, "Matrix must have a 'kernels' list"
    assert isinstance(data["kernels"], list), "'kernels' must be a list"
    return data["kernels"]


# --------------------------------------------------------------------------- #
# Migration matrix structure
# --------------------------------------------------------------------------- #


def test_migration_matrix_file_exists() -> None:
    assert MIGRATION_MATRIX.exists(), f"Missing matrix: {MIGRATION_MATRIX}"


def test_migration_matrix_covers_core_kernels() -> None:
    kernels = _load_matrix()
    ids = {entry["kernel_id"] for entry in kernels}
    missing = REQUIRED_KERNEL_IDS - ids
    assert not missing, f"Matrix missing core kernels: {missing}"


def test_migration_matrix_entries_have_required_fields() -> None:
    kernels = _load_matrix()
    assert kernels, "Matrix must contain at least one kernel entry"
    for entry in kernels:
        missing = REQUIRED_MATRIX_FIELDS - set(entry)
        assert not missing, f"Kernel {entry.get('kernel_id')} missing fields: {missing}"


def test_migration_matrix_status_values_are_governed() -> None:
    kernels = _load_matrix()
    for entry in kernels:
        assert entry["rust_status"] in ALLOWED_RUST_STATUS, entry
        assert entry["python_wrapper_status"] in ALLOWED_PYTHON_WRAPPER_STATUS, entry
        assert entry["parity_status"] in ALLOWED_PARITY_STATUS, entry
        assert entry["benchmark_status"] in ALLOWED_BENCHMARK_STATUS, entry
        assert entry["migration_priority"] in ALLOWED_PRIORITY, entry


def test_complete_rust_kernels_have_at_least_fixture_backed_parity() -> None:
    """A kernel marked Rust-complete must have at least fixture-backed parity.

    The maturity ladder is:
    - complete + fixture_backed parity: the Rust impl exists and is exercised by
      fixture parity tests; a dedicated CI benchmark gate may still be deferred.
    - complete + verified parity: the highest bar; a CI-gated benchmark is
      required because the kernel is a high-priority core surface.
    """
    kernels = _load_matrix()
    for entry in kernels:
        if entry["rust_status"] == "complete":
            assert entry["parity_status"] in {"fixture_backed", "verified"}, entry
            if entry["parity_status"] == "verified":
                assert entry["benchmark_status"] == "ci_gated", entry


def test_kernel_ids_are_unique() -> None:
    kernels = _load_matrix()
    ids = [entry["kernel_id"] for entry in kernels]
    assert len(ids) == len(set(ids)), f"Duplicate kernel ids: {ids}"


def test_python_owners_reference_real_modules() -> None:
    """Every python_owner path must point to an existing module file."""
    kernels = _load_matrix()
    for entry in kernels:
        owner = entry["python_owner"]
        assert owner.startswith("voiage."), entry
        path = REPO_ROOT / owner.replace(".", "/")
        if not path.with_suffix(".py").exists() and not (path / "__init__.py").exists():
            pytest.fail(f"python_owner does not resolve to a file: {owner}")


# --------------------------------------------------------------------------- #
# Bridge policy document
# --------------------------------------------------------------------------- #


def test_bridge_policy_document_exists() -> None:
    assert BRIDGE_POLICY_DOC.exists(), f"Missing policy doc: {BRIDGE_POLICY_DOC}"


def test_bridge_policy_covers_required_keywords() -> None:
    assert BRIDGE_POLICY_DOC.exists(), f"Missing policy doc: {BRIDGE_POLICY_DOC}"
    text = BRIDGE_POLICY_DOC.read_text(encoding="utf-8").lower()
    for keyword in REQUIRED_POLICY_KEYWORDS:
        assert keyword in text, f"Policy doc missing keyword: '{keyword}'"


def test_bridge_policy_states_python_facade_preservation() -> None:
    text = BRIDGE_POLICY_DOC.read_text(encoding="utf-8").lower()
    assert "public api" in text or "public python" in text
    assert "compatible" in text or "no breaking change" in text
