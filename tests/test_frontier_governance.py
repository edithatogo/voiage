"""Tests for the frontier VOI architecture and dependency governance policy.

These tests enforce the governance decisions defined by the
``voi-frontier-architecture-dependency-governance`` track:

- The method maturity taxonomy is well-defined with promotion criteria.
- Backend boundary ownership rules are explicit and testable.
- Optional bleeding-edge dependencies never leak into the base install.
- Frontier families carry consistent maturity labels across the registry.
- The governance document covers all required keywords.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voiage.governance import (
    BACKEND_OWNERSHIP,
    DEPENDENCY_POLICY,
    MATURITY_LEVELS,
    MATURITY_PROMOTION_ORDER,
    PROMOTION_MATRIX,
    validate_backend_boundary,
    validate_dependency_policy,
    validate_maturity_label,
    validate_promotion_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_DOC = REPO_ROOT / "docs" / "developer_guide" / "frontier_governance.rst"
REGISTRY_MANIFEST = REPO_ROOT / "specs" / "frontier" / "fixtures" / "manifest.json"
GOVERNANCE_SPEC_DIR = REPO_ROOT / "specs" / "frontier" / "governance"


# --------------------------------------------------------------------------- #
# Maturity taxonomy
# --------------------------------------------------------------------------- #


def test_maturity_levels_are_ordered() -> None:
    """The maturity taxonomy must define an ordered promotion ladder."""
    assert len(MATURITY_PROMOTION_ORDER) >= 4
    assert MATURITY_PROMOTION_ORDER[0] == "planned"
    assert MATURITY_PROMOTION_ORDER[-1] == "stable"


def test_promotion_matrix_covers_all_governed_states() -> None:
    """Promotion states must define evidence requirements and claim limits."""
    assert set(PROMOTION_MATRIX) == {
        "experimental",
        "fixture-backed",
        "cross-language-parity",
        "stable",
    }
    for state, metadata in PROMOTION_MATRIX.items():
        assert metadata["required_evidence"], state
        assert "forbidden_claims" in metadata, state


def test_promotion_matrix_json_matches_runtime_contract() -> None:
    matrix_path = (
        REPO_ROOT / "specs" / "frontier" / "governance" / "promotion-matrix.json"
    )
    with matrix_path.open(encoding="utf-8") as f:
        matrix = json.load(f)

    assert matrix["version"] == "v1"
    assert matrix["states"] == PROMOTION_MATRIX


def test_promotion_checklist_covers_every_registered_family() -> None:
    with REGISTRY_MANIFEST.open(encoding="utf-8") as f:
        registry = json.load(f)
    checklist_path = (
        REPO_ROOT / "specs" / "frontier" / "governance" / "promotion-checklist.json"
    )
    with checklist_path.open(encoding="utf-8") as f:
        checklist = json.load(f)

    registry_families = {family["name"] for family in registry["families"]}
    checklist_families = {family["name"] for family in checklist["families"]}
    assert checklist_families == registry_families
    for family in checklist["families"]:
        assert family["owner"]
        assert family["blocked_state"]
        assert family["artifact_paths"]
        assert family["stable_claim_allowed"] is False


def test_stable_promotion_requires_complete_evidence_boundary() -> None:
    evidence = {
        "runtime": True,
        "public_api": True,
        "deterministic_fixtures": True,
        "schema": True,
        "registry_entry": True,
        "cross_language_parity": True,
        "rust_kernel_parity": True,
        "full_documentation": True,
        "changelog_entry": True,
        "migration_guide_entry": True,
        "stable_promotion_approval": True,
        "public_api_compatibility": True,
    }
    validate_promotion_evidence("stable", evidence)


def test_stable_promotion_rejects_missing_migration_guidance() -> None:
    with pytest.raises(ValueError, match="migration_guide_entry"):
        validate_promotion_evidence(
            "stable",
            {
                "runtime": True,
                "public_api": True,
                "deterministic_fixtures": True,
                "schema": True,
                "registry_entry": True,
                "cross_language_parity": True,
                "rust_kernel_parity": True,
                "full_documentation": True,
                "changelog_entry": True,
                "stable_promotion_approval": True,
                "public_api_compatibility": True,
            },
        )


def test_each_maturity_level_has_metadata() -> None:
    """Every maturity level must carry a description and promotion criteria."""
    for level in MATURITY_PROMOTION_ORDER:
        assert level in MATURITY_LEVELS
        meta = MATURITY_LEVELS[level]
        assert "description" in meta
        assert "promotion_criteria" in meta
        assert isinstance(meta["promotion_criteria"], list)
        assert meta["promotion_criteria"]


def test_validate_maturity_label_accepts_known_levels() -> None:
    for level in MATURITY_PROMOTION_ORDER:
        validate_maturity_label(level)  # should not raise


def test_validate_maturity_label_rejects_unknown_levels() -> None:
    with pytest.raises(ValueError):
        validate_maturity_label("super-stable")


def test_registry_families_use_governed_maturity_labels() -> None:
    """Every frontier fixture family must use a maturity level from the taxonomy."""
    with REGISTRY_MANIFEST.open() as f:
        registry = json.load(f)

    for family in registry["families"]:
        validate_maturity_label(family["method_maturity"])


# --------------------------------------------------------------------------- #
# Backend boundary
# --------------------------------------------------------------------------- #


def test_backend_ownership_covers_all_layers() -> None:
    """Backend ownership must cover schema, methods, backends, CLI, and Rust core."""
    required_layers = {"schema", "methods", "backends", "cli", "rust_core"}
    assert required_layers.issubset(BACKEND_OWNERSHIP.keys())


def test_backend_ownership_defines_owner_and_boundary() -> None:
    for meta in BACKEND_OWNERSHIP.values():
        assert "owner" in meta
        assert "boundary" in meta
        assert meta["owner"]
        assert meta["boundary"]


def test_validate_backend_boundary_rejects_cross_layer_leak() -> None:
    with pytest.raises(ValueError):
        validate_backend_boundary(
            layer="methods", responsibility="backend selection dispatch"
        )


def test_validate_backend_boundary_accepts_owned_responsibility() -> None:
    validate_backend_boundary(
        layer="backends", responsibility="NumPy/JAX dispatch"
    )  # no raise


# --------------------------------------------------------------------------- #
# Dependency policy
# --------------------------------------------------------------------------- #


def test_dependency_policy_separates_base_from_optional() -> None:
    assert "base" in DEPENDENCY_POLICY
    assert "optional" in DEPENDENCY_POLICY
    assert isinstance(DEPENDENCY_POLICY["base"], list)
    assert isinstance(DEPENDENCY_POLICY["optional"], list)


def test_base_install_excludes_heavy_optional_backends() -> None:
    """JAX, PyTorch, and heavy ML deps must not be in the base install list."""
    base = DEPENDENCY_POLICY["base"]
    for heavy in ("jax", "torch", "pytorch"):
        assert not any(heavy in dep.lower() for dep in base), (
            f"'{heavy}' must not be a base dependency"
        )


def test_validate_dependency_policy_passes_for_clean_split() -> None:
    validate_dependency_policy()  # should not raise


# --------------------------------------------------------------------------- #
# Governance documentation
# --------------------------------------------------------------------------- #


def test_governance_document_exists_and_covers_keywords() -> None:
    """The governance RST must exist and cover all required keywords."""
    assert GOVERNANCE_DOC.exists(), f"Missing governance doc: {GOVERNANCE_DOC}"
    text = GOVERNANCE_DOC.read_text(encoding="utf-8").lower()

    for keyword in (
        "architecture",
        "dependency",
        "maturity taxonomy",
        "backend boundary",
        "no conflicts",
    ):
        assert keyword in text, f"Governance doc missing keyword: '{keyword}'"


def test_governance_spec_directory_has_readme() -> None:
    """The governance spec subtree must have a README."""
    readme = GOVERNANCE_SPEC_DIR / "README.md"
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert "maturity" in content.lower()
    assert "backend" in content.lower()
