"""Drift checks for the retained v1 binding matrix."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
MATRIX_PATH = REPO_ROOT / "specs" / "v1" / "binding-matrix.json"


def test_retained_binding_matrix_covers_supported_surfaces() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    bindings = matrix["bindings"]

    assert {entry["id"] for entry in bindings} == {
        "python",
        "mojo",
        "r",
        "julia",
        "rust",
    }
    assert {entry["status"] for entry in bindings} == {"retained", "external_boundary"}
    assert matrix["execution_authority"] == "rust"
    assert all(
        entry.get("status") == "external_boundary"
        or (REPO_ROOT / entry["path"]).exists()
        for entry in bindings
    )
    assert all(entry["registry"] and entry["tag_pattern"] for entry in bindings)


def test_binding_matrix_declares_ci_and_external_gate_for_each_surface() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    for entry in matrix["bindings"]:
        assert matrix["ci_workflow"] == ".github/workflows/bindings-ci.yml"
        assert matrix["contract_fixture_root"] == "specs/core-api/fixtures"
        assert entry["external_gate"]


def test_governance_docs_match_the_rust_execution_authority() -> None:
    product_guidelines = (REPO_ROOT / "conductor/product-guidelines.md").read_text(
        encoding="utf-8"
    )
    tech_stack = (REPO_ROOT / "conductor/tech-stack.md").read_text(encoding="utf-8")

    assert "Rust is the sole stable numerical execution authority" in product_guidelines
    assert "Target: >90% code coverage" in product_guidelines
    assert "Update `changelog.md` with each release" in product_guidelines
    assert "Rust result envelopes" in product_guidelines
    assert "NumPy, JAX" not in product_guidelines
    assert (
        "**Rust**: GitHub Releases for the internal `publish = false` workspace"
        in tech_stack
    )
    assert "**Rust**: crates.io" not in tech_stack
    assert "starlight-versions" not in tech_stack
    assert "starlight-links-validator" not in tech_stack
    assert "starlight-polyglot" not in tech_stack

    astro_architecture = (
        REPO_ROOT / "docs/astro-site/src/content/docs/developer-guide/architecture.mdx"
    ).read_text(encoding="utf-8")
    astro_backends = (
        REPO_ROOT / "docs/astro-site/src/content/docs/backends.mdx"
    ).read_text(encoding="utf-8")
    assert "sole execution authority" in astro_backends
    assert "NumPy remains the default runtime path" not in astro_architecture
    assert "Rust workspace" in astro_architecture
