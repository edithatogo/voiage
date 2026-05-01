from __future__ import annotations

from pathlib import Path


def test_numerical_equivalence_document_exists_and_has_expected_sections() -> None:
    document = Path("specs/core-api/numerical-equivalence.md")
    assert document.exists(), "expected the numerics contract document to exist"

    text = document.read_text(encoding="utf-8")
    assert "# Numerical Equivalence and Reproducibility" in text
    assert "## Numerical Equivalence" in text
    assert "## Tolerance Rules" in text
    assert "## Reproducibility and Provenance" in text
    assert "## Comparable Results" in text
    assert "## Non-Comparable Results" in text


def test_core_api_readme_references_numerics_contract_document() -> None:
    readme = Path("specs/core-api/README.md")
    text = readme.read_text(encoding="utf-8")
    assert "numerical-equivalence.md" in text
