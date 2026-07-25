from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_core_api_contract as validator


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


def test_diagnostics_document_exists_and_has_expected_sections() -> None:
    document = Path("specs/core-api/diagnostics.md")
    assert document.exists(), "expected the diagnostics contract document to exist"

    text = document.read_text(encoding="utf-8")
    assert "# Diagnostics, Warnings, and Approximation Caveats" in text
    assert "## Diagnostic Envelope" in text
    assert "## Warning Records" in text
    assert "## Normative Rules" in text
    assert "## Example Fragment" in text
    assert "## Relationship To Later Metadata" in text


def test_method_metadata_document_exists_and_has_expected_sections() -> None:
    document = Path("specs/core-api/method-metadata.md")
    assert document.exists(), "expected the method metadata contract to exist"

    text = document.read_text(encoding="utf-8")
    assert "# Capability, Stability, and Maturity Metadata" in text
    assert "## Metadata Envelope" in text
    assert "## Method Maturity" in text
    assert "## Capability Labels" in text
    assert "## Approximation Status" in text
    assert "approximate methods must never look exact by omission" in text


def test_extension_evolution_document_exists_and_cross_checks_stable_schemas() -> None:
    document = Path("specs/core-api/extension-evolution.md")
    assert document.exists(), "expected the extension-evolution contract to exist"

    text = document.read_text(encoding="utf-8")
    assert "# Contract Extension and Evolution Rules" in text
    assert "## Extension Principles" in text
    assert "## Versioning Rules" in text
    assert "## Deprecation Rules" in text
    assert "## Stable Schema Cross-Checks" in text
    assert "schemas/v1/method-metadata.schema.json" in text
    assert "schemas/v1/results/evsi.schema.json" in text


def test_core_api_readme_references_numerics_contract_document() -> None:
    readme = Path("specs/core-api/README.md")
    text = readme.read_text(encoding="utf-8")
    assert "numerical-equivalence.md" in text
    assert "diagnostics.md" in text


def test_diagnostics_schema_example_pair_conforms_to_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "specs/core-api/schemas/v1/diagnostics.schema.json"
    example_path = root / "specs/core-api/examples/v1/diagnostics.example.json"

    schema = validator._load_json(schema_path)
    example = validator._load_json(example_path)

    validator._validate(example, schema, "$", schema_path)


def test_method_metadata_schema_example_pair_conforms_to_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "specs/core-api/schemas/v1/method-metadata.schema.json"
    example_path = root / "specs/core-api/examples/v1/method-metadata.example.json"

    schema = validator._load_json(schema_path)
    example = validator._load_json(example_path)

    validator._validate(example, schema, "$", schema_path)


def test_method_metadata_requires_explicit_approximation_status() -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / "specs/core-api/schemas/v1/method-metadata.schema.json"
    schema = validator._load_json(schema_path)

    invalid = {
        "analysis_type": "method_metadata",
        "method_family": "evsi",
        "method_maturity": "fixture-backed",
        "capability_labels": ["surrogate-regression"],
    }

    with pytest.raises(validator.ValidationError, match="approximation_status"):
        validator._validate(invalid, schema, "$", schema_path)
