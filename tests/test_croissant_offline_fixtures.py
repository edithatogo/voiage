"""File-backed Croissant 1.1 profile conformance fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.croissant import CroissantProvider

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "croissant_1_1"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("unsupported/archive.json", "archives"),
        ("unsupported/checksum-mismatch.json", "SHA-256"),
        ("unsupported/key.json", "keys"),
        ("unsupported/multiple-distributions.json", "exactly one distribution"),
        ("unsupported/multiple-record-sets.json", "exactly one recordSet"),
        ("unsupported/nested-field.json", "nested fields"),
        ("unsupported/references.json", "field references"),
        ("unsupported/split.json", "splits"),
        ("unsupported/transform.json", "transformations"),
        ("unsupported/version-1-0.json", "version 1.1"),
        ("unsupported/wrong-columns.json", "exactly declare"),
    ],
)
def test_croissant_offline_unsupported_profile_fixtures_fail_closed(
    fixture: str, message: str
) -> None:
    """Every unsupported semantic has a stable, offline regression fixture."""
    descriptor_path = _FIXTURE_ROOT / fixture

    with pytest.raises(IngestionError, match=message):
        CroissantProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(_FIXTURE_ROOT)
        )


def test_croissant_offline_valid_profile_fixture_materializes() -> None:
    """The supported fixture is a one-resource, explicit CSV record set."""
    bundle = CroissantProvider().ingest(
        _FIXTURE_ROOT / "valid" / "croissant.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert tuple(field.field_id for field in bundle.manifest.tables[0].fields) == (
        "strategy_a",
        "strategy_b",
    )
    assert bundle.table("decision_samples").to_pylist() == [
        {"strategy_a": 100.0, "strategy_b": 80.0},
        {"strategy_a": 60.0, "strategy_b": 90.0},
    ]


def test_croissant_preserves_declared_field_data_type_as_descriptive_metadata(
    tmp_path,
) -> None:
    """A declared ML data type remains descriptive until a binding uses it."""
    (tmp_path / "samples.csv").write_text("score\n1.5\n", encoding="utf-8")
    descriptor = tmp_path / "croissant.json"
    descriptor.write_text(
        '{"@context":"https://mlcommons.org/croissant/1.1","distribution":[{"contentUrl":"samples.csv"}],"recordSet":[{"name":"samples","field":[{"name":"score","dataType":"sc:Float"}]}]}',
        encoding="utf-8",
    )

    bundle = CroissantProvider().ingest(descriptor, policy=SourceAccessPolicy(tmp_path))

    assert bundle.manifest.tables[0].fields[0].semantic_type == "sc:Float"


def test_croissant_rejects_a_non_scalar_declared_field_data_type(tmp_path) -> None:
    """Descriptor metadata cannot smuggle structured field semantics downstream."""
    (tmp_path / "samples.csv").write_text("score\n1.5\n", encoding="utf-8")
    descriptor = tmp_path / "croissant.json"
    descriptor.write_text(
        '{"@context":"https://mlcommons.org/croissant/1.1","distribution":[{"contentUrl":"samples.csv"}],"recordSet":[{"name":"samples","field":[{"name":"score","dataType":{"@id":"sc:Float"}}]}]}',
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="dataType must be a string"):
        CroissantProvider().ingest(descriptor, policy=SourceAccessPolicy(tmp_path))


def test_croissant_offline_identity_fixture_preserves_governance() -> None:
    """Croissant identities are retained as metadata, not VOI semantics."""
    bundle = CroissantProvider().ingest(
        _FIXTURE_ROOT / "unsupported" / "identity.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.extensions == {
        "mlcommons.org:croissant-governance": {"@id": "https://example.invalid/dataset"}
    }


def test_croissant_offline_governance_fixture_preserves_metadata() -> None:
    """Citation, PROV, usage, ODRL, and RAI metadata remain non-semantic."""
    bundle = CroissantProvider().ingest(
        _FIXTURE_ROOT / "valid" / "governed-croissant.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.provenance.license == "CC-BY-4.0"
    assert bundle.manifest.provenance.citation == "Example et al. (2026)"
    governance = bundle.manifest.extensions["mlcommons.org:croissant-governance"]
    assert governance["citation"] == "Example et al. (2026)"
    assert governance["usageInfo"] == "Synthetic offline fixture only."
    assert dict(governance["odrl"]) == {"permission": "use"}
    assert dict(governance["provenance"]) == {"wasGeneratedBy": "simulation"}
    assert dict(governance["rai"]) == {"risk": "low"}
