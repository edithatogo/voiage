"""File-backed Croissant 1.1 profile conformance fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.croissant import CroissantProvider, CroissantSelection
from voiage.ingestion.registry import default_registry

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "croissant_1_1"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("unsupported/non-object-root.json", "descriptor root must be a JSON object"),
        ("unsupported/archive.json", "archives"),
        ("unsupported/checksum-mismatch.json", "SHA-256"),
        ("unsupported/context-object.json", "string JSON-LD context entries"),
        ("unsupported/integrity-declaration.json", "integrity declarations"),
        ("unsupported/content-size-mismatch.json", "byte size does not match"),
        ("unsupported/content-size-text.json", "contentSize"),
        ("unsupported/key.json", "keys"),
        ("unsupported/non-csv-media-type.json", "CSV media type"),
        ("unsupported/multiple-distributions.json", "recordSet selection is required"),
        ("unsupported/multiple-record-sets.json", "recordSet selection is required"),
        ("unsupported/nested-field.json", "nested fields"),
        ("unsupported/references.json", "field references"),
        ("unsupported/field-source.json", "field sources"),
        ("unsupported/fileset-distribution.json", "FileObject or FileSet"),
        ("unsupported/malformed-distribution.json", "distribution object"),
        ("unsupported/malformed-record-set.json", "recordSet object"),
        ("unsupported/split.json", "splits"),
        ("unsupported/transform.json", "transformations"),
        ("unsupported/version-1-0.json", "version 1.1"),
        ("unsupported/version-1-10.json", "version 1.1"),
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
    receipt = bundle.manifest.resources[0]
    resource_path = _FIXTURE_ROOT / "valid" / "data.csv"
    assert receipt.resource_id == "decision_samples"
    assert receipt.uri == resource_path.resolve().as_uri()
    assert receipt.sha256 == hashlib.sha256(resource_path.read_bytes()).hexdigest()
    assert receipt.byte_size == resource_path.stat().st_size


def test_croissant_content_size_is_verified_and_retained_in_receipt() -> None:
    """The accepted Croissant contentSize is an actual byte-integrity check."""
    bundle = CroissantProvider().ingest(
        _FIXTURE_ROOT / "valid" / "content-size-croissant.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.resources[0].byte_size == 43


def test_croissant_profile_map_exercises_every_documented_boundary() -> None:
    """Fixtures make support and rejection explicit, never best-effort."""
    profile = json.loads((_FIXTURE_ROOT / "profile-map.json").read_text())

    for entry in profile["supported"]:
        bundle = CroissantProvider().ingest(
            _FIXTURE_ROOT / entry["fixture"],
            policy=SourceAccessPolicy(_FIXTURE_ROOT),
        )
        assert bundle.manifest.provenance.provider_id == "croissant"
    for entry in profile["rejected"]:
        with pytest.raises(IngestionError, match=entry["message"]):
            CroissantProvider().ingest(
                _FIXTURE_ROOT / entry["fixture"],
                policy=SourceAccessPolicy(_FIXTURE_ROOT),
            )


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


def test_croissant_offline_context_array_fixture_materializes() -> None:
    """A Croissant 1.1 context can coexist with other JSON-LD contexts."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "context-array-croissant.json"

    assert CroissantProvider().can_handle(
        {"@context": ["https://schema.org/", "https://mlcommons.org/croissant/1.1"]}
    )
    bundle = CroissantProvider().ingest(
        descriptor_path,
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.dataset_id == "context-array-decision-samples"
    assert bundle.table("decision_samples").num_rows == 2


def test_croissant_multi_pair_fixture_requires_explicit_local_selection() -> None:
    """Multi-entry descriptors cannot silently choose a record set or file."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "multi-pair-croissant.json"

    with pytest.raises(IngestionError, match="recordSet selection is required"):
        CroissantProvider().ingest(
            descriptor_path, policy=SourceAccessPolicy(_FIXTURE_ROOT)
        )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            CroissantSelection(record_set="missing", distribution="#baseline"),
            "selected Croissant recordSet is not declared",
        ),
        (
            CroissantSelection(record_set="baseline_samples", distribution="#missing"),
            "selected Croissant distribution is not declared",
        ),
        (
            CroissantSelection(
                record_set="baseline_samples", distribution="#alternate"
            ),
            "must explicitly reference the selected distribution",
        ),
    ],
)
def test_croissant_multi_pair_fixture_rejects_invalid_selection(
    selection: CroissantSelection, message: str
) -> None:
    """Missing and mismatched selectors have stable non-materializing errors."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "multi-pair-croissant.json"

    with pytest.raises(IngestionError, match=message):
        CroissantProvider().ingest(
            descriptor_path,
            policy=SourceAccessPolicy(_FIXTURE_ROOT),
            selection=selection,
        )


def test_croissant_multi_pair_fixture_preserves_selected_receipt_identity() -> None:
    """Explicit selections produce the selected table, receipt, and identity digest."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "multi-pair-croissant.json"
    provider = CroissantProvider()
    baseline = provider.ingest(
        descriptor_path,
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
        selection=CroissantSelection(
            record_set="baseline_samples", distribution="#baseline"
        ),
    )
    alternate = provider.ingest(
        descriptor_path,
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
        selection=CroissantSelection(
            record_set="alternate_samples", distribution="#alternate"
        ),
    )

    assert baseline.manifest.resources[0].uri.endswith("/valid/data.csv")
    assert alternate.manifest.resources[0].uri.endswith("/valid/alternate.csv")
    assert (
        baseline.manifest.resources[0].sha256 != alternate.manifest.resources[0].sha256
    )
    assert baseline.content_digest != alternate.content_digest
    assert baseline.manifest.extensions["mlcommons.org:croissant-selection"] == {
        "recordSet": "baseline_samples",
        "distribution": "#baseline",
    }
    assert alternate.manifest.extensions["mlcommons.org:croissant-selection"] == {
        "recordSet": "alternate_samples",
        "distribution": "#alternate",
    }


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


def test_croissant_context_array_governance_fixture_is_metadata_only() -> None:
    """One fixture proves context arrays coexist with retained governance metadata."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "context-array-governed-croissant.json"
    bundle = CroissantProvider().ingest(
        descriptor_path,
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.dataset_id == "context-array-governed-decision-samples"
    assert bundle.manifest.provenance.citation == "Example et al. (2026)"
    governance = bundle.manifest.extensions["mlcommons.org:croissant-governance"]
    assert governance["@id"] == "https://example.invalid/croissant/context-array"
    assert tuple(dict(creator) for creator in governance["creator"]) == (
        {"name": "Fixture maintainer"},
    )
    assert governance["usageInfo"] == "Synthetic offline fixture only."
    assert dict(governance["provenance"]) == {"wasGeneratedBy": "simulation"}


def test_croissant_fixture_inspection_is_descriptor_only() -> None:
    """Inspection identifies the profile without creating receipts or governance output."""
    descriptor_path = _FIXTURE_ROOT / "valid" / "context-array-governed-croissant.json"

    inspection = default_registry().inspect(descriptor_path)

    assert inspection["provider_id"] == "croissant"
    assert inspection["capabilities"]["format_versions"] == ("1.1",)
    assert set(inspection) == {"descriptor", "provider_id", "capabilities"}
