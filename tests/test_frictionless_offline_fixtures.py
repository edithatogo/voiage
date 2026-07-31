"""File-backed Frictionless Data Package profile conformance fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from voiage.ingestion import default_registry
from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.frictionless import FrictionlessProvider

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "frictionless_v1"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("unsupported/malformed-resource.json", "resources must be objects"),
        ("unsupported/non-object-root.json", "descriptor root must be a JSON object"),
        (
            "unsupported/non-comma-dialect.json",
            "CSV resources require a .csv path and comma delimiter",
        ),
        ("unsupported/non-csv-format.json", "Parquet resources require a .parquet"),
        ("unsupported/integrity-declaration.json", "hash must be a SHA-256"),
        ("unsupported/unsupported-type.json", "unsupported Data Package field type"),
        (
            "unsupported/unsupported-constraint.json",
            "unsupported Data Package field constraint",
        ),
        (
            "unsupported/declared-missing-values.json",
            "does not support schema missingValues",
        ),
        ("unsupported/required-null.json", "required field contains null"),
        ("unsupported/duplicate-primary-key.json", "primaryKey contains duplicate"),
        ("unsupported/unknown-primary-key.json", "primaryKey references an unknown"),
        (
            "unsupported/duplicate-schema-fields.json",
            "ambiguous duplicate field names",
        ),
        (
            "unsupported/unsupported-dialect-property.json",
            "CSV resources require a .csv path and comma delimiter",
        ),
        (
            "unsupported/tsv-wrong-dialect.json",
            "TSV resources require an explicit tab delimiter",
        ),
        (
            "unsupported/json-table-envelope.json",
            "JSON Table resource must contain a JSON array",
        ),
    ],
)
def test_frictionless_offline_unsupported_profile_fixtures_fail_closed(
    fixture: str, message: str
) -> None:
    """Every retained unsupported feature has a stable offline fixture."""
    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(
            _FIXTURE_ROOT / fixture, policy=SourceAccessPolicy(_FIXTURE_ROOT)
        )


def test_frictionless_offline_valid_profile_fixture_materializes() -> None:
    """The supported fixture preserves schema, key, and governance metadata."""
    bundle = FrictionlessProvider().ingest(
        _FIXTURE_ROOT / "valid" / "datapackage.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.manifest.tables[0].primary_key == ("scenario_id",)
    assert bundle.table("operations_samples").to_pylist() == [
        {"scenario_id": 1, "net_benefit": 100.0},
        {"scenario_id": 2, "net_benefit": 80.0},
    ]
    assert bundle.manifest.provenance.license == "CC-BY-4.0"
    assert bundle.manifest.extensions["frictionlessdata.org:profile"] == (
        "tabular-data-package"
    )


def test_frictionless_offline_tsv_profile_fixture_materializes() -> None:
    """The fixture corpus records the exact supported tab-separated profile."""
    bundle = FrictionlessProvider().ingest(
        _FIXTURE_ROOT / "valid" / "tsv-datapackage.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.table("operations_tsv").to_pylist() == [
        {"scenario_id": 1, "net_benefit": 100.0},
        {"scenario_id": 2, "net_benefit": 80.0},
    ]
    assert bundle.manifest.resources[0].media_type == "text/tab-separated-values"


def test_frictionless_offline_json_table_profile_fixture_materializes() -> None:
    """The fixture corpus records the exact supported JSON Table profile."""
    bundle = FrictionlessProvider().ingest(
        _FIXTURE_ROOT / "valid" / "json-table-datapackage.json",
        policy=SourceAccessPolicy(_FIXTURE_ROOT),
    )

    assert bundle.table("operations_json").to_pylist() == [
        {"scenario_id": 1, "net_benefit": 100.0},
        {"scenario_id": 2, "net_benefit": 80.0},
    ]
    assert bundle.manifest.resources[0].media_type == "application/json"


def test_frictionless_offline_receipted_fixture_preserves_declared_receipt() -> None:
    """A checked local CSV exposes its immutable receipt without inference."""
    descriptor = _FIXTURE_ROOT / "valid" / "receipted-datapackage.json"

    bundle = FrictionlessProvider().ingest(
        descriptor, policy=SourceAccessPolicy(_FIXTURE_ROOT)
    )

    receipt = bundle.manifest.resources[0]
    assert receipt.resource_id == "operations_samples"
    assert (
        receipt.sha256
        == "2fa13daf9b358e05326d2e3f04c5a9a27bf07097aa63e01f82817f0f1905149d"
    )
    assert receipt.byte_size == 39
    assert receipt.uri == (_FIXTURE_ROOT / "valid" / "receipted-data.csv").as_uri()
    assert bundle.manifest.provenance.citation == "Voiage synthetic fixture (2026)"


def test_frictionless_offline_receipted_fixture_replays_only_verified_data(
    tmp_path: Path,
) -> None:
    """The corpus proves provider-level offline replay, not only policy helpers."""
    fixture_root = tmp_path / "frictionless_v1"
    shutil.copytree(_FIXTURE_ROOT, fixture_root)
    descriptor = fixture_root / "valid" / "receipted-datapackage.json"
    cache_dir = tmp_path / "cache"

    FrictionlessProvider().ingest(
        descriptor,
        policy=SourceAccessPolicy(fixture_root, cache_dir=cache_dir),
    )
    (fixture_root / "valid" / "receipted-data.csv").unlink()

    replay = FrictionlessProvider().ingest(
        descriptor,
        policy=SourceAccessPolicy(fixture_root, cache_dir=cache_dir, offline=True),
    )

    assert replay.table("operations_samples").num_rows == 2
    assert replay.manifest.resources[0].sha256 == (
        "2fa13daf9b358e05326d2e3f04c5a9a27bf07097aa63e01f82817f0f1905149d"
    )


def test_frictionless_inspection_fixture_never_materializes_an_absent_resource() -> (
    None
):
    """Inspection is descriptor-only even when a declared local resource is absent."""
    descriptor = _FIXTURE_ROOT / "valid" / "inspect-only-datapackage.json"

    inspection = default_registry().inspect(descriptor)

    assert inspection["provider_id"] == "frictionless"
    assert inspection["descriptor"] == str(descriptor)
    capabilities = inspection["capabilities"]
    assert isinstance(capabilities, dict)
    assert cast("dict[str, object]", capabilities)["format_versions"] == ("1",)


def test_frictionless_rejects_non_object_resource_descriptors(tmp_path) -> None:
    """Resources must be objects before any materialization is attempted."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps({"resources": ["not-a-resource"]}), encoding="utf-8"
    )

    with pytest.raises(IngestionError, match="resources must be objects"):
        FrictionlessProvider().ingest(descriptor, policy=SourceAccessPolicy(tmp_path))


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {
                "fields": [
                    {
                        "name": "id",
                        "constraints": {"minimum": 1},
                    }
                ]
            },
            "unsupported Data Package field constraint",
        ),
        (
            {
                "fields": [
                    {
                        "name": "id",
                        "constraints": {"required": "yes"},
                    }
                ]
            },
            "constraints must be boolean",
        ),
        (
            {
                "fields": [{"name": "id"}],
                "missingValues": [""],
            },
            "does not support schema missingValues",
        ),
    ],
)
def test_frictionless_rejects_semantic_schema_claims_before_reading_resources(
    tmp_path: Path, schema: dict[str, object], message: str
) -> None:
    """Unsupported schema semantics fail before a descriptor can read its path."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "missing.csv",
                        "schema": schema,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(descriptor, policy=SourceAccessPolicy(tmp_path))


def test_frictionless_rejects_duplicate_resource_names(tmp_path) -> None:
    """Resource identifiers must remain unambiguous in normalized bundles."""
    (tmp_path / "first.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "second.csv").write_text("id\n2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "samples",
                        "path": "first.csv",
                        "schema": {"fields": [{"name": "id", "type": "integer"}]},
                    },
                    {
                        "name": "samples",
                        "path": "second.csv",
                        "schema": {"fields": [{"name": "id", "type": "integer"}]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="resource names must be unique"):
        FrictionlessProvider().ingest(descriptor, policy=SourceAccessPolicy(tmp_path))


def test_frictionless_multiple_resources_preserve_and_validate_foreign_keys(
    tmp_path,
) -> None:
    """A package relationship becomes a checked normalized KeyReference."""
    (tmp_path / "parents.csv").write_text("id\n1\n2\n", encoding="utf-8")
    (tmp_path / "children.csv").write_text("parent_id\n1\n2\n", encoding="utf-8")
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        '{"resources":[{"name":"parents","path":"parents.csv","schema":{"primaryKey":"id","fields":[{"name":"id","type":"integer"}]}},{"name":"children","path":"children.csv","schema":{"fields":[{"name":"parent_id","type":"integer"}],"foreignKeys":[{"fields":"parent_id","reference":{"resource":"parents","fields":"id"}}]}}]}',
        encoding="utf-8",
    )

    bundle = FrictionlessProvider().ingest(
        descriptor, policy=SourceAccessPolicy(tmp_path)
    )

    assert set(bundle.tables) == {"parents", "children"}
    assert bundle.manifest.key_references[0].target_table_id == "parents"


@pytest.mark.parametrize(
    ("foreign_key", "message"),
    [
        ("not-a-list", "foreignKeys must be a list"),
        (["not-an-object"], "foreignKeys require a reference"),
        (
            [
                {
                    "fields": "parent_id",
                    "reference": {"resource": 1, "fields": "id"},
                }
            ],
            "foreignKeys require resource and fields",
        ),
        (
            [
                {
                    "fields": "parent_id",
                    "reference": {"resource": "missing", "fields": "id"},
                }
            ],
            "foreignKey references an unknown resource",
        ),
        (
            [
                {
                    "fields": "parent_id",
                    "reference": {"resource": "parents", "fields": "missing"},
                }
            ],
            "foreignKey references an unknown field",
        ),
    ],
)
def test_frictionless_rejects_adversarial_foreign_key_descriptors(
    tmp_path, foreign_key: object, message: str
) -> None:
    """Relationship metadata is fail-closed before it reaches the conductor."""
    (tmp_path / "parents.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "children.csv").write_text("parent_id\n1\n", encoding="utf-8")
    descriptor = {
        "resources": [
            {
                "name": "parents",
                "path": "parents.csv",
                "schema": {"fields": [{"name": "id", "type": "integer"}]},
            },
            {
                "name": "children",
                "path": "children.csv",
                "schema": {
                    "fields": [{"name": "parent_id", "type": "integer"}],
                    "foreignKeys": foreign_key,
                },
            },
        ]
    }
    path = tmp_path / "datapackage.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(IngestionError, match=message):
        FrictionlessProvider().ingest(path, policy=SourceAccessPolicy(tmp_path))


def test_frictionless_rejects_foreign_key_values_absent_from_target(tmp_path) -> None:
    """A syntactically valid relationship must also be referentially valid."""
    (tmp_path / "parents.csv").write_text("id\n1\n", encoding="utf-8")
    (tmp_path / "children.csv").write_text("parent_id\n2\n", encoding="utf-8")
    path = tmp_path / "datapackage.json"
    path.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "name": "parents",
                        "path": "parents.csv",
                        "schema": {"fields": [{"name": "id", "type": "integer"}]},
                    },
                    {
                        "name": "children",
                        "path": "children.csv",
                        "schema": {
                            "fields": [{"name": "parent_id", "type": "integer"}],
                            "foreignKeys": [
                                {
                                    "fields": "parent_id",
                                    "reference": {
                                        "resource": "parents",
                                        "fields": "id",
                                    },
                                }
                            ],
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IngestionError, match="foreignKey references a missing value"):
        FrictionlessProvider().ingest(path, policy=SourceAccessPolicy(tmp_path))
