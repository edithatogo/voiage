"""File-backed Frictionless Data Package profile conformance fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.frictionless import FrictionlessProvider

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "frictionless_v1"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("unsupported/non-comma-dialect.json", "only CSV comma dialect"),
        ("unsupported/non-csv-format.json", "requires CSV format"),
        ("unsupported/integrity-declaration.json", "hash must be a SHA-256"),
        ("unsupported/unsupported-type.json", "unsupported Data Package field type"),
        ("unsupported/required-null.json", "required field contains null"),
        ("unsupported/duplicate-primary-key.json", "primaryKey contains duplicate"),
        ("unsupported/unknown-primary-key.json", "primaryKey references an unknown"),
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


def test_frictionless_rejects_non_object_resource_descriptors(tmp_path) -> None:
    """Resources must be objects before any materialization is attempted."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(
        json.dumps({"resources": ["not-a-resource"]}), encoding="utf-8"
    )

    with pytest.raises(IngestionError, match="resources must be objects"):
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
