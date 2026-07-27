"""File-backed Frictionless Data Package profile conformance fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.frictionless import FrictionlessProvider

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "frictionless_v1"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("unsupported/multiple-resources.json", "exactly one resource"),
        ("unsupported/non-comma-dialect.json", "only CSV comma dialect"),
        ("unsupported/non-csv-format.json", "requires CSV format"),
        ("unsupported/integrity-declaration.json", "integrity declarations"),
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
