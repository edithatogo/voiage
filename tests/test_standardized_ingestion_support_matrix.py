"""Keep published ingestion compatibility and security claims executable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from voiage.ingestion import SourceAccessPolicy
from voiage.ingestion.base import IngestionError
from voiage.ingestion.croissant import CroissantProvider
from voiage.ingestion.frictionless import FrictionlessProvider

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "specs/ingestion/support-matrix-v1.json"


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_support_matrix_matches_provider_capabilities_and_optional_extras() -> None:
    """The matrix cannot advertise a profile absent from the built-in adapters."""
    matrix = _matrix()
    assert matrix["schema_version"] == "voiage-ingestion-support-matrix-v1"
    assert matrix["runtime"] == {
        "python_supported": ["3.12", "3.13", "3.14"],
        "python_free_threaded": "observation-only",
    }
    providers = cast("dict[str, dict[str, object]]", matrix["providers"])
    assert providers["croissant"]["format_versions"] == list(
        CroissantProvider.capabilities.format_versions
    )
    assert providers["croissant"]["media_types"] == list(
        CroissantProvider.capabilities.media_types
    )
    assert providers["frictionless"]["format_versions"] == list(
        FrictionlessProvider.capabilities.format_versions
    )
    assert providers["frictionless"]["media_types"] == list(
        FrictionlessProvider.capabilities.media_types
    )
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras = cast("dict[str, list[object]]", matrix["optional_extras"])
    for extra in extras:
        assert f"{extra} = []" in project


def test_support_matrix_security_claims_are_local_and_workflow_scoped(
    tmp_path: Path,
) -> None:
    """Release/security rows do not silently claim hosted execution or transport."""
    matrix = _matrix()
    security = cast("dict[str, object]", matrix["security_and_release"])
    assert security["hosted_evidence"] == (
        "workflow declarations are not a completed release or hosted-check result"
    )
    controls = cast("list[str]", security["workflow_controls"])
    assert all((ROOT / path).is_file() for path in controls)
    policy = SourceAccessPolicy(tmp_path)
    with pytest.raises(IngestionError, match="network resource access is disabled"):
        policy.resolve("https://example.invalid/data.csv")
    with pytest.raises(IngestionError, match="archive resources"):
        policy.resolve("data.csv.zip")
