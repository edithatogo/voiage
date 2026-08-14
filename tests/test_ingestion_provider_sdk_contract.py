"""Versioned consumer contract tests for the provider SDK v1 surface."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from voiage.ingestion import (
    INGESTION_PROVIDER_SDK_VERSION,
    IngestionError,
    ProviderCapabilities,
    ProviderRegistry,
)

if TYPE_CHECKING:
    from voiage.contracts import NormalizedInputBundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "specs/core-api/fixtures/v2/ingestion-provider-sdk-v1.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_provider_sdk_v1_fixture_matches_the_public_consumer_surface() -> None:
    """The versioned fixture prevents accidental protocol-surface drift."""
    fixture = _fixture()

    assert fixture["sdk_version"] == INGESTION_PROVIDER_SDK_VERSION
    assert fixture["entry_point_group"] == "voiage.ingestion.providers"
    assert fixture["provider_protocol"] == {
        "attributes": ["provider_id", "capabilities"],
        "methods": {
            "can_handle": "(descriptor: dict[str, object]) -> bool",
            "ingest": (
                "(descriptor_path: Path, *, policy: SourceAccessPolicy) -> "
                "NormalizedInputBundle"
            ),
        },
    }
    assert fixture["capability_fields"] == [
        field.name for field in fields(ProviderCapabilities)
    ]


def test_provider_sdk_fixture_keeps_optional_profiles_dependency_neutral() -> None:
    """Named ingestion extras cannot silently become parser-runtime dependencies."""
    extras = _fixture()["optional_extra_profiles"]

    assert extras == {
        "croissant": {
            "dependencies": [],
            "status": "named-reservation",
            "reason": (
                "The built-in local CSV profile must not delegate materialization "
                "to an unreceipted parser stack."
            ),
        },
        "frictionless": {
            "dependencies": [],
            "status": "named-reservation",
            "reason": (
                "The built-in local CSV profile must not delegate materialization "
                "to an unreceipted parser stack."
            ),
        },
        "ingestion": {
            "dependencies": [],
            "status": "aggregate-reservation",
            "reason": (
                "The aggregate extra remains dependency-neutral while both built-in "
                "profiles use the base Arrow and JSON stack."
            ),
        },
    }


def test_registry_rejects_an_empty_provider_identity() -> None:
    """A consumer cannot publish an ambiguous provider capability manifest."""

    class EmptyProvider:
        provider_id = ""
        capabilities = ProviderCapabilities(
            provider_id="", format_versions=("1",), media_types=("text/csv",)
        )

        def can_handle(self, descriptor: dict[str, object]) -> bool:
            return False

        def ingest(self, *args: object, **kwargs: object) -> NormalizedInputBundle:
            raise AssertionError("the registry must reject this provider first")

    with pytest.raises(IngestionError, match="provider contract"):
        ProviderRegistry((EmptyProvider(),))
