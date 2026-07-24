"""Explicit provider registration without package-import side effects."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - public runtime annotation

from voiage.contracts.normalized_input import (
    NormalizedInputBundle,  # noqa: TC001 - public runtime API
)
from voiage.ingestion.base import IngestionError, IngestionProvider, SourceAccessPolicy


class ProviderRegistry:
    """A deterministic registry populated only by the application caller."""

    def __init__(self, providers: tuple[IngestionProvider, ...] = ()) -> None:
        self._providers = providers

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy | None = None
    ) -> NormalizedInputBundle:
        """Choose exactly one recognizer and convert its descriptor."""
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IngestionError("descriptor is not valid UTF-8 JSON") from error
        if not isinstance(descriptor, dict):
            raise IngestionError("descriptor root must be a JSON object")
        matches = tuple(
            provider for provider in self._providers if provider.can_handle(descriptor)
        )
        if len(matches) != 1:
            raise IngestionError(
                "descriptor must match exactly one registered provider"
            )
        return matches[0].ingest(
            descriptor_path, policy=policy or SourceAccessPolicy(descriptor_path.parent)
        )


def default_registry() -> ProviderRegistry:
    """Return the built-in parser set without discovering third-party code."""
    from voiage.ingestion.croissant import CroissantProvider
    from voiage.ingestion.frictionless import FrictionlessProvider

    return ProviderRegistry((CroissantProvider(), FrictionlessProvider()))
