"""Explicit provider registration without package-import side effects."""

from __future__ import annotations

from importlib.metadata import entry_points
import json
from pathlib import Path  # noqa: TC003 - public runtime annotation
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from importlib.metadata import EntryPoint

from voiage.contracts.normalized_input import (
    NormalizedInputBundle,  # noqa: TC001 - public runtime API
)
from voiage.ingestion.base import IngestionError, IngestionProvider, SourceAccessPolicy

_ENTRY_POINT_GROUP = "voiage.ingestion.providers"


def _validate_provider(provider: object) -> IngestionProvider:
    """Validate an explicitly loaded provider without importing its package."""
    if not (
        isinstance(getattr(provider, "provider_id", None), str)
        and callable(getattr(provider, "can_handle", None))
        and callable(getattr(provider, "ingest", None))
        and getattr(provider, "capabilities", None) is not None
    ):
        raise IngestionError("entry-point provider does not satisfy the provider contract")
    return cast("IngestionProvider", provider)


def discover_entry_point_providers(
    *,
    allowlist: Collection[str],
    resolver: Callable[..., Collection[EntryPoint]] = entry_points,
) -> tuple[IngestionProvider, ...]:
    """Load only explicitly allow-listed third-party provider entry points.

    Discovery is opt-in: an empty allow-list does not inspect installed entry
    points, and providers outside the allow-list are never imported. Entry
    points must return initialized provider instances rather than factories.
    """
    allowed = frozenset(allowlist)
    if not allowed:
        return ()
    candidates = resolver(group=_ENTRY_POINT_GROUP)
    providers: list[IngestionProvider] = []
    loaded_names: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.name):
        if candidate.name not in allowed:
            continue
        try:
            provider = candidate.load()
        except Exception as error:
            raise IngestionError("allow-listed provider could not be loaded") from error
        providers.append(_validate_provider(provider))
        loaded_names.add(candidate.name)
    missing = allowed - loaded_names
    if missing:
        raise IngestionError("allow-listed provider entry point is unavailable")
    return tuple(providers)


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
