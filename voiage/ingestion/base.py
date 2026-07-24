"""Source access and provider interfaces kept outside the conductor core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - protocol runtime annotation
from typing import Protocol

from voiage.contracts.normalized_input import (
    NormalizedInputBundle,  # noqa: TC001 - protocol API
)


class IngestionError(ValueError):
    """Stable, safe error raised when an external source cannot be ingested."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Conservative, source-neutral statement of a provider's support surface."""

    provider_id: str
    format_versions: tuple[str, ...]
    media_types: tuple[str, ...]
    supported_transforms: tuple[str, ...] = ()
    supports_projection: bool = False
    supports_filtering: bool = False
    supports_streaming: bool = False
    supports_random_access: bool = False


class SourceAccessPolicy:
    """Fail-closed local source policy for descriptor-relative resources."""

    def __init__(self, root: Path, *, allow_network: bool = False) -> None:
        self.root = root.resolve()
        self.allow_network = allow_network

    def resolve(self, reference: str) -> Path:
        """Resolve a relative local reference without allowing path traversal."""
        if "://" in reference:
            if not self.allow_network:
                raise IngestionError("network resource access is disabled by policy")
            raise IngestionError("network resource access is not implemented")
        candidate = (self.root / reference).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise IngestionError("resource path escapes the configured source root")
        if not candidate.is_file():
            raise IngestionError("declared resource does not exist")
        return candidate


class IngestionProvider(Protocol):
    """Adapter protocol; implementations must return only a normalized bundle."""

    provider_id: str
    capabilities: ProviderCapabilities

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Return whether this provider recognizes the already-read descriptor."""

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy
    ) -> NormalizedInputBundle:
        """Parse one descriptor and materialize its explicitly declared resources."""
