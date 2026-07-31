"""Source access and provider interfaces kept outside the conductor core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path  # noqa: TC003 - protocol runtime annotation
import shutil
from typing import Protocol, runtime_checkable

from voiage.contracts.normalized_input import (
    NormalizedInputBundle,  # noqa: TC001 - protocol API
)


class IngestionError(ValueError):
    """Stable, safe error raised when an external source cannot be ingested."""


INGESTION_PROVIDER_SDK_VERSION = "1"
"""Major version of the frozen public provider-SDK contract."""


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

    def __init__(
        self,
        root: Path,
        *,
        allow_network: bool = False,
        max_resource_bytes: int = 512 * 1024 * 1024,
        cache_dir: Path | None = None,
        cache_namespace: str | None = None,
        offline: bool = False,
    ) -> None:
        if max_resource_bytes <= 0:
            raise ValueError("max_resource_bytes must be positive")
        self.root = root.resolve()
        self.allow_network = allow_network
        self.max_resource_bytes = max_resource_bytes
        self.cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self.offline = offline
        context = cache_namespace or f"root={self.root};network={allow_network}"
        self._cache_context = hashlib.sha256(context.encode("utf-8")).hexdigest()

    def resolve(self, reference: str) -> Path:
        """Resolve a relative local reference without allowing path traversal."""
        candidate = self._local_candidate(reference)
        if not candidate.is_file():
            raise IngestionError("declared resource does not exist")
        if candidate.stat().st_size > self.max_resource_bytes:
            raise IngestionError("declared resource exceeds configured size limit")
        return candidate

    def materialize(
        self,
        reference: str,
        *,
        sha256: str | None = None,
        byte_size: int | None = None,
    ) -> Path:
        """Return a digest-verified local materialization, optionally from cache.

        This method never performs network I/O.  Offline replay is intentionally
        possible only with an expected digest, so a mutable source cannot be
        silently substituted for a previously reviewed materialization.
        """
        expected = self._validate_digest(sha256) if sha256 is not None else None
        if byte_size is not None and byte_size < 0:
            raise IngestionError("expected resource byte size must be non-negative")
        # Validate the descriptor reference even when an offline cache hit means
        # the original source is intentionally unavailable.
        self._local_candidate(reference)
        if self.offline:
            if expected is None:
                raise IngestionError(
                    "offline replay requires an expected SHA-256 digest"
                )
            cached = self._cache_path(expected)
            if not self._is_safe_cached_file(cached):
                raise IngestionError("no verified offline materialization is available")
            if self._digest(cached) != expected:
                raise IngestionError("cached materialization checksum does not match")
            if byte_size is not None and cached.stat().st_size != byte_size:
                raise IngestionError("cached materialization byte size does not match")
            return cached

        source = self.resolve(reference)
        if byte_size is not None and source.stat().st_size != byte_size:
            raise IngestionError("materialized resource byte size does not match")
        actual = self._digest(source)
        if expected is not None and actual != expected:
            raise IngestionError("materialized resource checksum does not match")
        if self.cache_dir is None:
            return source

        cached = self._cache_path(actual)
        assert cached is not None
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            if not self._is_safe_cached_file(cached) or self._digest(cached) != actual:
                raise IngestionError("cached materialization checksum does not match")
            return cached
        shutil.copyfile(source, cached)
        if self._digest(cached) != actual:
            raise IngestionError("cached materialization checksum does not match")
        return cached

    def source_uri(self, reference: str) -> str:
        """Return the validated logical local URI without requiring the source."""
        return self._local_candidate(reference).as_uri()

    def _cache_path(self, digest: str) -> Path | None:
        if self.cache_dir is None:
            return None
        candidate = self.cache_dir / self._cache_context / digest[:2] / digest
        resolved_candidate = candidate.parent.resolve()
        if (
            resolved_candidate != self.cache_dir
            and self.cache_dir not in resolved_candidate.parents
        ):
            raise IngestionError("cache path escapes the configured cache root")
        return candidate

    def _local_candidate(self, reference: str) -> Path:
        """Validate a descriptor-relative local path before any file operation."""
        if "://" in reference:
            if not self.allow_network:
                raise IngestionError("network resource access is disabled by policy")
            raise IngestionError("network resource access is not implemented")
        candidate = (self.root / reference).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise IngestionError("resource path escapes the configured source root")
        return candidate

    @staticmethod
    def _is_safe_cached_file(candidate: Path | None) -> bool:
        """Return whether a cache object has no alternate writable path."""
        if candidate is None or not candidate.is_file() or candidate.is_symlink():
            return False
        return candidate.stat().st_nlink == 1

    @staticmethod
    def _validate_digest(digest: str) -> str:
        candidate = digest.lower()
        if len(candidate) != 64 or any(
            char not in "0123456789abcdef" for char in candidate
        ):
            raise IngestionError(
                "expected SHA-256 digest must be lowercase hexadecimal"
            )
        return candidate

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


@runtime_checkable
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
