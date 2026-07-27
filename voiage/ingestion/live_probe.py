"""Fail-closed evidence helper for manually authorized provider probes."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.registry import default_registry

if TYPE_CHECKING:
    from pathlib import Path


class AuthoritativeProbeGateError(IngestionError):
    """Raised when authoritative interoperability evidence is not fully pinned."""


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of one locally staged probe artifact."""
    return sha256(path.read_bytes()).hexdigest()


def run_authoritative_probe(
    descriptor: Path,
    *,
    source_root: Path,
    expected_descriptor_sha256: str,
    expected_resource_sha256: str,
    enabled: bool,
) -> dict[str, str]:
    """Validate one approved local descriptor without performing network I/O.

    Callers must acquire and approve the source separately. The probe is disabled
    by default and checks descriptor and materialization digests before emitting
    a compact receipt suitable for evidence ledgers.
    """
    if not enabled:
        raise AuthoritativeProbeGateError(
            "authoritative live probe must be explicitly enabled"
        )
    if _digest(descriptor) != expected_descriptor_sha256:
        raise AuthoritativeProbeGateError("descriptor digest does not match approval")
    try:
        bundle = default_registry().ingest(
            descriptor, policy=SourceAccessPolicy(source_root)
        )
    except IngestionError as error:
        raise AuthoritativeProbeGateError(
            "approved descriptor did not materialize"
        ) from error
    if len(bundle.manifest.resources) != 1:
        raise AuthoritativeProbeGateError("probe requires exactly one resource receipt")
    resource = bundle.manifest.resources[0]
    if resource.sha256 != expected_resource_sha256:
        raise AuthoritativeProbeGateError("resource digest does not match approval")
    return {
        "descriptor_sha256": expected_descriptor_sha256,
        "provider_id": bundle.manifest.provenance.provider_id,
        "resource_sha256": resource.sha256,
    }
