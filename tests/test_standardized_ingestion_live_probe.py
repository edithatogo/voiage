"""Fail-closed evidence for manually authorized provider interoperability probes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiage.ingestion.live_probe import (
    AuthoritativeProbeGateError,
    run_authoritative_probe,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "croissant_1_1"
_DESCRIPTOR = _FIXTURE_ROOT / "valid" / "croissant.json"
_RESOURCE = _FIXTURE_ROOT / "valid" / "data.csv"


def test_authoritative_probe_requires_explicit_opt_in() -> None:
    """The gate rejects before reading a descriptor or materializing a resource."""
    with pytest.raises(AuthoritativeProbeGateError, match="explicitly enabled"):
        run_authoritative_probe(
            _DESCRIPTOR,
            source_root=_FIXTURE_ROOT,
            expected_descriptor_sha256="0" * 64,
            expected_resource_sha256="0" * 64,
            enabled=False,
        )


def test_authoritative_probe_records_pinned_provider_receipt() -> None:
    """An enabled, separately approved descriptor returns only stable evidence."""
    descriptor = _DESCRIPTOR
    resource = _RESOURCE

    evidence = run_authoritative_probe(
        descriptor,
        expected_descriptor_sha256=sha256(descriptor.read_bytes()).hexdigest(),
        expected_resource_sha256=sha256(resource.read_bytes()).hexdigest(),
        enabled=True,
        source_root=_FIXTURE_ROOT,
    )

    assert evidence == {
        "descriptor_sha256": sha256(descriptor.read_bytes()).hexdigest(),
        "provider_id": "croissant",
        "resource_sha256": sha256(resource.read_bytes()).hexdigest(),
    }


def test_authoritative_probe_rejects_unpinned_resource_content() -> None:
    """A source substitution cannot produce an authoritative-looking receipt."""
    descriptor = _DESCRIPTOR
    with pytest.raises(AuthoritativeProbeGateError, match="resource digest"):
        run_authoritative_probe(
            descriptor,
            expected_descriptor_sha256=sha256(descriptor.read_bytes()).hexdigest(),
            expected_resource_sha256="0" * 64,
            enabled=True,
            source_root=_FIXTURE_ROOT,
        )


def test_authoritative_probe_rejects_unpinned_descriptor() -> None:
    """A descriptor substitution cannot pass the approval gate."""
    with pytest.raises(AuthoritativeProbeGateError, match="descriptor digest"):
        run_authoritative_probe(
            _DESCRIPTOR,
            source_root=_FIXTURE_ROOT,
            expected_descriptor_sha256="0" * 64,
            expected_resource_sha256="0" * 64,
            enabled=True,
        )


def test_authoritative_probe_rejects_unmaterialized_descriptor() -> None:
    """A descriptor outside the approved source root fails closed."""
    with pytest.raises(AuthoritativeProbeGateError, match="did not materialize"):
        run_authoritative_probe(
            _DESCRIPTOR,
            source_root=Path(__file__).parent,
            expected_descriptor_sha256=sha256(_DESCRIPTOR.read_bytes()).hexdigest(),
            expected_resource_sha256="0" * 64,
            enabled=True,
        )


def test_authoritative_probe_rejects_multiple_resource_receipts(monkeypatch) -> None:
    """An approved probe remains single-artifact evidence, never an aggregate."""
    bundle = SimpleNamespace(
        manifest=SimpleNamespace(resources=(object(), object())),
    )
    monkeypatch.setattr(
        "voiage.ingestion.live_probe.default_registry",
        lambda: SimpleNamespace(ingest=lambda *_args, **_kwargs: bundle),
    )

    with pytest.raises(AuthoritativeProbeGateError, match="exactly one resource"):
        run_authoritative_probe(
            _DESCRIPTOR,
            source_root=_FIXTURE_ROOT,
            expected_descriptor_sha256=sha256(_DESCRIPTOR.read_bytes()).hexdigest(),
            expected_resource_sha256="0" * 64,
            enabled=True,
        )
