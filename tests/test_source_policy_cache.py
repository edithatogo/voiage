"""Behavioural evidence for verified offline source materializations."""

from __future__ import annotations

import hashlib

import pytest

from voiage.ingestion import IngestionError, SourceAccessPolicy


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_materialize_writes_a_content_addressed_verified_cache(tmp_path) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    cache = tmp_path / "cache"
    policy = SourceAccessPolicy(tmp_path, cache_dir=cache, cache_namespace="fixture")

    materialized = policy.materialize("source.csv", sha256=_digest(payload))

    assert materialized.read_bytes() == payload
    assert materialized.is_relative_to(cache)
    assert materialized.name == _digest(payload)


def test_offline_materialize_replays_a_verified_cache_without_source(tmp_path) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    digest = _digest(payload)
    cache = tmp_path / "cache"
    online = SourceAccessPolicy(tmp_path, cache_dir=cache, cache_namespace="fixture")
    online.materialize("source.csv", sha256=digest)
    source.unlink()

    offline = SourceAccessPolicy(
        tmp_path, cache_dir=cache, cache_namespace="fixture", offline=True
    )

    assert offline.materialize("source.csv", sha256=digest).read_bytes() == payload


def test_offline_materialize_requires_a_digest_and_never_refreshes(tmp_path) -> None:
    cache = tmp_path / "cache"
    policy = SourceAccessPolicy(tmp_path, cache_dir=cache, offline=True)

    with pytest.raises(IngestionError, match="requires an expected SHA-256"):
        policy.materialize("missing.csv")
    with pytest.raises(IngestionError, match="verified offline materialization"):
        policy.materialize("missing.csv", sha256=_digest(b"missing"))


def test_materialize_rejects_checksum_mismatch_and_cache_context_mismatch(
    tmp_path,
) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    cache = tmp_path / "cache"
    digest = _digest(payload)

    with pytest.raises(IngestionError, match="checksum does not match"):
        SourceAccessPolicy(tmp_path, cache_dir=cache).materialize(
            "source.csv", sha256=_digest(b"different")
        )

    SourceAccessPolicy(tmp_path, cache_dir=cache, cache_namespace="one").materialize(
        "source.csv", sha256=digest
    )
    other_context = SourceAccessPolicy(
        tmp_path, cache_dir=cache, cache_namespace="two", offline=True
    )
    with pytest.raises(IngestionError, match="verified offline materialization"):
        other_context.materialize("source.csv", sha256=digest)
