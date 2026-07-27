"""Behavioural evidence for verified offline source materializations."""

from __future__ import annotations

import hashlib
import shutil

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


def test_materialize_verifies_declared_byte_size_online_and_offline(tmp_path) -> None:
    """Byte-size declarations constrain both source materialization and replay."""
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    digest = _digest(payload)
    cache = tmp_path / "cache"
    online = SourceAccessPolicy(tmp_path, cache_dir=cache)

    with pytest.raises(IngestionError, match="byte size does not match"):
        online.materialize("source.csv", sha256=digest, byte_size=len(payload) + 1)
    online.materialize("source.csv", sha256=digest, byte_size=len(payload))
    source.unlink()

    with pytest.raises(IngestionError, match="byte size does not match"):
        SourceAccessPolicy(tmp_path, cache_dir=cache, offline=True).materialize(
            "source.csv", sha256=digest, byte_size=len(payload) + 1
        )


def test_materialize_rejects_negative_declared_byte_size(tmp_path) -> None:
    with pytest.raises(IngestionError, match="byte size must be non-negative"):
        SourceAccessPolicy(tmp_path).materialize("source.csv", byte_size=-1)


def test_offline_materialize_requires_a_digest_and_never_refreshes(tmp_path) -> None:
    cache = tmp_path / "cache"
    policy = SourceAccessPolicy(tmp_path, cache_dir=cache, offline=True)

    with pytest.raises(IngestionError, match="requires an expected SHA-256"):
        policy.materialize("missing.csv")
    with pytest.raises(IngestionError, match="verified offline materialization"):
        policy.materialize("missing.csv", sha256=_digest(b"missing"))


def test_offline_materialize_without_a_cache_rejects_a_valid_digest(tmp_path) -> None:
    with pytest.raises(IngestionError, match="verified offline materialization"):
        SourceAccessPolicy(tmp_path, offline=True).materialize(
            "missing.csv", sha256=_digest(b"missing")
        )


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


def test_materialize_without_cache_returns_the_verified_source(tmp_path) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)

    assert (
        SourceAccessPolicy(tmp_path).materialize("source.csv", sha256=_digest(payload))
        == source
    )


@pytest.mark.parametrize("digest", ["not-a-digest", "g" * 64])
def test_materialize_rejects_malformed_expected_digest(tmp_path, digest) -> None:
    with pytest.raises(IngestionError, match="lowercase hexadecimal"):
        SourceAccessPolicy(tmp_path).materialize("missing.csv", sha256=digest)


def test_materialize_rejects_corrupted_cached_data_on_replay_and_cache_hit(
    tmp_path,
) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    digest = _digest(payload)
    cache = tmp_path / "cache"
    policy = SourceAccessPolicy(tmp_path, cache_dir=cache)
    cached = policy.materialize("source.csv", sha256=digest)

    assert policy.materialize("source.csv", sha256=digest) == cached
    cached.write_bytes(b"modified")
    with pytest.raises(IngestionError, match="cached materialization checksum"):
        policy.materialize("source.csv", sha256=digest)
    with pytest.raises(IngestionError, match="cached materialization checksum"):
        SourceAccessPolicy(tmp_path, cache_dir=cache, offline=True).materialize(
            "source.csv", sha256=digest
        )


def test_materialize_rejects_a_symlinked_cache_entry(tmp_path) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    digest = _digest(payload)
    cache = tmp_path / "cache"
    policy = SourceAccessPolicy(tmp_path, cache_dir=cache)
    cached = policy.materialize("source.csv", sha256=digest)
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(payload)
    cached.unlink()
    cached.symlink_to(replacement)

    with pytest.raises(IngestionError, match="cached materialization checksum"):
        policy.materialize("source.csv", sha256=digest)


def test_materialize_rejects_a_corrupt_copy_result(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    payload = b"value\n1\n"
    source.write_bytes(payload)
    original_copyfile = shutil.copyfile

    def corrupt_copy(source_path, target_path):
        original_copyfile(source_path, target_path)
        target_path.write_bytes(b"corrupt")

    monkeypatch.setattr("voiage.ingestion.base.shutil.copyfile", corrupt_copy)
    with pytest.raises(IngestionError, match="cached materialization checksum"):
        SourceAccessPolicy(tmp_path, cache_dir=tmp_path / "cache").materialize(
            "source.csv", sha256=_digest(payload)
        )
