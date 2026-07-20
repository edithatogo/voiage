"""Tests for deterministic build-evidence comparison."""

from __future__ import annotations

import io
from pathlib import Path
import re
import shutil
import tarfile
import zipfile

import pytest

from scripts.reproducible_build import (
    _reproducible_environment,
    _source_identity,
    artifact_evidence,
    compare_build_directories,
    normalize_sdist,
)

ROOT = Path(__file__).resolve().parents[1]


def test_nested_build_identity_is_a_complete_git_commit_and_tree() -> None:
    """Git-less sdist builds receive the exact immutable source identity."""
    revision, tree = _source_identity(ROOT)

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert re.fullmatch(r"[0-9a-f]{40}", tree)


def test_nested_build_environment_carries_complete_clean_source_identity() -> None:
    """The sdist-to-wheel build cannot lose release provenance."""
    environment = _reproducible_environment(ROOT)
    revision, tree = _source_identity(ROOT)

    assert environment["VOIAGE_SOURCE_REVISION"] == revision
    assert environment["VOIAGE_SOURCE_TREE_GIT_OID"] == tree
    assert environment["VOIAGE_SOURCE_CLEAN"] == "true"
    assert environment["SOURCE_DATE_EPOCH"].isdigit()


def _write_wheel(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("package/module.py", payload)


def _write_sdist(path: Path, *, mtime: int) -> None:
    payload = b"value = 1\n"
    member = tarfile.TarInfo("package/module.py")
    member.size = len(payload)
    member.mtime = mtime
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(payload))


def test_identical_build_artifacts_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    artifact = first / "package-1.0-py3-none-any.whl"
    _write_wheel(artifact, b"value = 1\n")
    _ = shutil.copyfile(artifact, second / artifact.name)
    source = first / "package-1.0.tar.gz"
    _write_sdist(source, mtime=1)
    _ = shutil.copyfile(source, second / source.name)

    report = compare_build_directories(first, second)

    assert report["reproducible"] is True
    assert report["artifact_set_complete"] is True
    assert report["artifacts"][0]["byte_identical"] is True


def test_inventory_identity_does_not_hide_byte_drift(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "package-1.0-py3-none-any.whl"
    _write_wheel(first / name, b"value = 1\n")
    _write_wheel(second / name, b"value = 2\n")
    for root in (first, second):
        _write_sdist(root / "package-1.0.tar.gz", mtime=1)

    report = compare_build_directories(first, second)

    assert report["reproducible"] is False
    assert report["artifacts"][0]["inventory_identical"] is False


def test_missing_sdist_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        _write_wheel(root / "package-1.0-py3-none-any.whl", b"value = 1\n")

    report = compare_build_directories(first, second)

    assert report["artifact_set_complete"] is False
    assert report["reproducible"] is False


def test_non_archive_artifact_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "package.bin"
    _ = artifact.write_bytes(b"opaque")

    with pytest.raises(ValueError, match="unsupported build artifact"):
        _ = artifact_evidence(artifact)


def test_sdist_normalization_removes_archive_metadata_drift(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=1)
    _write_sdist(second, mtime=2)

    normalize_sdist(first, epoch=123)
    normalize_sdist(second, epoch=123)

    assert first.read_bytes() == second.read_bytes()
