"""Tests for deterministic build-evidence comparison."""

from __future__ import annotations

import io
import shutil
import tarfile
from typing import TYPE_CHECKING
import zipfile

import pytest

from scripts.reproducible_build import (
    artifact_evidence,
    compare_build_directories,
    normalize_sdist,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    report = compare_build_directories(first, second)

    assert report["reproducible"] is True
    assert report["artifacts"][0]["byte_identical"] is True


def test_inventory_identity_does_not_hide_byte_drift(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "package-1.0-py3-none-any.whl"
    _write_wheel(first / name, b"value = 1\n")
    _write_wheel(second / name, b"value = 2\n")

    report = compare_build_directories(first, second)

    assert report["reproducible"] is False
    assert report["artifacts"][0]["inventory_identical"] is False


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
