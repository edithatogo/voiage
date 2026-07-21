"""Tests for deterministic build-evidence comparison."""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
from typing import TYPE_CHECKING
import zipfile

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.reproducible_build import (
    _build_environment,
    _embed_sdist_provenance,
    artifact_evidence,
    compare_build_directories,
    normalize_sdist,
)


def test_sdist_provenance_helper_is_required_and_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproducible builds must create identity for the Git-less rebuild."""
    script = tmp_path / "scripts" / "embed_sdist_provenance.py"
    script.parent.mkdir()
    script.write_text("# fixed repository helper\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)

    _embed_sdist_provenance(tmp_path)

    assert observed["command"][-1] == str(script)
    assert observed["cwd"] == tmp_path
    assert observed["check"] is True


def test_missing_sdist_provenance_helper_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing sdist provenance helper"):
        _embed_sdist_provenance(tmp_path)


def test_windows_native_builds_share_a_reproducible_clean_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=x86-64")
    monkeypatch.setattr(
        "scripts.reproducible_build._source_identity", lambda _repo: ("rev", "tree")
    )
    monkeypatch.setattr(
        "scripts.reproducible_build._source_date_epoch", lambda _repo: "123"
    )
    target = tmp_path / "stable-target"

    environment = _build_environment(
        tmp_path, target, platform_name="nt", source_date_epoch="123"
    )

    assert environment["CARGO_TARGET_DIR"] == str(target.resolve())
    assert environment["RUSTFLAGS"] == "-C target-cpu=x86-64 -C link-arg=/Brepro"
    assert environment["VOIAGE_SOURCE_REVISION"] == "rev"
    assert environment["VOIAGE_SOURCE_TREE_GIT_OID"] == "tree"
    assert environment["VOIAGE_SOURCE_CLEAN"] == "true"


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
    assert report["artifacts"][0]["differing_entries"] == []


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
    assert report["artifacts"][0]["differing_entries"] == ["package/module.py"]


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
