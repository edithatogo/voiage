"""Focused contracts for the provenance-aware PEP 517 build backend."""

from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import TYPE_CHECKING

import pytest
import yaml

import build_backend

if TYPE_CHECKING:
    from typing import Any

REVISION = "0123456789abcdef0123456789abcdef01234567"
TREE = "89abcdef0123456789abcdef0123456789abcdef"
PROVENANCE = f"revision={REVISION}\ntree={TREE}\nclean=true\n".encode()
SOURCE_VARIABLES = (
    "VOIAGE_SOURCE_REVISION",
    "VOIAGE_SOURCE_TREE_GIT_OID",
    "VOIAGE_SOURCE_CLEAN",
)


def _run_git(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    assert git is not None
    return subprocess.run(
        [git, *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initialise_repository(root: Path) -> tuple[str, str]:
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.name", "Voiage Packaging Test")
    _run_git(root, "config", "user.email", "voiage@example.invalid")
    (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "--quiet", "-m", "fixture")
    return (
        _run_git(root, "rev-parse", "HEAD^{commit}"),
        _run_git(root, "rev-parse", "HEAD^{tree}"),
    )


@pytest.fixture
def isolated_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    provenance = root / "rust/crates/voiage-python/source-provenance.txt"
    monkeypatch.setattr(build_backend, "ROOT", root)
    monkeypatch.setattr(build_backend, "PROVENANCE", provenance)
    for name in SOURCE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return root


def _write_sdist(
    directory: Path,
    *,
    contents: bytes = PROVENANCE,
    member_type: bytes | None = None,
) -> str:
    filename = "voiage-1.0.0.tar.gz"
    member = tarfile.TarInfo(
        "voiage-1.0.0/rust/crates/voiage-python/source-provenance.txt"
    )
    member.size = len(contents)
    if member_type is not None:
        member.type = member_type
    with tarfile.open(directory / filename, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(contents))
    return filename


def test_clean_git_identity_is_used_for_an_ordinary_sdist(
    isolated_backend: Path,
) -> None:
    expected = _initialise_repository(isolated_backend)

    assert build_backend._source_identity() == expected


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_dirty_git_source_distribution_fails_closed(
    isolated_backend: Path,
    dirty_kind: str,
) -> None:
    _initialise_repository(isolated_backend)
    if dirty_kind == "tracked":
        (isolated_backend / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        (isolated_backend / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty source tree"):
        build_backend._source_identity()


def test_complete_environment_identity_supports_a_gitless_sdist(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", REVISION)
    monkeypatch.setenv("VOIAGE_SOURCE_TREE_GIT_OID", TREE)
    monkeypatch.setenv("VOIAGE_SOURCE_CLEAN", "true")

    assert build_backend._source_identity() == (REVISION, TREE)


def test_parent_repository_is_not_treated_as_the_archive_repository(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _initialise_repository(isolated_backend.parent)
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", REVISION)
    monkeypatch.setenv("VOIAGE_SOURCE_TREE_GIT_OID", TREE)
    monkeypatch.setenv("VOIAGE_SOURCE_CLEAN", "true")

    assert build_backend._git_identity() is None
    assert build_backend._source_identity() == (REVISION, TREE)


@pytest.mark.parametrize(
    "environment",
    [
        {"VOIAGE_SOURCE_REVISION": REVISION},
        {
            "VOIAGE_SOURCE_REVISION": REVISION,
            "VOIAGE_SOURCE_TREE_GIT_OID": TREE,
        },
        {
            "VOIAGE_SOURCE_REVISION": REVISION,
            "VOIAGE_SOURCE_TREE_GIT_OID": TREE,
            "VOIAGE_SOURCE_CLEAN": "false",
        },
        {
            "VOIAGE_SOURCE_REVISION": REVISION.upper(),
            "VOIAGE_SOURCE_TREE_GIT_OID": TREE,
            "VOIAGE_SOURCE_CLEAN": "true",
        },
    ],
)
def test_incomplete_or_noncanonical_environment_identity_fails_closed(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="requires all of"):
        build_backend._source_identity()


def test_supplied_identity_must_match_a_clean_git_checkout(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _initialise_repository(isolated_backend)
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", REVISION)
    monkeypatch.setenv("VOIAGE_SOURCE_TREE_GIT_OID", TREE)
    monkeypatch.setenv("VOIAGE_SOURCE_CLEAN", "true")

    with pytest.raises(RuntimeError, match="does not match"):
        build_backend._source_identity()


def test_build_sdist_verifies_archive_identity_and_restores_source(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision, tree = _initialise_repository(isolated_backend)
    expected = f"revision={revision}\ntree={tree}\nclean=true\n".encode()
    output = tmp_path / "dist"
    output.mkdir()
    observed: list[bytes] = []

    def fake_build_sdist(
        directory: str,
        _settings: dict[str, Any] | None,
    ) -> str:
        observed.append(build_backend.PROVENANCE.read_bytes())
        return _write_sdist(Path(directory), contents=expected)

    monkeypatch.setattr(build_backend.maturin, "build_sdist", fake_build_sdist)

    filename = build_backend.build_sdist(str(output))

    assert filename == "voiage-1.0.0.tar.gz"
    assert observed == [expected]
    assert not build_backend.PROVENANCE.exists()


def test_build_sdist_removes_an_archive_that_loses_identity(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _initialise_repository(isolated_backend)
    output = tmp_path / "dist"
    output.mkdir()

    def fake_build_sdist(
        directory: str,
        _settings: dict[str, Any] | None,
    ) -> str:
        return _write_sdist(Path(directory), contents=PROVENANCE)

    monkeypatch.setattr(build_backend.maturin, "build_sdist", fake_build_sdist)

    with pytest.raises(RuntimeError, match="did not preserve"):
        build_backend.build_sdist(str(output))

    assert not (output / "voiage-1.0.0.tar.gz").exists()
    assert not build_backend.PROVENANCE.exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"", "canonical three-line"),
        (
            f"revision={REVISION}\ntree={TREE}\nclean=false\n".encode(),
            "canonical three-line",
        ),
        (
            f"tree={TREE}\nrevision={REVISION}\nclean=true\n".encode(),
            "canonical three-line",
        ),
    ],
)
def test_gitless_wheel_rejects_malformed_embedded_identity(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: bytes,
    message: str,
) -> None:
    build_backend.PROVENANCE.parent.mkdir(parents=True)
    build_backend.PROVENANCE.write_bytes(contents)
    called = False

    def fake_build_wheel(*_args: Any, **_kwargs: Any) -> str:
        nonlocal called
        called = True
        return "unexpected.whl"

    monkeypatch.setattr(build_backend.maturin, "build_wheel", fake_build_wheel)

    with pytest.raises(RuntimeError, match=message):
        build_backend.build_wheel(str(tmp_path))

    assert called is False


def test_gitless_wheel_injects_embedded_identity_and_restores_environment(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_backend.PROVENANCE.parent.mkdir(parents=True)
    build_backend.PROVENANCE.write_bytes(PROVENANCE)
    observed: dict[str, str | None] = {}

    def fake_build_wheel(
        _directory: str,
        _settings: dict[str, Any] | None,
        _metadata: str | None,
    ) -> str:
        observed.update({name: os.environ.get(name) for name in SOURCE_VARIABLES})
        return "voiage.whl"

    monkeypatch.setattr(build_backend.maturin, "build_wheel", fake_build_wheel)

    assert build_backend.build_wheel(str(tmp_path)) == "voiage.whl"
    assert observed == {
        "VOIAGE_SOURCE_REVISION": REVISION,
        "VOIAGE_SOURCE_TREE_GIT_OID": TREE,
        "VOIAGE_SOURCE_CLEAN": "true",
    }
    assert all(os.environ.get(name) is None for name in SOURCE_VARIABLES)


def test_gitless_wheel_rejects_environment_that_conflicts_with_sdist(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_backend.PROVENANCE.parent.mkdir(parents=True)
    build_backend.PROVENANCE.write_bytes(PROVENANCE)
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", "0" * 40)
    monkeypatch.setenv("VOIAGE_SOURCE_TREE_GIT_OID", TREE)
    monkeypatch.setenv("VOIAGE_SOURCE_CLEAN", "true")

    with pytest.raises(RuntimeError, match="does not match embedded"):
        build_backend.build_wheel(str(tmp_path))


def test_gitless_wheel_rejects_incomplete_environment_identity(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_backend.PROVENANCE.parent.mkdir(parents=True)
    build_backend.PROVENANCE.write_bytes(PROVENANCE)
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", REVISION)

    with pytest.raises(RuntimeError, match="requires all of"):
        build_backend.build_wheel(str(tmp_path))


def test_wheel_rejects_clean_environment_identity_for_dirty_git(
    isolated_backend: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision, tree = _initialise_repository(isolated_backend)
    (isolated_backend / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setenv("VOIAGE_SOURCE_REVISION", revision)
    monkeypatch.setenv("VOIAGE_SOURCE_TREE_GIT_OID", tree)
    monkeypatch.setenv("VOIAGE_SOURCE_CLEAN", "true")

    with pytest.raises(RuntimeError, match="does not match the Git checkout"):
        build_backend.build_wheel(str(tmp_path))


def test_bridge_ci_runs_the_gitless_pep517_contract() -> None:
    path = Path(".github/workflows/python-rust-bridge.yml")
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)

    assert text.count('      - "build_backend.py"') == 2
    job = workflow["jobs"]["python-compatibility"]
    steps = {step["name"]: step for step in job["steps"]}
    assert (
        "tests/packaging/test_build_backend.py --no-cov"
        in steps["Test PEP 517 provenance wrapper"]["run"]
    )
    assert (
        "uv build --sdist --out-dir pep517-dist"
        in steps["Build ordinary clean source distribution"]["run"]
    )
    assert (
        "env -u VOIAGE_SOURCE_REVISION"
        in steps["Build ordinary clean source distribution"]["run"]
    )
    assert (
        steps["Build ordinary wheel from Git-less source distribution"][
            "working-directory"
        ]
        == "${{ runner.temp }}/voiage-sdist-source"
    )
    assert (
        "uv build --wheel"
        in steps["Build ordinary wheel from Git-less source distribution"]["run"]
    )
    assert (
        "env -u VOIAGE_SOURCE_REVISION"
        in steps["Build ordinary wheel from Git-less source distribution"]["run"]
    )
    verification = steps["Install and verify Git-less sdist-derived wheel"]["run"]
    assert (
        steps["Install and verify Git-less sdist-derived wheel"]["working-directory"]
        == "${{ runner.temp }}"
    )
    assert 'info["source_revision"]' in verification
    assert 'info["source_tree_git_oid"]' in verification
    assert 'info["source_dirty"] is False' in verification
