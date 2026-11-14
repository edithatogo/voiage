"""Exercise coverage artifact identity checks through their command line."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/coverage_artifact_identity.py"


@pytest.mark.parametrize("without_git", [False, True])
def test_artifact_identity_fails_closed_without_checkout_or_git(
    tmp_path: Path, without_git: bool
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "seal",
            "--repo",
            str(tmp_path),
            "--artifact",
            str(tmp_path / "coverage.json"),
            "--receipt",
            str(tmp_path / "identity.json"),
            "--features",
            "ci",
        ],
        env={**os.environ, "PATH": ""} if without_git else None,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    expected = (
        "git executable not found"
        if without_git
        else "Git checkout identity could not be read"
    )
    assert expected in result.stderr
    assert not (tmp_path / "identity.json").exists()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_artifact_identity_rejects_mismatches_and_dirty_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "uv.lock").write_text("lock", encoding="utf-8")
    (repo / "pyproject.toml").write_text("coverage-config", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", "uv.lock", "pyproject.toml", ".gitignore")
    _git(repo, "commit", "-q", "-m", "fixture")
    artifact = tmp_path / "coverage.json"
    artifact.write_text('{"coverage": 1}', encoding="utf-8")
    receipt = tmp_path / "identity.json"

    def run(
        mode: str, *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                mode,
                "--repo",
                str(repo),
                "--artifact",
                str(artifact),
                "--receipt",
                str(receipt),
                "--features",
                "ci,dev,producer-tests",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    assert run("seal", env={**os.environ, "PATH": ""}).returncode != 0
    assert (
        run(
            "seal", env={**os.environ, "GIT_PYTHON_GIT_EXECUTABLE": "invalid"}
        ).returncode
        == 0
    )
    assert run("seal").returncode == 0
    original = json.loads(receipt.read_text(encoding="utf-8"))
    assert original["source_head"] == _git(repo, "rev-parse", "HEAD")
    assert run("verify").returncode == 0
    (repo / "ignored.txt").write_text("ignored", encoding="utf-8")
    assert run("verify").returncode == 0
    for field in (
        "source_head",
        "lock_digest",
        "interpreter",
        "coverage_config_digest",
        "feature_configuration",
        "artifact_digest",
    ):
        modified = dict(original)
        modified[field] = "wrong"
        receipt.write_text(json.dumps(modified), encoding="utf-8")
        assert run("verify").returncode != 0, field
    receipt.write_text(json.dumps(original), encoding="utf-8")
    artifact.write_text("changed", encoding="utf-8")
    assert run("verify").returncode != 0
    artifact.unlink()
    assert run("verify").returncode != 0
    artifact.write_text('{"coverage": 1}', encoding="utf-8")
    for content in ("{}", "not json", "x" * 16_385):
        receipt.write_text(content, encoding="utf-8")
        assert run("verify").returncode != 0
    receipt.write_text(json.dumps(original), encoding="utf-8")
    untracked = repo / "untracked.txt"
    untracked.write_text("untracked", encoding="utf-8")
    assert run("verify").returncode != 0
    untracked.unlink()
    (repo / "uv.lock").write_text("changed", encoding="utf-8")
    assert run("verify").returncode != 0
    _git(repo, "add", "uv.lock")
    assert run("verify").returncode != 0
    _git(repo, "commit", "-q", "-m", "changed lock")
    assert run("verify").returncode != 0


def test_artifact_identity_rejects_dirty_submodules(tmp_path: Path) -> None:
    source = tmp_path / "submodule source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Fixture")
    _git(source, "config", "user.email", "fixture@example.invalid")
    (source / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-q", "-m", "fixture")

    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "uv.lock").write_text("lock", encoding="utf-8")
    (repo / "pyproject.toml").write_text("config", encoding="utf-8")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "nested",
    )
    _git(repo, "add", "uv.lock", "pyproject.toml", ".gitmodules", "nested")
    _git(repo, "commit", "-q", "-m", "fixture")
    artifact = tmp_path / "coverage.json"
    artifact.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "identity.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "seal",
        "--repo",
        str(repo),
        "--artifact",
        str(artifact),
        "--receipt",
        str(receipt),
        "--features",
        "ci",
    ]

    def seal() -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, check=False)

    assert seal().returncode == 0
    tracked = repo / "nested" / "tracked.txt"
    tracked.write_text("modified\n", encoding="utf-8")
    assert seal().returncode != 0
    tracked.write_text("original\n", encoding="utf-8")
    untracked = repo / "nested" / "untracked.txt"
    untracked.write_text("untracked", encoding="utf-8")
    assert seal().returncode != 0
