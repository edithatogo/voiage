"""Exercise coverage artifact identity checks through their command line."""

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/coverage_artifact_identity.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S607


def test_artifact_identity_rejects_mismatches_and_dirty_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "uv.lock").write_text("lock", encoding="utf-8")
    (repo / "pyproject.toml").write_text("coverage-config", encoding="utf-8")
    _git(repo, "add", "uv.lock", "pyproject.toml")
    _git(repo, "commit", "-q", "-m", "fixture")
    artifact = tmp_path / "coverage.json"
    artifact.write_text('{"coverage": 1}', encoding="utf-8")
    receipt = tmp_path / "identity.json"

    def run(mode: str) -> subprocess.CompletedProcess[str]:
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
        )

    assert run("seal").returncode == 0
    original = json.loads(receipt.read_text(encoding="utf-8"))
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
    (repo / "uv.lock").write_text("changed", encoding="utf-8")
    assert run("verify").returncode != 0
    _git(repo, "add", "uv.lock")
    _git(repo, "commit", "-q", "-m", "changed lock")
    assert run("verify").returncode != 0
