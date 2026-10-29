"""Exercise producer receipts and fail-closed aggregate preparation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_shard_receipts.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def setup(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[tool.coverage.run]\nbranch=true\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "add", "uv.lock", "pyproject.toml"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(
        [  # noqa: S607
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    root = tmp_path / "artifacts"
    for index in range(2):
        directory = root / f"shard-{index}"
        directory.mkdir(parents=True)
        (directory / "coverage.json").write_text('{"files": {}}', encoding="utf-8")
        (directory / "coverage.data").write_bytes(b"native-fixture")
        result = run(
            "prepare", "--repo", str(repo), "--root", str(root), "--index", str(index)
        )
        assert result.returncode == 0, result.stderr
    return repo, root


def test_successful_receipts_create_manifest(tmp_path: Path) -> None:
    repo, root = setup(tmp_path)
    result = run(
        "collect",
        "--repo",
        str(repo),
        "--root",
        str(root),
        "--matrix-result",
        "success",
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads((root / "manifest.json").read_text())["shards"]) == 2
    assert (root / "expected-identity.json").is_file()


@pytest.mark.parametrize("fault", ["failure", "cancelled", "missing", "identity"])
def test_incomplete_or_failed_collection_fails(tmp_path: Path, fault: str) -> None:
    repo, root = setup(tmp_path)
    matrix_result = fault if fault in {"failure", "cancelled"} else "success"
    if fault == "missing":
        (root / "shard-1" / "coverage.data").unlink()
    elif fault == "identity":
        path = root / "shard-1" / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["identity"]["source_head"] = "0" * 40
        path.write_text(json.dumps(receipt), encoding="utf-8")
    result = run(
        "collect",
        "--repo",
        str(repo),
        "--root",
        str(root),
        "--matrix-result",
        matrix_result,
    )
    assert result.returncode != 0
    assert not (root / "manifest.json").exists()


def test_collection_requires_explicit_matrix_result(tmp_path: Path) -> None:
    repo, root = setup(tmp_path)
    result = run("collect", "--repo", str(repo), "--root", str(root))
    assert result.returncode != 0
    assert not (root / "manifest.json").exists()
