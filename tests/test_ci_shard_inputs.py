"""Behavioral witnesses for fail-closed shard aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_shard_inputs.py"


def fixture(tmp_path: Path) -> dict:
    """Build two coverage artifacts with one shared execution identity."""
    identity = {
        "source_head": "a" * 40,
        "lock_digest": "b" * 64,
        "interpreter": "CPython 3.15.1",
        "features": "ci,dev",
        "coverage_config_digest": "c" * 64,
    }
    shards = []
    for index in range(2):
        path = tmp_path / f"coverage-{index}.json"
        path.write_text(
            '{"files": {"voiage/example.py": {"executed_lines": [1], "missing_lines": []}}}',
            encoding="utf-8",
        )
        shards.append(
            {
                "index": index,
                "conclusion": "success",
                "exit_status": 0,
                "identity": identity.copy(),
                "coverage": str(path),
                "coverage_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"expected_shards": 2, "identity": identity, "shards": shards}


def run(tmp_path: Path, manifest: dict) -> subprocess.CompletedProcess[str]:
    """Invoke the candidate CLI as the workflow would."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = tmp_path / "expected-identity.json"
    expected.write_text(
        json.dumps(
            {
                "source_head": "a" * 40,
                "lock_digest": "b" * 64,
                "interpreter": "CPython 3.15.1",
                "features": "ci,dev",
                "coverage_config_digest": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(path),
            "--expected-identity",
            str(expected),
            "--expected-shards",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_complete_matching_shards_pass(tmp_path: Path) -> None:
    assert run(tmp_path, fixture(tmp_path)).returncode == 0


@pytest.mark.parametrize(
    "fault",
    [
        "failure",
        "cancelled",
        "missing",
        "identity",
        "forged_identity",
        "bytes",
        "duplicate",
        "empty",
        "empty_coverage",
        "reduced_count",
        "no_execution",
        "malformed_lines",
    ],
)
def test_incomplete_or_mismatched_shards_fail(tmp_path: Path, fault: str) -> None:
    manifest = fixture(tmp_path)
    shards = manifest["shards"]
    if fault == "failure":
        shards[0]["exit_status"] = 1
    elif fault == "cancelled":
        shards[0]["conclusion"] = "cancelled"
    elif fault == "missing":
        Path(shards[0]["coverage"]).unlink()
    elif fault == "identity":
        shards[0]["identity"]["source_head"] = "d" * 40
    elif fault == "forged_identity":
        manifest["identity"]["source_head"] = "d" * 40
        for shard in shards:
            shard["identity"]["source_head"] = "d" * 40
    elif fault == "bytes":
        Path(shards[0]["coverage"]).write_text("changed", encoding="utf-8")
    elif fault == "duplicate":
        shards[1]["index"] = 0
    elif fault == "empty_coverage":
        path = Path(shards[0]["coverage"])
        path.write_text('{"files": {}}', encoding="utf-8")
        shards[0]["coverage_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault in {"no_execution", "malformed_lines"}:
        path = Path(shards[0]["coverage"])
        lines = [] if fault == "no_execution" else [True]
        path.write_text(
            json.dumps(
                {
                    "files": {
                        "voiage/example.py": {
                            "executed_lines": lines,
                            "missing_lines": [],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        shards[0]["coverage_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "reduced_count":
        manifest["expected_shards"] = 1
        manifest["shards"] = shards[:1]
    else:
        manifest["shards"] = []
    assert run(tmp_path, manifest).returncode != 0
