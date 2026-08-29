"""Contracts for the final repository-hardening evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "specs" / "submission-readiness"
_GIT = shutil.which("git")
if _GIT is None:
    raise RuntimeError("git is required to validate the hardened source binding")
GIT: str = _GIT


def _load(name: str) -> dict[str, Any]:
    return json.loads((READINESS / name).read_text(encoding="utf-8"))


def _git_bytes(revision: str, path: str) -> bytes:
    revision_available = (
        subprocess.run(
            [GIT, "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    if not revision_available:
        return (ROOT / path).read_bytes()
    return subprocess.run(
        [GIT, "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_hardened_source_is_hash_bound_without_false_release_claim() -> None:
    binding = _load("final-candidate-binding-20260829.json")
    hardened = binding["hardened_source"]
    assert isinstance(hardened, dict)
    revision = hardened["revision"]
    assert isinstance(revision, str)

    hardening_merge = binding["hardening_merge"]
    assert isinstance(hardening_merge, dict)
    assert hardening_merge["exact_tree_match"] is True
    assert hardening_merge["squash_merge_tree"] == hardened["tree"]

    for item in binding["bound_files"]:
        assert isinstance(item, dict)
        path = item["path"]
        assert isinstance(path, str)
        assert hashlib.sha256(_git_bytes(revision, path)).hexdigest() == item["sha256"]

    published = binding["published_release"]
    assert isinstance(published, dict)
    assert published["tagged_commit"] != revision
    assert binding["candidate_eligible_for_submission"] is False
    assert binding["release_performed"] is False
    assert binding["submission_performed"] is False


def test_preview_results_promote_nothing_and_record_every_lane() -> None:
    results = _load("preview-observation-results-20260829.json")
    policy = results["policy"]
    assert isinstance(policy, dict)
    assert policy == {
        "mode": "non_blocking_observation",
        "stable_bounds_changed": False,
        "candidate_promoted": False,
        "release_artifacts_published": False,
    }
    observations = results["observations"]
    assert isinstance(observations, list)
    assert {item["id"] for item in observations} == {
        "scipy-1.18",
        "pandas-3",
        "xarray-2026",
        "jax-0.11",
        "griffe-2",
        "ruff-0.16",
        "python-3.15",
        "deterministic-python-shards",
        "cargo-nextest-and-sccache",
    }


def test_governance_reconciliation_separates_repository_and_external_state() -> None:
    reconciliation = _load("governance-readiness-reconciliation-20260829.json")
    controls = reconciliation["repository_controls"]
    assert isinstance(controls, list)
    assert all(item["status"] == "satisfied" for item in controls)
    for item in controls:
        for evidence in item["evidence"]:
            assert (ROOT / evidence).exists(), evidence

    assert reconciliation["repository_readiness"] == "satisfied"
    assert reconciliation["external_readiness"] == "pending"
    assert reconciliation["external_action_performed"] is False
