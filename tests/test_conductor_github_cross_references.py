"""Tests for the Conductor GitHub cross-reference contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_conductor_github_cross_references.py"


def _validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conductor_github_xref", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_cross_references_are_complete() -> None:
    """Every local or proposed track has a valid unique GitHub record."""
    assert _validator().validate(ROOT) == []


def test_manifest_preserves_no_pr_evidence_boundary() -> None:
    """Legacy tracks without a provable PR state that boundary explicitly."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    completed_without_prs = [
        entry
        for entry in manifest["tracks"]
        if entry["lifecycle"] == "completed" and not entry["pull_requests"]
    ]
    assert completed_without_prs
    assert all(
        entry["pull_request_evidence"] == "none_found"
        for entry in completed_without_prs
    )
