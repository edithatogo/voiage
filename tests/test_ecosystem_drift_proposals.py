"""Evidence-preserving ecosystem drift proposal contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "propose_ecosystem_drift.py"
BASELINE = ROOT / "specs" / "software-landscape" / "ecosystem-drift-baseline.json"
SCHEMA = ROOT / "specs" / "software-landscape" / "ecosystem-drift-proposal.schema.json"


def test_checked_in_ecosystem_drift_baseline_is_current() -> None:
    """The reviewed baseline must match every locally observable frontier."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["snapshot"]) == {
        "software-registry",
        "toolchains",
        "lockfiles",
        "github-actions",
        "documentation-plugin",
    }
    plugin = baseline["snapshot"]["documentation-plugin"]
    assert plugin["path"] == ".repo-tools/astro-polyglot"
    assert len(plugin["gitlink_commit"]) == 40


def test_detected_drift_is_a_non_applying_review_proposal(tmp_path: Path) -> None:
    """Drift may produce evidence, but never approve or mutate the repository."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline["snapshot"]["documentation-plugin"]["gitlink_commit"] = "0" * 40
    changed_baseline = tmp_path / "baseline.json"
    changed_baseline.write_text(json.dumps(baseline), encoding="utf-8")
    output = tmp_path / "proposal.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(changed_baseline),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    proposal = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(proposal)
    assert proposal["has_drift"] is True
    assert proposal["auto_apply"] is False
    assert proposal["scientific_disposition_allowed"] is False
    assert any(
        item["category"] == "documentation-plugin" for item in proposal["proposals"]
    )
