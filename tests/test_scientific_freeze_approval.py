"""Fail-closed tests for recording the v1.1 scientific-freeze approval."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "record_scientific_freeze_approval.py"
CANDIDATE = (
    ROOT / "specs" / "software-landscape" / "v1.1-scientific-freeze-candidate.json"
)
SCHEMA = (
    ROOT
    / "specs"
    / "software-landscape"
    / "v1.1-scientific-freeze-approval.schema.json"
)


def _candidate_digest() -> str:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    return str(candidate["candidate_digest"])


def _run(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(CANDIDATE),
            "--output",
            str(output),
            "--candidate-digest",
            _candidate_digest(),
            "--reviewer",
            "Accountable Human Reviewer",
            "--approved-at",
            "2026-07-24T07:00:00Z",
            "--evidence",
            "https://github.com/edithatogo/voiage/issues/314#issuecomment-1",
            *extra,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_approval_record_is_schema_valid_and_candidate_bound(tmp_path: Path) -> None:
    """A complete approval must bind both the digest and candidate artifact."""
    output = tmp_path / "approval.json"

    result = _run(output)

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    assert record["candidate_digest"] == _candidate_digest()
    assert (
        record["candidate_artifact_sha256"]
        == sha256(CANDIDATE.read_bytes()).hexdigest()
    )
    assert record["status"] == "approved"
    assert record["decision_scope"] == "scientific-contract-only"


def test_approval_refuses_stale_digest(tmp_path: Path) -> None:
    """A reviewer cannot accidentally approve a different candidate."""
    output = tmp_path / "approval.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(CANDIDATE),
            "--output",
            str(output),
            "--candidate-digest",
            "0" * 64,
            "--reviewer",
            "Accountable Human Reviewer",
            "--approved-at",
            "2026-07-24T07:00:00Z",
            "--evidence",
            "https://github.com/edithatogo/voiage/issues/314#issuecomment-1",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not output.exists()


def test_approval_is_append_only(tmp_path: Path) -> None:
    """Existing approval evidence must never be silently overwritten."""
    output = tmp_path / "approval.json"
    assert _run(output).returncode == 0
    original = output.read_bytes()

    second = _run(output)

    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert output.read_bytes() == original


def test_approval_requires_external_evidence_reference(tmp_path: Path) -> None:
    """A local assertion is insufficient evidence for human approval."""
    output = tmp_path / "approval.json"

    result = _run(output, "--evidence", "not-a-uri")

    assert result.returncode != 0
    assert "absolute HTTPS URI" in result.stderr
    assert not output.exists()
