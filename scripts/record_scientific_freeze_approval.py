#!/usr/bin/env python3
"""Record an accountable, hash-bound v1.1 scientific-freeze approval."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).parents[1]
DEFAULT_CANDIDATE = (
    ROOT / "specs" / "software-landscape" / "v1.1-scientific-freeze-candidate.json"
)
DEFAULT_OUTPUT = (
    ROOT / "specs" / "software-landscape" / "v1.1-scientific-freeze-approval.json"
)
CANONICAL_CANDIDATE_PATH = (
    "specs/software-landscape/v1.1-scientific-freeze-candidate.json"
)


class ApprovalRecordError(ValueError):
    """Raised when approval evidence cannot be recorded safely."""


def _load_candidate(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApprovalRecordError(f"cannot read candidate: {error}") from error
    if not isinstance(value, dict):
        raise ApprovalRecordError("candidate must contain a JSON object")
    return value


def _validate_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ApprovalRecordError("approved-at must be a UTC RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ApprovalRecordError(
            "approved-at must be a UTC RFC 3339 timestamp"
        ) from error
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ApprovalRecordError("approved-at must be a UTC RFC 3339 timestamp")


def _validate_evidence(values: list[str]) -> list[str]:
    normalized = list(dict.fromkeys(value.strip() for value in values))
    for value in normalized:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ApprovalRecordError(
                "each evidence value must be an absolute HTTPS URI"
            )
    return normalized


def _requested_decisions_digest(candidate: dict[str, Any]) -> str:
    decisions = candidate.get("requested_decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or not all(isinstance(item, str) and item for item in decisions)
    ):
        raise ApprovalRecordError(
            "candidate requested_decisions must be a non-empty string array"
        )
    encoded = json.dumps(
        decisions,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _build_record(args: argparse.Namespace) -> dict[str, object]:
    candidate_path = args.candidate.resolve()
    if candidate_path != DEFAULT_CANDIDATE.resolve():
        raise ApprovalRecordError(
            f"candidate must be the canonical {CANONICAL_CANDIDATE_PATH}"
        )
    candidate = _load_candidate(candidate_path)
    actual_digest = candidate.get("candidate_digest")
    if actual_digest != args.candidate_digest:
        raise ApprovalRecordError(
            "provided candidate digest does not match the canonical candidate"
        )
    approval = candidate.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != (
        "pending-human-review"
    ):
        raise ApprovalRecordError("candidate is not pending human review")
    reviewer = args.reviewer.strip()
    if len(reviewer) < 3:
        raise ApprovalRecordError("reviewer must identify an accountable human")
    _validate_timestamp(args.approved_at)
    evidence = _validate_evidence(args.evidence)
    return {
        "schema_version": "1.0.0",
        "release_target": "v1.1",
        "status": "approved",
        "decision_scope": "scientific-contract-only",
        "candidate_digest": actual_digest,
        "candidate_artifact_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
        "candidate_path": CANONICAL_CANDIDATE_PATH,
        "approved_by": reviewer,
        "approved_at": args.approved_at,
        "evidence": evidence,
        "requested_decisions_sha256": _requested_decisions_digest(candidate),
        "non_waived_gates": [
            "implementation",
            "numerical-validation",
            "binding-conformance",
            "release",
            "publication",
            "external",
        ],
    }


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ApprovalRecordError(
            f"approval record already exists and is append-only: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    """Validate approval inputs and create one immutable JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-digest",
        required=True,
        help="Exact digest shown in the reviewed candidate.",
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="Accountable human reviewer identity.",
    )
    parser.add_argument(
        "--approved-at",
        required=True,
        help="UTC RFC 3339 approval timestamp.",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="Absolute HTTPS evidence URI; repeat for multiple references.",
    )
    args = parser.parse_args()
    try:
        record = _build_record(args)
        rendered = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        _write_exclusive(args.output, rendered)
    except ApprovalRecordError as error:
        print(f"approval not recorded: {error}", file=sys.stderr)
        return 1
    print(f"recorded scientific-freeze approval at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
