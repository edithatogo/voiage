"""Validate the SciCrunch/RRID registration packet without changing external state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

RRID_PATTERN = re.compile(r"^RRID:SCR_\d{6}$")
SWHID_PATTERN = re.compile(r"^swh:1:snp:[0-9a-f]{40}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ANSWERS = ("resource_name", "url", "description")
PROHIBITED_PLACEHOLDERS = ("todo", "tbd", "replace me", "human-selection-required")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("registration packet must be a JSON object")
    return value


def validate_registration(path: Path) -> dict[str, str | int]:
    """Validate prepared answers, evidence, and external-state boundaries."""
    packet = _load_object(path)
    errors: list[str] = []

    answers = packet.get("answers")
    if not isinstance(answers, dict):
        errors.append("answers must be an object")
        answers = {}
    for field in REQUIRED_ANSWERS:
        value = answers.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"answers.{field} is required")
        elif any(marker in value.lower() for marker in PROHIBITED_PLACEHOLDERS):
            errors.append(f"answers.{field} contains a placeholder")

    if answers.get("resource_name") != "voiage":
        errors.append("resource name must match the released software name")
    if answers.get("url") != "https://github.com/edithatogo/voiage":
        errors.append("URL must be the canonical repository")

    evidence = packet.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        evidence = {}
    release = evidence.get("release", {})
    if release.get("tag") != "v2.0.0":
        errors.append("release evidence must be bound to v2.0.0")
    if not SHA_PATTERN.fullmatch(str(release.get("commit", ""))):
        errors.append("release commit must be a full Git SHA")
    archive = evidence.get("software_heritage", {})
    if not SWHID_PATTERN.fullmatch(str(archive.get("swhid", ""))):
        errors.append("Software Heritage evidence must contain a snapshot SWHID")

    evidence_links = packet.get("evidence_links")
    if not isinstance(evidence_links, list) or not all(
        isinstance(value, str) and value.startswith("https://")
        for value in evidence_links
    ):
        errors.append("evidence_links must be a list of HTTPS URLs")

    submission = packet.get("submission", {})
    curation = packet.get("curation", {})
    state = packet.get("state")
    if state == "ready_for_account_submission":
        if submission.get("performed") is not False:
            errors.append("ready state cannot claim an external submission")
        if submission.get("submitted_at") is not None:
            errors.append("unsubmitted packet cannot contain submitted_at")
        if curation.get("status") != "not_started":
            errors.append("curation must remain not_started before submission")
        if curation.get("rrid") is not None:
            errors.append("RRID must be null until assigned by SciCrunch")
    elif state == "submitted_pending_curation":
        if submission.get("performed") is not True:
            errors.append("submitted state must record performed=true")
        submitted_at = submission.get("submitted_at")
        if not isinstance(submitted_at, str) or not submitted_at.endswith("Z"):
            errors.append("submitted state must contain a UTC submitted_at")
        if (
            submission.get("confirmation_url")
            != "https://scicrunch.org/scicrunch/about/thanks"
        ):
            errors.append("submitted state must contain the observed confirmation URL")
        if submission.get("confirmation_message") != "Thank you for your submission!":
            errors.append(
                "submitted state must contain the observed confirmation message"
            )
        duplicate_check = packet.get("duplicate_check", {})
        if duplicate_check.get("performed") is not True:
            errors.append("submitted state must record the portal duplicate check")
        if duplicate_check.get("result") != "no_similar_resource":
            errors.append("duplicate check must preserve the observed no-match result")
        if curation.get("status") != "pending":
            errors.append("curation must remain pending after submission")
        if curation.get("rrid") is not None:
            errors.append("RRID must be null until assigned by SciCrunch")
    else:
        errors.append(
            "state must be ready_for_account_submission or submitted_pending_curation"
        )
    rrid = curation.get("rrid")
    if rrid is not None and not RRID_PATTERN.fullmatch(str(rrid)):
        errors.append("assigned RRID must use the RRID:SCR_###### form")

    declarations = packet.get("declarations", {})
    for name in ("resource_owner", "information_accurate", "account_terms_accepted"):
        declaration = declarations.get(name)
        if not isinstance(declaration, dict) or "answer" not in declaration:
            errors.append(f"declarations.{name}.answer is required")
    declaration_answer = None if state == "ready_for_account_submission" else True
    if (
        declarations.get("information_accurate", {}).get("answer")
        is not declaration_answer
    ):
        errors.append("accuracy declaration does not match submission state")
    if (
        declarations.get("account_terms_accepted", {}).get("answer")
        is not declaration_answer
    ):
        errors.append("terms declaration does not match submission state")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "resource_name": str(answers["resource_name"]),
        "release": str(release["tag"]),
        "state": str(state),
        "required_answer_count": len(REQUIRED_ANSWERS),
        "evidence_count": len(evidence_links),
    }


def main() -> int:
    """Validate the configured packet and print its stable summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "packet",
        nargs="?",
        type=Path,
        default=Path("docs/release/scicrunch-rrid-registration.json"),
    )
    args = parser.parse_args()
    summary = validate_registration(args.packet)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
