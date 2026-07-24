#!/usr/bin/env python3
"""Generate the hash-bound v1.1 scientific-freeze review candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
JSON_OUTPUT = LANDSCAPE / "v1.1-scientific-freeze-candidate.json"
DOC_OUTPUT = (
    ROOT
    / "docs"
    / "astro-site"
    / "src"
    / "content"
    / "docs"
    / "v1-1-scientific-freeze-review.mdx"
)
SOURCE_PATHS = (
    LANDSCAPE / "methods.json",
    LANDSCAPE / "method-evidence.json",
    LANDSCAPE / "implementation-evidence.json",
    LANDSCAPE / "adjacent-method-dispositions.json",
    ROOT / "specs" / "core-api" / "schemas" / "v2" / "decision-problem.schema.json",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _source_digests() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _method_records() -> list[dict[str, str]]:
    methods = _load(LANDSCAPE / "methods.json")["methods"]
    evidence = {
        item["method_id"]: item
        for item in _load(LANDSCAPE / "method-evidence.json")["coverage"]
    }
    implementation = {
        item["method_id"]: item
        for item in _load(LANDSCAPE / "implementation-evidence.json")["records"]
    }
    records = []
    for method in methods:
        review = evidence[method["id"]]
        runtime = implementation.get(method["id"], {})
        records.append(
            {
                "method_id": method["id"],
                "label": method["label"],
                "class": method["class"],
                "moscow": method["moscow"],
                "maturity": method["maturity"],
                "voiage_state": method["voiage_state"],
                "review_state": review["review_state"],
                "disposition": review["disposition"],
                "promotion_gate": review["promotion_gate"],
                "authority_state": runtime.get("authority_state", "not-implemented"),
                "remaining_implementation_gate": runtime.get(
                    "remaining_gate", "implementation"
                ),
            }
        )
    return sorted(records, key=lambda item: item["method_id"])


def _candidate() -> dict[str, object]:
    digests = _source_digests()
    candidate_digest = sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    records = _method_records()
    by_maturity = {
        maturity: [record for record in records if record["maturity"] == maturity]
        for maturity in ("stable", "experimental", "planned")
    }
    return {
        "schema_version": "1.0.0",
        "release_target": "v1.1",
        "review_requested_on": "2026-07-24",
        "candidate_digest": candidate_digest,
        "source_digests": digests,
        "decision_contract": {
            "canonical_version": "v2",
            "compatibility": "additive-v1-read-and-lossless-map",
            "schema_path": "specs/core-api/schemas/v2/decision-problem.schema.json",
        },
        "stable_methods": by_maturity["stable"],
        "experimental_methods": by_maturity["experimental"],
        "planned_methods": by_maturity["planned"],
        "requested_decisions": [
            "Approve the listed stable definitions and dispositions for the v1.1 contract.",
            "Retain every experimental method outside the v1.1 stable guarantee.",
            "Retain every planned method as unavailable until its recorded promotion gate passes.",
            "Adopt DecisionProblemV2 additively while preserving v1 reads and lossless mapping.",
            "Keep implementation authority gates distinct from scientific-definition approval.",
        ],
        "approval": {
            "status": "pending-human-review",
            "approved_by": None,
            "approved_at": None,
            "evidence": None,
        },
    }


def _table(records: list[dict[str, str]]) -> list[str]:
    lines = [
        "| ID | Class | MoSCoW | Evidence | Authority | Remaining gate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    f"`{record['method_id']}`",
                    record["class"],
                    record["moscow"],
                    record["review_state"],
                    record["authority_state"],
                    record["remaining_implementation_gate"],
                ]
            )
            + " |"
        )
        for record in records
    )
    return lines


def _document(candidate: dict[str, object]) -> str:
    stable = candidate["stable_methods"]
    experimental = candidate["experimental_methods"]
    planned = candidate["planned_methods"]
    assert isinstance(stable, list)
    assert isinstance(experimental, list)
    assert isinstance(planned, list)
    lines = [
        "---",
        "title: v1.1 scientific freeze review",
        "description: Hash-bound human review candidate for VOIAGE method maturity and DecisionProblemV2",
        "---",
        "",
        "{/* Generated by scripts/generate_scientific_freeze_candidate.py. */}",
        "",
        "This page is a review request, not an approval or release claim.",
        "",
        f"Candidate digest: `{candidate['candidate_digest']}`.",
        "",
        "## Requested decision",
        "",
        "Approve the exact candidate digest above, or identify methods that need a",
        "different definition, maturity, disposition, or promotion gate. Scientific",
        "approval does not waive any remaining implementation or external gate.",
        "",
        "DecisionProblemV2 is proposed as the additive canonical interchange contract.",
        "v1 inputs remain readable and must losslessly map into the v2 representation.",
        "",
        f"## Stable candidate ({len(stable)})",
        "",
        *_table(stable),
        "",
        f"## Experimental, outside the stable guarantee ({len(experimental)})",
        "",
        *_table(experimental),
        "",
        f"## Planned, unavailable pending promotion ({len(planned)})",
        "",
        *_table(planned),
        "",
        "## Approval state",
        "",
        "`pending-human-review`. No agent, test, generated document, or CI result",
        "can change this state. Approval evidence must identify this exact digest and",
        "an accountable human reviewer.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Write or check the deterministic candidate and review page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    candidate = _candidate()
    json_rendered = json.dumps(candidate, indent=2, ensure_ascii=False) + "\n"
    doc_rendered = _document(candidate)
    if args.check:
        current = (
            JSON_OUTPUT.exists()
            and DOC_OUTPUT.exists()
            and JSON_OUTPUT.read_text(encoding="utf-8") == json_rendered
            and DOC_OUTPUT.read_text(encoding="utf-8") == doc_rendered
        )
        if not current:
            print(
                "scientific freeze candidate is stale; run "
                "scripts/generate_scientific_freeze_candidate.py",
                file=sys.stderr,
            )
            return 1
        print("scientific freeze candidate is current")
        return 0
    JSON_OUTPUT.write_text(json_rendered, encoding="utf-8")
    DOC_OUTPUT.write_text(doc_rendered, encoding="utf-8")
    print(f"wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {DOC_OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
