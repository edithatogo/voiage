"""Export installed qualitative-information schemas to the source contract tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voiage.contracts.qualitative_information import (
    QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1,
    qualitative_assessment_content_digest,
    qualitative_audit_event_digest,
)
from voiage.methods.qualitative_information import (
    qualitative_information_from_specification,
    render_qualitative_information_text,
)

ROOT = Path(__file__).parents[1]
SCHEMAS = ROOT / "specs/frontier/qualitative-information/v1/schemas"
NORMATIVE = ROOT / "specs/frontier/qualitative-information/v1/fixtures/normative"
CONTRACT = ROOT / "specs/frontier/qualitative-information/v1"
EVIDENCE_ARTIFACTS = (
    "scripts/export_qualitative_information_contracts.py",
    "specs/frontier/qualitative-information/v1/README.md",
    "specs/frontier/qualitative-information/v1/capabilities.json",
    "specs/frontier/qualitative-information/v1/fixtures/manifest.json",
    "specs/frontier/qualitative-information/v1/fixtures/normative/input.json",
    "specs/frontier/qualitative-information/v1/fixtures/normative/expected.json",
    "specs/frontier/qualitative-information/v1/fixtures/normative/rendering.json",
    "specs/frontier/qualitative-information/v1/fixtures/normative/request-result.json",
    "specs/frontier/qualitative-information/v1/fixtures/cases/disagreement.json",
    "specs/frontier/qualitative-information/v1/fixtures/cases/incomplete-ai.json",
    "specs/frontier/qualitative-information/v1/fixtures/cases/adversarial-audit.json",
    "specs/frontier/qualitative-information/v1/schemas/qualitative-information-assessment.schema.json",
    "specs/frontier/qualitative-information/v1/schemas/qualitative-information-audit-event.schema.json",
    "specs/frontier/qualitative-information/v1/schemas/qualitative-information-rendering.schema.json",
    "specs/frontier/qualitative-information/v1/schemas/qualitative-information-result.schema.json",
    "voiage/contracts/qualitative_information.py",
    "voiage/methods/qualitative_information.py",
    "tests/test_qualitative_information.py",
    "tests/test_qualitative_information_contract.py",
    "tests/test_qualitative_information_surfaces.py",
    "docs/astro-site/src/content/docs/examples/qualitative-information-assessment.mdx",
    "conductor/archive/supported_frontier_method_completion_20260723/qualitative-voi-reference-review.md",
    "conductor/archive/supported_frontier_method_completion_20260723/qualitative-information-implementation-review.md",
)


def main() -> None:
    """Write deterministic checked-in schema projections."""
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    values = {
        "qualitative-information-assessment.schema.json": QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
        "qualitative-information-audit-event.schema.json": QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1,
        "qualitative-information-rendering.schema.json": QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1,
        "qualitative-information-result.schema.json": QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1,
    }
    for name, value in values.items():
        (SCHEMAS / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    input_path = NORMATIVE / "input.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    assessment_digest = qualitative_assessment_content_digest(payload)
    previous_digest = None
    for event in payload["audit_history"]:
        event["assessment_content_digest"] = assessment_digest
        event["previous_content_digest"] = previous_digest
        event["content_digest"] = qualitative_audit_event_digest(event)
        previous_digest = event["content_digest"]
    input_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    result = qualitative_information_from_specification(payload)
    (NORMATIVE / "expected.json").write_text(
        json.dumps(result.to_contract_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    bundle_path = NORMATIVE / "request-result.json"
    bundle_path.write_text(
        json.dumps(
            {"request": payload, "result": result.to_contract_dict()},
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = CONTRACT / "fixtures/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixtures"][0]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    content = render_qualitative_information_text(result)
    rendering = {
        "schema_version": "1.0.0",
        "assessment_id": result.assessment_id,
        "media_type": "text/plain",
        "accessibility": {
            "wcag_reference": "WCAG-2.2",
            "headings_present": True,
            "no_colour_only_semantics": True,
            "redactions_preserved": True,
        },
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content": content,
    }
    (NORMATIVE / "rendering.json").write_text(
        json.dumps(rendering, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    evidence = {
        "method_family": "qualitative_information_assessment",
        "version": "v1",
        "maturity": "experimental",
        "stable_claim_allowed": False,
        "scope": "synthetic contract, Python runtime and accessible family documentation",
        "remaining_gates": [
            "practitioner, privacy, accessibility and scientific naming approval",
            "Rust, R and Julia execution parity disposition",
            "hosted exact-head and installed-wheel assurance",
            "stable promotion and release decision",
        ],
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            }
            for path in EVIDENCE_ARTIFACTS
        ],
    }
    (CONTRACT / "fixtures/evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
