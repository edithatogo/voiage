"""Export installed qualitative-information schemas to the source contract tree."""

from __future__ import annotations

import json
from pathlib import Path

from voiage.contracts.qualitative_information import (
    QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1,
)

ROOT = Path(__file__).parents[1]
SCHEMAS = ROOT / "specs/frontier/qualitative-information/v1/schemas"


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


if __name__ == "__main__":
    main()
