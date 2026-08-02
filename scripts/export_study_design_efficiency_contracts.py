#!/usr/bin/env python3
"""Export authoritative study-design Pydantic models as portable JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path

from voiage.contracts.study_design import (
    CossRequestV1,
    CossResultV1,
    InformationEfficiencyRequestV1,
    InformationEfficiencyResultV1,
)

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "specs/frontier/study-design-efficiency/v1/schemas"
MODELS = {
    "coss-request.schema.json": CossRequestV1,
    "coss-result.schema.json": CossResultV1,
    "efficiency-request.schema.json": InformationEfficiencyRequestV1,
    "efficiency-result.schema.json": InformationEfficiencyResultV1,
}


def main() -> None:
    """Write deterministic schemas for review and cross-language consumers."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        payload = json.dumps(
            model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        (OUTPUT / filename).write_text(f"{payload}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
