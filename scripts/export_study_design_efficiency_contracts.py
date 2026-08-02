#!/usr/bin/env python3
"""Export authoritative study-design Pydantic models as portable JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from voiage.contracts.study_design import (
    CossRequestV1,
    CossResultV1,
    InformationEfficiencyRequestV1,
    InformationEfficiencyResultV1,
)

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "specs/frontier/study-design-efficiency/v1/schemas"
CONTRACT = OUTPUT.parent
PACKAGED = (
    ROOT / "voiage/resources/frontier/study-design-efficiency/v1"
)
MODELS = {
    "coss-request.schema.json": CossRequestV1,
    "coss-result.schema.json": CossResultV1,
    "efficiency-request.schema.json": InformationEfficiencyRequestV1,
    "efficiency-result.schema.json": InformationEfficiencyResultV1,
}


def main() -> None:
    """Write deterministic schemas for review and cross-language consumers."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    packaged_schemas = PACKAGED / "schemas"
    packaged_schemas.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        payload = json.dumps(
            model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        (OUTPUT / filename).write_text(f"{payload}\n", encoding="utf-8")
        (packaged_schemas / filename).write_text(f"{payload}\n", encoding="utf-8")
    for relative in (
        Path("capabilities.json"),
        Path("fixtures/manifest.json"),
        Path("fixtures/normative/coss-efficiency.json"),
        Path("fixtures/normative/joint-enbs-replicates.json"),
        Path("fixtures/normative/paired-efficiency-replicates.json"),
    ):
        destination = PACKAGED / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CONTRACT / relative, destination)


if __name__ == "__main__":
    main()
