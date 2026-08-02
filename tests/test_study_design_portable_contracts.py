"""Portable study-design schemas and capability metadata."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from voiage.contracts.study_design import (
    CossRequestV1,
    CossResultV1,
    InformationEfficiencyRequestV1,
    InformationEfficiencyResultV1,
)

ROOT = Path(__file__).parents[1] / "specs/frontier/study-design-efficiency/v1"


def test_portable_schemas_match_authoritative_models() -> None:
    models = {
        "coss-request.schema.json": CossRequestV1,
        "coss-result.schema.json": CossResultV1,
        "efficiency-request.schema.json": InformationEfficiencyRequestV1,
        "efficiency-result.schema.json": InformationEfficiencyResultV1,
    }
    for filename, model in models.items():
        committed = json.loads(
            (ROOT / "schemas" / filename).read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(committed)
        generated = model.model_json_schema()
        assert committed == generated


def test_capability_metadata_is_honest_about_installed_parity() -> None:
    capability = json.loads((ROOT / "capabilities.json").read_text(encoding="utf-8"))

    assert capability["contract_version"] == "1.0.0"
    assert capability["maturity"] == "experimental"
    assert capability["surfaces"]["rust"]["status"] == "kernel"
    assert capability["surfaces"]["python"]["status"] == "executable"
    for language in ("r", "julia"):
        assert capability["surfaces"][language]["status"] == "unsupported"
    assert capability["installed_wheel_verified"] is False
    assert "scientific review" in " ".join(capability["remaining_gates"])
