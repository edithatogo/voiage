"""Portable study-design schemas and capability metadata."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from voiage.contracts.study_design import (
    CossRequestV1,
    CossResultV1,
    InformationEfficiencyRequestV1,
    InformationEfficiencyResultV1,
    SelectionUncertaintyV1,
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
    registry = json.loads(
        (ROOT.parents[1] / "fixtures/manifest.json").read_text(encoding="utf-8")
    )
    registered = next(
        item
        for item in registry["families"]
        if item["name"] == "study_design_efficiency"
    )

    assert capability["contract_version"] == "1.0.0"
    assert capability["maturity"] == "experimental"
    assert registered["method_maturity"] == "experimental"
    assert capability["surfaces"]["rust"]["status"] == "kernel"
    assert capability["surfaces"]["python"]["status"] == "executable"
    for language in ("r", "julia"):
        assert capability["surfaces"][language]["status"] == "unsupported"
    assert capability["installed_wheel_verified"] is True
    assert "scientific review" in " ".join(capability["remaining_gates"])


def test_portable_coss_request_carries_replayable_joint_replicates() -> None:
    fixture = json.loads(
        (ROOT / "fixtures/normative/coss-efficiency.json").read_text(encoding="utf-8")
    )
    replicates = json.loads(
        (ROOT / "fixtures/normative/joint-enbs-replicates.json").read_text(
            encoding="utf-8"
        )
    )["joint_enbs_replicates"]
    request = CossRequestV1.model_validate_json(
        json.dumps(
            {
                "context": fixture["input"]["context"],
                "designs": fixture["input"]["designs"],
                "joint_enbs_replicates": replicates,
                "replay_artifact": "fixtures/normative/joint-enbs-replicates.json",
            }
        )
    )

    assert len(request.joint_enbs_replicates or ()) == 4
    assert request.selection_uncertainty is None
    conflicting = request.model_dump(mode="json")
    conflicting["selection_uncertainty"] = SelectionUncertaintyV1().model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CossRequestV1.model_validate_json(json.dumps(conflicting))


def test_packaged_study_design_resources_match_canonical_specs() -> None:
    packaged = files("voiage").joinpath("resources/frontier/study-design-efficiency/v1")
    for relative in (
        "capabilities.json",
        "fixtures/manifest.json",
        "fixtures/normative/coss-efficiency.json",
        "fixtures/normative/joint-enbs-replicates.json",
        "fixtures/normative/paired-efficiency-replicates.json",
        "schemas/coss-request.schema.json",
        "schemas/coss-result.schema.json",
        "schemas/efficiency-request.schema.json",
        "schemas/efficiency-result.schema.json",
    ):
        assert (
            packaged.joinpath(relative).read_bytes() == (ROOT / relative).read_bytes()
        )
