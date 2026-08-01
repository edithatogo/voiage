"""CLI, installed contract and lazy discovery tests for issue #558."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.qualitative_information import (
    QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
)
from voiage.methods.qualitative_information import (
    QualitativeInformationResult,
    QualitativeQuestionResult,
    qualitative_information_from_specification,
    render_qualitative_information_text,
)

FIXTURE = "specs/frontier/qualitative-information/v1/fixtures/normative/input.json"


def test_cli_returns_exact_versioned_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "assess-qualitative-information",
            FIXTURE,
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "qualitative_information_assessment"
    assert payload["workflow_status"] == "complete"
    assert payload["numerical_estimand"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cli_text_is_deterministic_accessible_and_redaction_safe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.txt"
    result = CliRunner().invoke(
        app, ["assess-qualitative-information", FIXTURE, "-o", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    assert "Qualitative information assessment" in result.stdout
    assert "[REDACTED]" in result.stdout
    assert "ordinal workflow only" in result.stdout
    assert output.read_text(encoding="utf-8") in result.stdout


def test_installed_schema_runtime_and_lazy_exports_are_identity_preserving() -> None:
    payload = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    Draft202012Validator(QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1).validate(payload)
    result = qualitative_information_from_specification(payload)
    assert isinstance(result, QualitativeInformationResult)
    assert render_qualitative_information_text(result).startswith("Qualitative")
    assert voiage.QualitativeInformationResult is QualitativeInformationResult
    assert voiage.QualitativeQuestionResult is QualitativeQuestionResult
    assert (
        voiage.qualitative_information_from_specification
        is qualitative_information_from_specification
    )
    assert (
        voiage.render_qualitative_information_text
        is render_qualitative_information_text
    )


def test_cli_rejects_non_object_invalid_and_semantically_bad_requests(
    tmp_path: Path,
) -> None:
    valid = Path(FIXTURE).read_text(encoding="utf-8")
    for index, content in enumerate(
        ["[]", "{", "{}", valid.replace('"sequence": 2', '"sequence": 7')]
    ):
        request = tmp_path / f"bad-{index}.json"
        request.write_text(content, encoding="utf-8")
        result = CliRunner().invoke(
            app, ["assess-qualitative-information", str(request)]
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
