"""CLI, installed schema and public discovery tests for issue #557."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.distributional_information import (
    VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1,
)
from voiage.methods.distributional_information import (
    DistributionalInformationResult,
    distributional_information_from_specification,
    value_of_distributional_information,
)

FIXTURE = (
    "specs/frontier/value-of-distributional-information/v1/"
    "fixtures/normative/input.json"
)


def test_cli_returns_exact_versioned_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-value-of-distributional-information",
            FIXTURE,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "distribution_family_information_value"
    assert payload["information_target"] == "model_family_index"
    assert payload["gross_vdi"] == 2.0
    assert payload["net_vdi"] == 1.5
    assert payload["current_optimal_alternatives"] == ["B"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cli_text_output_reports_gross_and_signed_net_values(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    result = CliRunner().invoke(
        app,
        [
            "calculate-value-of-distributional-information",
            FIXTURE,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Distribution-family information value: 2.000000" in result.stdout
    assert "signed net VDI: 1.500000" in result.stdout
    assert f"Result saved to {output}" in result.stdout
    assert "signed net VDI: 1.500000" in output.read_text(encoding="utf-8")


def test_cli_and_installed_runtime_share_the_checked_schema() -> None:
    payload = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    Draft202012Validator(VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1).validate(
        payload
    )
    result = distributional_information_from_specification(payload)
    assert isinstance(result, DistributionalInformationResult)


def test_experimental_public_exports_are_identity_preserving() -> None:
    assert voiage.DistributionalInformationResult is DistributionalInformationResult
    assert (
        voiage.distributional_information_from_specification
        is distributional_information_from_specification
    )
    assert (
        voiage.value_of_distributional_information
        is value_of_distributional_information
    )


def test_cli_rejects_non_object_invalid_and_semantically_bad_requests(
    tmp_path: Path,
) -> None:
    for index, content in enumerate(
        [
            "[]",
            "{",
            json.dumps({}),
            Path(FIXTURE)
            .read_text(encoding="utf-8")
            .replace(
                '"model_probabilities": [0.5, 0.5]', '"model_probabilities": [0.4, 0.4]'
            ),
        ]
    ):
        request = tmp_path / f"bad-{index}.json"
        request.write_text(content, encoding="utf-8")
        result = CliRunner().invoke(
            app,
            ["calculate-value-of-distributional-information", str(request)],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
