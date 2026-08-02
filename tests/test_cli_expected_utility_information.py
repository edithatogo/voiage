from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from voiage import cli

if TYPE_CHECKING:
    import pytest

FIXTURES = (
    Path(__file__).parents[1]
    / "specs/frontier/expected-utility-information-pricing/v1/fixtures/normative"
)
RUNNER = CliRunner()


def _request(tmp_path: Path, fixture: str) -> Path:
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))["request"]
    path = tmp_path / fixture
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_emits_the_canonical_result_without_a_voc_kernel(tmp_path: Path) -> None:
    request = _request(tmp_path, "log-buy-sell-asymmetry.json")
    response = RUNNER.invoke(
        cli.app,
        [
            "--format",
            "json",
            "calculate-expected-utility-information",
            str(request),
            "--measure",
            "bpi",
        ],
    )

    assert response.exit_code == 0, response.stdout
    payload = json.loads(response.stdout)
    assert payload["command"] == "calculate-expected-utility-information"
    assert payload["selected_measure"] == "bpi"
    assert payload["selected_value"] == payload["result"]["bpi"]["value"]
    presentation = payload["result"]["presentation"]
    assert {
        key: presentation[key]
        for key in ("canonical_result_ref", "presentation_label", "selected_measure")
    } == {
        "canonical_result_ref": "self",
        "presentation_label": "canonical",
        "selected_measure": "bpi",
    }
    assert presentation["presentation_contract_version"] == "1.0.0"
    assert (
        presentation["canonical_input_digest"]
        == payload["result"]["input_digest"]["value"]
    )
    digest_input = {
        "canonical_input_digest": presentation["canonical_input_digest"],
        "presentation_contract_version": "1.0.0",
        "presentation_label": "canonical",
        "selected_measure": "bpi",
    }
    encoded = json.dumps(
        digest_input, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert presentation["presentation_digest"] == hashlib.sha256(encoded).hexdigest()
    assert "voc" not in payload["result"]


def test_cli_voc_evpi_is_an_affine_presentation(tmp_path: Path) -> None:
    request = _request(tmp_path, "affine-clairvoyant.json")
    response = RUNNER.invoke(
        cli.app,
        [
            "--format",
            "json",
            "calculate-expected-utility-information",
            str(request),
            "--presentation",
            "voc",
            "--measure",
            "evpi",
        ],
    )

    assert response.exit_code == 0, response.stdout
    payload = json.loads(response.stdout)
    assert payload["result"]["presentation"]["selected_measure"] == "evpi"
    assert payload["selected_value"] == payload["result"]["affine_reduction"]["value"]


def test_cli_rejects_nonlinear_monetary_evpi_and_has_no_calculate_voc(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, "log-buy-sell-asymmetry.json")
    response = RUNNER.invoke(
        cli.app,
        [
            "calculate-expected-utility-information",
            str(request),
            "--presentation",
            "voc",
            "--measure",
            "evpi",
        ],
    )

    assert response.exit_code == 1
    assert "affine utility reduction" in response.stderr
    help_response = RUNNER.invoke(cli.app, ["--help"])
    assert "calculate-expected-utility-information" in help_response.stdout
    assert "calculate-voc" not in help_response.stdout


def test_cli_surfaces_failed_price_root_instead_of_reporting_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    request = _request(tmp_path, "log-buy-sell-asymmetry.json")
    original = cli.expected_utility_information_value

    def failed_result(payload: dict[str, object]) -> dict[str, object]:
        result = original(payload)
        result["bpi"] = {
            **result["bpi"],
            "status": "failed",
            "value": None,
            "diagnostics_ref": "bpi_root",
        }
        result["bpi_root"] = {
            **result["bpi_root"],
            "status": "discontinuous_no_root",
            "termination_reason": "discontinuous_no_root",
        }
        return result

    monkeypatch.setattr(cli, "expected_utility_information_value", failed_result)
    response = RUNNER.invoke(
        cli.app,
        ["calculate-expected-utility-information", str(request), "--measure", "bpi"],
    )

    assert response.exit_code == 1
    assert "bpi failed: discontinuous_no_root" in response.stderr
    assert "unavailable" not in response.stderr


def test_cli_rejects_non_object_requests_and_canonical_evpi(tmp_path: Path) -> None:
    """The command owns both request-shape and presentation-alias diagnostics."""
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    shape_response = RUNNER.invoke(
        cli.app,
        ["calculate-expected-utility-information", str(non_object)],
    )
    assert shape_response.exit_code == 1
    assert "must contain a JSON object" in shape_response.stderr

    request = _request(tmp_path, "affine-clairvoyant.json")
    evpi_response = RUNNER.invoke(
        cli.app,
        [
            "calculate-expected-utility-information",
            str(request),
            "--measure",
            "evpi",
        ],
    )
    assert evpi_response.exit_code == 1
    assert "requires --presentation voc" in evpi_response.stderr


def test_cli_writes_and_announces_output_when_status_messages_are_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output persistence remains separate from optional terminal status text."""
    request = _request(tmp_path, "affine-clairvoyant.json")
    output = tmp_path / "result.json"
    monkeypatch.setattr(cli, "_should_echo_status_messages", lambda: True)

    response = RUNNER.invoke(
        cli.app,
        [
            "--format",
            "json",
            "calculate-expected-utility-information",
            str(request),
            "--output",
            str(output),
        ],
    )

    assert response.exit_code == 0, response.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["command"] == (
        "calculate-expected-utility-information"
    )
    assert f"Result saved to {output}" in response.stdout


def test_cli_converts_unexpected_expected_utility_failures_to_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected facade failures do not escape the command boundary."""
    request = _request(tmp_path, "affine-clairvoyant.json")

    def unexpected(_: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(cli, "expected_utility_information_value", unexpected)
    response = RUNNER.invoke(
        cli.app,
        ["calculate-expected-utility-information", str(request)],
    )

    assert response.exit_code == 1
    assert "An error occurred: synthetic unexpected failure" in response.stderr
