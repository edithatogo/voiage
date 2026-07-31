from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage import cli

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
    assert payload["result"]["presentation"] == {
        "canonical_result_ref": "self",
        "presentation_label": "canonical",
        "selected_measure": "bpi",
    }
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
