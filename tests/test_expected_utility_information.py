from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.utility_information import (
    expected_utility_information_value,
    value_of_clairvoyance,
)

FIXTURE = (
    Path(__file__).parents[1]
    / "specs/frontier/expected-utility-information-pricing/v1/fixtures/normative"
)


def _request(name: str) -> dict[str, object]:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))["request"]


def test_python_facade_reproduces_nonlinear_reference() -> None:
    request = _request("log-buy-sell-asymmetry.json")
    result = expected_utility_information_value(request)
    assert result["schema_version"] == "expected-utility-information-result-v1"
    assert result["method_maturity"] == "experimental"
    assert result["bpi"]["value"] == pytest.approx(3.7521886610, abs=1e-7)
    assert result["spi"]["value"] == pytest.approx(3.4085030261, abs=1e-7)
    assert "voc" not in result
    repeated = expected_utility_information_value(request)
    assert json.dumps(result, sort_keys=True) == json.dumps(repeated, sort_keys=True)


def test_voc_presentation_delegates_to_canonical_callable(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def canonical(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return {"presentation": {"presentation_label": "voc"}, "eui": {"value": 2.0}}

    monkeypatch.setattr(
        "voiage.methods.utility_information.expected_utility_information_value",
        canonical,
    )
    result = value_of_clairvoyance(
        _request("affine-clairvoyant.json"), selected_measure="bpi"
    )
    assert len(calls) == 1
    assert calls[0]["presentation_label"] == "voc"
    assert result["presentation"]["selected_measure"] == "bpi"
    assert "voc" not in result


def test_voc_rejects_finite_signal_presentation() -> None:
    request = _request("affine-clairvoyant.json")
    request["information"]["kind"] = "finite_signal"
    with pytest.raises(Exception, match="clairvoyant"):
        value_of_clairvoyance(request)


def test_decision_analysis_exposes_explicit_state_contract() -> None:
    analysis = DecisionAnalysis(nb_array=np.array([[0.0, 1.0], [1.0, 0.0]]))
    result = analysis.expected_utility_information(_request("affine-clairvoyant.json"))
    assert result["affine_reduction"]["monetary_measure"] == "evpi"
