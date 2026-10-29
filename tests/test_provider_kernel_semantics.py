"""Independent provider-to-kernel semantic witness checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.check_consumer_matrix import evaluate_provider_semantics

ROOT = Path(__file__).resolve().parents[1]


def test_semantic_reference_is_calculated_from_provider_samples(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/compatibility_witnesses/provider-semantics.json"
        ).read_text()
    )
    case = fixture["cases"][0]
    case["expected_voi"] = 1.2
    for provider in case["providers"]:
        if "rows" in provider:
            provider["rows"][0]["weight"] = 0.8
            provider["rows"][1]["weight"] = 0.2
        else:
            provider["samples"][0]["probability"] = 0.2
            provider["samples"][1]["probability"] = 0.8
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(fixture))
    result = evaluate_provider_semantics(ROOT, path)
    assert result["passed"] is True
    assert result["cases"][0]["reference_voi"] == pytest.approx(1.2)


def test_semantic_reference_fails_when_provider_values_disagree_with_oracle(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/compatibility_witnesses/provider-semantics.json"
        ).read_text()
    )
    fixture["cases"][0]["expected_voi"] = 999.0
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(fixture))
    result = evaluate_provider_semantics(ROOT, path)
    assert result["passed"] is False
    assert result["cases"][0]["reference_passed"] is False


def test_provider_witness_handles_zero_weight_and_single_action(tmp_path: Path) -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/compatibility_witnesses/provider-semantics.json"
        ).read_text()
    )
    case = fixture["cases"][0]
    provider = case["providers"][0]
    case["providers"] = [provider]
    case["expected_voi"] = 0.0
    provider["rows"][0].update(weight=1.0, net_benefit=[10.0])
    provider["rows"][1].update(weight=0.0, net_benefit=[2.0])
    path = tmp_path / "single-action.json"
    path.write_text(json.dumps(fixture))
    result = evaluate_provider_semantics(ROOT, path)
    assert result["passed"] is True
    assert result["cases"][0]["reference_voi"] == pytest.approx(0.0)


def test_cli_fails_when_provider_witness_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.check_consumer_matrix as consumer

    output = tmp_path / "provider.json"
    monkeypatch.setattr(consumer, "evaluate_matrix", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        consumer, "evaluate_provider_semantics", lambda *_args: {"passed": False}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_consumer_matrix.py", "--provider-semantics-output", str(output)],
    )
    assert consumer.main() == 2
    assert json.loads(output.read_text())["passed"] is False


def test_provider_witness_rejects_null_provider(tmp_path: Path) -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/compatibility_witnesses/provider-semantics.json"
        ).read_text()
    )
    fixture["cases"][1]["providers"] = [None]
    path = tmp_path / "null-provider.json"
    path.write_text(json.dumps(fixture))
    result = evaluate_provider_semantics(ROOT, path)
    assert result["passed"] is True
    assert result["cases"][1]["actual"] == "rejected"
