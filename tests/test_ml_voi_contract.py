"""Contract tests distinguishing information gain from decision VOI."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/ai-assisted-evidence-triage/v1/fixtures/normative"
    / "eig-versus-voi.json"
)


def test_eig_fixture_keeps_information_and_decision_values_distinct() -> None:
    """Entropy reduction must not be mislabeled as economic VOI."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["expected_information_gain_nats"] > 0
    assert payload["expected_decision_voi"] == (
        payload["posterior_expected_utility"]
        - payload["current_expected_utility"]
        - payload["information_cost"]
    )
    assert "utility" in payload["interpretation"]
    assert "cost" in payload["interpretation"]
    assert payload["expected_decision_voi"] != payload["expected_information_gain_nats"]


def test_ml_contract_requires_offline_cpu_and_optional_backends() -> None:
    """ML/LLM methods must not require providers or private-data transport."""
    spec = (ROOT / "conductor/tracks/ml_llm_agent_voi_20260723/spec.md").read_text(
        encoding="utf-8"
    )
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = "\n".join(metadata["project"]["dependencies"]).lower()

    spec = spec.lower()
    assert "offline tables" in spec
    assert "cpu deterministic" in spec
    assert "no network or" in spec
    assert "private-data transmission" in spec
    assert "pyro" in spec
    assert "botorch" in spec
    assert "pyro" not in base
    assert "botorch" not in base
