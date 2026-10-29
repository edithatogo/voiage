"""Independent finite-world witnesses for information about structure only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voiage.methods.information_source_portfolio import (
    information_source_portfolio_value,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("worlds", "expected"),
    [
        # Both sample outcomes have the same structure. Structure is already
        # known, although a sample-revealing oracle could be worth five.
        ([("same", (10.0, 0.0), 0.5), ("same", (0.0, 10.0), 0.5)], 0.0),
        # Structure distinguishes the states and switches the optimal action.
        ([("first", (10.0, 0.0), 0.5), ("second", (0.0, 10.0), 0.5)], 5.0),
        # The fixed-action baseline is 8 and revealing structure yields 10.
        ([("first", (10.0, 0.0), 0.8), ("second", (0.0, 10.0), 0.2)], 2.0),
        # Each structure contains two outcomes. Knowing structure yields 5
        # against a baseline of 4; knowing the individual outcome yields 8.
        (
            [
                ("first", (10.0, 0.0), 0.25),
                ("first", (0.0, 6.0), 0.25),
                ("second", (0.0, 10.0), 0.25),
                ("second", (6.0, 0.0), 0.25),
            ],
            1.0,
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_finite_world_structure_observation(
    worlds: list[tuple[str, tuple[float, float], float]], expected: float, reverse: bool
) -> None:
    """Do not accidentally reveal the sample within an observed structure."""
    path = (
        ROOT
        / "specs/frontier/information-source-portfolio/v1/fixtures/normative/input.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["sources"][0]
    source.update(cost=0.0, latency=0.0, privacy_cost=0.0, must_precede=[], excludes=[])
    source_id = source["source_id"]
    payload.update(
        actions=["a", "b"],
        sources=[source],
        states=[
            {
                "state_id": f"world-{index}",
                "probability": probability,
                "action_values": dict(zip(("a", "b"), values, strict=True)),
                "source_observations": {source_id: structure},
            }
            for index, (structure, values, probability) in enumerate(
                reversed(worlds) if reverse else worlds
            )
        ],
    )
    payload["value_context"]["delay_cost_per_time"] = 0.0
    payload["constraints"].update(max_sources=1, required_coverage=[])
    result = information_source_portfolio_value(payload).to_contract_dict()
    assert result["optimum"]["gross_value"] == pytest.approx(expected)
