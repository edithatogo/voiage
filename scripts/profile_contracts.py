#!/usr/bin/env python3
"""Bounded profiling workload for the opt-in v2 contract adapters."""

from __future__ import annotations

import numpy as np

from voiage.contracts.analysis import AnalysisSpec, NumericalPolicy
from voiage.contracts.kernel import run_evpi
from voiage.contracts.perspective import run_perspective


def main() -> None:
    """Exercise scalar and perspective envelopes with deterministic inputs."""
    rng = np.random.default_rng(20260719)
    values = rng.normal(size=(2_000, 4))
    policy = NumericalPolicy(backend_preference=("numpy",))
    spec = AnalysisSpec(
        analysis_id="profile-evpi-v2",
        decision_problem_id="profile-decision",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("A", "B", "C", "D"),
        numerical_policy=policy,
    )
    _ = run_evpi(values, spec=spec, policy=policy)

    perspective_values = np.stack((values, values * 1.05), axis=2)
    _ = run_perspective(
        perspective_values,
        analysis_id="profile-perspective-v2",
        decision_problem_id="profile-decision",
        strategy_names=spec.strategy_names,
        perspective_names=("payer", "societal"),
        policy=policy,
    )


if __name__ == "__main__":
    main()
