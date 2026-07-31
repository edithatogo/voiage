"""Contracts and numerical evidence for estimation-focused variance VOI."""

from __future__ import annotations

import pytest

from voiage.methods.estimation import (
    ESTIMATION_VARIANCE_METHODS,
    estimation_variance_method,
)


def test_estimation_variance_registry_separates_decision_and_sensitivity_methods() -> (
    None
):
    assert set(ESTIMATION_VARIANCE_METHODS) == {"evppi_var", "evsi_var"}
    for method_id, descriptor in ESTIMATION_VARIANCE_METHODS.items():
        assert descriptor["method_id"] == method_id
        assert descriptor["family"] == "estimation-focused-variance-voi"
        assert descriptor["estimand_kind"] == "variance_reduction"
        assert descriptor["decision_focused"] is False
        assert descriptor["sensitivity_index"] is False
        assert descriptor["estimator_uncertainty"] is False
        assert method_id not in {"evppi", "evsi"}


@pytest.mark.parametrize(
    ("name", "method_id"),
    [
        ("evppi_var", "evppi_var"),
        ("evppi_variance", "evppi_var"),
        ("evsi_var", "evsi_var"),
        ("evsi_variance", "evsi_var"),
    ],
)
def test_estimation_variance_aliases_are_explicit(name: str, method_id: str) -> None:
    assert estimation_variance_method(name)["method_id"] == method_id


@pytest.mark.parametrize("name", ["evppi", "evsi", "sobol", "posterior_variance"])
def test_decision_and_adjacent_names_are_not_estimation_variance_aliases(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="estimation-focused variance"):
        estimation_variance_method(name)
