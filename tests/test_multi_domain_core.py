"""Focused coverage for the concrete multi-domain VOI helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from voiage.exceptions import InputError
from voiage.multi_domain import (
    DomainParameters,
    DomainType,
    EngineeringParameters,
    EnvironmentalParameters,
    FinanceParameters,
    ManufacturingParameters,
    MultiDomainVOI,
    calculate_domain_evpi,
    compare_domain_performance,
    create_engineering_voi,
    create_environmental_voi,
    create_finance_voi,
    create_manufacturing_voi,
)


@dataclass
class _FakeDecisionAnalysis:
    evpi_value: float = 10.0

    def evpi(self) -> float:
        return self.evpi_value

    def calculate_evpi(self) -> float:
        return self.evpi_value

    def evppi(self, _parameters: list[str]) -> float:
        return self.evpi_value / 2.0

    def get_decision_recommendations(self) -> list[dict[str, object]]:
        return [{"strategy": "A", "recommended": True}]


def test_domain_factory_helpers_create_expected_analysis_types() -> None:
    """Factory helpers should attach the matching domain type and defaults."""
    manufacturing = create_manufacturing_voi(ManufacturingParameters())
    finance = create_finance_voi(FinanceParameters())
    environmental = create_environmental_voi(EnvironmentalParameters())
    engineering = create_engineering_voi(EngineeringParameters())

    assert manufacturing.domain_type is DomainType.MANUFACTURING
    assert finance.domain_type is DomainType.FINANCE
    assert environmental.domain_type is DomainType.ENVIRONMENTAL
    assert engineering.domain_type is DomainType.ENGINEERING
    assert callable(manufacturing.outcome_function)
    assert callable(finance.outcome_function)
    assert callable(environmental.outcome_function)
    assert callable(engineering.outcome_function)


@pytest.mark.parametrize(
    ("domain_type", "params", "decision_vars", "uncertainties"),
    [
        (
            DomainType.MANUFACTURING,
            ManufacturingParameters(),
            jnp.array([500.0, 0.95, 100.0]),
            jnp.array([0.1, 0.05, 0.02]),
        ),
        (
            DomainType.FINANCE,
            FinanceParameters(),
            jnp.array([0.7, 0.5, 3.0]),
            jnp.array([0.08, 0.2, 0.3]),
        ),
        (
            DomainType.ENVIRONMENTAL,
            EnvironmentalParameters(),
            jnp.array([100.0, 2.0, 1.0]),
            jnp.array([0.1, 0.2, 0.05]),
        ),
        (
            DomainType.ENGINEERING,
            EngineeringParameters(),
            jnp.array([3.0, 2.0, 5.0]),
            jnp.array([0.1, 0.0, 0.0]),
        ),
    ],
)
def test_default_domain_outcomes_return_finite_metric_vectors(
    domain_type: DomainType,
    params: object,
    decision_vars: jnp.ndarray,
    uncertainties: jnp.ndarray,
) -> None:
    """Each concrete default outcome function should produce five metrics."""
    analysis = MultiDomainVOI(domain_type, params)

    result = analysis.outcome_function(decision_vars, uncertainties, params)

    assert result.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_generic_domain_outcome_and_wrapper_error_path() -> None:
    """Unsupported domains should use the generic outcome and validate wrappers."""
    params = DomainParameters(
        domain_type=DomainType.EDUCATION,
        name="Education",
        description="Education analysis",
    )
    analysis = MultiDomainVOI(DomainType.EDUCATION, params)

    result = analysis.outcome_function(jnp.array([1.0, 2.0]), jnp.array([3.0]), params)

    assert float(result) == pytest.approx(6.0)

    analysis.outcome_function = None
    with pytest.raises(InputError, match="No outcome function set"):
        analysis._domain_outcome_wrapper()(jnp.array([1.0]))


def test_generic_parameter_conversion_feeds_domain_specific_outcomes() -> None:
    """Generic domain parameters should convert into concrete parameter classes."""
    generic = DomainParameters(
        domain_type=DomainType.MANUFACTURING,
        name="Generic",
        description="Converted",
        additional_params={"production_capacity": 250.0},
    )
    analysis = MultiDomainVOI(DomainType.MANUFACTURING, generic)

    converted = analysis._convert_to_manufacturing_params(generic)
    result = analysis._manufacturing_outcome(
        jnp.array([100.0, 0.9, 10.0]),
        jnp.array([0.1, 0.0, 0.0]),
        generic,
    )

    assert converted.production_capacity == 250.0
    assert result.shape == (5,)


@pytest.mark.parametrize(
    ("domain_type", "expected_key"),
    [
        (DomainType.MANUFACTURING, "operational_risk"),
        (DomainType.FINANCE, "liquidity_risk"),
        (DomainType.ENVIRONMENTAL, "scientific_uncertainty"),
        (DomainType.ENGINEERING, "safety_risk"),
    ],
)
def test_domain_specific_metrics_insights_and_risks(
    domain_type: DomainType, expected_key: str
) -> None:
    """Domain metrics, insights, and risks should vary by domain."""
    params = DomainParameters(domain_type, "Domain", "Description")
    analysis = MultiDomainVOI(domain_type, params)
    decision_analysis = _FakeDecisionAnalysis(evpi_value=20.0)

    metrics = analysis.domain_specific_evpi(decision_analysis)
    risks = analysis._assess_domain_risks(decision_analysis)
    insights = analysis._generate_domain_insights(decision_analysis, metrics)

    assert metrics["evpi"] == 20.0
    assert expected_key in risks
    assert insights


def test_domain_report_and_comparison_helpers() -> None:
    """Report and comparison helpers should summarize available analyses."""
    params = ManufacturingParameters(name="Factory")
    analysis = MultiDomainVOI(DomainType.MANUFACTURING, params)
    decision_analysis = _FakeDecisionAnalysis(evpi_value=12.0)

    report = analysis.create_domain_report(decision_analysis)

    assert report["analysis_summary"]["domain_name"] == "Factory"
    assert report["voi_metrics"]["evpi"] == 12.0
    assert report["voi_metrics"]["evppi"] == 6.0
    assert report["decision_recommendations"] == [
        {"strategy": "A", "recommended": True}
    ]

    analysis.decision_analysis = decision_analysis
    empty = MultiDomainVOI(DomainType.FINANCE, FinanceParameters())

    assert calculate_domain_evpi(decision_analysis, DomainType.MANUFACTURING) == 12.0
    assert compare_domain_performance([analysis, empty]) == {
        "manufacturing_evpi": 12.0,
        "finance_evpi": 0.0,
    }

    with pytest.raises(InputError, match="No decision analysis"):
        analysis.create_domain_report(None)
