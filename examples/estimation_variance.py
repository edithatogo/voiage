"""Runnable scalar EVPPI-var example using the experimental governed API."""

from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationTargetSpec,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
)
from voiage.methods.estimation import evppi_var
from voiage.reporting import build_estimation_variance_reporting


def main() -> None:
    """Run an enumerable discrete-conditioning example."""
    specification = EstimationVarianceSpec(
        method_id="evppi_var",
        target=EstimationTargetSpec(
            target_id="net_cases",
            shape="scalar",
            component_units=("count",),
            covariance_functional="variance",
        ),
        prior_model_id="enumerable_prior",
        conditioning=ConditioningSpec(
            parameter_subset=("risk_state",),
            sigma_field="sigma_risk_state",
            averaging_convention="empirical_reference",
        ),
        estimator=EstimatorAssuranceSpec(
            estimator_id="discrete_conditioning",
            seed=17,
            bootstrap_replicates=128,
            convergence_threshold=1.0,
        ),
    )
    result = evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=specification,
    )
    report = build_estimation_variance_reporting(result)
    print(
        " ".join(
            (
                f"{report['method_id']}: reduction={report['absolute_reduction']}",
                str(report["functional_units"]),
            )
        )
    )


if __name__ == "__main__":
    main()
