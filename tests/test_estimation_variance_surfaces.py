"""CLI, report, example and plot surfaces for estimation variance VOI."""

# pyright: reportAny=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportUnknownMemberType=false

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.contracts.estimation import (
    ConditioningSpec,
    EstimationTargetSpec,
    EstimationVarianceResult,
    EstimationVarianceSpec,
    EstimatorAssuranceSpec,
    SamplingModelSpec,
)
from voiage.methods.estimation import evppi_var
from voiage.plot.estimation_variance import plot_estimation_variance
from voiage.reporting import build_estimation_variance_reporting

runner = CliRunner()


def _evppi_spec() -> EstimationVarianceSpec:
    return EstimationVarianceSpec(
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
        ),
    )


def _result() -> EstimationVarianceResult:
    return evppi_var(
        [0.0, 2.0, 1.0, 3.0],
        ["a", "a", "b", "b"],
        specification=_evppi_spec(),
    )


def test_estimation_reporting_preserves_values_assurance_and_provenance() -> None:
    report = build_estimation_variance_reporting(_result())
    assert report["method_family"] == "estimation-focused-variance-voi"
    assert report["absolute_reduction"] == pytest.approx(0.25)
    assert report["functional_units"] == "count^2"
    assert report["maturity"] == "experimental"
    assert report["provenance"]["backend"] == "rust"


def test_estimation_variance_plot_has_direct_labels_and_hatch() -> None:
    pytest.importorskip("matplotlib")
    axes = plot_estimation_variance(_result())
    assert axes.get_ylabel() == "Variance functional (count^2)"
    assert axes.get_title() == "Estimation uncertainty: net_cases"
    assert [patch.get_height() for patch in axes.patches] == [1.25, 1.0]
    assert axes.patches[1].get_hatch() == "//"
    same_axes = plot_estimation_variance(_result(), ax=axes)
    assert same_axes is axes


def test_estimation_variance_cli_emits_versioned_result(tmp_path: Path) -> None:
    specification_path = tmp_path / "specification.json"
    data_path = tmp_path / "data.json"
    _ = specification_path.write_text(
        _evppi_spec().model_dump_json(indent=2),
        encoding="utf-8",
    )
    _ = data_path.write_text(
        json.dumps(
            {
                "target_samples": [0.0, 2.0, 1.0, 3.0],
                "conditioning_groups": ["a", "a", "b", "b"],
            }
        ),
        encoding="utf-8",
    )
    invocation = runner.invoke(
        app,
        [
            "calculate-estimation-variance",
            str(specification_path),
            str(data_path),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(invocation.output)
    assert payload["result"]["schema_version"] == "1.0.0"
    assert payload["result"]["absolute_reduction"] == pytest.approx(0.25)
    assert payload["reporting"]["provenance"]["backend"] == "rust"


def test_estimation_variance_cli_evsi_writes_output(tmp_path: Path) -> None:
    specification = EstimationVarianceSpec(
        method_id="evsi_var",
        target=_evppi_spec().target,
        prior_model_id="study_prior",
        sampling_model=SamplingModelSpec(
            design_id="study",
            likelihood_id="likelihood",
            conditioning_sigma_field="sigma_y",
            averaging_convention="prior_predictive",
        ),
        estimator=EstimatorAssuranceSpec(
            estimator_id="posterior_variance_aggregation",
            seed=17,
        ),
    )
    specification_path = tmp_path / "specification.json"
    data_path = tmp_path / "data.json"
    output_path = tmp_path / "result.json"
    _ = specification_path.write_text(specification.model_dump_json(), encoding="utf-8")
    _ = data_path.write_text(
        json.dumps(
            {
                "prior_target_samples": [0.0, 1.0, 2.0, 3.0],
                "posterior_variances": [0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    invocation = runner.invoke(
        app,
        [
            "calculate-estimation-variance",
            str(specification_path),
            str(data_path),
            "--output",
            str(output_path),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["result"]["method_id"]
        == "evsi_var"
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "runtime sample JSON must be an object"),
        ("{}", "target_samples"),
    ],
)
def test_estimation_variance_cli_reports_invalid_runtime_payload(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    specification_path = tmp_path / "specification.json"
    data_path = tmp_path / "data.json"
    _ = specification_path.write_text(
        _evppi_spec().model_dump_json(),
        encoding="utf-8",
    )
    _ = data_path.write_text(payload, encoding="utf-8")
    invocation = runner.invoke(
        app,
        [
            "calculate-estimation-variance",
            str(specification_path),
            str(data_path),
        ],
    )
    assert invocation.exit_code == 1
    assert message in invocation.output


def test_estimation_variance_example_is_runnable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = (
        Path(__file__).resolve().parents[1] / "examples" / "estimation_variance.py"
    )
    namespace = runpy.run_path(str(example), run_name="test_example")
    namespace["main"]()
    assert "evppi_var: reduction=0.25 count^2" in capsys.readouterr().out
