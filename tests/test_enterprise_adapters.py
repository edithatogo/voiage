"""Tests for Enterprise Model, Lineage, Metric, and Experiment Adapters (#583)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.enterprise_adapters import (
    EnterpriseAdapterRecord,
    adapt_causal_cate_artifact,
    adapt_dbt_semantic_metric,
    adapt_experiment_export,
    adapt_mlflow_model_metadata,
    adapt_openlineage_job_facet,
    adapt_probabilistic_forecast,
    load_enterprise_adapters_from_fixture,
    validate_enterprise_adapter_record,
)
from voiage.exceptions import InputError


def test_load_and_validate_enterprise_adapters_fixture() -> None:
    adapters = load_enterprise_adapters_from_fixture()
    assert len(adapters) == 6

    for adapter in adapters:
        assert validate_enterprise_adapter_record(adapter.to_dict()) is True


def test_adapt_mlflow_model_metadata() -> None:
    record = adapt_mlflow_model_metadata(
        source_system="mlflow://tracking/models/churn_xgb",
        run_id="run_123",
        experiment_id="exp_01",
        metrics={"auc": 0.88},
        parameters={"max_depth": 5},
        target_variable="churn",
        model_flavor="xgboost",
    )
    assert record.adapter_type == "mlflow_model"
    assert record.metadata["run_id"] == "run_123"
    assert record.payload["metrics"]["auc"] == 0.88
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_adapt_openlineage_job_facet() -> None:
    record = adapt_openlineage_job_facet(
        source_system="openlineage://airflow/dags/job_voi",
        job_name="voi_pricing_job",
        namespace="prod_analytics",
        inputs=[{"namespace": "snowflake", "name": "dw.pricing"}],
        outputs=[{"namespace": "s3", "name": "cards/pricing.json"}],
    )
    assert record.adapter_type == "openlineage_facet"
    assert len(record.payload["inputs"]) == 1
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_adapt_dbt_semantic_metric() -> None:
    record = adapt_dbt_semantic_metric(
        source_system="dbt://models/metrics.yml",
        metric_name="cac",
        grain="month",
        expression="spend / new_customers",
        filter_clause="channel = 'search'",
        dimensions=["channel", "region"],
    )
    assert record.adapter_type == "dbt_semantic_metric"
    assert record.payload["dimensions"] == ["channel", "region"]
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_adapt_experiment_export() -> None:
    record = adapt_experiment_export(
        source_system="growthbook://experiments/exp_1",
        experiment_id="exp_1",
        variations=["control", "variant"],
        sample_sizes={"control": 5000, "variant": 5000},
        conversion_rates={"control": 0.05, "variant": 0.06},
        lift_mean=0.01,
        lift_ci_95=[0.002, 0.018],
    )
    assert record.adapter_type == "experiment_export"
    assert record.payload["lift_mean"] == 0.01
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_adapt_causal_cate_artifact() -> None:
    record = adapt_causal_cate_artifact(
        source_system="econml://causal_forest",
        treatment="discount",
        outcome="retention",
        heterogeneity_features=["tenure", "spend"],
        average_treatment_effect=0.15,
        ate_standard_error=0.02,
        estimator_type="CausalForestDML",
    )
    assert record.adapter_type == "causal_cate_artifact"
    assert record.payload["treatment"] == "discount"
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_adapt_probabilistic_forecast() -> None:
    record = adapt_probabilistic_forecast(
        source_system="forecast://demand_v1",
        sku_id="SKU-100",
        quantiles={"p10": [10.0, 12.0], "p90": [20.0, 22.0]},
        mean=[15.0, 17.0],
        forecast_horizon_weeks=2,
    )
    assert record.adapter_type == "probabilistic_forecast"
    assert record.payload["quantiles"]["p10"] == [10.0, 12.0]
    assert validate_enterprise_adapter_record(record.to_dict()) is True


def test_enterprise_adapter_error_handling() -> None:
    with pytest.raises(InputError, match="must be a dictionary"):
        EnterpriseAdapterRecord.from_dict("invalid")  # type: ignore[arg-type]

    non_existent = Path("specs/integrations/enterprise/schemas/v1/missing.schema.json")
    with pytest.raises(InputError, match="not found"):
        validate_enterprise_adapter_record({}, schema_path=non_existent)

    non_existent_fixture = Path("specs/integrations/enterprise/missing_fixture.json")
    with pytest.raises(InputError, match="not found"):
        load_enterprise_adapters_from_fixture(fixture_path=non_existent_fixture)
