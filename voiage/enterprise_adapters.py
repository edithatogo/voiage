"""Enterprise Model, Lineage, Metric, and Experiment Adapters (#583).

This module integrates VOIAGE as decision-value middleware across modern analytics,
experimentation, and ML stacks (MLflow, OpenLineage, dbt Semantic Layer,
GrowthBook/Statsig, EconML/CausalML, and probabilistic forecasts) without
requiring heavy enterprise SDKs as hard base dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import jsonschema

from voiage.exceptions import raise_input_error

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCHEMA_PATH = (
    _ROOT
    / "specs"
    / "integrations"
    / "enterprise"
    / "schemas"
    / "v1"
    / "enterprise-adapters.schema.json"
)
_DEFAULT_FIXTURE_PATH = (
    _ROOT
    / "specs"
    / "integrations"
    / "enterprise"
    / "fixtures"
    / "normative"
    / "adapters-fixture.json"
)


def _current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EnterpriseAdapterRecord:
    """Canonical envelope for enterprise stack adapter payloads.

    Attributes
    ----------
    adapter_type : str
        Integration category (e.g. mlflow_model, openlineage_facet).
    adapter_version : str
        Semantic version of the adapter format.
    source_system : str
        Originating system URI or identifier.
    extracted_at : str
        ISO 8601 extraction timestamp.
    metadata : dict[str, Any]
        Source system metadata dictionary.
    payload : dict[str, Any]
        Adapter-specific payload dictionary.
    """

    adapter_type: str
    adapter_version: str
    source_system: str
    extracted_at: str
    metadata: dict[str, Any]
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnterpriseAdapterRecord:
        """Instantiate adapter record from dictionary."""
        if not isinstance(data, dict):
            raise_input_error("EnterpriseAdapterRecord data must be a dictionary.")
        return cls(
            adapter_type=str(data["adapter_type"]),
            adapter_version=str(data.get("adapter_version", "1.0.0")),
            source_system=str(data["source_system"]),
            extracted_at=str(data.get("extracted_at", _current_iso_timestamp())),
            metadata=dict(data.get("metadata", {})),
            payload=dict(data.get("payload", {})),
        )


def adapt_mlflow_model_metadata(
    source_system: str,
    run_id: str,
    experiment_id: str,
    metrics: dict[str, float],
    parameters: dict[str, Any],
    target_variable: str,
    model_flavor: str = "custom",
) -> EnterpriseAdapterRecord:
    """Create an adapter record from MLflow model run metadata."""
    return EnterpriseAdapterRecord(
        adapter_type="mlflow_model",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"run_id": run_id, "experiment_id": experiment_id},
        payload={
            "model_flavor": model_flavor,
            "metrics": metrics,
            "parameters": parameters,
            "target_variable": target_variable,
        },
    )


def adapt_openlineage_job_facet(
    source_system: str,
    job_name: str,
    namespace: str,
    inputs: list[dict[str, str]],
    outputs: list[dict[str, str]],
) -> EnterpriseAdapterRecord:
    """Create an adapter record for OpenLineage dataset and job lineage."""
    return EnterpriseAdapterRecord(
        adapter_type="openlineage_facet",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"job_name": job_name, "namespace": namespace},
        payload={"inputs": inputs, "outputs": outputs},
    )


def adapt_dbt_semantic_metric(
    source_system: str,
    metric_name: str,
    grain: str,
    expression: str,
    filter_clause: str = "",
    dimensions: list[str] | None = None,
) -> EnterpriseAdapterRecord:
    """Create an adapter record from dbt Semantic Layer metric definitions."""
    return EnterpriseAdapterRecord(
        adapter_type="dbt_semantic_metric",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"metric_name": metric_name, "grain": grain},
        payload={
            "type": "derived",
            "expression": expression,
            "filter": filter_clause,
            "dimensions": dimensions or [],
        },
    )


def adapt_experiment_export(
    source_system: str,
    experiment_id: str,
    variations: list[str],
    sample_sizes: dict[str, int],
    conversion_rates: dict[str, float],
    lift_mean: float,
    lift_ci_95: list[float],
) -> EnterpriseAdapterRecord:
    """Create an adapter record from A/B experiment platform exports."""
    return EnterpriseAdapterRecord(
        adapter_type="experiment_export",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"experiment_id": experiment_id, "status": "exported"},
        payload={
            "variations": variations,
            "sample_sizes": sample_sizes,
            "conversion_rates": conversion_rates,
            "lift_mean": lift_mean,
            "lift_ci_95": lift_ci_95,
        },
    )


def adapt_causal_cate_artifact(
    source_system: str,
    treatment: str,
    outcome: str,
    heterogeneity_features: list[str],
    average_treatment_effect: float,
    ate_standard_error: float,
    estimator_type: str = "CausalForest",
) -> EnterpriseAdapterRecord:
    """Create an adapter record for CATE heterogeneous uplift estimators."""
    return EnterpriseAdapterRecord(
        adapter_type="causal_cate_artifact",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"estimator_type": estimator_type},
        payload={
            "treatment": treatment,
            "outcome": outcome,
            "heterogeneity_features": heterogeneity_features,
            "average_treatment_effect": average_treatment_effect,
            "ate_standard_error": ate_standard_error,
        },
    )


def adapt_probabilistic_forecast(
    source_system: str,
    sku_id: str,
    quantiles: dict[str, list[float]],
    mean: list[float],
    forecast_horizon_weeks: int = 12,
) -> EnterpriseAdapterRecord:
    """Create an adapter record for probabilistic quantile demand forecasts."""
    return EnterpriseAdapterRecord(
        adapter_type="probabilistic_forecast",
        adapter_version="1.0.0",
        source_system=source_system,
        extracted_at=_current_iso_timestamp(),
        metadata={"sku_id": sku_id, "forecast_horizon_weeks": forecast_horizon_weeks},
        payload={"quantiles": quantiles, "mean": mean},
    )


def validate_enterprise_adapter_record(
    record_dict: dict[str, Any], schema_path: Path | None = None
) -> bool:
    """Validate a raw dictionary against the Enterprise Adapters JSON schema."""
    s_path = schema_path or _DEFAULT_SCHEMA_PATH
    if not s_path.is_file():
        raise_input_error(f"Enterprise adapter schema not found at {s_path}")

    schema = json.loads(s_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=record_dict, schema=schema)
    return True


def load_enterprise_adapters_from_fixture(
    fixture_path: Path | None = None,
) -> list[EnterpriseAdapterRecord]:
    """Load and parse enterprise adapter records from a fixture file."""
    f_path = fixture_path or _DEFAULT_FIXTURE_PATH
    if not f_path.is_file():
        raise_input_error(f"Enterprise adapter fixture not found at {f_path}")

    raw = json.loads(f_path.read_text(encoding="utf-8"))
    return [EnterpriseAdapterRecord.from_dict(item) for item in raw.get("adapters", [])]
