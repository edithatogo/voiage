"""Run deterministic ML, engineering, and business ingestion reference cases."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from voiage.contracts import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
    VOIBinding,
    prepare_analysis_inputs,
)
from voiage.ingestion import SourceAccessPolicy, default_registry
from voiage.methods.basic import evpi

_ROOT = Path(__file__).parents[2]
_FIXTURES = _ROOT / "tests" / "fixtures" / "standardized_ingestion"


def _binding() -> VOIBinding:
    return VOIBinding(
        role="net_benefit",
        table_id="samples",
        field_ids=("strategy_a", "strategy_b"),
        strategy_names=("A", "B"),
    )


def _bound(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=bundle.manifest.model_copy(update={"bindings": (_binding(),)}),
        tables=bundle.tables,
    )


def _direct() -> NormalizedInputBundle:
    table = pa.table(
        {"strategy_a": [10.0, 30.0, 20.0], "strategy_b": [20.0, 10.0, 25.0]}
    )
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="canonical-decision-fixture",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=tuple(
                        FieldManifest(field_id=field.name, dtype=str(field.type))
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct-reference-case",
                source_uri="urn:voiage:reference-case:business",
                descriptor_digest="f" * 64,
            ),
            bindings=(_binding(),),
        ),
        tables={"samples": table},
    )


def _cost_outcome_bindings() -> tuple[VOIBinding, VOIBinding]:
    return (
        VOIBinding(
            role="cost",
            table_id="samples",
            field_ids=("cost_a", "cost_b"),
            strategy_names=("A", "B"),
            unit="currency",
        ),
        VOIBinding(
            role="outcome",
            table_id="samples",
            field_ids=("outcome_a", "outcome_b"),
            strategy_names=("A", "B"),
            unit="outcome",
        ),
    )


def _direct_cost_outcome() -> NormalizedInputBundle:
    table = pa.table(
        {
            "cost_a": [100.0, 180.0, 130.0],
            "cost_b": [150.0, 120.0, 160.0],
            "outcome_a": [0.010, 0.016, 0.009],
            "outcome_b": [0.014, 0.012, 0.013],
        }
    )
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="cost-outcome-decision-fixture",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=tuple(
                        FieldManifest(field_id=field.name, dtype=str(field.type))
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct-reference-case",
                source_uri="urn:voiage:reference-case:business-cost-outcome",
                descriptor_digest="e" * 64,
            ),
            bindings=_cost_outcome_bindings(),
        ),
        tables={"samples": table},
    )


def run_reference_cases() -> dict[str, object]:
    """Calculate one explicit EVPI case through each supported input surface."""
    policy = SourceAccessPolicy(_FIXTURES)
    bundles = {
        "ml": _bound(
            default_registry().ingest(
                _FIXTURES / "canonical-decision.croissant.json", policy=policy
            )
        ),
        "engineering": _bound(
            default_registry().ingest(
                _FIXTURES / "canonical-decision.datapackage.json", policy=policy
            )
        ),
        "business": _direct(),
    }
    values = {
        domain: float(evpi(prepare_analysis_inputs(bundle).net_benefits))
        for domain, bundle in bundles.items()
    }
    if len(set(values.values())) != 1:
        raise RuntimeError("reference cases must have identical explicit EVPI")
    return {"binding": _binding().model_dump(mode="json"), "evpi": values}


def run_cost_outcome_reference_cases() -> dict[str, float]:
    """Derive net benefit from explicit cost/outcome bindings in each surface."""
    policy = SourceAccessPolicy(_FIXTURES)
    bindings = _cost_outcome_bindings()

    def with_bindings(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
        return NormalizedInputBundle(
            manifest=bundle.manifest.model_copy(update={"bindings": bindings}),
            tables=bundle.tables,
        )

    bundles = {
        "ml": with_bindings(
            default_registry().ingest(
                _FIXTURES / "cost-outcome-decision.croissant.json", policy=policy
            )
        ),
        "engineering": with_bindings(
            default_registry().ingest(
                _FIXTURES / "cost-outcome-decision.datapackage.json", policy=policy
            )
        ),
        "business": _direct_cost_outcome(),
    }
    values = {
        domain: float(
            evpi(
                prepare_analysis_inputs(
                    bundle, willingness_to_pay=20_000.0
                ).net_benefits
            )
        )
        for domain, bundle in bundles.items()
    }
    if len(set(values.values())) != 1:
        raise RuntimeError("cost/outcome reference cases must have identical EVPI")
    return values


if __name__ == "__main__":
    print(json.dumps(run_reference_cases(), sort_keys=True))
