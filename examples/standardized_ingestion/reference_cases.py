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


if __name__ == "__main__":
    print(json.dumps(run_reference_cases(), sort_keys=True))
