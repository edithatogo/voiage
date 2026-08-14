"""Run deterministic ML, engineering, and business ingestion reference cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
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
from voiage.ingestion import SourceAccessPolicy, default_registry, from_dataframe
from voiage.methods.basic import evpi
from voiage.methods.ceaf import calculate_ceaf
from voiage.schema import ValueArray

_ROOT = Path(__file__).parents[2]
_FIXTURES = _ROOT / "tests" / "fixtures" / "standardized_ingestion"


def _artifact(stem: str) -> dict[str, Any]:
    """Load the review-owned identity for one deterministic synthetic fixture."""
    payload = json.loads(
        (_FIXTURES / f"{stem}.manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):  # pragma: no cover - static JSON fixture
        raise TypeError("reference fixture manifest must be a JSON object")
    return payload


def _binding() -> VOIBinding:
    return VOIBinding(
        role="net_benefit",
        table_id="samples",
        field_ids=("strategy_a", "strategy_b"),
        strategy_names=("A", "B"),
    )


def _long_binding() -> VOIBinding:
    """Bind the deterministic long-form fixture without inferring its columns."""
    return VOIBinding(
        role="net_benefit",
        table_id="samples",
        field_ids=("net_benefit",),
        strategy_names=("A", "B"),
        layout="long",
        sample_id_field_id="sample_id",
        strategy_field_id="strategy",
        value_field_id="net_benefit",
    )


def _bound(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=bundle.manifest.model_copy(update={"bindings": (_binding(),)}),
        tables=bundle.tables,
    )


def _direct_long() -> NormalizedInputBundle:
    """Build the long-form decision table without an external descriptor."""
    table = pa.table(
        {
            "sample_id": [1, 1, 2, 2, 3, 3],
            "strategy": ["A", "B", "A", "B", "A", "B"],
            "net_benefit": [10.0, 20.0, 30.0, 10.0, 20.0, 25.0],
        }
    )
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="long-decision-fixture",
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
                source_uri="urn:voiage:reference-case:direct-long",
                descriptor_digest=_artifact("long-decision")["normalized"][
                    "resource_sha256"
                ],
            ),
            bindings=(_long_binding(),),
        ),
        tables={"samples": table},
    )


def _business_dataframe() -> NormalizedInputBundle:
    """Model a business user's in-memory decision table through the SDK."""
    table = pa.table(
        {"strategy_a": [10.0, 30.0, 20.0], "strategy_b": [20.0, 10.0, 25.0]}
    )
    return from_dataframe(
        table,
        dataset_id="canonical-decision-fixture",
        table_id="samples",
        bindings=(_binding(),),
        allow_copy=False,
    )


def _direct() -> NormalizedInputBundle:
    """Build the canonical decision without an external descriptor."""
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
                source_uri="urn:voiage:reference-case:direct",
                descriptor_digest=_artifact("canonical-decision")["normalized"][
                    "resource_sha256"
                ],
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


def _business_cost_outcome_dataframe() -> NormalizedInputBundle:
    """Use the same SDK path for an explicit cost/outcome decision table."""
    table = pa.table(
        {
            "cost_a": [100.0, 180.0, 130.0],
            "cost_b": [150.0, 120.0, 160.0],
            "outcome_a": [0.010, 0.016, 0.009],
            "outcome_b": [0.014, 0.012, 0.013],
        }
    )
    return from_dataframe(
        table,
        dataset_id="cost-outcome-decision-fixture",
        table_id="samples",
        bindings=_cost_outcome_bindings(),
        allow_copy=False,
    )


def _direct_cost_outcome() -> NormalizedInputBundle:
    """Build the cost/outcome decision without an external descriptor."""
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
                source_uri="urn:voiage:reference-case:direct-cost-outcome",
                descriptor_digest=_artifact("cost-outcome-decision")["normalized"][
                    "resource_sha256"
                ],
            ),
            bindings=_cost_outcome_bindings(),
        ),
        tables={"samples": table},
    )


def _net_benefit_surfaces(
    policy: SourceAccessPolicy,
) -> dict[str, NormalizedInputBundle]:
    """Materialize the canonical decision through every supported input surface."""
    return {
        "croissant": _bound(
            default_registry().ingest(
                _FIXTURES / "canonical-decision.croissant.json", policy=policy
            )
        ),
        "frictionless": _bound(
            default_registry().ingest(
                _FIXTURES / "canonical-decision.datapackage.json", policy=policy
            )
        ),
        "direct": _direct(),
        "dataframe": _business_dataframe(),
    }


def _long_net_benefit_surfaces(
    policy: SourceAccessPolicy,
) -> dict[str, NormalizedInputBundle]:
    """Materialize the same net-benefit decision through explicit long rows."""

    def with_binding(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
        return NormalizedInputBundle(
            manifest=bundle.manifest.model_copy(
                update={"bindings": (_long_binding(),)}
            ),
            tables=bundle.tables,
        )

    return {
        "croissant": with_binding(
            default_registry().ingest(
                _FIXTURES / "long-decision.croissant.json", policy=policy
            )
        ),
        "frictionless": with_binding(
            default_registry().ingest(
                _FIXTURES / "long-decision.datapackage.json", policy=policy
            )
        ),
        "direct": _direct_long(),
    }


def _cost_outcome_surfaces(
    policy: SourceAccessPolicy, bindings: tuple[VOIBinding, VOIBinding]
) -> dict[str, NormalizedInputBundle]:
    """Materialize the cost/outcome decision through every supported input surface."""

    def with_bindings(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
        return NormalizedInputBundle(
            manifest=bundle.manifest.model_copy(update={"bindings": bindings}),
            tables=bundle.tables,
        )

    return {
        "croissant": with_bindings(
            default_registry().ingest(
                _FIXTURES / "cost-outcome-decision.croissant.json", policy=policy
            )
        ),
        "frictionless": with_bindings(
            default_registry().ingest(
                _FIXTURES / "cost-outcome-decision.datapackage.json", policy=policy
            )
        ),
        "direct": _direct_cost_outcome(),
        "dataframe": _business_cost_outcome_dataframe(),
    }


def _repeat_for_domains(values: dict[str, float]) -> dict[str, dict[str, float]]:
    """Associate one cross-surface decision with each documented domain."""
    return {domain: dict(values) for domain in ("ml", "engineering", "business")}


def run_reference_cases() -> dict[str, object]:
    """Calculate every documented domain through every supported input surface."""
    policy = SourceAccessPolicy(_FIXTURES)
    bundles = _net_benefit_surfaces(policy)
    values = {
        surface: float(evpi(prepare_analysis_inputs(bundle).net_benefits))
        for surface, bundle in bundles.items()
    }
    if len(set(values.values())) != 1:
        raise RuntimeError("reference cases must have cross-surface EVPI parity")
    return {
        "binding": _binding().model_dump(mode="json"),
        "evpi": _repeat_for_domains(values),
        "schema": {
            surface: str(bundle.table("samples").schema)
            for surface, bundle in bundles.items()
        },
        "provenance_digests": {
            surface: bundle.manifest.provenance.descriptor_digest
            for surface, bundle in bundles.items()
        },
        "resource_digests": {
            surface: [resource.sha256 for resource in bundle.manifest.resources]
            for surface, bundle in bundles.items()
        },
    }


def run_cost_outcome_reference_cases() -> dict[str, dict[str, float]]:
    """Derive net benefit in every input surface for each documented domain."""
    policy = SourceAccessPolicy(_FIXTURES)
    bindings = _cost_outcome_bindings()
    values = {
        surface: float(
            evpi(
                prepare_analysis_inputs(
                    bundle, willingness_to_pay=20_000.0
                ).net_benefits
            )
        )
        for surface, bundle in _cost_outcome_surfaces(policy, bindings).items()
    }
    if len(set(values.values())) != 1:
        raise RuntimeError("cost/outcome cases must have cross-surface EVPI parity")
    return _repeat_for_domains(values)


def run_long_reference_cases() -> dict[str, object]:
    """Prove that declared long rows normalize to the canonical wide decision."""
    policy = SourceAccessPolicy(_FIXTURES)
    wide = _net_benefit_surfaces(policy)
    long = _long_net_benefit_surfaces(policy)
    wide_values = {
        surface: tuple(
            tuple(float(value) for value in row)
            for row in prepare_analysis_inputs(bundle).net_benefits.numpy_values
        )
        for surface, bundle in wide.items()
        if surface != "dataframe"
    }
    long_values = {
        surface: tuple(
            tuple(float(value) for value in row)
            for row in prepare_analysis_inputs(bundle).net_benefits.numpy_values
        )
        for surface, bundle in long.items()
    }
    if len(set(long_values.values())) != 1 or set(wide_values.values()) != set(
        long_values.values()
    ):
        raise RuntimeError("long and wide reference cases must normalize identically")
    values = {
        surface: float(evpi(prepare_analysis_inputs(bundle).net_benefits))
        for surface, bundle in long.items()
    }
    if len(set(values.values())) != 1:
        raise RuntimeError("long reference cases must have cross-surface EVPI parity")
    return {"normalized_net_benefit": long_values, "evpi": values}


def run_cost_outcome_ceaf_reference_cases() -> dict[str, dict[str, object]]:
    """Calculate CEAF only where the deterministic fixture supplies its inputs.

    The engineering cost/outcome rows provide paired PSA samples for two
    strategies.  Reconstructing net benefit at declared WTP thresholds creates
    the required samples-by-strategies-by-thresholds surface without inventing
    a conditional model, sample-information study, or perspective split.
    """
    thresholds = np.asarray([10_000.0, 20_000.0, 30_000.0])
    policy = SourceAccessPolicy(_FIXTURES)
    results: dict[str, dict[str, object]] = {}
    for surface, bundle in _cost_outcome_surfaces(
        policy, _cost_outcome_bindings()
    ).items():
        table = bundle.table("samples")
        costs = np.column_stack(
            [table[name].to_numpy() for name in ("cost_a", "cost_b")]
        )
        outcomes = np.column_stack(
            [table[name].to_numpy() for name in ("outcome_a", "outcome_b")]
        )
        net_benefits = (
            outcomes[:, :, None] * thresholds[None, None, :] - costs[:, :, None]
        )
        result = calculate_ceaf(
            ValueArray.from_numpy_perspectives(
                net_benefits,
                strategy_names=["A", "B"],
                perspective_names=[str(int(value)) for value in thresholds],
            ),
            thresholds,
        )
        results[surface] = {
            "optimal_strategy_names": tuple(result.optimal_strategy_names),
            "acceptability_probabilities": tuple(
                float(value) for value in result.acceptability_probabilities
            ),
            "expected_net_benefit": tuple(
                float(value) for value in result.expected_net_benefit
            ),
            "normalized_net_benefit_at_20000": tuple(
                tuple(float(value) for value in row)
                for row in prepare_analysis_inputs(
                    bundle, willingness_to_pay=20_000.0
                ).net_benefits.numpy_values
            ),
        }
    if len({repr(value) for value in results.values()}) != 1:
        raise RuntimeError("cost/outcome CEAF must have cross-surface parity")
    return results


def method_applicability_matrix() -> dict[str, dict[str, str]]:
    """State only methods supported by the current deterministic fixtures."""
    unavailable = (
        "not applicable: fixture lacks the required conditional or study-design inputs"
    )
    return {
        "ml": {
            "EVPI": "evaluated: explicit wide net-benefit samples",
            "CEAF": "not applicable: fixture lacks cost/outcome WTP thresholds",
            "EVPPI/EVSI/ENBS": unavailable,
            "wide/long/cost-outcome/perspective": (
                "wide only; no long, cost/outcome, or perspective split fixture"
            ),
        },
        "engineering": {
            "EVPI": "evaluated: explicit cost/outcome net benefits at WTP 20,000",
            "CEAF": (
                "evaluated: paired cost/outcome PSA rows at WTP 10,000/20,000/30,000"
            ),
            "EVPPI/EVSI/ENBS": unavailable,
            "wide/long/cost-outcome/perspective": (
                "cost/outcome wide rows; no long or perspective split fixture"
            ),
        },
        "business": {
            "EVPI": "evaluated: explicit wide net-benefit samples",
            "CEAF": "not applicable: fixture lacks cost/outcome WTP thresholds",
            "EVPPI/EVSI/ENBS": unavailable,
            "wide/long/cost-outcome/perspective": (
                "wide only; no long, cost/outcome, or perspective split fixture"
            ),
        },
    }


def run_cross_format_reference_cases() -> dict[str, dict[str, object]]:
    """Return reproducible P9 evidence for every community and input surface.

    The ML and business scenarios intentionally share the small explicit
    net-benefit fixture; the engineering scenario uses explicit cost/outcome
    bindings.  All forms are local, deterministic repository fixtures.  Their
    accompanying records make no claim about a live registry, external source
    rights, or inferred domain semantics.
    """
    net_benefit = cast("dict[str, Any]", run_reference_cases())
    cost_outcome = run_cost_outcome_reference_cases()

    def case(
        *,
        artifact_stem: str,
        method: str,
        surfaces: dict[str, float],
    ) -> dict[str, object]:
        artifact = _artifact(artifact_stem)
        receipts = cast("dict[str, list[str]]", net_benefit["resource_digests"])
        if artifact_stem == "cost-outcome-decision":
            policy = SourceAccessPolicy(_FIXTURES)
            receipts = {
                surface: [resource.sha256 for resource in bundle.manifest.resources]
                for surface, bundle in _cost_outcome_surfaces(
                    policy, _cost_outcome_bindings()
                ).items()
            }
        provider_receipts = {
            surface: values[0]
            for surface, values in receipts.items()
            if surface in {"croissant", "frictionless"}
        }
        resource_sha256 = artifact["normalized"]["resource_sha256"]
        if set(provider_receipts.values()) != {resource_sha256}:
            raise RuntimeError("reference-case receipts must match pinned artifacts")
        required = {
            surface: surfaces[surface]
            for surface in ("croissant", "direct", "frictionless")
        }
        if len(set(required.values())) != 1:
            raise RuntimeError("reference-case surfaces must have identical EVPI")
        return {
            "artifact": artifact,
            "expected_evpi": next(iter(required.values())),
            "governance": artifact["governance"],
            "method": method,
            "receipt_sha256": resource_sha256,
            "surfaces": surfaces,
        }

    return {
        "ml": case(
            artifact_stem="canonical-decision",
            method="EVPI on explicit model-selection net benefits",
            surfaces=cast("dict[str, float]", net_benefit["evpi"]["ml"]),
        ),
        "engineering": case(
            artifact_stem="cost-outcome-decision",
            method="EVPI on explicit cost/outcome net benefits at WTP 20,000",
            surfaces=cost_outcome["engineering"],
        ),
        "business": case(
            artifact_stem="canonical-decision",
            method="EVPI on explicit investment-strategy net benefits",
            surfaces=cast("dict[str, float]", net_benefit["evpi"]["business"]),
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_cross_format_reference_cases(), sort_keys=True))
