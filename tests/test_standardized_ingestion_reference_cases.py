"""Executable evidence for the cross-domain standardized-ingestion examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.ingestion import SourceAccessPolicy, default_registry


def test_reference_cases_use_one_binding_and_one_evpi() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_reference_cases()

    assert result["binding"]["role"] == "net_benefit"
    assert result["binding"]["field_ids"] == ["strategy_a", "strategy_b"]
    assert result["evpi"] == {
        domain: {
            "croissant": pytest.approx(5.0),
            "frictionless": pytest.approx(5.0),
            "direct": pytest.approx(5.0),
            "dataframe": pytest.approx(5.0),
        }
        for domain in ("ml", "engineering", "business")
    }
    assert len(set(result["schema"].values())) == 1
    assert (
        result["resource_digests"]["croissant"]
        == result["resource_digests"]["frictionless"]
    )
    assert len(result["provenance_digests"]["dataframe"]) == 64
    assert module._business_dataframe().manifest.provenance.provider_id == (
        "dataframe-interchange"
    )


def test_cost_outcome_reference_cases_are_equivalent_across_input_surfaces() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_cost_outcome_reference_cases()

    assert result == {
        domain: {
            "croissant": pytest.approx(20.0 / 3.0),
            "frictionless": pytest.approx(20.0 / 3.0),
            "direct": pytest.approx(20.0 / 3.0),
            "dataframe": pytest.approx(20.0 / 3.0),
        }
        for domain in ("ml", "engineering", "business")
    }


def test_cost_outcome_ceaf_is_equivalent_across_input_surfaces() -> None:
    """CEAF is evidence only for the paired engineering cost/outcome fixture."""
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_cost_outcome_ceaf_reference_cases()

    assert set(result) == {"croissant", "frictionless", "direct", "dataframe"}
    assert len({repr(value) for value in result.values()}) == 1
    assert result["croissant"]["optimal_strategy_names"] == ("B", "B", "B")
    assert result["croissant"]["acceptability_probabilities"] == pytest.approx(
        (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)
    )


def test_method_applicability_matrix_does_not_overclaim_fixture_support() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    matrix = module.method_applicability_matrix()

    assert set(matrix) == {"ml", "engineering", "business"}
    assert matrix["engineering"]["CEAF"].startswith("evaluated:")
    for domain in ("ml", "business"):
        assert matrix[domain]["CEAF"].startswith("not applicable:")
    for row in matrix.values():
        assert row["EVPPI/EVSI/ENBS"].startswith("not applicable:")
        assert "no long" in row["wide/long/cost-outcome/perspective"]
        assert "perspective split" in row["wide/long/cost-outcome/perspective"]


def test_each_domain_reference_case_has_all_supported_input_surfaces() -> None:
    """Each documented community can reproduce its case through every surface."""
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = module.run_cross_format_reference_cases()

    assert set(cases) == {"ml", "engineering", "business"}
    for case in cases.values():
        for surface in ("croissant", "direct", "frictionless"):
            assert case["surfaces"][surface] == pytest.approx(case["expected_evpi"])
        assert (
            case["receipt_sha256"] == case["artifact"]["normalized"]["resource_sha256"]
        )
        assert case["governance"]["synthetic"] is True
        assert case["governance"]["third_party_data"] is False
    assert (
        module._business_cost_outcome_dataframe().manifest.provenance.provider_id
        == ("dataframe-interchange")
    )


def test_cross_format_reference_descriptors_preserve_identical_schema_order() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "standardized_ingestion"
    policy = SourceAccessPolicy(fixtures)
    registry = default_registry()

    croissant = registry.ingest(
        fixtures / "canonical-decision.croissant.json", policy=policy
    )
    frictionless = registry.ingest(
        fixtures / "canonical-decision.datapackage.json", policy=policy
    )

    assert croissant.table("samples").schema == frictionless.table("samples").schema
    assert tuple(
        field.field_id for field in croissant.manifest.tables[0].fields
    ) == tuple(field.field_id for field in frictionless.manifest.tables[0].fields)


def test_direct_dataframe_provenance_digest_changes_when_content_changes() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = module._business_dataframe()
    changed = module.from_dataframe(
        module.pa.table({"strategy_a": [11.0], "strategy_b": [20.0]}),
        dataset_id="canonical-decision-fixture",
        table_id="samples",
        bindings=(module._binding(),),
        allow_copy=False,
    )

    assert (
        original.manifest.provenance.descriptor_digest
        != changed.manifest.provenance.descriptor_digest
    )


@pytest.mark.parametrize(
    "descriptor_name",
    ["canonical-decision.croissant.json", "canonical-decision.datapackage.json"],
)
def test_reference_case_cli_walkthrough_validates_inspects_and_calculates(
    descriptor_name: str,
) -> None:
    """Published fixture walkthroughs expose stable records before calculation."""
    fixtures = Path(__file__).parent / "fixtures" / "standardized_ingestion"
    descriptor = fixtures / descriptor_name
    runner = CliRunner()
    options = ["--table", "samples", "--field", "strategy_a", "--field", "strategy_b"]

    validated = runner.invoke(app, ["ingest", "validate", str(descriptor)])
    inspected = runner.invoke(app, ["ingest", "inspect", str(descriptor)])
    calculated = runner.invoke(
        app, ["ingest", "calculate-from-dataset", str(descriptor), *options]
    )

    assert validated.exit_code == 0
    assert json.loads(validated.output)["valid"] is True
    inspection = json.loads(inspected.output)
    assert inspected.exit_code == 0
    assert inspection["binding_resolution"] is None
    assert inspection["provider"] in {"croissant", "frictionless"}
    assert calculated.exit_code == 0
    assert json.loads(calculated.output)["evpi"] == pytest.approx(5.0)
