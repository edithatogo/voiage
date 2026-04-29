from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, cast

from voiage import HeomlRunBundle, load_heoml_run_bundle

FIXTURE_ROOT = Path("specs/integrations/lifecourse/v1/fixtures")
NORMATIVE_ROOT = FIXTURE_ROOT / "normative"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_lifecourse_fixture_manifest_and_bundle_are_consistent() -> None:
    manifest = cast("dict[str, Any]", _load_json(FIXTURE_ROOT / "manifest.json"))
    assert manifest["version"] == "v1"
    assert manifest["status"] == "scaffold"
    assert manifest["heoml_profile_version"] == "0.1"
    assert manifest["heoml_manifest_id"] == "heoml-run-bundle-v1-example"
    normative = cast("list[dict[str, Any]]", manifest["normative"])
    assert len(normative) == 1

    entry = normative[0]
    assert entry["name"] == "screening program deterministic bundle"
    assert entry["method_family"] == "evpi"
    assert (
        entry["expected_output_artifact"] == "normative/screening_program_bundle.json"
    )
    assert entry["tolerance_policy"] == "exact"
    assert entry["provenance"] == {"seed": 101, "execution_mode": "deterministic"}

    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    assert bundle["profile_version"] == "v1"
    assert bundle["heoml_profile_version"] == "0.1"
    assert bundle["heoml_manifest_id"] == "heoml-run-bundle-v1-example"
    assert bundle["producer"] == "lifecourse"
    assert bundle["schema_profile"] == "heoml"
    assert bundle["strategy_names"] == ["Usual care", "Targeted screening"]
    assert bundle["willingness_to_pay_thresholds"] == [50000]
    assert bundle["population"] == 10000
    assert bundle["time_horizon_years"] == 10
    assert bundle["discount_rate"] == 0.035

    artifacts = cast("dict[str, str]", bundle["artifacts"])
    for relpath in artifacts.values():
        assert (NORMATIVE_ROOT / relpath).is_file()


def test_lifecourse_heoml_loader_builds_core_contract_objects() -> None:
    loaded = load_heoml_run_bundle(
        str(NORMATIVE_ROOT / "screening_program_bundle.json")
    )

    manifest = cast("dict[str, Any]", loaded.manifest)
    assert isinstance(loaded, HeomlRunBundle)
    assert manifest["heoml_version"] == "0.1"
    assert manifest["profile"] == "heoml.run_bundle"

    value_array = loaded.value_array
    parameter_set = loaded.parameter_set
    provenance = cast("dict[str, Any]", loaded.provenance)

    assert value_array is not None
    assert parameter_set is not None
    assert value_array.strategy_names == ["Usual care", "Targeted screening"]
    assert parameter_set.parameter_names == [
        "prob_screen_positive",
        "incremental_cost",
        "incremental_effect",
    ]
    assert provenance["heoml_version"] == "0.1"
    assert provenance["profile"] == "heoml.run_bundle"
    assert provenance["manifest_id"] == "heoml-run-bundle-v1-example"


def test_lifecourse_heoml_loader_is_publicly_exported() -> None:
    from voiage import load_heoml_run_bundle as exported_loader

    assert exported_loader is load_heoml_run_bundle
    exported_bundle = HeomlRunBundle
    assert exported_bundle is HeomlRunBundle


def test_lifecourse_fixture_evpi_payload_matches_net_benefits() -> None:
    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    artifacts = cast("dict[str, str]", bundle["artifacts"])
    net_benefits = _load_csv(NORMATIVE_ROOT / artifacts["net_benefits"])
    expected_evpi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evpi"])
    )

    strategy_names = cast("list[str]", bundle["strategy_names"])
    rows_by_strategy = {
        strategy: [float(row[strategy]) for row in net_benefits]
        for strategy in strategy_names
    }
    expected_current_value = max(mean(values) for values in rows_by_strategy.values())
    expected_perfect_information = mean(
        max(float(row[strategy]) for strategy in strategy_names) for row in net_benefits
    )
    expected_value = expected_perfect_information - expected_current_value

    assert expected_evpi["analysis_type"] == "evpi"
    assert expected_evpi["decision_problem_id"] == bundle["decision_problem_id"]
    assert expected_evpi["strategy_names"] == strategy_names
    assert expected_evpi["expected_current_value"] == expected_current_value
    assert expected_evpi["expected_perfect_information"] == expected_perfect_information
    assert expected_evpi["evpi"] == expected_value
    assert expected_evpi["method"] == bundle["method_settings"]["evpi"]["method"]


def test_lifecourse_fixture_evppi_payload_is_structured() -> None:
    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    artifacts = cast("dict[str, str]", bundle["artifacts"])
    expected_evppi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evppi"])
    )
    expected_evpi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evpi"])
    )

    assert expected_evppi["analysis_type"] == "evppi"
    assert expected_evppi["decision_problem_id"] == bundle["decision_problem_id"]
    assert expected_evppi["parameter_names"] == [
        "prob_screen_positive",
        "incremental_cost",
    ]
    assert expected_evppi["method"] == "gam"
    assert (
        expected_evppi["expected_current_value"]
        == expected_evpi["expected_current_value"]
    )
    assert (
        expected_evppi["expected_perfect_information"]
        == expected_evpi["expected_perfect_information"]
    )
    assert 0 <= expected_evppi["evppi"] <= expected_evpi["evpi"]
    assert "diagnostics" in expected_evppi
    assert expected_evppi["method"] == bundle["method_settings"]["evppi"]["method"]


def test_lifecourse_fixture_conformance_expectations_are_deterministic() -> None:
    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    artifacts = cast("dict[str, str]", bundle["artifacts"])
    expected_evpi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evpi"])
    )
    expected_evppi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evppi"])
    )

    assert expected_evpi["analysis_id"] == "lifecourse-screening-001-evpi"
    assert expected_evppi["analysis_id"] == "lifecourse-screening-001-evppi"
    assert expected_evpi["decision_problem_id"] == expected_evppi["decision_problem_id"]
    assert expected_evpi["method"] == "nested-monte-carlo"
    assert expected_evppi["method"] == "gam"
    assert (
        expected_evpi["willingness_to_pay"]
        == bundle["willingness_to_pay_thresholds"][0]
    )
    assert expected_evppi["evppi"] <= expected_evpi["evpi"]
