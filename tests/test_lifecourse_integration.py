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
    assert manifest["compatibility"] == {
        "voiage_version": "0.2.0",
        "lifecourse_profile_version": "v1",
        "heoml_profile_version": "0.1",
    }
    assert manifest["heoml_profile_version"] == "0.1"
    assert manifest["heoml_manifest_id"] == "heoml-run-bundle-v1-example"
    normative = cast("list[dict[str, Any]]", manifest["normative"])
    assert len(normative) == 1
    illustrative = cast("list[dict[str, Any]]", manifest["illustrative"])
    assert len(illustrative) == 1

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

    illustrative_entry = illustrative[0]
    assert illustrative_entry["name"] == "result envelope metadata contract"
    assert illustrative_entry["method_family"] == "result-envelope"
    assert (
        illustrative_entry["expected_output_artifact"]
        == "illustrative/voi_result_envelope.json"
    )
    assert illustrative_entry["tolerance_policy"] == "structure-only"
    assert illustrative_entry["provenance"] == {
        "seed": 101,
        "execution_mode": "deterministic",
    }
    assert (FIXTURE_ROOT / illustrative_entry["expected_output_artifact"]).is_file()


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
    assert bundle["method_settings"]["evsi"]["method"] == "two-loop-monte-carlo"
    assert bundle["method_settings"]["enbs"]["method"] == "decision-rule"
    assert (
        expected_evpi["willingness_to_pay"]
        == bundle["willingness_to_pay_thresholds"][0]
    )
    assert expected_evppi["evppi"] <= expected_evpi["evpi"]


def test_lifecourse_fixture_evsi_payload_is_structured() -> None:
    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    artifacts = cast("dict[str, str]", bundle["artifacts"])
    expected_evsi = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_evsi"])
    )

    assert expected_evsi["analysis_type"] == "evsi"
    assert expected_evsi["decision_problem_id"] == bundle["decision_problem_id"]
    assert expected_evsi["trial_design"] == {
        "sample_size": 250,
        "design": "illustrative",
    }
    assert expected_evsi["value"] == 2.25
    assert expected_evsi["method"] == bundle["method_settings"]["evsi"]["method"]


def test_lifecourse_fixture_enbs_payload_is_structured() -> None:
    bundle = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / "screening_program_bundle.json")
    )
    artifacts = cast("dict[str, str]", bundle["artifacts"])
    expected_enbs = cast(
        "dict[str, Any]", _load_json(NORMATIVE_ROOT / artifacts["expected_enbs"])
    )

    assert expected_enbs["analysis_type"] == "enbs"
    assert expected_enbs["decision_problem_id"] == bundle["decision_problem_id"]
    assert expected_enbs["research_cost"] == 25000.0
    assert expected_enbs["net_value"] == -24997.75
    assert expected_enbs["method"] == bundle["method_settings"]["enbs"]["method"]


def test_lifecourse_dependency_policy_and_result_envelope_contract_are_documented() -> (
    None
):
    docs_text = Path(
        "docs/astro-site/src/content/docs/integrations/lifecourse.mdx"
    ).read_text(encoding="utf-8")
    profile_text = Path("specs/integrations/lifecourse/v1/README.md").read_text(
        encoding="utf-8"
    )
    examples_text = Path(
        "specs/integrations/lifecourse/v1/examples/README.md"
    ).read_text(encoding="utf-8")
    schemas_text = Path("specs/integrations/lifecourse/v1/schemas/README.md").read_text(
        encoding="utf-8"
    )
    fixtures_text = Path(
        "specs/integrations/lifecourse/v1/fixtures/README.md"
    ).read_text(encoding="utf-8")

    assert "`voiage` should not depend on `lifecourse`." in docs_text
    assert "optional extra such as `lifecourse[voiage]`" in docs_text
    assert "No new mandatory extra is required" in docs_text
    assert "Compatibility Versioning" in docs_text
    assert "Artifact Exchange And Fixture Validation" in docs_text
    assert "0.2.0" in docs_text
    assert "HEOML profile `0.1`" in docs_text
    assert "Result Envelope Contract" in docs_text
    assert "supported result families" in docs_text
    assert "package-version information" in docs_text
    assert "illustrative envelope fixture" in docs_text

    assert "dependency posture explicit" in profile_text
    assert "schemas/voi-result-envelope.schema.json" in profile_text
    assert "fixtures/illustrative/voi_result_envelope.json" in profile_text
    assert "Compatibility Versioning" in profile_text
    assert "Validation Path" in profile_text
    assert "illustrative result-envelope contract" in profile_text

    assert "illustrative result envelopes" in examples_text
    assert "fixtures/illustrative/voi_result_envelope.json" in examples_text
    assert "compatibility anchors" in examples_text
    assert "0.2.0" in examples_text
    assert "voi-result-envelope.schema.json" in schemas_text
    assert "illustrative result-envelope fixture" in fixtures_text
    assert "compatibility" in fixtures_text
    assert "0.2.0" in fixtures_text


def test_lifecourse_result_envelope_fixture_is_structured() -> None:
    envelope = cast(
        "dict[str, Any]",
        _load_json(FIXTURE_ROOT / "illustrative" / "voi_result_envelope.json"),
    )
    schema = cast(
        "dict[str, Any]",
        _load_json(
            Path(
                "specs/integrations/lifecourse/v1/schemas/voi-result-envelope.schema.json"
            )
        ),
    )

    assert envelope["schema_version"] == "v1"
    assert envelope["supported_result_families"] == [
        "evpi",
        "evppi",
        "evsi",
        "enbs",
    ]
    assert envelope["analysis_id"] == "lifecourse-screening-001"
    assert envelope["decision_problem_id"] == "screening-program-001"
    assert set(envelope["method_settings"]) == {"evpi", "evppi", "evsi", "enbs"}
    assert set(envelope["result_payloads"]) == {"evpi", "evppi", "evsi", "enbs"}
    assert envelope["compatibility"] == {
        "voiage_version": "0.2.0",
        "lifecourse_profile_version": "v1",
        "heoml_profile_version": "0.1",
    }
    assert envelope["package_version"]["voiage"] == "0.2.0"
    assert envelope["provenance"]["heoml_version"] == "0.1"
    assert envelope["provenance"]["heoml_manifest_id"] == "heoml-run-bundle-v1-example"
    assert envelope["result_payloads"]["evpi"]["value"] == expected_evpi_value(envelope)
    assert schema["required"] == [
        "schema_version",
        "analysis_id",
        "decision_problem_id",
        "supported_result_families",
        "compatibility",
        "method_settings",
        "scaling",
        "diagnostics",
        "package_version",
        "provenance",
        "result_payloads",
    ]


def expected_evpi_value(envelope: dict[str, Any]) -> float:
    return (
        envelope["result_payloads"]["evpi"]["expected_perfect_information"]
        - envelope["result_payloads"]["evpi"]["expected_current_value"]
    )
