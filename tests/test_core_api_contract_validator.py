from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_core_api_contract as validator
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

CORE_API_CONTRACT_PAIRS: tuple[tuple[Path, Path], ...] = (
    (
        Path("specs/core-api/schemas/v1/decision-problem.schema.json"),
        Path("specs/core-api/examples/v1/decision-problem.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/intervention.schema.json"),
        Path("specs/core-api/examples/v1/intervention.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/trial-design.schema.json"),
        Path("specs/core-api/examples/v1/trial-design.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/parameter-set.schema.json"),
        Path("specs/core-api/examples/v1/parameter-set.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/value-array.schema.json"),
        Path("specs/core-api/examples/v1/value-array.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/diagnostics.schema.json"),
        Path("specs/core-api/examples/v1/diagnostics.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/method-metadata.schema.json"),
        Path("specs/core-api/examples/v1/method-metadata.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evpi.schema.json"),
        Path("specs/core-api/examples/v1/evpi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evppi.schema.json"),
        Path("specs/core-api/examples/v1/evppi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/evsi.schema.json"),
        Path("specs/core-api/examples/v1/evsi.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/enbs.schema.json"),
        Path("specs/core-api/examples/v1/enbs.example.json"),
    ),
    (
        Path("specs/core-api/schemas/v1/results/ceac.schema.json"),
        Path("specs/core-api/examples/v1/ceac.example.json"),
    ),
)


def test_validate_fixture_manifest_accepts_minimal_v1_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    input_artifact = fixture_root / "normative" / "inputs" / "input.json"
    input_artifact.parent.mkdir(parents=True)
    input_artifact.write_text("{}", encoding="utf-8")

    artifact = fixture_root / "normative" / "result.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")

    manifest = {
        "version": "v1",
        "normative": [
            {
                "name": "basic normative case",
                "method_family": "evpi",
                "input_artifact": "normative/inputs/input.json",
                "expected_output_artifact": "normative/result.json",
                "tolerance_policy": "exact",
                "provenance": {"seed": 101, "execution_mode": "deterministic"},
            }
        ],
        "illustrative": [],
    }
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(validator, "FIXTURE_MANIFEST", manifest_path)

    validator._validate_fixture_manifest(manifest_path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            {"version": "v2", "normative": [], "illustrative": []},
            "manifest version",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "missing provenance",
                        "method_family": "evpi",
                        "input_artifact": "normative/result.json",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                    }
                ],
                "illustrative": [],
            },
            "provenance",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "missing artifact",
                        "method_family": "evpi",
                        "input_artifact": "normative/result.json",
                        "expected_output_artifact": "normative/missing.json",
                        "tolerance_policy": "exact",
                        "provenance": {"seed": 101, "execution_mode": "deterministic"},
                    }
                ],
                "illustrative": [],
            },
            "missing artifact",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "bad provenance seed",
                        "method_family": "evpi",
                        "input_artifact": "normative/result.json",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                        "provenance": {
                            "seed": "101",
                            "execution_mode": "deterministic",
                        },
                    }
                ],
                "illustrative": [],
            },
            "seed",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "bad provenance mode",
                        "method_family": "evpi",
                        "input_artifact": "normative/result.json",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                        "provenance": {"seed": 101, "execution_mode": "random"},
                    }
                ],
                "illustrative": [],
            },
            "execution_mode",
        ),
        (
            {
                "version": "v1",
                "normative": [
                    {
                        "name": "missing input artifact",
                        "method_family": "evpi",
                        "expected_output_artifact": "normative/result.json",
                        "tolerance_policy": "exact",
                        "provenance": {"seed": 101, "execution_mode": "deterministic"},
                    }
                ],
                "illustrative": [],
            },
            "input_artifact",
        ),
    ],
)
def test_validate_fixture_manifest_rejects_invalid_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    message: str,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(validator, "FIXTURE_MANIFEST", manifest_path)

    with pytest.raises(validator.ValidationError, match=message):
        validator._validate_fixture_manifest(manifest_path)


def test_iter_fixture_cases_exposes_normative_input_output_pairs() -> None:
    cases = validator.iter_fixture_cases()
    assert [case.method_family for case in cases] == [
        "evpi",
        "evppi",
        "evsi",
        "enbs",
        "ceac",
    ]
    assert all(case.tolerance_policy == "exact" for case in cases)

    for case in cases:
        input_payload = validator.load_fixture_payload(
            validator.resolve_fixture_artifact(case.input_artifact)
        )
        output_payload = validator.load_fixture_payload(
            validator.resolve_fixture_artifact(case.expected_output_artifact)
        )
        assert input_payload["decision_problem"]["decision_problem_id"] == output_payload[
            "decision_problem_id"
        ]
        assert input_payload["decision_problem"]["analysis_type"] == "net-benefit-first"
        assert input_payload["parameter_set"]["parameter_set_id"] == "screening-psa-001"
        assert input_payload["trial_design"]["trial_design_id"] == "screening-trial-design-001"
        assert output_payload["analysis_type"] == case.method_family


def test_core_api_fixture_bundle_drives_population_scaled_analysis_paths() -> None:
    bundle = validator.load_fixture_payload(
        Path("specs/core-api/fixtures/v1/normative/inputs/screening-program-001.json")
    )
    nb_values = np.array(
        [
            [1500.0, 1625.0],
            [1500.0, 1625.0],
            [1500.0, 1625.0],
        ],
        dtype=float,
    )
    parameter_samples = {
        name: np.asarray(values, dtype=float)
        for name, values in bundle["parameter_set"]["parameters"].items()
    }
    analysis = DecisionAnalysis(nb_values, parameter_samples=parameter_samples)

    assert (
        analysis.calculate_evpi(
            population=100.0,
            time_horizon=2.0,
            discount_rate=0.03,
        )
        >= 0.0
    )
    assert analysis.evpi(chunk_size=2) >= 0.0
    assert (
        analysis.evppi(
            population=100.0,
            time_horizon=2.0,
            discount_rate=0.03,
        )
        >= 0.0
    )
    assert (
        analysis.evppi(
            parameters_of_interest=["incremental_cost"],
            n_regression_samples=2,
            chunk_size=2,
            population=100.0,
            time_horizon=2.0,
            discount_rate=0.03,
        )
        >= 0.0
    )
    assert (
        analysis.enbs(
            research_cost=1000.0,
            strategy_of_interest="Targeted screening",
            population=100.0,
            time_horizon=2.0,
            discount_rate=0.03,
        )
        >= 0.0
    )


def test_core_api_fixture_bundle_covers_validation_branches() -> None:
    bundle = validator.load_fixture_payload(
        Path("specs/core-api/fixtures/v1/normative/inputs/screening-program-001.json")
    )
    nb_values = np.array(
        [
            [1500.0, 1625.0],
            [1500.0, 1625.0],
            [1500.0, 1625.0],
        ],
        dtype=float,
    )
    parameter_samples = {
        name: np.asarray(values, dtype=float)
        for name, values in bundle["parameter_set"]["parameters"].items()
    }
    analysis = DecisionAnalysis(nb_values, parameter_samples=parameter_samples)

    with pytest.raises(ValueError, match="Population"):
        analysis.evpi(population="bad", time_horizon=2.0)
    with pytest.raises(ValueError, match="parameters_of_interest"):
        analysis.evppi(parameters_of_interest="incremental_cost")
    with pytest.raises(ValueError, match="ParameterSet"):
        analysis.evppi(parameters_of_interest=["missing"])
    with pytest.raises(ValueError, match="n_regression_samples"):
        analysis.evppi(
            parameters_of_interest=["incremental_cost"],
            n_regression_samples=4,
        )
    with pytest.raises(ValueError, match="Population"):
        analysis.enbs(
            research_cost=1000.0,
            population="bad",
            time_horizon=2.0,
        )


def test_core_api_fixture_bundle_covers_streaming_update_paths() -> None:
    bundle = validator.load_fixture_payload(
        Path("specs/core-api/fixtures/v1/normative/inputs/screening-program-001.json")
    )
    nb_values = np.array(
        [
            [1500.0, 1625.0],
            [1500.0, 1625.0],
        ],
        dtype=float,
    )
    parameter_samples = {
        name: np.asarray(values[:2], dtype=float)
        for name, values in bundle["parameter_set"]["parameters"].items()
    }

    streaming_analysis = DecisionAnalysis(
        nb_values,
        parameter_samples=parameter_samples,
        streaming_window_size=2,
    )
    streaming_analysis.update_with_new_data(
        ValueArray.from_numpy(
            np.array(
                [
                    [1500.0, 1625.0],
                ],
                dtype=float,
            )
        ),
        new_parameter_samples=parameter_samples,
    )
    assert next(streaming_analysis.streaming_evpi()) >= 0.0
    assert next(streaming_analysis.streaming_evppi()) >= 0.0

    append_analysis = DecisionAnalysis(nb_values)
    append_analysis.update_with_new_data(np.array([[1500.0, 1625.0]], dtype=float))
    assert append_analysis.nb_array.n_samples == 3

    append_with_params = DecisionAnalysis(
        nb_values,
        parameter_samples=parameter_samples,
    )
    append_with_params.update_with_new_data(
        np.array([[1500.0, 1625.0]], dtype=float),
        new_parameter_samples=parameter_samples,
    )
    assert append_with_params.nb_array.n_samples == 3
    assert append_with_params.parameter_samples is not None


def test_resolve_fixture_artifact_rejects_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures" / "v1"
    fixture_root.mkdir(parents=True)

    monkeypatch.setattr(validator, "FIXTURE_ROOT", fixture_root)

    with pytest.raises(validator.ValidationError, match="escapes"):
        validator._resolve_fixture_artifact("../escape.json")


@pytest.mark.parametrize(("schema_relpath", "example_relpath"), CORE_API_CONTRACT_PAIRS)
def test_schema_example_pair_conforms_to_contract(
    schema_relpath: Path,
    example_relpath: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    schema_path = root / schema_relpath
    example_path = root / example_relpath

    schema = validator._load_json(schema_path)
    example = validator._load_json(example_path)

    validator._validate(example, schema, "$", schema_path)
