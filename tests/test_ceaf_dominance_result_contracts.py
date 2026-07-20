from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from scripts import validate_core_api_contract as contract_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PAIRS = (
    (
        REPO_ROOT / "specs/core-api/schemas/v1/results/ceaf.schema.json",
        REPO_ROOT / "specs/core-api/examples/v1/ceaf.example.json",
    ),
    (
        REPO_ROOT / "specs/core-api/schemas/v1/results/dominance.schema.json",
        REPO_ROOT / "specs/core-api/examples/v1/dominance.example.json",
    ),
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("schema_path", "example_path"), CONTRACT_PAIRS)
def test_result_example_conforms_to_draft_2020_12_schema(
    schema_path: Path,
    example_path: Path,
) -> None:
    schema = _load_json(schema_path)
    example = _load_json(example_path)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(example)
    contract_validator._validate(example, schema, "$", schema_path)


def test_ceaf_contract_requires_probability_and_frontier_outputs() -> None:
    schema = _load_json(CONTRACT_PAIRS[0][0])

    assert {
        "wtp_thresholds",
        "optimal_strategy_indices",
        "optimal_strategy_names",
        "acceptability_probabilities",
        "probability_lower",
        "probability_upper",
        "expected_net_benefit",
    } <= set(schema["required"])


def test_dominance_contract_freezes_status_vocabulary_and_icer_outputs() -> None:
    schema = _load_json(CONTRACT_PAIRS[1][0])
    properties = schema["properties"]

    assert set(properties["status"]["items"]["enum"]) == {
        "frontier",
        "strongly_dominated",
        "extended_dominated",
    }
    assert {
        "frontier_indices",
        "strongly_dominated_indices",
        "extended_dominated_indices",
        "incremental_costs",
        "incremental_effects",
        "icers",
    } <= set(schema["required"])


def test_contract_index_lists_both_result_pairs() -> None:
    index = (REPO_ROOT / "specs/core-api/contract-index.md").read_text(encoding="utf-8")

    assert (
        "`schemas/v1/results/ceaf.schema.json` -> `examples/v1/ceaf.example.json`"
    ) in index
    assert (
        "`schemas/v1/results/dominance.schema.json` -> "
        "`examples/v1/dominance.example.json`"
    ) in index
