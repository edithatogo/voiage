"""Governance tests for stable-core scalability and resource claims."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT_PATH = Path("specs/v1/stable-core-scalability.json")
SCHEMA_PATH = Path("specs/v1/stable-core-scalability.schema.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scalability_contract_conforms_to_its_schema() -> None:
    Draft202012Validator(_load(SCHEMA_PATH)).validate(_load(CONTRACT_PATH))


def test_stable_rust_reduction_order_is_explicit_and_not_parallel_by_implication() -> (
    None
):
    reduction = _load(CONTRACT_PATH)["reduction"]

    assert reduction["stable_mode"] == "sequential-fixed-input-order"
    assert reduction["parallel_mode"] == "unsupported"
    assert reduction["promotion_gate"] == (
        "fixed-partition-indexed-partials-and-fixed-tree-reduction-with-cross-worker-fixtures"
    )


def test_evsi_rng_stream_is_indexed_splittable_but_execution_remains_sequential() -> (
    None
):
    rng = _load(CONTRACT_PATH)["rng"]

    assert rng["algorithm"] == "xorshift64-star-v1"
    assert rng["stream_identity"] == "seed,resample-index"
    assert rng["splitting_rule"] == "seed-xor-(resample-index+1)-times-mix64"
    assert rng["parallel_execution"] == "unsupported"
    assert Path(rng["implementation"]).is_file()


def test_streaming_and_out_of_core_claims_fail_closed() -> None:
    streaming = _load(CONTRACT_PATH)["streaming"]

    assert streaming["stable_rust_input"] == "materialized-sample-containers"
    assert streaming["out_of_core"] == "unsupported"
    assert (
        streaming["python_chunk_hint"]
        == "compatibility-orchestration-not-rust-streaming"
    )
    assert streaming["unsupported_error"] == "backend_unavailable"


def test_every_resource_profile_has_bounded_claims() -> None:
    contract = _load(CONTRACT_PATH)
    profiles = contract["resource_profiles"]
    expected = {
        "net-benefit",
        "expected-loss",
        "evpi",
        "evppi-regression",
        "evsi-nested-mc",
        "evsi-regression",
        "evsi-moment-matching",
        "enbs",
        "ceaf",
        "dominance",
        "structural-voi",
    }

    assert {profile["method_id"] for profile in profiles} == expected
    for profile in profiles:
        assert profile["memory_model"]
        assert profile["latency_evidence"]
        assert profile["energy_evidence"] == "not-measured-no-claim"
