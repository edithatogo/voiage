"""Contracts for deterministic cross-runner pytest partitions."""

from __future__ import annotations

import pytest

from scripts.pytest_sharding import parse_shard_environment, shard_for_nodeid


def test_shard_assignment_is_stable_complete_and_order_independent() -> None:
    nodeids = [f"tests/test_module_{index}.py::test_case" for index in range(1000)]
    forward = {nodeid: shard_for_nodeid(nodeid, 4) for nodeid in nodeids}
    reverse = {nodeid: shard_for_nodeid(nodeid, 4) for nodeid in reversed(nodeids)}

    assert forward == reverse
    assert set(forward.values()) == {0, 1, 2, 3}
    counts = [sum(value == shard for value in forward.values()) for shard in range(4)]
    assert max(counts) - min(counts) < 100


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, None),
        (
            {"VOIAGE_TEST_SHARD_INDEX": "1", "VOIAGE_TEST_SHARD_COUNT": "3"},
            (1, 3),
        ),
    ],
)
def test_shard_environment_parses_complete_values(
    environment: dict[str, str], expected: tuple[int, int] | None
) -> None:
    assert parse_shard_environment(environment) == expected


@pytest.mark.parametrize(
    "environment",
    [
        {"VOIAGE_TEST_SHARD_INDEX": "0"},
        {"VOIAGE_TEST_SHARD_COUNT": "2"},
        {"VOIAGE_TEST_SHARD_INDEX": "x", "VOIAGE_TEST_SHARD_COUNT": "2"},
        {"VOIAGE_TEST_SHARD_INDEX": "0", "VOIAGE_TEST_SHARD_COUNT": "0"},
        {"VOIAGE_TEST_SHARD_INDEX": "2", "VOIAGE_TEST_SHARD_COUNT": "2"},
    ],
)
def test_shard_environment_fails_closed(environment: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        parse_shard_environment(environment)
