# tests/conftest.py

"""
Configuration for pytest.

This file can be used to define fixtures, hooks, and plugins that are
shared across multiple test files.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from scripts.pytest_sharding import parse_shard_environment, shard_for_nodeid
from voiage.schema import ParameterSet, ValueArray


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply coarse-grained markers based on the test module name.

    Most modules in this repository are unit tests. A small number of files are
    explicitly integration or benchmark suites, and we classify those centrally
    here to avoid duplicating marker boilerplate across many edited test files.
    """
    try:
        shard = parse_shard_environment(os.environ)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if shard is not None:
        shard_index, shard_count = shard
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            target = shard_for_nodeid(item.nodeid, shard_count)
            (selected if target == shard_index else deselected).append(item)
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    for item in items:
        file_name = item.path.name
        marker_names = {mark.name for mark in item.iter_markers()}

        if "benchmark" in file_name or "performance" in file_name:
            if "benchmark" not in marker_names:
                item.add_marker(pytest.mark.benchmark)
            continue

        if "e2e" in file_name:
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
            continue

        if "integration" in file_name:
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
            continue

        if "unit" not in marker_names:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def sample_value_array() -> ValueArray:
    """Provide a ValueArray instance."""
    data = np.array(
        [[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]],
        dtype=np.float64,
    )
    return ValueArray.from_numpy(data, ["Strategy A", "Strategy B"])


@pytest.fixture(scope="session")
def sample_parameter_set() -> ParameterSet:
    """Provide a ParameterSet instance."""
    params = {
        "param1": np.array([1, 2, 3, 4, 5]),
        "param2": np.array([10, 20, 30, 40, 50]),
    }
    return ParameterSet.from_numpy_or_dict(params)
