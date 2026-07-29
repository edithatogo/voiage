"""Tests for the lock-aware dependency-frontier audit."""

from packaging.requirements import Requirement

from scripts.dependency_frontier import latest_compatible, locked_versions


def test_latest_compatible_respects_reviewed_upper_bounds() -> None:
    requirement = Requirement("example>=1.0,<2")

    assert latest_compatible(requirement, ["1.0", "1.9", "2.0", "2.1rc1"]) == "1.9"


def test_locked_versions_normalizes_names_and_keeps_the_greatest_resolution() -> None:
    lock = {
        "package": [
            {"name": "types_setuptools", "version": "82.0"},
            {"name": "types-setuptools", "version": "83.0"},
            {"name": "local-package"},
        ]
    }

    assert locked_versions(lock) == {"types-setuptools": "83.0"}
