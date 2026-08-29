"""Contracts for the lock-derived Python runtime licence gate."""

from __future__ import annotations

from scripts.check_python_licenses import DENIED, validate


def test_reviewed_permissive_runtime_licences_pass() -> None:
    report = [
        {"Name": "numpy", "Version": "2.5.2", "License": "BSD-3-Clause"},
        {"Name": "pyarrow", "Version": "25.0.1", "License": "Apache-2.0"},
    ]

    assert validate(report) == []


def test_missing_or_denied_licence_fails_closed() -> None:
    report = [
        {"Name": "missing", "Version": "1.0", "License": "UNKNOWN"},
        {"Name": "copyleft", "Version": "2.0", "License": "GPL-3.0"},
    ]

    findings = validate(report)
    assert any("missing reviewed licence" in finding for finding in findings)
    assert any("denied licence GPL-3.0" in finding for finding in findings)
    assert DENIED == ("GPL-2.0", "GPL-3.0", "AGPL-1.0", "AGPL-3.0")
