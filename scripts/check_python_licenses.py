#!/usr/bin/env python3
"""Apply the hosted dependency-review licence policy to a runtime report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

DENIED = ("GPL-2.0", "GPL-3.0", "AGPL-1.0", "AGPL-3.0")


def validate(report: object) -> list[str]:
    """Return fail-closed policy findings for a pip-licenses JSON report."""
    if not isinstance(report, list) or not report:
        return ["runtime licence report must contain at least one package"]
    findings: list[str] = []
    for row in report:
        if not isinstance(row, dict):
            findings.append("runtime licence report contains a non-object row")
            continue
        name = row.get("Name")
        version = row.get("Version")
        licence = row.get("License")
        identity = f"{name or '<missing>'}=={version or '<missing>'}"
        if not isinstance(name, str) or not isinstance(version, str):
            findings.append(f"{identity}: missing package identity")
        if not isinstance(licence, str) or licence.strip().lower() in {
            "",
            "unknown",
            "none",
        }:
            findings.append(f"{identity}: missing reviewed licence")
            continue
        findings.extend(
            f"{identity}: denied licence {denied}"
            for denied in DENIED
            if re.search(rf"(?<![A-Z]){re.escape(denied)}(?![A-Z0-9.-])", licence)
        )
    return findings


def main() -> int:
    """Validate one generated runtime licence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    findings = validate(json.loads(args.report.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "policy": {"denied": list(DENIED), "missing_licence": "fail"},
                "passed": not findings,
                "findings": findings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
