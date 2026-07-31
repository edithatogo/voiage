#!/usr/bin/env python3
"""Fail closed when the C16 projection and VOIAGE governance diverge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_ISSUES = {
    318: ("supported_frontier_method_completion_20260723", {"M16", "M17"}),
    571: ("study_design_efficiency_20260727", {"M15", "M17"}),
    595: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
    694: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
    695: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
    696: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
    697: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
    619: ("estimation_focused_variance_voi_20260727", {"M14", "M17"}),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def validate(projection_path: Path, repository_root: Path) -> None:
    """Verify that a canonical C16 projection remains safe for this consumer."""
    projection = _load(projection_path)
    for field, expected in {
        "schema_version": "1.0.0",
        "projection_id": "specialized-voi-v1.2.0",
        "contract_version": "v1.2.0",
        "canonical_repository": "edithatogo/vop_poc_nz",
        "canonical_track": "C16",
    }.items():
        if projection.get(field) != expected:
            raise ValueError(f"projection {field} must equal {expected!r}")

    registered = projection.get("registered_repositories")
    if not isinstance(registered, list) or not any(
        entry.get("repository") == "edithatogo/voiage"
        and entry.get("managed_projection") is True
        for entry in registered
        if isinstance(entry, dict)
    ):
        raise ValueError("VOIAGE is not an explicitly managed C16 consumer")

    issues = projection.get("issues")
    if not isinstance(issues, list):
        raise TypeError("projection issues must be a list")
    actual = {issue.get("number"): issue for issue in issues if isinstance(issue, dict)}
    if set(actual) != set(EXPECTED_ISSUES):
        raise ValueError("projection must contain exactly the governed specialized issues")

    for number, (track_id, requirement_ids) in EXPECTED_ISSUES.items():
        issue = actual[number]
        if issue.get("repository") != "edithatogo/voiage":
            raise ValueError(f"#{number} is not owned by VOIAGE")
        if issue.get("track_id") != track_id:
            raise ValueError(f"#{number} track does not match its consumer track")
        if set(issue.get("requirement_ids", [])) != requirement_ids:
            raise ValueError(f"#{number} requirement IDs do not match C16")

    for track_id, requirement_ids in EXPECTED_ISSUES.values():
        metadata = _load(repository_root / "conductor/tracks" / track_id / "metadata.json")
        if metadata.get("canonical_track") != "C16":
            raise ValueError(f"{track_id} is not linked to C16")
        if metadata.get("planned_version") != "1.2.0":
            raise ValueError(f"{track_id} planned version is not 1.2.0")
        if set(metadata.get("requirement_ids", [])) != requirement_ids:
            raise ValueError(f"{track_id} requirement IDs drift from the projection")


def main() -> int:
    """Run the C16 consumer projection validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("conductor/canonical-projections/specialized-voi-v1.2.0.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    validate(args.projection, args.repository_root)
    print("C16 specialized-VOI consumer projection is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
