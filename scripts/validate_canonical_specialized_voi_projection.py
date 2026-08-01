#!/usr/bin/env python3
"""Fail closed when a versioned specialized-VOI projection diverges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

IssueContracts = dict[int, tuple[str, set[str]]]

PROJECTIONS: dict[str, dict[str, object]] = {
    "specialized-voi-v1.2.0": {
        "contract_version": "v1.2.0",
        "canonical_track": "C16",
        "issues": {
            318: (
                "supported_frontier_method_completion_20260723",
                {"M16", "M17"},
            ),
            571: ("study_design_efficiency_20260727", {"M15", "M17"}),
            595: ("risk_adjusted_information_pricing_20260731", {"M16", "M17"}),
            619: ("estimation_focused_variance_voi_20260727", {"M14", "M17"}),
        },
        "extension_version": None,
        "project_fields": {
            "MoSCoW": "Must",
            "Contract Version": "v1.2.0",
            "Priority": "P1",
            "Status": "In Progress",
            "Lifecycle": "Open",
            "Gate": "Local",
            "Evidence State": "Unverified",
            "Sync State": "Clean",
        },
    },
    "specialized-voi-v1.3.0": {
        "contract_version": "v1.3.0",
        "canonical_track": "C17",
        "issues": {
            318: (
                "supported_frontier_method_completion_20260723",
                {"M17", "M21"},
            ),
            560: (
                "supported_frontier_method_completion_20260723",
                {"M17", "M21"},
            ),
        },
        "extension_version": "1.3.0",
        "project_fields": {
            "MoSCoW": "Should",
            "Contract Version": "v1.3.0",
            "Priority": "P1",
            "Status": "In Progress",
            "Lifecycle": "Open",
            "Gate": "External",
            "Evidence State": "Unverified",
            "Sync State": "Clean",
        },
    },
}
REQUIRED_SYNC_POLICY = {
    "stable_markers_required": True,
    "bounded_managed_sections_only": True,
    "preserve_human_content": True,
    "three_way_conflict_detection": True,
    "fail_closed_on_missing_credentials": True,
    "new_repositories_require_explicit_registration": True,
    "automatic_merge": False,
    "automatic_issue_closure": False,
    "automatic_release": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def validate(
    projection_path: Path,
    repository_root: Path,
    expected_projection_id: str | None = None,
) -> None:
    """Verify that a registered canonical projection is safe for this consumer."""
    projection = _load(projection_path)
    projection_id = projection.get("projection_id")
    if expected_projection_id is not None and projection_id != expected_projection_id:
        raise ValueError(
            f"projection_id {projection_id!r} does not match selected path"
        )
    projection_contract = PROJECTIONS.get(projection_id)
    if projection_contract is None:
        raise ValueError(f"projection_id {projection_id!r} is not registered")
    for field, expected in {
        "schema_version": "1.0.0",
        "projection_id": projection_id,
        "contract_version": projection_contract["contract_version"],
        "canonical_repository": "edithatogo/vop_poc_nz",
        "canonical_track": projection_contract["canonical_track"],
    }.items():
        if projection.get(field) != expected:
            raise ValueError(f"projection {field} must equal {expected!r}")

    registered_repositories = projection.get("registered_repositories")
    if not isinstance(registered_repositories, list) or not any(
        entry.get("repository") == "edithatogo/voiage"
        and entry.get("managed_projection") is True
        for entry in registered_repositories
        if isinstance(entry, dict)
    ):
        raise ValueError("VOIAGE is not an explicitly managed consumer")

    if projection.get("sync_policy") != REQUIRED_SYNC_POLICY:
        raise ValueError("projection does not preserve the fail-closed sync policy")

    project = projection.get("github_project")
    expected_project_fields = projection_contract["project_fields"]
    if (
        not isinstance(project, dict)
        or project.get("owner") != "edithatogo"
        or project.get("number") != 28
        or project.get("fields") != expected_project_fields
    ):
        raise ValueError("projection does not match the governed Project 28 fields")

    issues = projection.get("issues")
    if not isinstance(issues, list):
        raise TypeError("projection issues must be a list")
    actual: dict[int, dict[str, Any]] = {}
    for issue in issues:
        if not isinstance(issue, dict):
            raise TypeError("each projection issue must be an object")
        number = issue.get("number")
        if not isinstance(number, int) or number <= 0 or number in actual:
            raise ValueError(
                "projection issue numbers must be unique positive integers"
            )
        actual[number] = issue
    expected_issues = cast("IssueContracts", projection_contract["issues"])
    if set(actual) != set(expected_issues):
        raise ValueError(
            "projection must contain exactly the governed specialized issues"
        )

    for number, (track_id, requirement_ids) in expected_issues.items():
        issue = actual[number]
        if issue.get("repository") != "edithatogo/voiage":
            raise ValueError(f"#{number} is not owned by VOIAGE")
        if issue.get("track_id") != track_id:
            raise ValueError(f"#{number} track does not match its consumer track")
        if set(issue.get("requirement_ids", [])) != requirement_ids:
            raise ValueError(f"#{number} requirement IDs do not match {projection_id}")

    if projection_id == "specialized-voi-v1.2.0":
        voc = actual[595]
        if voc.get("subissues") != [694, 695, 696, 697]:
            raise ValueError("#595 must own the four native utility-price subissues")
        if voc.get("capability_contract") != (
            "conductor/tracks/risk_adjusted_information_pricing_20260731/contract.md"
        ):
            raise ValueError(
                "#595 capability contract does not match its delivery track"
            )
    else:
        mcda = actual[560]
        if mcda.get("parent") != 318 or mcda.get("subissues") != [
            746,
            747,
            748,
            749,
            750,
        ]:
            raise ValueError("#560 must preserve its native parent and delivery issues")
        if mcda.get("capability_contract") != (
            "specs/frontier/mcda-information/v1/capabilities.json"
        ):
            raise ValueError("#560 capability contract does not match MCDA delivery")
        if (
            mcda.get("implementation_pr") != 751
            or mcda.get("maturity") != "experimental"
            or mcda.get("stable_claim_allowed") is not False
            or mcda.get("implementation_status") != "experimental_repository_evidence"
            or mcda.get("record_id") != "mcda-voi"
            or mcda.get("remaining_gates")
            != [
                "independent-scientific-review",
                "hosted-c17-exact-head",
                "rust-r-julia-parity",
                "stable-promotion",
                "release-and-issue-closure",
            ]
        ):
            raise ValueError("#560 must remain exact experimental PR #751 evidence")

    checked_tracks: set[str] = set()
    for track_id, requirement_ids in expected_issues.values():
        if track_id in checked_tracks:
            continue
        checked_tracks.add(track_id)
        metadata = _load(
            repository_root / "conductor/tracks" / track_id / "metadata.json"
        )
        if metadata.get("canonical_track") != "C16":
            raise ValueError(f"{track_id} is not linked to C16")
        if metadata.get("planned_version") != "1.2.0":
            raise ValueError(f"{track_id} planned version is not 1.2.0")
        metadata_requirements = set(metadata.get("requirement_ids", []))
        if not requirement_ids.issubset(metadata_requirements):
            raise ValueError(f"{track_id} is missing projected requirement IDs")
        extension_version = cast("str | None", projection_contract["extension_version"])
        if extension_version is not None:
            extensions = metadata.get("planned_version_extensions")
            if not isinstance(extensions, dict) or not requirement_ids.issubset(
                set(extensions.get(extension_version, [])) | {"M17"}
            ):
                raise ValueError(
                    f"{track_id} is missing the {extension_version} extension"
                )


def main() -> int:
    """Run the C16 consumer projection validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("conductor/canonical-projections/specialized-voi-v1.3.0.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--expected-projection-id")
    args = parser.parse_args()
    validate(args.projection, args.repository_root, args.expected_projection_id)
    print("Versioned specialized-VOI consumer projection is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
