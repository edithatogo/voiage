#!/usr/bin/env python3
"""Validate the local, unposted pyOpenSci submission staging packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

STAGING_PATH = Path("specs/submission-readiness/pyopensci-submission-staging.json")
EXPECTED_CANDIDATE_VERSION = "2.2.0"
EXPECTED_EXTERNAL_OUTCOMES = {
    "pyopensci_review": "not_started",
    "pyopensci_acceptance": "pending_external",
    "joss_referral": "not_started",
    "joss_acceptance": "pending_external",
}
REQUIRED_HUMAN_ATTESTATIONS = {
    "code_of_conduct",
    "maintenance_commitment_form_checkbox",
    "submitted_version",
    "joss_partnership_option",
    "reviewer_direct_issue_permission",
    "author_guide_read",
    "pre_review_survey",
}
EXPECTED_HUMAN_ATTESTATIONS = {
    "code_of_conduct": "pending",
    "maintenance_commitment_form_checkbox": "pending",
    "submitted_version": "confirmed",
    "joss_partnership_option": "pending",
    "reviewer_direct_issue_permission": "pending",
    "author_guide_read": "pending",
    "pre_review_survey": "pending",
}
REQUIRED_EXTERNAL_ACTIONS = {
    "pre_review_survey_completed",
    "pyopensci_issue_created",
    "pyopensci_contact_made",
    "joss_submission_created",
    "badge_added",
    "doi_archive_created",
}
PENDING_DRAFT_CHECKBOX_MARKERS = (
    "I agree to abide by",
    "I have read and will commit",
    "Do you wish to automatically submit",
    "Maintainer confirmation pending. If confirmed",
    "I have read the pyOpenSci author guide",
    "Last but not least please fill out our pre-review survey",
)
CONFIRMED_SUBMITTED_VERSION_LINE = (
    "Version submitted: 2.2.0 (confirmed by maintainer; submission not performed)"
)
PLACEHOLDER = re.compile(r"\b(?:TBD|TODO|FILL(?:\s+THIS)?\s+IN)\b", re.IGNORECASE)


def _load_json(path: Path, findings: list[str]) -> dict[str, Any] | None:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(f"cannot load {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        findings.append(f"{path} must contain a JSON object")
        return None
    return payload


def _safe_file(root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        return None
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bound_file(
    root: Path,
    binding: object,
    label: str,
    findings: list[str],
    *,
    require_all_files: bool,
) -> tuple[Path | None, dict[str, Any] | None]:
    if not isinstance(binding, dict):
        findings.append(f"{label} binding must be an object")
        return None, None
    path = _safe_file(root, binding.get("path"))
    if path is None:
        if require_all_files:
            findings.append(f"{label} path must identify a repository file")
        return None, None
    expected = binding.get("sha256")
    if expected != _sha256(path):
        findings.append(f"{label} SHA-256 does not match the staged artifact")
    if path.suffix == ".json":
        return path, _load_json(path, findings)
    return path, None


def _validate_published_candidate(
    root: Path,
    candidate: dict[str, Any],
    recommended: dict[str, Any],
    findings: list[str],
) -> None:
    """Bind a published projection to the separately reviewed local receipt."""
    _, receipt = _validate_bound_file(
        root,
        recommended.get("publication_receipt"),
        "publication receipt",
        findings,
        require_all_files=True,
    )
    if receipt is None:
        findings.append("publication receipt must contain JSON evidence")
        return
    release = receipt.get("release")
    github = receipt.get("github")
    workflows = receipt.get("workflows")
    if not all(isinstance(item, dict) for item in (release, github, workflows)):
        findings.append(
            "publication receipt lacks release, GitHub or workflow evidence"
        )
        return
    assert isinstance(release, dict)
    assert isinstance(github, dict)
    assert isinstance(workflows, dict)
    publication = workflows.get("publication")
    version = EXPECTED_CANDIDATE_VERSION
    if (
        release.get("version") != version
        or release.get("tag") != f"v{version}"
        or recommended.get("tag") != release.get("tag")
        or any(
            not isinstance(release.get(key), str)
            or re.fullmatch(r"[0-9a-f]{40}", release[key]) is None
            or recommended.get(key) != release[key]
            for key in ("commit", "tree", "tag_object")
        )
        or release.get("tag_signature_verified") is not True
        or recommended.get("tag_signature_verified") is not True
        or github.get("draft") is not False
        or github.get("prerelease") is not False
        or github.get("immutable") is not True
        or recommended.get("immutable_github_release") is not True
        or not isinstance(github.get("published_at"), str)
        or not github["published_at"]
        or recommended.get("published_at") != github["published_at"]
        or github.get("url")
        != f"https://github.com/edithatogo/voiage/releases/tag/v{version}"
        or recommended.get("github_release") != github.get("url")
        or recommended.get("pypi_release")
        != f"https://pypi.org/project/voiage/{version}/"
        or recommended.get("latest_on_pypi_when_observed") is not True
        or not isinstance(publication, dict)
        or publication.get("conclusion") != "success"
        or publication.get("head") != release.get("commit")
    ):
        findings.append(
            "published candidate identity does not match successful immutable release evidence"
        )
    expected_names = {
        f"voiage-{version}-cp312-abi3-macosx_11_0_arm64.whl",
        f"voiage-{version}-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        f"voiage-{version}-cp312-abi3-win_amd64.whl",
        f"voiage-{version}.tar.gz",
    }
    digests = receipt.get("reviewed_digests")
    if (
        not isinstance(digests, dict)
        or set(digests) != expected_names
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in digests.values()
        )
        or candidate.get("artifact_sha256") != digests
    ):
        findings.append(
            "published candidate must match all four reviewed artifact digests"
        )
        return
    for name, host in (("pypi", "pypi.org"), ("testpypi", "test.pypi.org")):
        registry = receipt.get(name)
        artifacts = registry.get("artifacts") if isinstance(registry, dict) else None
        if (
            not isinstance(registry, dict)
            or registry.get("url") != f"https://{host}/project/voiage/{version}/"
            or (name == "pypi" and registry.get("latest_version") != version)
            or not isinstance(artifacts, list)
            or len(artifacts) != 4
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("filename"), str)
                or item.get("yanked") is not False
                for item in artifacts
            )
            or {item.get("filename"): item.get("sha256") for item in artifacts}
            != digests
        ):
            findings.append(
                f"published candidate lacks exact non-yanked {name} artifact evidence"
            )


def validate_staging_packet(
    root: Path | str,
    *,
    require_all_files: bool = True,
) -> list[str]:
    """Return fail-closed findings for the repository staging packet."""
    resolved_root = Path(root).resolve()
    findings: list[str] = []
    staging = _load_json(resolved_root / STAGING_PATH, findings)
    if staging is None:
        return findings

    if staging.get("schema_version") != "voiage.pyopensci-submission-staging.v1":
        findings.append("unsupported pyOpenSci staging schema")
    if staging.get("state") != "prepared_local_unposted":
        findings.append("staging state must remain prepared_local_unposted")
    if staging.get("candidate_version") != EXPECTED_CANDIDATE_VERSION:
        findings.append("candidate_version must match the selected v2.2.0 candidate")
    if staging.get("candidate_confirmation") != "confirmed_maintainer":
        findings.append("candidate confirmation must record the maintainer selection")
    if staging.get("ordinary_fields") != "complete_for_local_review":
        findings.append("ordinary draft fields must be complete for local review")

    attestations = staging.get("human_attestations")
    if not isinstance(attestations, dict) or not attestations:
        findings.append("human attestations must be a non-empty object")
    else:
        if set(attestations) != REQUIRED_HUMAN_ATTESTATIONS:
            findings.append("human attestation key set is incomplete or unexpected")
        if attestations != EXPECTED_HUMAN_ATTESTATIONS:
            findings.append(
                "only the submitted version may be confirmed in the local packet"
            )

    external_actions = staging.get("external_actions")
    if not isinstance(external_actions, dict) or not external_actions:
        findings.append("external actions must be a non-empty object")
    else:
        if set(external_actions) != REQUIRED_EXTERNAL_ACTIONS:
            findings.append("external action key set is incomplete or unexpected")
        if any(value is not False for value in external_actions.values()):
            findings.append("external actions must remain false in an unposted packet")

    if staging.get("external_outcomes") != EXPECTED_EXTERNAL_OUTCOMES:
        findings.append("external outcomes must remain not started or pending external")

    draft_path, _ = _validate_bound_file(
        resolved_root,
        staging.get("draft"),
        "draft",
        findings,
        require_all_files=require_all_files,
    )
    _, template = _validate_bound_file(
        resolved_root,
        staging.get("template"),
        "template",
        findings,
        require_all_files=require_all_files,
    )
    _, candidate = _validate_bound_file(
        resolved_root,
        staging.get("candidate"),
        "candidate",
        findings,
        require_all_files=require_all_files,
    )

    if template is not None:
        if template.get("schema_version") != (
            "voiage.pyopensci-submission-template.v1"
        ):
            findings.append("unsupported pyOpenSci template provenance schema")
        if template.get("state") != "reference_only_unposted":
            findings.append("template provenance must remain reference-only")
        if template.get("submission_performed") is not False:
            findings.append("template provenance cannot claim submission")
        upstream = template.get("upstream")
        if not isinstance(upstream, dict):
            findings.append("template upstream provenance must be an object")
        else:
            if upstream.get("commit") != staging.get("template", {}).get(
                "upstream_commit"
            ):
                findings.append("template upstream commit binding mismatch")
            if upstream.get("content_sha256") != staging.get("template", {}).get(
                "upstream_content_sha256"
            ):
                findings.append("template upstream content digest binding mismatch")

    if candidate is not None:
        if candidate.get("schema_version") != (
            "voiage.pyopensci-submission-candidate.v1"
        ):
            findings.append("unsupported pyOpenSci candidate schema")
        published = candidate.get("state") == "published_release_maintainer_confirmed"
        if not published and candidate.get("state") != (
            "release_candidate_prepublication_maintainer_confirmed"
        ):
            findings.append(
                "candidate state must record a confirmed candidate or published release"
            )
        if candidate.get("maintainer_version_confirmation") != "confirmed":
            findings.append("candidate maintainer confirmation must be recorded")
        if candidate.get("submission_performed") is not False:
            findings.append("candidate cannot claim submission")
        recommended = candidate.get("recommended_candidate")
        if not isinstance(recommended, dict) or recommended.get("version") != (
            staging.get("candidate_version")
        ):
            findings.append("recommended candidate does not match staging version")
        elif published:
            _validate_published_candidate(
                resolved_root, candidate, recommended, findings
            )
        elif (
            recommended.get("tag") != f"v{EXPECTED_CANDIDATE_VERSION}"
            or any(
                key not in recommended or recommended[key] is not None
                for key in (
                    "commit",
                    "tree",
                    "tag_object",
                    "published_at",
                    "publication_receipt",
                )
            )
            or recommended.get("tag_signature_verified") is not False
            or recommended.get("latest_on_pypi_when_observed") is not False
            or recommended.get("immutable_github_release") is not False
        ):
            findings.append(
                "prepublication candidate must not claim release identity or publication"
            )
        if not published and candidate.get("artifact_sha256") != {}:
            findings.append(
                "prepublication candidate must not claim published artifact digests"
            )
        joss_handoff = candidate.get("joss_handoff")
        if not isinstance(joss_handoff, dict) or joss_handoff.get("state") != (
            "blocked_pending_refresh_and_external_evidence"
        ):
            findings.append("JOSS handoff must remain blocked")

    if draft_path is not None:
        draft = draft_path.read_text(encoding="utf-8")
        if (
            "UNPOSTED LOCAL DRAFT" not in draft
            or "Submission performed: **No**" not in draft
        ):
            findings.append("draft must state that it is local and unposted")
        if PLACEHOLDER.search(draft):
            findings.append("draft contains an unresolved ordinary placeholder")
        required_sections = template.get("required_sections") if template else None
        if not isinstance(required_sections, list) or not required_sections:
            findings.append("template required sections must be a non-empty array")
        else:
            findings.extend(
                f"draft is missing template section: {section}"
                for section in required_sections
                if not isinstance(section, str) or f"## {section}" not in draft
            )
        lines = draft.splitlines()
        checkbox_markers_valid = True
        for marker in PENDING_DRAFT_CHECKBOX_MARKERS:
            unchecked = f"- [ ] {marker}"
            checked = re.compile(rf"^- \[\s*[xX]\s*\] {re.escape(marker)}")
            if sum(line.startswith(unchecked) for line in lines) != 1 or any(
                checked.match(line) for line in lines
            ):
                checkbox_markers_valid = False
        if not checkbox_markers_valid:
            findings.append(
                "draft human-attestation markers must remain uniquely unchecked"
            )

        version_lines = [
            line for line in lines if line.startswith("Version submitted:")
        ]
        if version_lines != [CONFIRMED_SUBMITTED_VERSION_LINE]:
            findings.append(
                "draft submitted version must match the confirmed maintainer selection"
            )

    return findings


def main(argv: list[str] | None = None) -> int:
    """Validate the supplied repository root and print actionable findings."""
    arguments = sys.argv[1:] if argv is None else argv
    root = Path(arguments[0] if arguments else ".")
    findings = validate_staging_packet(root)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        return 1
    print("pyOpenSci submission staging packet valid; external actions unperformed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
