"""Governance contract for the supported-frontier umbrella programme."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).parents[1]
INVENTORY = (
    ROOT
    / "conductor/archive/supported_frontier_method_completion_20260723"
    / "child-dispositions.json"
)
EXPECTED_CHILDREN = {
    556,
    557,
    558,
    559,
    560,
    570,
    571,
    572,
    582,
    593,
    594,
    595,
    596,
    597,
    598,
    599,
    600,
    619,
}


def _inventory() -> dict[str, object]:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _resolve_historical_artifact(relative: str) -> Path:
    """Resolve evidence paths across the active-to-archive track migration."""
    path = ROOT / relative
    if path.exists():
        return path
    return ROOT / relative.replace("conductor/tracks/", "conductor/archive/", 1)


def test_scientific_review_plan_requires_orchestrated_independent_panel() -> None:
    track = INVENTORY.parent
    protocol = (track / "scientific-review-panel.md").read_text(encoding="utf-8")
    plan = (track / "plan.md").read_text(encoding="utf-8")
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    design = (track / "design.md").read_text(encoding="utf-8")
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))

    for role in (
        "orchestrating agent",
        "Estimand and domain reviewer",
        "Estimator-assurance reviewer",
        "Cross-language and API reviewer",
        "Governance and publication reviewer",
    ):
        assert role in protocol
    for evidence in (
        "review-packet.json",
        "artifact-manifest.json",
        "reviewer-attestations.json",
        "finding-dispositions.json",
        "disagreement-register.json",
        "orchestrator-synthesis.md",
        "adjudication.json",
        "scientific-approval.json",
        "promotion-receipt.json",
    ):
        assert evidence in protocol
    for automatic_blocker in (
        "unresolved Critical/High finding",
        "disputed scientific validity",
        "approval represented only as a Boolean",
    ):
        assert automatic_blocker in protocol

    for task in range(1, 11):
        assert re.search(
            rf"^- (?:\[[ x~]\] |\*\*Migrated:\*\* )\*\*SR{task}(?:\s|\s*/)",
            plan,
            re.MULTILINE,
        )
    for issue in (570, 571, 595, 619):
        assert f"#{issue}" in plan
    for requirement in range(1, 7):
        assert f"**M17-R{requirement}:**" in requirements
    for requirement in range(1, 8):
        assert f"**M17-X{requirement}:**" in requirements
    assert "Critical or High finding may be dispositioned only" in protocol
    assert "Every delta invalidates approval by default" in protocol
    assert "governance reviewer and an affected scientific reviewer" in protocol
    assert "repository owner records the accountable" in protocol
    assert "not independent review" in protocol
    for receipt_field in (
        "identity",
        "qualifications",
        "conflict and independence status",
        "candidate commit/tree and packet hash",
        "family and capability scope",
        "conditions and dissent references",
        "date, expiry and supersession link",
    ):
        assert receipt_field in protocol
    assert "Scientific acceptance does not imply installed parity" in " ".join(
        protocol.split()
    )
    assert "Reject dirty,\n  moving or unreconciled candidates" in plan
    assert "Orchestrating agent" in design
    assert "Independent structured reports" in design

    scientific_gate = next(
        gate
        for gate in metadata["gates"]
        if gate["id"] == "scientific-and-contract-review"
    )
    assert scientific_gate["status"] == "pending"
    assert (
        "accepts M22-M31 scientifically at experimental maturity only"
        in scientific_gate["evidence"]
    )
    assert "remains pending for other families" in scientific_gate["evidence"]

    owning_requirements = {
        "estimation_focused_variance_voi_20260727": (
            range(6, 10),
            "M14-E",
            range(19, 24),
            "E",
        ),
        "study_design_efficiency_20260727": (
            range(7, 12),
            "M15-S",
            range(20, 25),
            "S",
        ),
        "risk_adjusted_information_pricing_20260731": (
            range(5, 7),
            "M16-U",
            range(18, 22),
            "U",
        ),
    }
    for track_id, (
        req_ids,
        req_prefix,
        task_ids,
        task_prefix,
    ) in owning_requirements.items():
        owning_track = ROOT / "conductor" / "tracks" / track_id
        if not owning_track.is_dir():
            owning_track = ROOT / "conductor" / "archive" / track_id
        owning_req_text = (owning_track / "requirements.md").read_text(encoding="utf-8")
        owning_plan_text = (owning_track / "plan.md").read_text(encoding="utf-8")
        for req_id in req_ids:
            assert f"**{req_prefix}{req_id}:**" in owning_req_text
        for task_id in task_ids:
            assert re.search(
                rf"^- (?:\[[ x~]\] |\*\*Migrated:\*\* )"
                rf"\*\*{task_prefix}{task_id}:\*\*",
                owning_plan_text,
                re.MULTILINE,
            )


def test_sampling_harm_scoping_is_canonical_and_fail_closed() -> None:
    track = INVENTORY.parent
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    plan = (track / "plan.md").read_text(encoding="utf-8")
    canonical_requirements = (ROOT / "conductor/requirements.md").read_text(
        encoding="utf-8"
    )
    canonical_design = (ROOT / "conductor/design.md").read_text(encoding="utf-8")

    assert "**M32 / planned v1.3.0:**" in canonical_requirements
    assert "| v1.3.0 | Must | M32 |" in canonical_requirements
    assert "C18 / M32" in canonical_design
    assert "sampling_acquisition_harm_voi_20260802" in canonical_design

    assert "M32" in metadata["requirement_ids"]
    assert "M32" in metadata["planned_version_extensions"]["1.3.0"]
    assert "M32" in metadata["canonical_track_extensions"]["C18"]
    for issue in range(850, 854):
        issue_url = f"https://github.com/edithatogo/voiage/issues/{issue}"
        assert issue_url in metadata["github_subissues"]
        assert issue_url in metadata["github_cross_reference"]["subissues"]

    scope_gate = next(
        gate for gate in metadata["gates"] if gate["id"] == "sampling-harm-scoping"
    )
    assert scope_gate["status"] == "satisfied"
    for boundary in (
        "sampling_acquisition_harm_voi_20260802",
        "fail-closed",
        "owner",
        "runtime",
    ):
        assert boundary in scope_gate["evidence"]

    assert re.search(r"^- \[x\] \*\*SR4 / #850", plan, re.MULTILINE)
    for reference in (
        "#851\N{EN DASH}#853",
        "sampling_acquisition_harm_voi_20260802",
        "fail-closed",
        "human",
        "runtime",
    ):
        assert reference in plan


def test_scientific_review_candidate_freezes_live_governance_and_artifacts() -> None:
    track = INVENTORY.parent
    candidate = json.loads(
        (track / "scientific-review-candidate-20260802.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (track / candidate["artifact_manifest"]).read_text(encoding="utf-8")
    )
    readback = json.loads(
        (track / candidate["governance_readback"]).read_text(encoding="utf-8")
    )

    assert candidate["commit"] == "3b9024c503c171e7e321ddfaacc3665589e4d5e8"
    assert candidate["tree"] == "e4f13cb9eea99a62a1330b39b3ac6f13faa76559"
    assert candidate["owning_issue"] == 841
    assert candidate["implementation_issues"] == list(range(842, 851))
    assert candidate["review_entry_state"] == "ready_with_explicit_external_gates"
    assert (
        "any candidate change invalidates"
        in candidate["invalidation_policy"]["default"]
    )

    assert manifest["candidate_id"] == candidate["candidate_id"]
    assert len(manifest["artifacts"]) >= 15
    git = shutil.which("git")
    assert git is not None
    for artifact in manifest["artifacts"]:
        content = subprocess.run(
            [git, "show", f"{candidate['commit']}:{artifact['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]

    hierarchy = {item["issue"]: item for item in readback["hierarchy"]}
    assert set(hierarchy) == set(range(841, 851))
    assert hierarchy[841]["parent"] == 318
    assert hierarchy[843]["parent"] == 619
    assert hierarchy[844]["parent"] == 571
    assert hierarchy[845]["parent"] == 595
    assert hierarchy[850]["parent"] == 570
    assert hierarchy[850]["dependency"] == 571
    assert readback["normalized_existing_issue"] == {
        "issue": 570,
        "release_target": "v1.3.0 (C18/M22)",
        "project_contract_version": "1.3.0",
        "project_gate": "Human",
        "project_risk": "High",
        "project_sync_state": "Clean",
        "review_due": "2026-11-30",
    }

    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    metadata_issues = set(metadata["github_subissues"])
    for issue in range(841, 851):
        assert f"https://github.com/edithatogo/voiage/issues/{issue}" in metadata_issues


def test_inventory_covers_exact_live_native_hierarchy() -> None:
    inventory = _inventory()
    assert inventory["schema_version"] == "1.0.0"
    assert inventory["track_id"] == "supported_frontier_method_completion_20260723"
    assert inventory["parent_issue"] == 313
    assert inventory["issue"] == 318
    children = inventory["children"]
    assert isinstance(children, list)
    assert {child["issue"] for child in children} == EXPECTED_CHILDREN
    assert len(children) == len(EXPECTED_CHILDREN)


def test_programme_evidence_map_closes_only_repository_owned_g5_to_g15() -> None:
    track = INVENTORY.parent
    evidence_map = json.loads(
        (track / "g5-g13-evidence-map.json").read_text(encoding="utf-8")
    )
    inventory = _inventory()
    plan = (track / "plan.md").read_text(encoding="utf-8")

    assert evidence_map["source_revision"] == (
        "163825d8b09e064a65cbab8a5807904629bbf05e"
    )
    assert {item["issue"] for item in evidence_map["families"]} == EXPECTED_CHILDREN
    assert set(evidence_map["completed_repository_gates"]) == {
        "G5",
        "G6",
        "G7",
        "G8",
        "G9",
        "G10",
        "G11",
        "G12",
        "G13",
        "G14",
        "G15",
    }
    assert evidence_map["pending_programme_gates"] == []
    for family in evidence_map["families"]:
        for gate in ("G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12"):
            assert family["evidence"][gate]
        for artifact in family["artifacts"]:
            assert _resolve_historical_artifact(artifact).is_file(), artifact
    by_mapped_issue = {item["issue"]: item for item in evidence_map["families"]}
    for issue in (571, 595, 619):
        assert "independent" in by_mapped_issue[issue]["evidence"]["G8"]

    by_issue = {item["issue"]: item for item in inventory["children"]}
    assert inventory["source_revision"] == evidence_map["source_revision"]
    assert by_issue[558]["issue_state"] == "closed"
    assert by_issue[558]["project_status"] == "Done"
    for issue in (556, 557, 558, 559):
        assert by_issue[issue]["disposition"] == "experimental_merged"
    assert evidence_map["project_normalization_eligibility"]["mutation_performed"]
    assert evidence_map["project_normalization_eligibility"]["observations"] == {
        "558": "Done / Resolved / Verified / Clean",
        "724": "Done / Resolved / Verified / Clean",
        "556": "In Progress / Mitigating / Unverified / Clean",
    }
    assert evidence_map["project_normalization_eligibility"]["issues"] == [
        558,
        724,
        725,
        726,
        727,
        728,
        731,
        732,
        733,
        734,
        735,
        738,
        739,
        740,
        741,
        742,
    ]
    readback_path = _resolve_historical_artifact(
        evidence_map["project_normalization_eligibility"]["readback"]
    )
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    assert {item["issue"] for item in readback["closed_items"]} == {
        558,
        724,
        725,
        726,
        727,
        728,
        731,
        732,
        733,
        734,
        735,
        738,
        739,
        740,
        741,
        742,
    }
    assert {item["issue"] for item in readback["open_items"]} == {
        318,
        556,
        557,
        559,
        560,
        570,
        571,
        572,
        582,
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        600,
        619,
    }
    assert {
        (item["status"], item["lifecycle"], item["evidence_state"], item["sync_state"])
        for item in readback["closed_items"]
    } == {("Done", "Resolved", "Verified", "Clean")}
    assert {
        (item["status"], item["lifecycle"], item["evidence_state"], item["sync_state"])
        for item in readback["open_items"]
    } == {("In Progress", "Mitigating", "Unverified", "Clean")}
    assert readback["issue_619"]["state"] == "OPEN"
    assert readback["issue_619"]["body_synchronized"] is True
    assert "- [x] **G8:**" in plan
    for gate in ("G14", "G15"):
        assert f"- [x] **{gate}:" in plan

    completion = evidence_map["repository_completion"]
    assert completion["exact_head"] == "8f1d70cb6bc67d5f1b07d95cd254171d8d3a913d"
    assert completion["merge_commit"] == ("163825d8b09e064a65cbab8a5807904629bbf05e")
    assert completion["checks"] == {
        "success": 38,
        "skipped": 3,
        "neutral": 1,
        "failed": 0,
        "cancelled": 0,
        "pending": 0,
    }
    assert completion["review_threads"] == 0
    assert completion["repository_complete"] is True
    assert completion["issue_318_closed"] is False
    receipt = _resolve_historical_artifact(completion["receipt"])
    assert receipt.is_file()
    receipt_text = receipt.read_text(encoding="utf-8")
    for boundary in (
        "scientific approval",
        "cross-language parity",
        "stable promotion",
        "release",
        "issue closure",
    ):
        assert boundary in receipt_text

    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    umbrella_xref = next(
        item
        for item in cross_references["tracks"]
        if item["track_id"] == "supported_frontier_method_completion_20260723"
    )
    pr_836 = next(
        item for item in umbrella_xref["pull_requests"] if item["number"] == 836
    )
    assert pr_836["status"] == "merged"
    assert completion["exact_head"] in pr_836["evidence"]
    assert completion["merge_commit"] in pr_836["evidence"]

    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    hosted_gate = next(
        gate for gate in metadata["gates"] if gate["id"] == "hosted-required-checks"
    )
    assert hosted_gate["status"] == "satisfied"
    assert not [
        gate
        for gate in metadata["gates"]
        if gate["kind"] == "hosted_validation" and gate["status"] == "pending"
    ]
    metadata_prs = set(metadata["github_cross_reference"]["pull_requests"])
    central_prs = {item["url"] for item in umbrella_xref["pull_requests"]}
    pr_836_url = "https://github.com/edithatogo/voiage/pull/836"
    assert pr_836_url in metadata_prs
    assert pr_836_url in central_prs
    assert metadata["updated_at"] >= "2026-08-01T17:57:13Z"
    assert cross_references["generated_at"] >= "2026-08-01T17:57:13Z"

    index = (track / "index.md").read_text(encoding="utf-8")
    assert "is being reconciled" not in index
    assert "repository-owned programme work is complete" in index


def test_stage_one_roadmap_does_not_reopen_merged_delivery_prs() -> None:
    roadmap = (ROOT / "roadmap.md").read_text(encoding="utf-8")

    for stale_claim in (
        "draft PR #65",
        "open estimation-family sync PR #64",
        "contract on draft PR #723",
        "engine on draft PR #723",
        "hosted exact-head and installed-wheel evidence",
    ):
        assert stale_claim not in roadmap
    for merge_receipt in (
        "cedc6fbb",
        "ac61bb9f",
        "44e0067a",
        "e8aaba82",
    ):
        assert merge_receipt in roadmap


def test_inventory_never_promotes_adjacent_artifacts_to_delivery_evidence() -> None:
    children = _inventory()["children"]
    assert isinstance(children, list)
    for child in children:
        assert child["disposition"] in {
            "planned_only",
            "adjacent_only",
            "contract_in_progress",
            "experimental_branch",
            "experimental_merged",
            "reviewed_exclusion",
        }
        if child["disposition"] in {"planned_only", "adjacent_only"}:
            assert child["satisfies_ac06"] is False
            assert child["implementation_pull_requests"] == []
        if child["disposition"] == "reviewed_exclusion":
            assert child["review_artifacts"]


def test_positive_delivery_claims_are_bound_to_pull_requests_and_tracks() -> None:
    children = _inventory()["children"]
    assert isinstance(children, list)
    delivered = {
        child["issue"]: child
        for child in children
        if child["disposition"] in {"experimental_branch", "experimental_merged"}
    }
    assert set(delivered) == {
        556,
        557,
        558,
        559,
        560,
        570,
        571,
        572,
        582,
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        600,
        619,
    }
    for child in delivered.values():
        assert child["delivery_track"]
        assert child["implementation_pull_requests"]
        assert child["maturity"] == "experimental"
    assert delivered[571]["implementation_pull_requests"] == [679]
    assert delivered[570]["implementation_pull_requests"] == [769]
    assert delivered[572]["implementation_pull_requests"] == [770]
    assert delivered[594]["implementation_pull_requests"] == [798]
    assert delivered[596]["implementation_pull_requests"] == [804]
    assert delivered[596]["review_artifacts"][-1].endswith(
        "event-localized-information-final-review.md"
    )
    assert delivered[597]["implementation_pull_requests"] == [807]
    assert delivered[597]["review_artifacts"][-1].endswith(
        "belief-state-information-fifth-review.md"
    )
    assert delivered[594]["review_artifacts"][-1].endswith(
        "uncertainty_modelling_value_20260801/independent-implementation-review.md"
    )
    assert delivered[572]["review_artifacts"][-1].endswith(
        "forecast-signal-implementation-review.md"
    )
    assert delivered[582]["implementation_pull_requests"] == [772]
    assert delivered[556]["implementation_pull_requests"] == [723]
    assert delivered[556]["review_artifacts"] == [
        "conductor/tracks/supported_frontier_method_completion_20260723/"
        "deterministic-sensitivity-implementation-review.md"
    ]
    assert delivered[557]["implementation_pull_requests"] == [736]
    assert delivered[557]["review_artifacts"] == [
        "conductor/tracks/supported_frontier_method_completion_20260723/"
        "distribution-family-information-implementation-review.md"
    ]
    assert delivered[558]["implementation_pull_requests"] == [743, 744]
    assert delivered[558]["review_artifacts"] == [
        "conductor/tracks/supported_frontier_method_completion_20260723/"
        "qualitative-information-implementation-review.md"
    ]
    assert delivered[559]["implementation_pull_requests"] == [723]
    assert delivered[560]["implementation_pull_requests"] == [751]
    assert delivered[560]["review_artifacts"][-1].endswith(
        "mcda-information-implementation-review.md"
    )
    assert delivered[593]["implementation_pull_requests"] == [787]
    assert delivered[593]["disposition"] == "experimental_merged"
    assert delivered[595]["implementation_pull_requests"] == [712]
    assert delivered[600]["implementation_pull_requests"] == [831]
    assert delivered[619]["implementation_pull_requests"] == [676, 837]


def test_issue_571_delivery_closeout_preserves_later_gates() -> None:
    umbrella = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"
    dedicated = ROOT / "conductor/archive/study_design_efficiency_20260727"
    child = next(item for item in _inventory()["children"] if item["issue"] == 571)
    umbrella_metadata = json.loads(
        (umbrella / "metadata.json").read_text(encoding="utf-8")
    )
    dedicated_metadata = json.loads(
        (dedicated / "metadata.json").read_text(encoding="utf-8")
    )
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    governed_track = next(
        item
        for item in cross_references["tracks"]
        if item["track_id"] == "study_design_efficiency_20260727"
    )
    pull_request = next(
        item for item in governed_track["pull_requests"] if item["number"] == 679
    )
    umbrella_gate = next(
        gate
        for gate in umbrella_metadata["gates"]
        if gate["id"] == "study-design-efficiency-hosted-assurance"
    )
    dedicated_gate = next(
        gate
        for gate in dedicated_metadata["gates"]
        if gate["id"] == "hosted-required-checks"
    )
    pending_text = " ".join(
        (
            (umbrella / "plan.md").read_text(encoding="utf-8"),
            (dedicated / "plan.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["delivery_subissues"] == [680, 681, 682]
    assert pull_request["status"] == "merged"
    assert "hosted-required-checks" in pull_request["evidence"]
    assert "ce5d712779897bdd7d398e367de6a7e0bc743692" in pull_request["evidence"]
    assert "5d059a80447afc85cee63eb85971fc1c9e80f40c" in pull_request["evidence"]
    assert umbrella_gate["status"] == "satisfied"
    assert dedicated_gate["status"] == "satisfied"
    for evidence in (umbrella_gate["evidence"], dedicated_gate["evidence"]):
        assert "65 terminal conclusions" in evidence
        assert "Both review threads were resolved" in evidence
    for gate in (
        "scientific review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #571 closure",
        "umbrella #318 closure",
    ):
        assert gate in pending_text


def test_issue_595_delivery_closeout_preserves_alias_and_later_gates() -> None:
    umbrella = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"
    dedicated = ROOT / "conductor/archive/risk_adjusted_information_pricing_20260731"
    child = next(item for item in _inventory()["children"] if item["issue"] == 595)
    umbrella_metadata = json.loads(
        (umbrella / "metadata.json").read_text(encoding="utf-8")
    )
    dedicated_metadata = json.loads(
        (dedicated / "metadata.json").read_text(encoding="utf-8")
    )
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    governed_track = next(
        item
        for item in cross_references["tracks"]
        if item["track_id"] == "risk_adjusted_information_pricing_20260731"
    )
    pull_request = next(
        item for item in governed_track["pull_requests"] if item["number"] == 712
    )
    umbrella_gate = next(
        gate
        for gate in umbrella_metadata["gates"]
        if gate["id"] == "expected-utility-pricing-hosted-assurance"
    )
    dedicated_gate = next(
        gate
        for gate in dedicated_metadata["gates"]
        if gate["id"] == "hosted-required-checks"
    )
    governed_text = " ".join(
        (
            (umbrella / "plan.md").read_text(encoding="utf-8"),
            (dedicated / "plan.md").read_text(encoding="utf-8"),
            (dedicated / "index.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["delivery_subissues"] == [694, 695, 696, 697]
    assert pull_request["status"] == "merged"
    assert "hosted-required-checks" in pull_request["evidence"]
    assert "1048c4bc4354acdb0c468da17cb0b5d581961690" in pull_request["evidence"]
    assert "b8395abfc603509a2f1a2c87c9c33e6260fb5840" in pull_request["evidence"]
    assert umbrella_gate["status"] == "satisfied"
    assert dedicated_gate["status"] == "satisfied"
    for evidence in (umbrella_gate["evidence"], dedicated_gate["evidence"]):
        assert "65 terminal conclusions" in evidence
        assert "Both review threads were resolved" in evidence
    assert "VoC remains a presentation/delegating alias" in governed_text
    for gate in (
        "scientific review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #595 closure",
        "umbrella #318 closure",
    ):
        assert gate in governed_text


def test_issue_593_delivery_closeout_preserves_later_gates() -> None:
    track = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    pull_requests = [
        pull_request
        for governed_track in cross_references["tracks"]
        for pull_request in governed_track["pull_requests"]
        if pull_request["number"] == 787
    ]
    hosted_gate = next(
        gate
        for gate in metadata["gates"]
        if gate["id"] == "implementation-information-hosted-assurance"
    )
    pending_text = " ".join(
        (
            (track / "plan.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert len(pull_requests) >= 2
    for pull_request in pull_requests:
        assert pull_request["status"] == "merged"
        assert "hosted-required-checks" in pull_request["evidence"]
        assert "de31458b556136359cb9195f8ced82cff9182ece" in pull_request["evidence"]
        assert "20e0c606fb02f282134e9cc876fa475178edfe40" in pull_request["evidence"]
    assert hosted_gate["status"] == "satisfied"
    assert "38 successful checks" in hosted_gate["evidence"]
    for gate in (
        "scientific review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #593 closure",
        "umbrella #318 closure",
    ):
        assert gate in pending_text


def test_issue_594_delivery_closeout_preserves_later_gates() -> None:
    umbrella = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"
    dedicated = ROOT / "conductor/archive/uncertainty_modelling_value_20260801"
    inventory = _inventory()
    child = next(item for item in inventory["children"] if item["issue"] == 594)
    umbrella_metadata = json.loads(
        (umbrella / "metadata.json").read_text(encoding="utf-8")
    )
    dedicated_metadata = json.loads(
        (dedicated / "metadata.json").read_text(encoding="utf-8")
    )
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    pull_requests = [
        pull_request
        for governed_track in cross_references["tracks"]
        for pull_request in governed_track["pull_requests"]
        if pull_request["number"] == 798
    ]
    umbrella_gate = next(
        gate
        for gate in umbrella_metadata["gates"]
        if gate["id"] == "uncertainty-modelling-hosted-assurance"
    )
    dedicated_gate = next(
        gate
        for gate in dedicated_metadata["gates"]
        if gate["id"] == "hosted-required-checks"
    )
    pending_text = " ".join(
        (
            (umbrella / "plan.md").read_text(encoding="utf-8"),
            (dedicated / "plan.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert child["disposition"] == "experimental_merged"
    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert len(pull_requests) >= 2
    for pull_request in pull_requests:
        assert pull_request["status"] == "merged"
        assert "hosted-required-checks" in pull_request["evidence"]
        assert "aa5d9fd86a42fecd5e8746e77c74ba23e33bb092" in pull_request["evidence"]
        assert "c5adca8fd49b74a04312111168283fbdffc2dcbd" in pull_request["evidence"]
    assert umbrella_gate["status"] == "satisfied"
    assert dedicated_gate["status"] == "satisfied"
    assert "42 hosted checks" in umbrella_gate["evidence"]
    for gate in (
        "scientific review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #594 closure",
        "umbrella #318 closure",
    ):
        assert gate in pending_text


def test_issue_600_contract_and_native_delivery_children_are_governed() -> None:
    track = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"
    inventory = _inventory()
    child = next(item for item in inventory["children"] if item["issue"] == 600)
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    cross_reference = next(
        item
        for item in cross_references["tracks"]
        if item["track_id"] == "supported_frontier_method_completion_20260723"
    )
    governed_text = " ".join(
        (
            (track / "requirements.md").read_text(encoding="utf-8"),
            (track / "design.md").read_text(encoding="utf-8"),
            (track / "plan.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["maturity"] == "experimental"
    assert child["delivery_subissues"] == [790, 791, 792]
    assert child["implementation_pull_requests"] == [831]
    assert child["satisfies_ac06"] is True
    for issue in (600, 790, 791, 792):
        issue_url = f"https://github.com/edithatogo/voiage/issues/{issue}"
        assert issue_url in metadata["github_subissues"]
        assert issue_url in cross_reference["subissues"]
    for required_boundary in (
        "weighted population",
        "expectation-only",
        "rVSI0",
        "independent implementation review",
        "All five CodeQL review threads were resolved",
        "Rust/R/Julia parity",
        "stable promotion",
        "parent #600",
        "umbrella #318",
    ):
        assert required_boundary in governed_text
    assert "eb5a201d82350631dc3ba0b636dfdf43563ea64f" in governed_text
    assert "ac1d31bf900c3ee6e817047202cae4229918d48f" in governed_text


def test_issue_619_repository_delivery_and_open_scientific_gate_are_governed() -> None:
    dedicated = ROOT / "conductor/archive/estimation_focused_variance_voi_20260727"
    child = next(item for item in _inventory()["children"] if item["issue"] == 619)
    metadata = json.loads((dedicated / "metadata.json").read_text(encoding="utf-8"))
    cross_references = json.loads(
        (ROOT / "conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    cross_reference = next(
        item
        for item in cross_references["tracks"]
        if item["track_id"] == "estimation_focused_variance_voi_20260727"
    )
    governed_text = " ".join(
        (
            (dedicated / "index.md").read_text(encoding="utf-8"),
            (dedicated / "plan.md").read_text(encoding="utf-8"),
            (dedicated / "delivery-closeout-20260801.md").read_text(encoding="utf-8"),
            (ROOT / "roadmap.md").read_text(encoding="utf-8"),
            (ROOT / "todo.md").read_text(encoding="utf-8"),
        )
    )

    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["delivery_subissues"] == [671, 672, 673, 674]
    assert child["implementation_pull_requests"] == [676, 837]
    assert child["satisfies_ac06"] is True
    scientific_gate = next(
        gate
        for gate in metadata["gates"]
        if gate["id"] == "scientific-classification-review"
    )
    delivery_gate = next(
        gate
        for gate in metadata["gates"]
        if gate["id"] == "implementation-and-canonical-sync-merge"
    )
    assert metadata["status"] == "completed"
    assert metadata["legacy_outcome"] == "superseded"
    assert scientific_gate["status"] == "pending"
    assert delivery_gate["status"] == "satisfied"
    assert cross_reference["issue"]["state"] == "open"
    assert cross_reference["pull_request_evidence"] == (
        "merged_track_delivery_and_canonical_sync"
    )
    pull_requests = {item["number"]: item for item in cross_reference["pull_requests"]}
    assert pull_requests[676]["status"] == "merged"
    assert pull_requests[837]["status"] == "merged"
    assert "076a29075e839e3cad49d0487dff0c4e2639845f" in pull_requests[837]["evidence"]
    assert "366186b358abd775bea5fd2440d7e0ececb3ebaa" in pull_requests[837]["evidence"]
    assert pull_requests[64]["status"] == "merged"
    for evidence in (
        "5e2c097fbdda8965d1907d7e930e910238fa24da",
        "9495fc3f372b9564701a180c6cf611a3ddc010dd",
        "6c3fd72358f3feef6c542e0a374d7ea74889f915",
        "cedc6fbb17a5d999cb12bb300a01f87d976ec02e",
        "076a29075e839e3cad49d0487dff0c4e2639845f",
        "366186b358abd775bea5fd2440d7e0ececb3ebaa",
        "65 terminal",
        "two resolved review threads",
        "E17 scientific classification",
        "parent #619 closure",
        "umbrella #318 closure",
    ):
        assert evidence in governed_text


def test_programme_records_unfinished_census_dependency() -> None:
    dependencies = _inventory()["dependencies"]
    assert dependencies == [
        {
            "track_id": "voi_method_census_contract_reconciliation_20260723",
            "status": "classification_checkpoint_satisfied",
            "blocking_claim": "accepted-family classification complete",
        },
        {
            "track_id": "stable_voi_rust_core_completion_20260723",
            "status": "new",
            "blocking_claim": "stable-core dependency complete",
        },
    ]


def test_event_localized_information_experimental_delivery_is_governed() -> None:
    track = INVENTORY.parent
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    design = (track / "design.md").read_text(encoding="utf-8")
    plan = (track / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    canonical = (ROOT / "conductor/requirements.md").read_text(encoding="utf-8")
    canonical_design = (ROOT / "conductor/design.md").read_text(encoding="utf-8")
    child = next(child for child in _inventory()["children"] if child["issue"] == 596)

    assert {"M27-S1", "M27-S2", "M27-S3", "M27-S4"} <= {
        line.split(":", maxsplit=1)[0].removeprefix("- **")
        for line in requirements.splitlines()
        if line.startswith("- **M27-")
    }
    assert "Event-localized information value" in design
    assert "M27" in metadata["requirement_ids"]
    assert "M27" in metadata["canonical_track_extensions"]["C18"]
    assert "C18 governed event-localized" in canonical
    assert "C18/M27 policy-relative EUI density" in canonical_design
    for issue in range(777, 780):
        assert f"#{issue}" in plan
    assert child["disposition"] == "experimental_merged"
    assert child["implementation_pull_requests"] == [804]
    assert child["satisfies_ac06"] is True


def test_belief_state_information_experimental_delivery_is_governed() -> None:
    track = INVENTORY.parent
    plan = (track / "plan.md").read_text(encoding="utf-8")
    todo = (ROOT / "todo.md").read_text(encoding="utf-8")
    child = next(child for child in _inventory()["children"] if child["issue"] == 597)

    for issue in range(780, 783):
        assert f"#{issue}" in plan
    assert "#780--#782" in todo
    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["implementation_pull_requests"] == [807]
    assert child["satisfies_ac06"] is True
    assert "35cfe522c1b23b8dae3542442a8900b14f9bbcc0" in plan
    assert "39de9c6ab2079b55a4666243baff2a5db7f10604" in plan
    for gate in (
        "scientific panel",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #597 closure",
        "umbrella #318 closure",
    ):
        assert gate in f"{plan}\n{todo}"


def test_parallel_m26_to_m31_frontier_governance_is_additively_preserved() -> None:
    track = INVENTORY.parent
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    children = _inventory()["children"]
    assert isinstance(children, list)
    by_issue = {child["issue"]: child for child in children}
    expected_requirements = {"M26", "M27", "M28", "M29", "M30", "M31"}

    assert expected_requirements <= set(metadata["requirement_ids"])
    assert expected_requirements <= set(metadata["planned_version_extensions"]["1.3.0"])
    assert expected_requirements <= set(metadata["canonical_track_extensions"]["C18"])
    assert by_issue[597]["delivery_subissues"] == [780, 781, 782]
    assert by_issue[597]["implementation_pull_requests"] == [807]
    assert by_issue[598]["delivery_subissues"] == [783, 784, 785]
    assert by_issue[598]["implementation_pull_requests"] == [808]
    assert by_issue[598]["disposition"] == "experimental_merged"
    assert by_issue[599]["delivery_subissues"] == [786, 788, 789]
    assert by_issue[599]["implementation_pull_requests"] == [809]
    assert by_issue[599]["disposition"] == "experimental_merged"
    assert by_issue[600]["delivery_subissues"] == [790, 791, 792]
    assert by_issue[600]["implementation_pull_requests"] == [831]
    assert by_issue[600]["disposition"] == "experimental_merged"


def test_heterogeneity_value_experimental_delivery_is_governed() -> None:
    track = INVENTORY.parent
    plan = (track / "plan.md").read_text(encoding="utf-8")
    todo = (ROOT / "todo.md").read_text(encoding="utf-8")
    child = next(child for child in _inventory()["children"] if child["issue"] == 599)

    for issue in (786, 788, 789):
        assert f"#{issue}" in plan
        assert f"#{issue}" in todo
    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["implementation_pull_requests"] == [809]
    assert child["satisfies_ac06"] is True
    assert "b0fc8db75796ffac9e66720ab45fdcf341c0b516" in plan
    assert "1a37526af0ee87acc57dd14a629eb52aef2e182c" in plan
    assert "zero review threads" in plan
    for gate in (
        "Scientific review",
        "selection-bias and sparse-subgroup validity review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #599 closure",
        "umbrella #318 closure",
    ):
        assert gate in f"{plan}\n{todo}"


def test_signed_social_information_experimental_delivery_is_governed() -> None:
    track = INVENTORY.parent
    plan = (track / "plan.md").read_text(encoding="utf-8")
    todo = (ROOT / "todo.md").read_text(encoding="utf-8")
    child = next(child for child in _inventory()["children"] if child["issue"] == 598)

    for issue in range(783, 786):
        assert f"#{issue}" in plan
    assert "#783" in todo
    assert "#785" in todo
    assert child["issue_state"] == "open"
    assert child["project_status"] == "In Progress"
    assert child["disposition"] == "experimental_merged"
    assert child["implementation_pull_requests"] == [808]
    assert child["satisfies_ac06"] is True
    assert "4d121b29bb50492bcc84b1cdfa6fb46df9e5e51c" in plan
    assert "d649c344ef2493abe445fb9e3ef20da89c53fb75" in plan
    assert "all 10 review threads were resolved" in plan
    for gate in (
        "Scientific review",
        "Rust/R/Julia parity",
        "stable promotion",
        "release",
        "parent #598 closure",
        "umbrella #318 closure",
    ):
        assert gate in f"{plan}\n{todo}"


def test_dsa_governance_is_versioned_and_cross_referenced() -> None:
    track = INVENTORY.parent
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    design = (track / "design.md").read_text(encoding="utf-8")
    plan = (track / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    canonical = (ROOT / "conductor/requirements.md").read_text(encoding="utf-8")
    canonical_design = (ROOT / "conductor/design.md").read_text(encoding="utf-8")

    assert "M18-U1" in requirements
    assert "M18-U2" in requirements
    assert "M18-U3" in requirements
    assert "Deterministic sensitivity analysis" in design
    assert "M18 / planned v1.2.0" in canonical
    assert "DSA baseline + direction + units" in canonical_design
    assert "M18" in metadata["requirement_ids"]
    for issue in range(724, 729):
        assert f"#{issue}" in plan


def test_distribution_family_information_is_governed_and_cross_referenced() -> None:
    track = INVENTORY.parent
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    design = (track / "design.md").read_text(encoding="utf-8")
    plan = (track / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    canonical = (ROOT / "conductor/requirements.md").read_text(encoding="utf-8")
    canonical_design = (ROOT / "conductor/design.md").read_text(encoding="utf-8")

    assert {"M19-U1", "M19-U2", "M19-U3"} <= {
        line.split(":", maxsplit=1)[0].removeprefix("- **")
        for line in requirements.splitlines()
        if line.startswith("- **M19-")
    }
    assert "Value of Distribution-Family Information" in design
    assert "M19 / planned v1.2.0" in canonical
    assert "Declared model-family index" in canonical_design
    assert "M19" in metadata["requirement_ids"]
    for issue in range(731, 736):
        assert f"#{issue}" in plan


def test_qualitative_voi_is_governed_and_cross_referenced() -> None:
    track = INVENTORY.parent
    requirements = (track / "requirements.md").read_text(encoding="utf-8")
    design = (track / "design.md").read_text(encoding="utf-8")
    plan = (track / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    canonical = (ROOT / "conductor/requirements.md").read_text(encoding="utf-8")
    canonical_design = (ROOT / "conductor/design.md").read_text(encoding="utf-8")

    assert {"M20-U1", "M20-U2", "M20-U3", "M20-U4"} <= {
        line.split(":", maxsplit=1)[0].removeprefix("- **")
        for line in requirements.splitlines()
        if line.startswith("- **M20-")
    }
    assert "Qualitative value of information" in design
    assert "M20 / planned v1.3.0" in canonical
    assert "Versioned qualitative assessment" in canonical_design
    assert "M20" in metadata["requirement_ids"]
    for issue in range(738, 743):
        assert f"#{issue}" in plan
