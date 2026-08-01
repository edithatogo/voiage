"""Governance contract for the supported-frontier umbrella programme."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
INVENTORY = (
    ROOT
    / "conductor/tracks/supported_frontier_method_completion_20260723"
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
    assert delivered[619]["implementation_pull_requests"] == [676]


def test_issue_593_delivery_closeout_preserves_later_gates() -> None:
    track = ROOT / "conductor/tracks/supported_frontier_method_completion_20260723"
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

    assert len(pull_requests) == 2
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
    umbrella = ROOT / "conductor/tracks/supported_frontier_method_completion_20260723"
    dedicated = ROOT / "conductor/tracks/uncertainty_modelling_value_20260801"
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
    assert len(pull_requests) == 2
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


def test_parallel_m26_to_m30_frontier_governance_is_additively_preserved() -> None:
    track = INVENTORY.parent
    metadata = json.loads((track / "metadata.json").read_text(encoding="utf-8"))
    children = _inventory()["children"]
    assert isinstance(children, list)
    by_issue = {child["issue"]: child for child in children}
    expected_requirements = {"M26", "M27", "M28", "M29", "M30"}

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
