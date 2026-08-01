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
    assert set(delivered) == {556, 559, 571, 595, 619}
    for child in delivered.values():
        assert child["delivery_track"]
        assert child["implementation_pull_requests"]
        assert child["maturity"] == "experimental"
    assert delivered[571]["implementation_pull_requests"] == [679]
    assert delivered[556]["implementation_pull_requests"] == [723]
    assert delivered[556]["review_artifacts"] == [
        "conductor/tracks/supported_frontier_method_completion_20260723/"
        "deterministic-sensitivity-implementation-review.md"
    ]
    assert delivered[559]["implementation_pull_requests"] == [723]
    assert delivered[595]["implementation_pull_requests"] == [712]
    assert delivered[619]["implementation_pull_requests"] == [676]


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
