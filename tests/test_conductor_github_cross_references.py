"""Tests for the Conductor GitHub cross-reference contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_conductor_github_cross_references.py"


def _validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conductor_github_xref", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_cross_references_are_complete() -> None:
    """Every local or proposed track has a valid unique GitHub record."""
    assert _validator().validate(ROOT) == []


def test_local_track_discovery_allows_an_empty_active_queue(tmp_path: Path) -> None:
    """A repository with only archived tracks needs no placeholder directory."""
    archived = tmp_path / "conductor" / "archive" / "completed_track"
    archived.mkdir(parents=True)

    assert _validator()._local_tracks(tmp_path) == {"completed_track"}


def test_cross_reference_manifest_rejects_paths_outside_track_roots(
    tmp_path: Path,
) -> None:
    """A manifest path cannot escape the two Conductor track containers."""
    conductor = tmp_path / "conductor"
    conductor.mkdir()
    manifest = {
        "schema_version": "1.0",
        "project_url": "https://github.com/users/edithatogo/projects/28",
        "tracks": [
            {
                "track_id": "escaped_track",
                "path": "../outside/escaped_track",
                "lifecycle": "active",
                "issue": {"url": "https://github.com/edithatogo/voiage/issues/9999"},
                "parent_issue_url": "https://github.com/edithatogo/voiage/issues/1",
                "subissues": [],
                "pull_requests": [],
            }
        ],
    }
    (conductor / "github-cross-references.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (conductor / "tracks").mkdir()
    (conductor / "archive").mkdir()

    errors = _validator().validate(tmp_path)

    assert (
        "escaped_track: path must stay within conductor/tracks or conductor/archive"
        in errors
    )


def test_sampling_acquisition_harm_track_owns_native_issue_hierarchy() -> None:
    """The fail-closed C18/M32 track is distinct from its umbrella listing."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "sampling_acquisition_harm_voi_20260802"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    metadata = json.loads((ROOT / entry["path"] / "metadata.json").read_text())

    assert entry["issue"]["number"] == 850
    assert entry["parent_issue_url"].endswith("/570")
    assert entry["subissues"] == [
        "https://github.com/edithatogo/voiage/issues/851",
        "https://github.com/edithatogo/voiage/issues/852",
        "https://github.com/edithatogo/voiage/issues/853",
        "https://github.com/edithatogo/voiage/issues/864",
        "https://github.com/edithatogo/voiage/issues/867",
        "https://github.com/edithatogo/voiage/issues/870",
        "https://github.com/edithatogo/voiage/issues/873",
        "https://github.com/edithatogo/voiage/issues/876",
    ]
    assert metadata["planned_version"] == "1.3.0"
    assert metadata["moscow"] == "must"
    assert metadata["canonical_track"] == "C18"
    assert "M32" in metadata["requirement_ids"]
    assert metadata["github_cross_reference"]["subissues"] == entry["subissues"]


def test_manifest_preserves_no_pr_evidence_boundary() -> None:
    """Legacy tracks without a provable PR state that boundary explicitly."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    completed_without_prs = [
        entry
        for entry in manifest["tracks"]
        if entry["lifecycle"] == "completed" and not entry["pull_requests"]
    ]
    assert completed_without_prs
    assert all(
        entry["pull_request_evidence"] == "none_found"
        for entry in completed_without_prs
    )


def test_expected_utility_track_metadata_and_manifest_share_delivery_prs() -> None:
    """The #595 track owns its VOIAGE delivery PR as well as canonical planning."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "risk_adjusted_information_pricing_20260731"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    metadata = json.loads(
        (ROOT / "conductor" / "archive" / track_id / "metadata.json").read_text()
    )

    manifest_urls = {item["url"] for item in entry["pull_requests"]}
    assert manifest_urls == set(metadata["github_cross_reference"]["pull_requests"])
    assert "https://github.com/edithatogo/voiage/pull/712" in manifest_urls


def test_external_landscape_track_preserves_phase_one_pr_boundaries() -> None:
    """The canonical manifest owns both planning and open delivery PRs."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "external_voi_library_feature_parity_20260723"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    track = ROOT / entry["path"]
    metadata = json.loads((track / "metadata.json").read_text())
    index = (track / "index.md").read_text()
    plan = (track / "plan.md").read_text()
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    manifest_prs = {item["url"]: item["status"] for item in entry["pull_requests"]}
    assert set(manifest_prs) == set(metadata["github_cross_reference"]["pull_requests"])
    assert manifest_prs["https://github.com/edithatogo/voiage/pull/621"] == "merged"
    assert manifest_prs["https://github.com/edithatogo/voiage/pull/819"] == "open"
    assert "Merged planning PR #621" in index
    assert "Delivery PR #819" in index
    assert "- [x] **G4:**" in plan
    assert "- **Migrated:** **G5:**" in plan
    assert "- **Migrated:** **G15:**" in plan
    assert gates["scientific-and-contract-review"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "pending"


def test_ml_llm_agent_track_syncs_pr_states_lifecycle_and_pending_gates() -> None:
    """The #319 projections agree without promoting delivery or external gates."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "ml_llm_agent_voi_20260723"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    track_root = ROOT / entry["path"]
    metadata = json.loads((track_root / "metadata.json").read_text())

    manifest_urls = {item["url"] for item in entry["pull_requests"]}
    assert manifest_urls == set(metadata["github_cross_reference"]["pull_requests"])
    planning = next(item for item in entry["pull_requests"] if item["number"] == 621)
    delivery = next(item for item in entry["pull_requests"] if item["number"] == 820)
    assert planning["status"] == "merged"
    assert "b86a7d1aa08896eec2f83ab786c13c25a7fff3a3" in planning["evidence"]
    assert delivery["status"] == "open"
    assert entry["lifecycle"] == "archived"
    assert metadata["status"] == "completed"
    assert metadata["legacy_outcome"] == "superseded"

    gates = {gate["id"]: gate["status"] for gate in metadata["gates"]}
    for gate_id in (
        "scientific-and-contract-review",
        "hosted-required-checks",
        "installed-polyglot-parity",
        "rights-and-human-approval",
    ):
        assert gates[gate_id] == "pending"

    index = (track_root / "index.md").read_text()
    registry = (ROOT / "conductor" / "tracks.md").read_text()
    umbrella_section = registry.split("\n---", 1)[0]
    ml_section = registry.split(
        "## [x] Track: ML, LLM and Agent Value of Information", 1
    )[1].split("\n---", 1)[0]
    assert "Merged planning PR #621" in index
    assert "Status: superseded on 2026-08-29" in index
    assert "bounded governance" not in umbrella_section
    assert "programme completed on 2026-08-30" in umbrella_section
    assert "not active Conductor implementation tasks" in umbrella_section
    assert "Status: superseded on 2026-08-29" in ml_section
    assert "pending work migrated" in ml_section


def test_polyglot_parity_track_projects_current_delivery_state() -> None:
    """The #320 projections agree on merged planning and active delivery."""
    track_id = "polyglot_abi_binding_parity_20260723"
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    metadata = json.loads((ROOT / entry["path"] / "metadata.json").read_text())
    pull_requests = {item["number"]: item for item in entry["pull_requests"]}

    assert {item["url"] for item in entry["pull_requests"]} == set(
        metadata["github_cross_reference"]["pull_requests"]
    )
    assert pull_requests[621]["status"] == "merged"
    assert "b86a7d1aa08896eec2f83ab786c13c25a7fff3a3" in pull_requests[621]["evidence"]
    assert pull_requests[821]["status"] == "open"
    assert "polyglot-parity-governance-delivery" in pull_requests[821]["evidence"]
    assert entry["lifecycle"] == "archived"
    assert metadata["status"] == "completed"
    assert metadata["legacy_outcome"] == "superseded"

    registry = (ROOT / "conductor" / "tracks.md").read_text()
    umbrella_section = registry.split("\n---", 1)[0]
    section = registry.split("## [x] Track: Polyglot ABI and Binding Parity", 1)[1]
    section = section.split("\n---\n", 1)[0]
    assert "governance and capability reconciliation" not in umbrella_section
    assert "*Status: superseded on 2026-08-29" in section
    assert "*Status: new" not in section

    index = (ROOT / entry["path"] / "index.md").read_text()
    assert "Merged planning PR #621" in index
    assert "Delivery PR #821" in index


def test_final_governed_delivery_reconciliation_is_exact_and_additive() -> None:
    """Merged queue facts agree across manifest, metadata, index, and gates."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    deliveries = {
        "conductor-github-cross-reference-reconciliation_20260724": (
            810,
            "8eccc1581729b10a29db957f827ff3cb752f010e",
            "50cf258fe317d5d2e331b060083dbad70a4dd691",
        ),
        "information_source_portfolio_voi_20260801": (
            812,
            "3a2227d12721ffa418d6bf0d7e925ebe70182c59",
            "286f1700b3c06824b6ab56cc6afb84348958190d",
        ),
        "research_software_registry_readiness_20260721": (
            813,
            "144c0e0fbc20a5a55225f2b884e73eee11c9db64",
            "9d49572ade4a27bd340c30f1fa1869c090f2bb6d",
        ),
        "voi_method_census_contract_reconciliation_20260723": (
            816,
            "1a159e02af95fc3a6bce46f2bf8909561be0b9bd",
            "68cb15dfcb8706ab653f8a1631b433a7f63ba322",
        ),
        "controlled_live_dataset_interoperability_20260801": (
            817,
            "d40f5b617df847fe517759f5892a1562f25bc4d9",
            "33537325ad7262dc15bcddb4283a6aa51cfdb323",
        ),
        "quality_release_automation_20260723": (
            822,
            "7e12a5fbc6f7091166d7f5d64c6f2b5b45764f72",
            "0df988125b89f8d0bad08def0bd5b2ea03cd54f5",
        ),
        "remote_dataset_ingestion_security_20260801": (
            824,
            "3635b5d3ca9b680fdfffcddec092db94854cc8e0",
            "9934da329ca3c06bd54094ba95c18fe282c42bb6",
        ),
        "research_contribution_ai_transparency_20260723": (
            825,
            "f00e63d05562d4fc5165aa261c5ab0a296265dd2",
            "4d890aafeb760a0df84a03efa5db95ba5ec85005",
        ),
        "rust_polyglot_voi_completion_20260723": (
            826,
            "e23155e4fa2c39167d7d92a36d34029b6d9c1ee6",
            "0e19b46815af61031e4879a60158864b72748be4",
        ),
        "stable_voi_rust_core_completion_20260723": (
            827,
            "d2881921d5fb17dc3b5fb10ad4c9374b047b6a9f",
            "211044c5fd1ce773ea64a161a92293d00c987f81",
        ),
        "value_of_perspective_completion_20260723": (
            828,
            "e63f31e4929081201c2ca5df3372ab73c9714eba",
            "168156a3e0910e99babecbf4ec06bbfb86b85f56",
        ),
    }

    for track_id, (number, head, merge) in deliveries.items():
        entry = next(
            item for item in manifest["tracks"] if item["track_id"] == track_id
        )
        track_root = ROOT / entry["path"]
        metadata = json.loads((track_root / "metadata.json").read_text())
        pull_request = next(
            item for item in entry["pull_requests"] if item["number"] == number
        )
        assert pull_request["status"] == "merged"
        assert head in pull_request["evidence"]
        assert merge in pull_request["evidence"]
        assert {item["url"] for item in entry["pull_requests"]} == set(
            metadata["github_cross_reference"]["pull_requests"]
        )
        assert f"pull/{number}" in (track_root / "index.md").read_text()

    registry = (ROOT / "conductor" / "tracks.md").read_text()
    root_section = registry.split("\n---\n", 1)[0]
    assert "programme completed on 2026-08-30" in root_section
    assert "not active Conductor implementation tasks" in root_section
    assert "workstream tracks" not in root_section

    registry_entry = next(
        item
        for item in manifest["tracks"]
        if item["track_id"] == "research_software_registry_readiness_20260721"
    )
    registry_prs = {item["number"]: item for item in registry_entry["pull_requests"]}
    for number in (480, 561, 813):
        assert registry_prs[number]["status"] == "merged"
    registry_metadata = json.loads(
        (
            ROOT
            / "conductor"
            / "archive"
            / "research_software_registry_readiness_20260721"
            / "metadata.json"
        ).read_text()
    )
    assert set(registry_entry["subissues"]) == set(
        registry_metadata["github_subissues"]
    )
    assert (
        registry_entry["path"]
        == "conductor/archive/research_software_registry_readiness_20260721"
    )
    assert registry_entry["lifecycle"] == "completed"
    assert registry_entry["issue_closure"] == "external_gate_tracking"

    archive_entry = next(
        item
        for item in manifest["tracks"]
        if item["track_id"]
        == "conductor-github-cross-reference-reconciliation_20260724"
    )
    assert archive_entry["lifecycle"] == "completed"
    assert archive_entry["path"].startswith("conductor/archive/")
