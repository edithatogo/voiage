"""Keep the out-of-tree recovery archive receipt honest and machine-checkable."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "conductor/tracks/remaining_backlog_delivery_20260831"
    / "recovery-archive-verification-20261101.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text())


def test_receipt_declares_both_archives_as_non_repositories() -> None:
    receipt = _receipt()
    archives = {archive["root"]: archive for archive in receipt["archives"]}
    assert set(archives) == {
        "/Volumes/PortableSSD/GitHub/voiage-recovery",
        "/Volumes/PortableSSD/voiage-qualification-20261029-main",
    }
    for archive in archives.values():
        assert archive["is_git_repository"] is False
        assert archive["mergeable"] is False
        assert archive["role"]
        assert archive["disposition"]


def test_every_bundle_digest_and_verification_is_recorded() -> None:
    receipt = _receipt()
    bundles = receipt["bundles"]
    assert len(bundles) == receipt["ref_census"]["bundle_count"]
    assert len(bundles) == 5
    for bundle in bundles:
        assert SHA256.fullmatch(bundle["sha256"])
        assert bundle["bundle_verify"] == "passed"
        assert bundle["bytes"] > 0
        assert bundle["ref_count"] > 0
    assert len({bundle["relative_path"] for bundle in bundles}) == len(bundles)


def test_easybuild_recovery_digest_matches_the_closeout_contract() -> None:
    receipt = _receipt()
    easybuild = next(
        bundle
        for bundle in receipt["bundles"]
        if bundle["relative_path"].endswith("18e076b4.bundle")
    )
    assert (
        easybuild["sha256"]
        == "5f5beb6f9926792fb1424c933e6bdb5908034c7f7c86f25c587af2f64a9c5196"
    )
    restore = receipt["restore_verification"]
    assert restore["sha256"] == (
        "606f9df8d29313b4f25ee6daef9f2a560104545ec680b8da0bc764ad85e247a0"
    )
    assert restore["restored_commit"] == receipt["superseded_ref"]["commit"]
    assert restore["restored_tree"] == "e49a881b6aa2e9f27343f2cbfc63ea0cc72e52ba"
    assert restore["recorded_fsck"] == "passed"


def test_ref_census_partitions_every_pair() -> None:
    census = _receipt()["ref_census"]
    reachable = census["pairs_reachable_from_main"]
    unreachable = census["pairs_not_reachable_from_main"]
    absent = census["pairs_whose_object_is_absent_from_the_clone"]
    assert reachable + unreachable + absent == census["unique_ref_sha_pairs"]
    assert reachable < census["unique_ref_names"] <= census["unique_ref_sha_pairs"]
    assert census["interpretation"]


def test_superseded_ref_records_the_content_comparison() -> None:
    ref = _receipt()["superseded_ref"]
    assert ref["commit"] == "18e076b4c00bf0f9bc5664de4ff12040088db65d"
    assert ref["verdict"] == "superseded, no merge required"
    assert "156 deletions" in ref["evidence"]
