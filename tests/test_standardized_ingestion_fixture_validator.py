"""Tests for deterministic standardized-ingestion fixture digest manifests."""

from __future__ import annotations

from pathlib import Path
import shutil

from scripts import validate_standardized_ingestion_fixtures as validator

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "standardized_ingestion"


def test_checked_in_fixture_corpus_has_current_digests() -> None:
    """Both canonical source-format corpora are pinned by their manifests."""
    manifests = validator.fixture_manifests(FIXTURE_ROOT)

    assert [path.name for path in manifests] == [
        "canonical-decision.manifest.json",
        "cost-outcome-decision.manifest.json",
    ]
    assert validator.main([str(FIXTURE_ROOT)]) == 0


def test_fixture_validator_detects_and_deterministically_refreshes_changes(
    tmp_path: Path,
) -> None:
    """Digest refreshes are reproducible and require an explicit write command."""
    root = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_ROOT, root)
    resource = root / "canonical-decision.csv"
    resource.write_text(resource.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert validator.main([str(root)]) == 1
    assert validator.main([str(root), "--write"]) == 0
    assert validator.main([str(root)]) == 0
