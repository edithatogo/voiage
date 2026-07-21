from __future__ import annotations

import json
from pathlib import Path

REGISTRY_ROOT = Path("specs/dataset-registry")


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


# Registry structure tests


def test_dataset_registry_directory_exists() -> None:
    """The dataset registry directory must exist under specs/."""
    assert REGISTRY_ROOT.is_dir(), (
        f"Dataset registry directory not found at {REGISTRY_ROOT}. "
        "Create specs/dataset-registry/ with the required structure."
    )


def test_dataset_registry_has_registry_json() -> None:
    """The dataset registry must have a registry.json file at its root."""
    registry_path = REGISTRY_ROOT / "registry.json"
    assert registry_path.is_file(), (
        f"registry.json not found at {registry_path}. "
        "Define the dataset registry with source URL, license, citation, "
        "transform command, snapshot hash, schema, and method tags."
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert isinstance(registry, dict)
    for entry_key, entry in registry.items():
        assert isinstance(entry, dict), f"Entry {entry_key!r} must be a dict"
        assert "source_url" in entry, f"Entry {entry_key!r} missing source_url"
        assert "license" in entry, f"Entry {entry_key!r} missing license"
        assert "citation" in entry, f"Entry {entry_key!r} missing citation"
        assert "snapshot_hash" in entry, f"Entry {entry_key!r} missing snapshot_hash"
        assert "schema" in entry, f"Entry {entry_key!r} missing schema"
        assert "method_tags" in entry, f"Entry {entry_key!r} missing method_tags"


def test_dataset_registry_has_transform_scripts() -> None:
    """Transform scripts directory must exist for live refresh."""
    transforms_dir = REGISTRY_ROOT / "transforms"
    assert transforms_dir.is_dir(), (
        f"Transform scripts directory not found at {transforms_dir}. "
        "Create specs/dataset-registry/transforms/ with download-and-transform "
        "scripts for each registered dataset."
    )


def test_dataset_registry_has_snapshots() -> None:
    """Committed snapshot directory must exist for offline-capable tests."""
    snapshots_dir = REGISTRY_ROOT / "snapshots"
    assert snapshots_dir.is_dir(), (
        f"Snapshot directory not found at {snapshots_dir}. "
        "Create specs/dataset-registry/snapshots/ with small committed "
        "snapshots so default tests do not require live network access."
    )


# Synthetic dataset tests


def test_synthetic_datasets_directory_exists() -> None:
    """Synthetic datasets must live under specs/dataset-registry/synthetic/."""
    synthetic_dir = REGISTRY_ROOT / "synthetic"
    assert synthetic_dir.is_dir(), (
        f"Synthetic dataset directory not found at {synthetic_dir}. "
        "Create specs/dataset-registry/synthetic/ with deterministic "
        "synthetic datasets for each analysis family."
    )


ANALYSIS_FAMILIES = (
    "evpi",
    "evppi",
    "evsi",
    "enbs",
    "nma",
    "structural_voi",
    "calibration_voi",
    "adaptive_trial_voi",
    "portfolio_voi",
    "sequential_voi",
    "ceac",
    "heterogeneity",
)


def test_every_analysis_family_has_synthetic_dataset() -> None:
    """Each analysis family must have at least one synthetic dataset."""
    synthetic_dir = REGISTRY_ROOT / "synthetic"
    if not synthetic_dir.is_dir():
        return
    manifest_path = synthetic_dir / "manifest.json"
    assert manifest_path.is_file(), (
        f"Synthetic dataset manifest not found at {manifest_path}. "
        "Create specs/dataset-registry/synthetic/manifest.json listing "
        "all synthetic datasets and the analysis families they cover."
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    covered_families = set(manifest.keys())
    missing = set(ANALYSIS_FAMILIES) - covered_families
    assert not missing, (
        f"Missing synthetic datasets for analysis families: {missing}. "
        f"Each family in {ANALYSIS_FAMILIES} must have at least one "
        "synthetic dataset registered in the manifest."
    )


# Open data source mapping tests


REQUIRED_OPEN_DATA_SOURCES = (
    "NHANES",
    "MEPS",
    "ClinicalTrials.gov",
    "World Bank",
    "NOAA",
    "EPA",
)


def test_required_open_data_sources_are_mapped() -> None:
    """Required open data sources must have a mapping entry in the registry."""
    registry_path = REGISTRY_ROOT / "registry.json"
    if not registry_path.is_file():
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for source in REQUIRED_OPEN_DATA_SOURCES:
        found = any(
            source.lower() in entry.get("source_url", "").lower()
            or source.lower() in key.lower().replace("_", " ")
            for key, entry in registry.items()
        )
        assert found, (
            f"Required open data source {source!r} not found in registry.json."
        )


# Snapshot isolation tests


def test_snapshot_files_are_committed() -> None:
    """Snapshot files must be small and committed for offline test use."""
    snapshots_dir = REGISTRY_ROOT / "snapshots"
    if not snapshots_dir.is_dir():
        return
    snapshot_files = list(snapshots_dir.iterdir())
    assert len(snapshot_files) > 0, (
        f"No snapshot files found in {snapshots_dir}. "
        "Add at least one committed snapshot file for offline test use."
    )
    for f in snapshot_files:
        size_kb = f.stat().st_size / 1024
        assert size_kb < 512, (
            f"Snapshot file {f.name} is {size_kb:.1f} KB. "
            "Snapshots should be small (<512 KB) to keep the repo lean."
        )


# Open data sources documentation test


def test_open_data_documentation_exists() -> None:
    """Docs must explain data licenses, citations, transforms, and refresh policy."""
    doc_path = Path("docs/astro-site/src/content/docs/dataset-registry.mdx")
    assert doc_path.is_file(), (
        "No dataset-registry documentation found in the Astro site."
    )
    doc_text = doc_path.read_text(encoding="utf-8").lower()
    for keyword in ("license", "citation", "transform", "refresh"):
        assert keyword in doc_text, (
            f"Keyword {keyword!r} not found in dataset-registry documentation."
        )
