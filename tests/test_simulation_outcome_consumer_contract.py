"""Contract and integration tests for isolated simulation-outcome consumer (O08)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_simulation_outcome_consumer import (  # noqa: E402
    inspect_dependency_graph,
    parse_cargo_test_counts,
    run_isolated_consumer_check,
)


def _valid_metadata(unpacked_root: Path, version: str = "3.1.0") -> dict[str, object]:
    return {
        "packages": [
            {
                "name": "voiage-domain",
                "version": version,
                "manifest_path": str(
                    unpacked_root / f"voiage-domain-{version}" / "Cargo.toml"
                ),
            },
            {
                "name": "voiage-diagnostics",
                "version": version,
                "manifest_path": str(
                    unpacked_root / f"voiage-diagnostics-{version}" / "Cargo.toml"
                ),
            },
            {
                "name": "voiage-numerics",
                "version": version,
                "manifest_path": str(
                    unpacked_root / f"voiage-numerics-{version}" / "Cargo.toml"
                ),
            },
            {
                "name": "serde_json",
                "version": "1.0.154",
                "manifest_path": "/external/.cargo/registry/serde_json/Cargo.toml",
            },
        ]
    }


@pytest.mark.parametrize("producer_name", ["kairos", "paignia", "tobac-core", "vop_nz"])
def test_inspect_dependency_graph_rejects_producer_engines(
    tmp_path: Path, producer_name: str
) -> None:
    unpacked = tmp_path / "unpacked"
    metadata = _valid_metadata(unpacked)
    assert isinstance(metadata["packages"], list)
    metadata["packages"].append(
        {
            "name": producer_name,
            "version": "0.1.0",
            "manifest_path": str(tmp_path / "external" / producer_name / "Cargo.toml"),
        }
    )

    with pytest.raises(RuntimeError, match="forbidden producer engine"):
        inspect_dependency_graph(
            metadata,
            repo_root=REPO_ROOT,
            unpacked_root=unpacked,
            expected_version="3.1.0",
        )


def test_inspect_dependency_graph_rejects_in_repo_and_wrong_version_crates(
    tmp_path: Path,
) -> None:
    unpacked = tmp_path / "unpacked"
    in_repo = _valid_metadata(unpacked)
    assert isinstance(in_repo["packages"], list)
    in_repo["packages"][0]["manifest_path"] = str(
        REPO_ROOT / "rust" / "crates" / "voiage-domain" / "Cargo.toml"
    )
    with pytest.raises(RuntimeError, match="inside source repository"):
        inspect_dependency_graph(
            in_repo,
            repo_root=REPO_ROOT,
            unpacked_root=unpacked,
            expected_version="3.1.0",
        )

    wrong_version = _valid_metadata(unpacked, version="3.1.0")
    assert isinstance(wrong_version["packages"], list)
    wrong_version["packages"][2]["version"] = "0.0.1"
    with pytest.raises(RuntimeError, match="version 0.0.1 != expected 3.1.0"):
        inspect_dependency_graph(
            wrong_version,
            repo_root=REPO_ROOT,
            unpacked_root=unpacked,
            expected_version="3.1.0",
        )


def test_parse_cargo_test_counts_rejects_zero_and_failures() -> None:
    with pytest.raises(RuntimeError, match="zero passing tests"):
        parse_cargo_test_counts("test result: ok. 0 passed; 0 failed; 0 ignored;")
    with pytest.raises(RuntimeError, match="failure"):
        parse_cargo_test_counts("test result: ok. 3 passed; 1 failed; 0 ignored;")
    assert parse_cargo_test_counts("test result: ok. 4 passed; 0 failed;") == {
        "passed": 4,
        "failed": 0,
    }


def test_isolated_consumer_end_to_end_without_producer_engines(tmp_path: Path) -> None:
    work_dir = tmp_path / "isolated-work"
    receipt = run_isolated_consumer_check(REPO_ROOT, work_dir=work_dir)
    assert receipt["test_counts"]["passed"] >= 4
    assert receipt["test_counts"]["failed"] == 0
    assert receipt["synthetic_fixtures_only"] is True
    assert receipt["real_producer_qualified"] is False
    artifacts = {entry["name"]: entry for entry in receipt["crate_artifacts"]}
    assert set(artifacts) == {"voiage-domain", "voiage-diagnostics", "voiage-numerics"}
    assert all(len(entry["sha256"]) == 64 for entry in artifacts.values())
