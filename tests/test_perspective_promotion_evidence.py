from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "specs" / "frontier" / "perspective" / "v1"


def test_perspective_catalog_and_promotion_gates_are_explicit() -> None:
    catalog = json.loads((ROOT / "fixtures" / "perspective-catalog.json").read_text())
    evidence = json.loads((ROOT / "fixtures" / "evidence.json").read_text())

    assert catalog["method_maturity"] == "fixture-backed"
    assert catalog["stable_claim_allowed"] is False
    assert [item["id"] for item in catalog["perspectives"]] == [
        "payer",
        "societal",
        "patient",
        "provider",
        "regulator",
        "equity-weighted",
        "custom",
    ]
    assert catalog["assumptions"]["objective_uncertainty"]
    assert evidence["open_data"]["status"] == "blocked_external"
    assert evidence["open_data"]["next_action"]
    assert evidence["parity"]["python"] == "verified"
    assert evidence["stable_claim_allowed"] is False

    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        if artifact["sha256"] != "pending":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_perspective_readme_preserves_unsupported_estimand_boundary() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "current-information estimand" in readme
    assert "Perfect perspective information" in readme
    assert "Partial or sample perspective information" in readme
    assert "Unsupported rows must remain fail-closed" in readme


def test_perspective_capability_dispositions_are_explicit() -> None:
    capabilities = json.loads((ROOT / "capabilities.json").read_text())
    assert capabilities["execution"] == {
        "python": "executable-experimental",
        "rust": "unsupported",
        "r": "unsupported",
        "julia": "unsupported",
        "mojo": "external-boundary",
    }
    assert capabilities["installed_shared_fixture"]["status"] == "pending"
    assert capabilities["stable_claim_allowed"] is False


def test_cli_reference_links_to_repository_perspective_contracts() -> None:
    cli_reference = (
        ROOT.parents[3] / "docs/astro-site/src/content/docs/cli-reference.mdx"
    ).read_text()
    prefix = "../../../../../specs/frontier/perspective/v1/"
    assert f"{prefix}README.md" in cli_reference
    assert f"{prefix}capabilities.json" in cli_reference
