import json
from pathlib import Path

from scripts.refresh_binding_registry_audit import (
    Channel,
    _evaluator_external_manual,
    _snapshot_entry,
)


def _confirmed_evaluator(
    _status: int | None,
    _body: bytes | None,
    _error: str | None,
) -> str:
    return "confirmed"


def test_registry_audit_documents_current_live_status() -> None:
    root = Path.cwd()
    audit_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text(encoding="utf-8")

    assert "Live Registry Audit" in audit_text
    assert "Python `voiage` is present on PyPI" in audit_text
    assert "Rust core crates are publishable on crates.io" in audit_text
    assert "R `voiageR` on CRAN returned `404`" in audit_text
    assert "conda-forge-feedstock-publication_20260625" in audit_text
    assert "spack-package-merge-followthrough_20260625" in audit_text
    assert "e4s-inclusion-followthrough_20260625" in audit_text
    assert (
        "The retired Go, TypeScript, and .NET channels are not v1.0 targets."
        in audit_text
    )


def test_registry_audit_snapshot_matches_expected_channels() -> None:
    root = Path.cwd()
    snapshot = json.loads(
        (root / "docs" / "release" / "registry_audit_snapshot.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(snapshot["snapshot"].keys()) == {
        "python",
        "r",
        "julia",
        "rust",
        "conda_forge",
        "r_universe",
        "spack",
        "easybuild",
        "hpsf",
        "e4s",
    }
    assert snapshot["snapshot"]["python"]["package"] == "voiage"
    assert snapshot["snapshot"]["conda_forge"]["registry"] == "conda-forge"
    assert snapshot["snapshot"]["r_universe"]["registry"] == "r-universe"
    assert (
        "core crates are publishable on crates.io"
        in snapshot["snapshot"]["rust"]["notes"]
    )
    assert snapshot["snapshot"]["spack"]["registry"] == "Spack"
    assert snapshot["snapshot"]["easybuild"]["registry"] == "EasyBuild"
    assert snapshot["snapshot"]["hpsf"]["status"] == "external_manual"
    assert snapshot["snapshot"]["e4s"]["status"] == "external_manual"


def test_snapshot_entry_is_stable_shape() -> None:
    entry = _snapshot_entry(
        Channel(
            key="python",
            package="voiage",
            registry="PyPI",
            registry_url="https://example",
            check_url="https://example",
            notes="notes",
            evaluator=_confirmed_evaluator,
            confidence="high",
        ),
        "confirmed",
    )
    assert entry["package"] == "voiage"
    assert entry["status"] == "confirmed"
    assert "checked_at" in entry


def test_external_manual_classifier_keeps_curation_targets_conservative() -> None:
    assert _evaluator_external_manual(200, b"available", None) == "external_manual"
    assert _evaluator_external_manual(None, None, "network unavailable") == (
        "external_manual"
    )
