import json
from pathlib import Path

from scripts.refresh_binding_registry_audit import (
    Channel,
    _evaluator_go_module_proxy,
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
    assert "Python `voiage` on PyPI returned `404`" in audit_text
    assert "TypeScript `@voiage/core` on npm returned `404`" in audit_text
    assert "Rust `voiage-core` on crates.io returned `404`" in audit_text
    assert ".NET `Voiage.Core` on NuGet returned `404`" in audit_text
    assert "R `voiageR` on CRAN returned `404`" in audit_text
    assert "reported no released versions" in audit_text
    assert (
        "submissions have not been confirmed as published for these package names"
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
        "typescript",
        "go",
        "rust",
        "dotnet",
    }
    assert snapshot["snapshot"]["python"]["package"] == "voiage"
    assert snapshot["snapshot"]["typescript"]["package"] == "@voiage/core"


def test_go_module_proxy_classifier_no_released_versions() -> None:
    status = _evaluator_go_module_proxy(
        200,
        b'["not found: module github.com/edithatogo/voiage/bindings/go: no such file"]',
        None,
    )
    assert status == "no_released_versions"


def test_go_module_proxy_classifier_confirmed_with_version_lines() -> None:
    status = _evaluator_go_module_proxy(200, b"v0.1.0\nv0.1.1\n", None)
    assert status == "confirmed"


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
