from pathlib import Path


def test_rust_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    crates_workflow_text = (
        root / ".github" / "workflows" / "rust-crates-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert '"rust-v*"' not in workflow_text
    assert "contents: write" not in workflow_text
    assert '"julia-v*"' in workflow_text
    assert '"r-v*"' in workflow_text
    assert '"rust-v*"' in crates_workflow_text
    assert "cargo publish --locked --package voiage-domain" in crates_workflow_text
    assert "cargo publish --locked --package voiage-diagnostics" in crates_workflow_text
    assert "cargo publish --locked --package voiage-numerics" in crates_workflow_text
    assert (
        "cargo publish --locked --package voiage-serialization" in crates_workflow_text
    )
    assert "CARGO_REGISTRY_TOKEN" in crates_workflow_text

    assert (
        "The Rust crate remains the canonical execution core and contract owner."
        in checklist_text
    )
    assert "core crates are publishable on crates.io" in checklist_text
