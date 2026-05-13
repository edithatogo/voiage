from pathlib import Path


def test_rust_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (root / ".github" / "workflows" / "bindings-release.yml").read_text()
    checklist_text = (root / "docs" / "release" / "binding-submission-checklist.md").read_text()

    assert "rust-v*" in workflow_text
    assert "cargo fmt --check" in workflow_text
    assert "cargo clippy --all-targets --locked -- -D warnings" in workflow_text
    assert "cargo publish --locked" in workflow_text
    assert "softprops/action-gh-release@v2" in workflow_text

    assert "The Rust crate remains the canonical execution core and contract owner." in checklist_text
    assert "crates.io publication is automated on `rust-v*` tags when credentials are present." in checklist_text
