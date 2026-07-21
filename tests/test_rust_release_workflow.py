from pathlib import Path


def test_rust_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert "rust-v*" in workflow_text
    assert "cargo fmt --check" in workflow_text
    assert "cargo clippy --all-targets --locked -- -D warnings" in workflow_text
    assert "Validate Rust core release" in workflow_text
    assert "publish=false" in workflow_text
    assert "cargo publish --locked" not in workflow_text
    assert 'gh release create "$GITHUB_REF_NAME"' in workflow_text
    assert "--generate-notes --verify-tag" in workflow_text
    assert "GH_TOKEN: ${{ github.token }}" in workflow_text

    assert (
        "The Rust crate remains the canonical execution core and contract owner."
        in checklist_text
    )
    assert (
        "The Rust workspace is validated and released through GitHub Releases;"
        in checklist_text
    )
