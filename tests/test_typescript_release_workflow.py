from pathlib import Path


def test_typescript_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert "typescript-v*" in workflow_text
    assert "npm version --no-git-tag-version --allow-same-version" in workflow_text
    assert "npm publish --provenance --access public" in workflow_text
    assert 'gh release create "$GITHUB_REF_NAME"' in workflow_text
    assert "--generate-notes --verify-tag" in workflow_text
    assert "GH_TOKEN: ${{ github.token }}" in workflow_text

    assert (
        "The TypeScript binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "npm publication is automated on `typescript-v*` tags with provenance."
        in checklist_text
    )
