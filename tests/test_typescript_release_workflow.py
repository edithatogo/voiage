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
    assert (
        "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65"
        in workflow_text
    )

    assert (
        "The TypeScript binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "npm publication is automated on `typescript-v*` tags with provenance."
        in checklist_text
    )
