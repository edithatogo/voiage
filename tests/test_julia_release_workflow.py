from pathlib import Path


def test_julia_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert "julia-v*" in workflow_text
    assert "Project.toml version" in workflow_text
    assert "Pkg.test()" in workflow_text
    assert 'gh release create "${GITHUB_REF_NAME}"' not in workflow_text
    assert "GH_TOKEN: ${{ github.token }}" not in workflow_text
    assert "contents: read" in workflow_text

    assert (
        "The Julia binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "Julia General registry submission/approval remains external/manual."
        in checklist_text
    )
