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
    assert (
        "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65"
        in workflow_text
    )

    assert (
        "The Julia binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "Julia General registry submission/approval remains external/manual."
        in checklist_text
    )
