from pathlib import Path


def test_go_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert "bindings/go/v*" in workflow_text
    assert "go test -mod=readonly ./..." in workflow_text
    assert "go vet -mod=readonly ./..." in workflow_text
    assert (
        "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65"
        in workflow_text
    )

    assert (
        "The Go binding remains the thin adapter over the shared contract."
        in checklist_text
    )
    assert (
        "Module publication is driven by semver tag pushes and downstream module proxy indexing."
        in checklist_text
    )
