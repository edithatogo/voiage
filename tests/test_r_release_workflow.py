from pathlib import Path


def test_r_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (root / ".github" / "workflows" / "bindings-release.yml").read_text(
        encoding="utf-8"
    )
    ci_workflow_text = (root / ".github" / "workflows" / "bindings-ci.yml").read_text(
        encoding="utf-8"
    )
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text(encoding="utf-8")

    assert "r-v*" in workflow_text
    assert "R CMD build ." in workflow_text
    assert (
        'rcmdcheck::rcmdcheck(args = "--no-manual", error_on = "warning")'
        in workflow_text
    )
    assert "Rscript tools/build-manual.R" in workflow_text
    assert (
        "r-lib/actions/setup-tinytex@d3c5be51b12e724e68f33216ca3c148b66d5f0b6"
        in workflow_text
    )
    assert (
        "r-lib/actions/setup-tinytex@d3c5be51b12e724e68f33216ca3c148b66d5f0b6"
        in ci_workflow_text
    )
    assert (
        "softprops/action-gh-release@fe965f7af51af5f2602596916f38a38df2e33de0"
        in workflow_text
    )
    assert "tag_name: ${{ github.ref_name }}" in workflow_text
    assert "generate_release_notes: true" in workflow_text
    assert "GH_TOKEN: ${{ github.token }}" in workflow_text

    assert (
        "The R package remains the thin reticulate bridge over the shared contract."
        in checklist_text
    )
    assert "CRAN submission remains external/manual." in checklist_text
