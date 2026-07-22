from pathlib import Path


def test_arxiv_readiness_pipeline_is_non_submitting_and_pinned() -> None:
    root = Path.cwd()
    workflow = (root / ".github/workflows/arxiv-readiness.yml").read_text()
    manifest = (root / "paper/arxiv/manifest.json").read_text()
    builder = (root / "scripts/build_arxiv_submission.py").read_text()

    assert "workflow_dispatch:" in workflow
    assert "permissions: {}" in workflow
    assert "pandoc texlive-latex-base texlive-latex-extra" in workflow
    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990" in workflow
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    )
    assert "submission_performed" in builder
    assert "out_of_scope_for_this_execution" in manifest
    assert "arxiv.org" not in builder
