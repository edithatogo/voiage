from __future__ import annotations

from pathlib import Path


def test_python_release_workflow_matches_documented_tag_flow() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    conda_workflow = Path(".github/workflows/conda-update.yml").read_text(
        encoding="utf-8"
    )

    assert "tags:\n      - 'v*'" in release_workflow
    assert "Publish to TestPyPI" in release_workflow
    assert "Publish to PyPI" in release_workflow
    assert release_workflow.index("Publish to TestPyPI") < release_workflow.index(
        "Publish to PyPI"
    )
    assert "scripts/reproducible_build.py" in release_workflow
    assert "--dist-dir dist" in release_workflow
    assert "workflow_dispatch:" in release_workflow
    assert 'description: "Existing release tag to publish"' in release_workflow
    assert (
        "RELEASE_TAG: ${{ inputs.release_tag || github.ref_name }}" in release_workflow
    )
    assert 'gh release create "$RELEASE_TAG" --draft' in release_workflow
    assert 'gh release edit "$RELEASE_TAG" --draft=false' in release_workflow
    assert "--generate-notes --verify-tag" in release_workflow
    assert "GH_TOKEN: ${{ github.token }}" in release_workflow

    assert "release:\n    types: [published]" in conda_workflow
    assert "startsWith(github.event.release.tag_name, 'v')" in conda_workflow
    assert "Update conda/meta.yaml" in conda_workflow
    assert "steps.release.outputs.version" in conda_workflow
    assert "steps.source.outputs.sha256" in conda_workflow


def test_python_release_docs_keep_conda_boundary_explicit() -> None:
    release_docs = Path("docs/release/polyglot-bindings.md").read_text(encoding="utf-8")
    checklist = Path("docs/release/binding-submission-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "Python uses repository tags" in release_docs
    assert "conda-forge feedstock" in release_docs
    assert "The Python façade remains the stable release surface" in checklist
    assert "External conda-forge feedstock merge remains manual" in checklist
