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
    assert (
        'description: "Existing signed release tag to stage or publish"'
        in release_workflow
    )
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


def test_python_release_keeps_staging_separate_from_publication() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert (
        'description: "Existing signed release tag to stage or publish"'
        in release_workflow
    )
    assert "publish:\n        description:" in release_workflow
    assert "type: boolean\n        default: false" in release_workflow
    assert (
        "group: release-${{ inputs.release_tag || github.ref_name }}"
        in release_workflow
    )
    assert "stage:\n    name: Build and Stage Private Draft" in release_workflow
    assert "publish:\n    name: Publish Reviewed Draft" in release_workflow
    assert "needs: stage" in release_workflow
    assert (
        "if: github.event_name == 'workflow_dispatch' && inputs.publish"
        in release_workflow
    )
    assert release_workflow.index(
        "Create or refresh private draft release"
    ) < release_workflow.index("publish:\n    name: Publish Reviewed Draft")
    publish_job = release_workflow.index("publish:\n    name: Publish Reviewed Draft")
    assert "environment: pypi" not in release_workflow[:publish_job]
    assert "id-token: write" not in release_workflow[:publish_job]
    assert "environment: pypi" in release_workflow[publish_job:]
    assert "id-token: write" in release_workflow[publish_job:]
    assert "expected_wheel_sha256" in release_workflow
    assert "expected_sdist_sha256" in release_workflow
    assert "EXPECTED_WHEEL_SHA256" in release_workflow
    assert "EXPECTED_SDIST_SHA256" in release_workflow
    assert "sha256sum" in release_workflow
    assert "release-payload-${{ env.RELEASE_TAG }}" in release_workflow
    assert "Revalidate exact signed remote release tag" in release_workflow
    assert "Attest reviewed release artifacts and SBOM" in release_workflow
    assert 'gh release view "$RELEASE_TAG" --json isDraft' in release_workflow


def test_python_release_docs_keep_conda_boundary_explicit() -> None:
    release_docs = Path("docs/release/polyglot-bindings.md").read_text(encoding="utf-8")
    checklist = Path("docs/release/binding-submission-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "Python uses repository tags" in release_docs
    assert "conda-forge feedstock" in release_docs
    assert "The Python façade remains the stable release surface" in checklist
    assert "External conda-forge feedstock merge remains manual" in checklist
