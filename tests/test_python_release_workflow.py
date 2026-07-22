from __future__ import annotations

from pathlib import Path

import yaml


def test_release_checklist_matches_the_fail_closed_v1_workflow() -> None:
    checklist = Path("RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for required in (
        "CI=true uv run tox -q",
        "pnpm install --frozen-lockfile",
        "maturin build --locked --release",
        "signed annotated `v1.0.0` tag",
        "TestPyPI",
        "expected_wheel_sha256",
        "expected_sdist_sha256",
        "SBOM",
        "provenance",
        "conda-forge",
        "Julia General",
        "approved R registry",
    ):
        assert required in checklist

    for stale in (
        "CHANGELOG.md",
        "python -m build",
        "automatically publish to PyPI",
        "Docker Hub",
        "docker run",
        "v0.X.X",
    ):
        assert stale not in checklist


def test_python_release_workflow_builds_and_publishes_aggregated_artifacts() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    conda_workflow = Path(".github/workflows/conda-update.yml").read_text(
        encoding="utf-8"
    )

    assert 'tags:\n      - "v*"' in release_workflow
    assert "Publish to TestPyPI" in release_workflow
    assert "Publish to PyPI" in release_workflow
    assert release_workflow.index("Publish to TestPyPI") < release_workflow.index(
        "Publish to PyPI"
    )
    assert "maturin build --locked --release" in release_workflow
    assert "maturin sdist --out dist" in release_workflow
    assert "maturin sdist --locked" not in release_workflow
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '{ path = "specs/**/*.json", format = "sdist" }' in pyproject
    assert (
        "PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b"
        in release_workflow
    )
    assert "os: ubuntu-24.04" in release_workflow
    assert "os: macos-14" in release_workflow
    assert "os: windows-2025" in release_workflow
    assert "cp312-abi3" in release_workflow
    assert "voiage/_core." in release_workflow
    assert "workflow_dispatch:" in release_workflow
    assert (
        'description: "Existing signed release tag to stage or publish"'
        in release_workflow
    )
    assert (
        "RELEASE_TAG: ${{ inputs.release_tag || github.ref_name }}" in release_workflow
    )
    assert (
        'python -m voiage.versioning --release-tag "$RELEASE_TAG"' in release_workflow
    )
    assert (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
        in release_workflow
    )
    assert (
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
        in release_workflow
    )
    assert (
        "actions/attest-build-provenance@0f67c3f4856b2e3261c31976d6725780e5e4c373"
        in release_workflow
    )
    assert (
        "pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b"
        in release_workflow
    )
    assert (
        'gh release create "$RELEASE_TAG" --draft --generate-notes --verify-tag --target "$TAG_SHA"'
        in release_workflow
    )
    assert "permissions: {}" in release_workflow
    assert "release:\n    types: [published]" in conda_workflow
    assert "startsWith(github.event.release.tag_name, 'v')" in conda_workflow
    assert "Update conda-recipe/meta.yaml" in conda_workflow
    assert "steps.release.outputs.version" in conda_workflow
    assert "steps.source.outputs.sha256" in conda_workflow


def test_conda_release_recipe_is_single_native_maturin_contract() -> None:
    recipe = Path("conda-recipe/meta.yaml").read_text(encoding="utf-8")

    assert not Path("conda/meta.yaml").exists()
    assert "noarch: python" not in recipe
    assert "{{ compiler('rust') }}" in recipe
    assert "maturin >=1.9,<2.0" in recipe
    assert "python >= {{ python_min }}" in recipe
    assert "sha256: UPDATE_ON_RELEASE" in recipe

    workflow = Path(".github/workflows/conda-update.yml").read_text(encoding="utf-8")
    assert 'meta = Path("conda-recipe/meta.yaml")' in workflow
    assert 'r"sha256: [0-9a-fA-F_]+"' in workflow


def test_python_release_publish_job_is_exact_tag_and_least_privilege() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "resolve-tag:" in release_workflow
    assert (
        'git ls-remote --exit-code --refs origin "refs/tags/$RELEASE_TAG"'
        in release_workflow
    )
    assert (
        'git rev-parse --verify "refs/tags/$RELEASE_TAG^{commit}"' in release_workflow
    )
    assert "TAG_SHA: ${{ needs.resolve-tag.outputs.tag_sha }}" in release_workflow
    assert "ref: ${{ needs.resolve-tag.outputs.tag_sha }}" in release_workflow
    assert "needs: [resolve-tag, wheels, sdist]" in release_workflow
    assert "contents: write" in release_workflow
    assert "id-token: write" in release_workflow
    assert "attestations: write" in release_workflow
    assert "merge-multiple: true" in release_workflow
    assert (
        'python -m voiage.versioning --release-tag "$RELEASE_TAG"' in release_workflow
    )
    assert "wheel-manifest-*.txt" in release_workflow
    assert "expected one wheel per architecture manifest" in release_workflow
    assert "expected exactly one source distribution" in release_workflow
    assert "attest:" in release_workflow
    assert "test-pypi:" in release_workflow
    assert "pypi:" in release_workflow
    assert "github-release:" in release_workflow
    assert "needs: [resolve-tag, stage]" in release_workflow
    assert "needs: [resolve-tag, reviewed-payload]" in release_workflow
    assert "needs: [resolve-tag, test-pypi]" in release_workflow
    assert "needs: [resolve-tag, test-pypi-smoke]" in release_workflow
    assert "needs: [resolve-tag, pypi]" in release_workflow


def test_python_release_keeps_staging_separate_from_publication() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "publish:" in release_workflow
    assert "type: boolean\n        default: false" in release_workflow
    assert "stage:\n    name: Build and Stage Private Draft" in release_workflow
    assert (
        "test-pypi:\n    name: Publish Reviewed Draft to TestPyPI" in release_workflow
    )
    assert (
        "if: github.event_name == 'workflow_dispatch' && inputs.publish"
        in release_workflow
    )
    stage_job = release_workflow.index(
        "stage:\n    name: Build and Stage Private Draft"
    )
    publish_job = release_workflow.index(
        "test-pypi:\n    name: Publish Reviewed Draft to TestPyPI"
    )
    assert stage_job < publish_job
    assert "environment: testpypi" not in release_workflow[:publish_job]
    assert "expected_wheel_sha256" in release_workflow
    assert "expected_sdist_sha256" in release_workflow
    assert "sha256sum" in release_workflow
    assert 'gh release view "$RELEASE_TAG" --json isDraft' in release_workflow
    assert (
        "if: github.event_name != 'workflow_dispatch' || !inputs.publish"
        in release_workflow
    )
    assert (
        release_workflow.count(
            "gh release download \"$RELEASE_TAG\" --repo \"$GITHUB_REPOSITORY\" --dir dist --pattern '*.whl' --pattern '*.tar.gz'"
        )
        == 1
    )
    assert (
        "reviewed-payload:\n    name: Validate Reviewed Private Draft"
        in release_workflow
    )
    assert release_workflow.count("name: reviewed-release-payload") == 3
    assert (
        'gh release view "$RELEASE_TAG" --repo "$GITHUB_REPOSITORY" --json isDraft'
        in release_workflow
    )
    assert 'gh release edit "$RELEASE_TAG" --draft=false' in release_workflow
    assert "git cat-file -t" in release_workflow
    assert ".verification.verified == true" in release_workflow


def test_release_wheels_are_installed_and_exercised_before_upload() -> None:
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    exercise = release_workflow.index("Exercise wheel outside checkout")
    upload = release_workflow.index("Upload native wheel")
    assert exercise < upload
    assert "tests/packaging/test_wheel_black_box.py" in release_workflow
    assert "--import-mode=importlib" in release_workflow
    assert "WHEEL_VENV" in release_workflow


def test_release_builds_embed_immutable_source_and_platform_provenance() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    for job_name in ("wheels", "sdist"):
        job = workflow["jobs"][job_name]
        rendered = str(job)
        assert job["env"]["TAG_SHA"] == "${{ needs.resolve-tag.outputs.tag_sha }}"
        assert (
            job["env"]["VOIAGE_SOURCE_REVISION"]
            == "${{ needs.resolve-tag.outputs.tag_sha }}"
        )
        assert job["env"]["VOIAGE_SOURCE_CLEAN"] == "true"
        assert "VOIAGE_SOURCE_TREE_GIT_OID" in rendered
        assert "EXPECTED_SOURCE_TREE_GIT_OID" in rendered
        assert "SOURCE_DATE_EPOCH" in rendered
        assert "git show -s --format=%ct HEAD" in rendered
        assert "EXPECTED_SOURCE_REVISION" in rendered
        assert "EXPECTED_PLATFORM_SUFFIX" in rendered

    sdist_steps = workflow["jobs"]["sdist"]["steps"]
    rendered_sdist_steps = str(sdist_steps)
    assert "scripts/embed_sdist_provenance.py" in rendered_sdist_steps
    reconcile = next(
        step
        for step in sdist_steps
        if step["name"] == "Reconcile extracted Rust lockfile"
    )
    assert reconcile["working-directory"] == ".sdist-source"
    assert "cargo generate-lockfile --manifest-path rust/Cargo.toml" in reconcile["run"]
    extracted_build = next(
        step
        for step in sdist_steps
        if step["name"] == "Build manylinux2014 wheel from clean extracted sdist"
    )
    assert extracted_build["with"]["working-directory"] == ".sdist-source"
    assert extracted_build["with"]["manylinux"] == "2014"
    assert extracted_build["with"]["docker-options"] == "-e SOURCE_DATE_EPOCH"
    assert "VOIAGE_" not in extracted_build["with"]["docker-options"]


def test_release_wheel_resolves_metadata_in_a_blank_environment() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["wheels"]["steps"]
    install = next(
        step for step in steps if step["name"] == "Install wheel into clean environment"
    )

    assert "locked-requirements.txt" not in install["run"]
    assert "--no-deps" not in install["run"]
    assert "dist/*.whl" in install["run"]
    assert "uv pip check --python .release-wheel-venv" in install["run"]


def test_sdist_derived_wheel_is_installed_and_smoked() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["sdist"]["steps"]
    names = [step["name"] for step in steps]

    assert "Install and smoke sdist-derived wheel" in names
    smoke = next(
        step
        for step in steps
        if step["name"] == "Install and smoke sdist-derived wheel"
    )
    assert (
        "uv pip install --python .sdist-wheel-venv .sdist-wheel/*.whl" in smoke["run"]
    )
    assert "uv pip check --python .sdist-wheel-venv" in smoke["run"]
    assert "test_wheel_black_box.py --import-mode=importlib" in smoke["run"]


def test_release_validates_real_wheel_platform_suffixes() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["wheels"]["steps"]
    validation = next(
        step for step in steps if step["name"] == "Verify wheel tag and native module"
    )

    matrix = workflow["jobs"]["wheels"]["strategy"]["matrix"]["include"]
    assert {entry["platform"]: entry["platform-suffix"] for entry in matrix} == {
        "linux-x86_64": "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "macos-arm64": "macosx_11_0_arm64",
        "windows-x86_64": "win_amd64",
    }
    assert (
        validation["env"]["EXPECTED_PLATFORM_SUFFIX"] == "${{ matrix.platform-suffix }}"
    )
    assert "WHEEL_PLATFORM" in validation["run"]
    assert "endswith(expected)" in validation["run"]


def test_release_architecture_policy_is_explicit_and_extensible() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    release_workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "[tool.voiage.release]" in pyproject
    assert 'python-abi = "cp312-abi3"' in pyproject
    assert 'architecture-policy = "native-runner"' in pyproject
    assert (
        'supported-native-architectures = ["linux-x86_64", "macos-arm64", "windows-x86_64"]'
        in pyproject
    )
    assert "platform: linux-x86_64" in release_workflow
    assert "platform: macos-arm64" in release_workflow
    assert "platform: windows-x86_64" in release_workflow
    assert (
        "EXPECTED_NATIVE_ARCHITECTURES: linux-x86_64,macos-arm64,windows-x86_64"
        in release_workflow
    )
    assert "wheel-manifest-${{ matrix.platform }}.txt" in release_workflow


def test_registry_jobs_revalidate_tag_immediately_before_publish() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )

    for job_name in ("test-pypi", "pypi"):
        steps = workflow["jobs"][job_name]["steps"]
        assert "Revalidate remote tag binding immediately" in steps[-2]["name"]
        assert steps[-1]["name"].startswith("Publish to ")
        assert "refs/tags/$RELEASE_TAG^{commit}" in steps[-2]["run"]


def test_testpypi_smoke_is_bounded_and_blocks_pypi() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    smoke = workflow["jobs"]["test-pypi-smoke"]
    rendered = str(smoke)

    assert smoke["needs"] == ["resolve-tag", "test-pypi"]
    assert workflow["jobs"]["pypi"]["needs"] == [
        "resolve-tag",
        "test-pypi-smoke",
    ]
    assert "for attempt in 1 2 3 4 5 6" in rendered
    assert "sleep 20" in rendered
    assert "https://test.pypi.org/simple/" in rendered
    assert "--no-deps" in rendered
    assert "test_wheel_black_box.py --import-mode=importlib" in rendered


def test_python_release_docs_keep_conda_boundary_explicit() -> None:
    release_docs = Path("docs/release/polyglot-bindings.md").read_text(encoding="utf-8")
    checklist = Path("docs/release/binding-submission-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "Python uses repository tags" in release_docs
    assert "conda-forge feedstock" in release_docs
    assert "The Python façade remains the stable release surface" in checklist
    assert "External conda-forge feedstock merge remains manual" in checklist
