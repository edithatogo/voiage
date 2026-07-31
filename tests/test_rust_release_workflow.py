from pathlib import Path

import yaml


def test_rust_release_workflow_and_checklist_align() -> None:
    root = Path.cwd()
    workflow_text = (
        root / ".github" / "workflows" / "bindings-release.yml"
    ).read_text()
    crates_workflow_text = (
        root / ".github" / "workflows" / "rust-crates-release.yml"
    ).read_text()
    checklist_text = (
        root / "docs" / "release" / "binding-submission-checklist.md"
    ).read_text()

    assert '"rust-v*"' not in workflow_text
    assert "contents: write" not in crates_workflow_text
    assert '"julia-v*"' in workflow_text
    assert '"r-v*"' in workflow_text
    assert '"rust-v*"' in crates_workflow_text
    assert "release_tag:" in crates_workflow_text
    assert "runs-on: ubuntu-24.04" in crates_workflow_text
    assert 'toolchain: "1.85"' in crates_workflow_text
    assert (
        '[[ "$RELEASE_TAG" =~ ^rust-v[0-9]+\\.[0-9]+\\.[0-9]+' in crates_workflow_text
    )
    assert 'git cat-file -t "refs/tags/$RELEASE_TAG"' in crates_workflow_text
    assert ".verification.verified == true" in crates_workflow_text
    assert "vars.RELEASE_TAGGER_EMAIL" in crates_workflow_text
    assert 'gh release download "v${release_version}"' in crates_workflow_text
    assert ".source_commit == $source_commit" in crates_workflow_text
    assert "Verify all public crate versions match the release tag" in (
        crates_workflow_text
    )
    assert "cargo publish --locked --package voiage-domain" in crates_workflow_text
    assert "cargo publish --locked --package voiage-diagnostics" in crates_workflow_text
    assert "cargo publish --locked --package voiage-numerics" in crates_workflow_text
    assert (
        "cargo publish --locked --package voiage-serialization" in crates_workflow_text
    )
    assert "wait_for_crate voiage-domain" in crates_workflow_text
    assert "wait_for_crate voiage-diagnostics" in crates_workflow_text
    assert "wait_for_crate voiage-numerics" in crates_workflow_text
    assert "wait_for_crate voiage-serialization" in crates_workflow_text
    assert "CARGO_REGISTRY_TOKEN" in crates_workflow_text
    assert "secrets.CARGO_REGISTRY_TOKEN" not in crates_workflow_text
    assert "rust-lang/crates-io-auth-action@" in crates_workflow_text

    assert (
        "The Rust crate remains the canonical execution core and contract owner."
        in checklist_text
    )
    assert "core crates are publishable on crates.io" in checklist_text
    assert "short-lived crates.io Trusted Publishing credential" in checklist_text


def test_rust_release_workflow_has_least_privilege_oidc_permissions() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/rust-crates-release.yml").read_text()
    )
    assert workflow["permissions"] == {}
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "crates-io"
    assert publish["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
