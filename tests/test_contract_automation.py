"""Governance-mirror tests for the C13 contract automation surface."""

from pathlib import Path

from scripts.repo_harness import check_contract_governance

ROOT = Path(__file__).resolve().parents[1]


def test_contract_governance_is_mirrored_across_runner_surfaces() -> None:
    assert check_contract_governance(ROOT) == []


def test_hosted_mirror_validation_checks_out_the_manifest_commit() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    manifest = (
        ROOT / "specs/integration/vop-voiage/governance/UPSTREAM.json"
    ).read_text(encoding="utf-8")
    assert "ref: 51a61340dde73868c319f97a7ce436b0650255be" in workflow
    assert (
        '"canonical_git_commit": "51a61340dde73868c319f97a7ce436b0650255be"' in manifest
    )
    assert "VOP_GOVERNANCE_REPOSITORY_ROOT: .upstream/vop_poc_nz" in workflow
    frontier = workflow.split("  frontier-contract:", 1)[1].split("  version-sync:", 1)[
        0
    ]
    assert frontier.index("Install dependencies") < frontier.index(
        "Checkout pinned VOP governance source"
    )


def test_missing_contract_task_is_reported(tmp_path: Path) -> None:
    for relative_path in (
        "pyproject.toml",
        "noxfile.py",
        "pixi.toml",
        ".github/workflows/ci.yml",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# empty\n", encoding="utf-8")
    findings = check_contract_governance(tmp_path)
    assert {item.path for item in findings} == {
        ".github/workflows/ci.yml",
        "noxfile.py",
        "pixi.toml",
        "pyproject.toml",
    }
