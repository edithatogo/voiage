"""Governance-mirror tests for the C13 contract automation surface."""

from pathlib import Path

from scripts.repo_harness import check_contract_governance

ROOT = Path(__file__).resolve().parents[1]


def test_contract_governance_is_mirrored_across_runner_surfaces() -> None:
    assert check_contract_governance(ROOT) == []


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
