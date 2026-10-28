"""Producer-compatibility absence controls for the complete reference policy."""

from pathlib import Path

import pytest
import yaml

from scripts.check_reference_policy import check_files, check_repository, digest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "text",
    [
        '"import pandas"',
        '"xarray.Dataset"',
        '"pd.DataFrame([])"',
        '"xr.DataArray([])"',
        '"results.itertuples()"',
    ],
)
@pytest.mark.parametrize("suffix", [".py", ".ipynb", ".md", ".yml"])
def test_new_reference_including_generated_strings_fails(
    tmp_path: Path, text: str, suffix: str
) -> None:
    name = "new" + suffix
    (tmp_path / name).write_text(text)
    result = check_files(tmp_path, [name], {"files": {}})
    assert not result["passed"]
    assert len(result["unapproved"]) == 1


def test_exact_line_permissions_cannot_hide_added_or_duplicate_reference(
    tmp_path: Path,
) -> None:
    text = 'producer = "pandas"'
    path = tmp_path / "producer.py"
    path.write_text(text + "\n")
    policy = {
        "files": {
            path.name: {
                "category": "producer_compatibility",
                "reason": "External caller test",
                "lines": [{"sha256": digest(text.encode()), "count": 1}],
            }
        }
    }
    assert check_files(tmp_path, [path.name], policy)["passed"]
    path.write_text(text + "\n" + text + "\n")
    assert not check_files(tmp_path, [path.name], policy)["passed"]
    path.write_text('producer = "xarray"\n')
    assert not check_files(tmp_path, [path.name], policy)["passed"]


def test_frozen_evidence_is_immutable_and_missing_entries_fail(tmp_path: Path) -> None:
    path = tmp_path / "record.md"
    path.write_text("Historical pandas observation\n")
    policy = {
        "files": {
            path.name: {
                "category": "frozen_historical_evidence",
                "reason": "Dated evidence",
                "sha256": digest(path.read_bytes()),
            }
        }
    }
    assert check_files(tmp_path, [path.name], policy)["passed"]
    path.write_text("Changed pandas observation\n")
    assert not check_files(tmp_path, [path.name], policy)["passed"]
    assert not check_files(tmp_path, [], policy)["passed"]


def test_symlink_and_empty_inventory_fail_closed(tmp_path: Path) -> None:
    assert not check_files(tmp_path, [], {"files": {}})["passed"]
    (tmp_path / "link.py").symlink_to(ROOT / "voiage/schema.py")
    assert not check_files(tmp_path, ["link.py"], {"files": {}})["passed"]


def test_current_repository_reference_policy_passes() -> None:
    assert check_repository(ROOT)["passed"]


def test_existing_protected_lint_and_local_tox_run_reference_gate() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    lint = workflow["jobs"]["lint"]
    commands = "\n".join(step.get("run", "") for step in lint["steps"])
    assert "python scripts/check_reference_policy.py" in commands
    assert not lint.get("continue-on-error", False)
    assert "python scripts/check_reference_policy.py" in (ROOT / "tox.ini").read_text()
