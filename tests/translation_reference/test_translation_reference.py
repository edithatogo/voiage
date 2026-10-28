"""Executable contracts for the source-derived migration reference."""

from pathlib import Path

import pytest

from scripts.check_translation_reference import load_cases, run_case, validate

ROOT = Path(__file__).parents[2]
IDS = [
    "index",
    "columns",
    "rows",
    "grouping",
    "concat",
    "alignment",
    "reduction",
    "rolling",
    "resample",
    "interpolation",
    "nulls",
    "float-format",
    "attrs",
    "nd-interpolation",
]


def test_derived_inventory_and_documentation_are_complete() -> None:
    assert validate(ROOT) == []
    assert [row["id"] for row in load_cases(ROOT)] == IDS


@pytest.mark.parametrize("row_id", IDS)
def test_polars_target(row_id: str) -> None:
    row = next(row for row in load_cases(ROOT) if row["id"] == row_id)
    run_case(row, "target")


@pytest.mark.parametrize("row_id", IDS)
def test_external_legacy_oracle(row_id: str) -> None:
    row = next(row for row in load_cases(ROOT) if row["id"] == row_id)
    pytest.importorskip(
        row["legacy_module"],
        reason="External oracle is unavailable; it is not a runtime dependency",
    )
    run_case(row, "legacy")


@pytest.mark.parametrize(
    "damage", ["excerpt", "citation", "publication", "missing-row", "notice"]
)
def test_reference_drift_is_rejected(tmp_path: Path, damage: str) -> None:
    import json
    import shutil

    from scripts.check_translation_reference import NATIVE_FILES, PAGE, TRACK

    for name in [
        *(str(TRACK / p.name) for p in (ROOT / TRACK).glob("*.json")),
        *NATIVE_FILES,
        str(TRACK / "consumer-notices/tobac-LICENSE"),
        str(TRACK / "consumer-notices/tobac-NOTICE"),
        str(PAGE),
    ]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    if damage == "excerpt":
        path = tmp_path / TRACK / "tobac-consumer-idioms.json"
        data = json.loads(path.read_text())
        data["files"][0]["snippets"][0]["text"] = "Changed source\n"
        path.write_text(json.dumps(data))
    elif damage == "notice":
        (tmp_path / TRACK / "consumer-notices/tobac-NOTICE").write_text(
            "Changed attribution\n"
        )
    elif damage == "publication":
        (tmp_path / PAGE).write_text("Unqualified wording\n")
    else:
        path = tmp_path / TRACK / "semantic-cases.json"
        data = json.loads(path.read_text())
        if damage == "citation":
            data["rows"][0]["citations"][0]["id"] = "absent:1:index"
        else:
            data["rows"].pop()
        path.write_text(json.dumps(data))
    assert validate(tmp_path)


def test_wrong_semantic_result_is_rejected() -> None:
    row = dict(load_cases(ROOT)[0], target_code="result = [999]\n")
    with pytest.raises(AssertionError, match="index:target"):
        run_case(row, "target")
