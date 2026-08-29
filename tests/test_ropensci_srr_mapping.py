from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
R_PACKAGE = ROOT / "r-package" / "voiageR"

GENERAL = {
    "G1.0",
    "G1.1",
    "G1.2",
    "G1.3",
    "G1.4",
    "G1.4a",
    "G1.5",
    "G1.6",
    "G2.0",
    "G2.0a",
    "G2.1",
    "G2.1a",
    "G2.2",
    "G2.3",
    "G2.3a",
    "G2.3b",
    "G2.4",
    "G2.4a",
    "G2.4b",
    "G2.4c",
    "G2.4d",
    "G2.4e",
    "G2.5",
    "G2.6",
    "G2.7",
    "G2.8",
    "G2.9",
    "G2.10",
    "G2.11",
    "G2.12",
    "G2.13",
    "G2.14",
    "G2.14a",
    "G2.14b",
    "G2.14c",
    "G2.15",
    "G2.16",
    "G3.0",
    "G3.1",
    "G3.1a",
    "G4.0",
    "G5.0",
    "G5.1",
    "G5.2",
    "G5.2a",
    "G5.2b",
    "G5.3",
    "G5.4",
    "G5.4a",
    "G5.4b",
    "G5.4c",
    "G5.5",
    "G5.6",
    "G5.6a",
    "G5.6b",
    "G5.7",
    "G5.8",
    "G5.8a",
    "G5.8b",
    "G5.8c",
    "G5.8d",
    "G5.9",
    "G5.9a",
    "G5.9b",
    "G5.10",
    "G5.11",
    "G5.11a",
    "G5.12",
}
BAYESIAN = {
    "BS1.0",
    "BS1.1",
    "BS1.2",
    "BS1.2a",
    "BS1.2b",
    "BS1.2c",
    "BS1.3",
    "BS1.3a",
    "BS1.3b",
    "BS1.4",
    "BS1.5",
    "BS2.1",
    "BS2.1a",
    "BS2.2",
    "BS2.3",
    "BS2.4",
    "BS2.5",
    "BS2.6",
    "BS2.7",
    "BS2.8",
    "BS2.9",
    "BS2.10",
    "BS2.11",
    "BS2.12",
    "BS2.13",
    "BS2.14",
    "BS2.15",
    "BS3.0",
    "BS3.1",
    "BS3.2",
    "BS4.0",
    "BS4.1",
    "BS4.2",
    "BS4.3",
    "BS4.4",
    "BS4.5",
    "BS4.6",
    "BS4.7",
    "BS5.0",
    "BS5.1",
    "BS5.2",
    "BS5.3",
    "BS5.4",
    "BS5.5",
    "BS6.0",
    "BS6.1",
    "BS6.2",
    "BS6.3",
    "BS6.4",
    "BS6.5",
    "BS7.0",
    "BS7.1",
    "BS7.2",
    "BS7.3",
    "BS7.4",
    "BS7.4a",
}


def _tagged_standards(tag: str, text: str) -> set[str]:
    tagged: set[str] = set()
    for line in text.splitlines():
        if f"@{tag}" not in line:
            continue
        match = re.search(r"\{([^}]+)\}", line)
        assert match, f"missing standard list on {line!r}"
        tagged.update(part.strip() for part in match.group(1).split(","))
    return tagged


def test_srr_mapping_is_complete_disjoint_and_has_no_todos() -> None:
    files = [
        *R_PACKAGE.rglob("*.R"),
        *R_PACKAGE.rglob("*.Rmd"),
        *R_PACKAGE.rglob("*.rs"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    addressed = _tagged_standards("srrstats", text)
    not_applicable = _tagged_standards("srrstatsNA", text)

    assert "@srrstatsTODO" not in text
    assert not addressed.intersection(not_applicable)
    assert addressed.union(not_applicable) == GENERAL.union(BAYESIAN)
    assert "Roxygen: list(markdown = TRUE" in (R_PACKAGE / "DESCRIPTION").read_text(
        encoding="utf-8"
    )


def test_every_na_standard_has_an_item_level_justification() -> None:
    mapping = (R_PACKAGE / "R" / "srr-stats-standards.R").read_text(encoding="utf-8")
    for line in mapping.splitlines():
        if "@srrstatsNA" in line:
            assert "{" in line and "}" in line
            assert len(line.split("}", 1)[1].strip()) >= 20
