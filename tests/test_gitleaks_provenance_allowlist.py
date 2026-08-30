"""A public gitlink exception must not hide other values, keys or files."""

import json
from pathlib import Path
import re
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    "conductor/tracks/v2_2_release_and_venue_submissions_20260830/"
    "venue-authentext-audit-20260830.json"
)


@pytest.mark.parametrize("mutation", ["none", "value", "key", "file", "suffix"])
def test_public_provenance_exception_is_exact(mutation: str) -> None:
    config = tomllib.loads((ROOT / ".gitleaks.toml").read_text())
    rule = next(item for item in config["rules"] if item["id"] == "generic-api-key")
    exception = rule["allowlists"][0]
    assert exception["condition"] == "AND"
    assert exception["regexTarget"] == "line"
    value = json.loads((ROOT / REPORT).read_text())["authentext_commit"]
    key = "authentext_commit"
    path = REPORT
    if mutation == "value":
        value = "0" + value[1:]
    elif mutation == "key":
        key = "auth_token"
    elif mutation == "file":
        path = "other.json"
    elif mutation == "suffix":
        path += ".backup"
    line = f'  "{key}": "{value}",'
    matches = any(re.search(pattern, path) for pattern in exception["paths"]) and any(
        re.search(pattern, line) for pattern in exception["regexes"]
    )
    assert bool(matches) is (mutation == "none")
