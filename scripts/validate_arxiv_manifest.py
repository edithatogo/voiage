#!/usr/bin/env python3
"""Validate manuscript metadata and non-submission gates."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "paper/readiness-manifest.json").read_text())
metadata = json.loads((ROOT / "paper/metadata.json").read_text())
assert manifest["submission_performed"] is False
assert manifest["source_provenance"] == {
    "sourceright": "submodule",
    "authentext": "submodule",
}
assert all(manifest["human_gates"])
assert manifest["schema_version"].startswith("voiage.arxiv-readiness.")
assert {
    "required_tools",
    "optional_tools",
    "source_provenance",
    "human_gates",
    "joss_submission",
} <= manifest.keys()
assert {"title", "authors", "abstract", "categories", "license", "comments"} <= (
    metadata.keys()
)
assert isinstance(metadata["authors"], list)
assert metadata["authors"]
assert isinstance(metadata["categories"], list)
assert metadata["categories"]
print("arXiv readiness manifest: pass")
