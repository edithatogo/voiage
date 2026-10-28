"""Validate the captured semantic reference without consumer checkout access."""

from pathlib import Path
from typing import Any


def load_cases(root: Path) -> list[dict[str, Any]]:
    """Load the planned semantic rows."""
    import json

    return json.loads((root / "conductor/tracks/translation_reference_20261026/semantic-cases.json").read_text())["rows"]


def run_case(row: dict[str, Any], mode: str) -> Any:
    """Execute one qualified reference form."""
    raise NotImplementedError("Semantic reference is not implemented")


def validate(root: Path) -> list[str]:
    """Reject an incomplete inventory or missing publication."""
    return ["Captured inventory and reference publication are not implemented"]
