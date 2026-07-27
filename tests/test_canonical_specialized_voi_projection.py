from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/validate_canonical_specialized_voi_projection.py"
SPEC = importlib.util.spec_from_file_location("c16_projection", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_c16_projection_matches_voiage_tracks() -> None:
    MODULE.validate(
        ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json", ROOT
    )


def test_c16_projection_rejects_missing_consumer_registration(tmp_path: Path) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json"
        ).read_text(encoding="utf-8")
    )
    projection["registered_repositories"] = []
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly managed"):
        MODULE.validate(path, ROOT)
