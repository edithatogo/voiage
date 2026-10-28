"""Current declarations and historical graphs have separate acceptance scopes."""

import json
from pathlib import Path
import shutil

from scripts.check_current_overlays import spack_requirement, validate

ROOT = Path(__file__).parents[1]


def test_current_declarations_and_frozen_graphs_validate() -> None:
    assert validate(ROOT) == []


def test_bound_translation_keeps_minor_and_major_ceilings() -> None:
    assert spack_requirement("jsonschema>=4.26.0,<5.0") == "py-jsonschema@4.26.0:4"
    assert spack_requirement("numpy>=2.5.4,<3") == "py-numpy@2.5.4:2"
    assert spack_requirement("scipy>=1.18.1,<1.19") == "py-scipy@1.18.1:1.18"


def test_current_overlay_drift_and_unearned_native_claim_fail(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "packaging/current", tmp_path / "packaging/current")
    (tmp_path / "specs").mkdir()
    shutil.copy2(
        ROOT / "specs/dependency-runtime-policy.json",
        tmp_path / "specs/dependency-runtime-policy.json",
    )
    lifecycle = tmp_path / "packaging/current/overlay-lifecycle.json"
    lifecycle.write_text(json.dumps({"historical_files": {}, "historical_tools": {}}))
    runtime = tmp_path / "packaging/current/runtime.json"
    data = json.loads(runtime.read_text())
    data["runtime"].append("pandas>=3")
    data["native_build_executed"] = True
    runtime.write_text(json.dumps(data))
    errors = validate(tmp_path)
    assert "Current overlay policy drift: runtime" in errors
    assert "Unqualified native claim: native_build_executed" in errors
