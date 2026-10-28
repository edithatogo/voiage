"""Current declarations and historical graphs have separate acceptance scopes."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.check_current_overlays import spack_requirement, validate

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "damage",
    ["missing", "checksum", "revision", "archive", "invalid", "native", "policy"],
)
def test_source_claim_requires_exact_receipt(tmp_path: Path, damage: str) -> None:
    shutil.copytree(ROOT / "packaging/current", tmp_path / "packaging/current")
    (tmp_path / "specs").mkdir()
    shutil.copy2(
        ROOT / "specs/dependency-runtime-policy.json",
        tmp_path / "specs/dependency-runtime-policy.json",
    )
    (tmp_path / "packaging/current/overlay-lifecycle.json").write_text(
        json.dumps({"historical_files": {}, "historical_tools": {}})
    )
    receipt_name = "source-proof.json"
    original = json.loads(
        (
            ROOT
            / "conductor/tracks/remaining_backlog_delivery_20260831/source-qualification-20261028.json"
        ).read_text()
    )
    runtime_path = tmp_path / "packaging/current/runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime.update(
        source_archive_qualified=True,
        source_qualification_receipt=receipt_name,
        source_archive_sha256=original["archive_sha256"],
    )
    original["source_revision"] = runtime["source_commit"]
    if damage == "revision":
        original["source_revision"] = "0" * 40
    if damage == "archive":
        original["archive_sha256"] = "0" * 64
    if damage == "native":
        original["native_build_executed"] = True
    if damage == "policy":
        original["runtime_policy_sha256"] = "0" * 64
    receipt = tmp_path / receipt_name
    receipt.write_text(json.dumps(original))
    if damage == "invalid":
        receipt.write_text("not JSON")
    runtime["source_qualification_receipt_sha256"] = hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    if damage == "missing":
        receipt.unlink()
    if damage == "checksum":
        receipt.write_text(receipt.read_text() + "\n")
    runtime_path.write_text(json.dumps(runtime))
    assert "Current source qualification receipt mismatch" in validate(tmp_path)


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
