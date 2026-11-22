"""Current declarations and historical graphs have separate acceptance scopes."""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts.check_current_overlays import spack_requirement, validate

ROOT = Path(__file__).parents[1]


def test_spack_pins_the_qualified_cpython_source() -> None:
    current = ROOT / "packaging/current"
    recipe = ast.parse((current / "spack/packages/python/package.py").read_text())
    (release,) = (
        node
        for node in ast.walk(recipe)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "version"
    )
    assert ast.literal_eval(release.args[0]) == "3.15.2"
    source = {kw.arg: ast.literal_eval(kw.value) for kw in release.keywords}
    assert source["url"] == (
        "https://www.python.org/ftp/python/3.15.2/Python-3.15.2.tar.xz"
    )
    for generation in ("2023a", "2024a"):
        (easyconfig,) = (current / "easybuild" / generation).glob(
            "Python-3.15.2-GCCcore-*.eb"
        )
        checksum = next(
            ast.literal_eval(node.value)[0]
            for node in ast.parse(easyconfig.read_text()).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "checksums"
        )
        assert source["sha256"] == checksum


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
            / "conductor/tracks/remaining_backlog_delivery_20260831/source-qualification-20261115-python3152.json"
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
    baseline_receipt = tmp_path / receipt_name
    baseline_receipt.write_text(json.dumps(original))
    runtime["source_qualification_receipt_sha256"] = hashlib.sha256(
        baseline_receipt.read_bytes()
    ).hexdigest()
    runtime_path.write_text(json.dumps(runtime))
    assert validate(tmp_path) == []
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


def test_python3151_overlay_snapshot_retains_every_recorded_file() -> None:
    base = ROOT / "conductor/tracks/remaining_backlog_delivery_20260831/historical"
    archive = base / "python3151-current-overlay-20261115.tar.gz"
    manifest = json.loads(
        (base / "python3151-current-overlay-20261115.json").read_text()
    )
    assert manifest["source_revision"] == "7f1bbab88a266fbb23b818dd82d3d55666ca1b8d"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    expected = {entry["path"]: entry for entry in manifest["files"]}
    assert len(expected) == len(manifest["files"]) == 147
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            if member.isdir():
                continue
            assert member.name not in seen
            assert member.name in expected
            seen.add(member.name)
            entry = expected[member.name]
            if member.issym():
                assert entry["mode"] == "120000"
                data = member.linkname.encode()
            else:
                assert member.isfile()
                assert entry["mode"] in ("100644", "100755")
                assert member.mode & 0o777 == int(entry["mode"][-3:], 8)
                stream = source.extractfile(member)
                assert stream is not None
                data = stream.read()
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert seen == set(expected)


def test_historical_smoke_bytes_and_lint_only_derivation_are_preserved() -> None:
    lifecycle = json.loads(
        (ROOT / "packaging/current/overlay-lifecycle.json").read_text()
    )
    source = "scripts/hpc_package_smoke.py"
    archive = "conductor/tracks/pandas_free_io_20261026/historical/hpc_package_smoke-20261115.py.txt"
    original_digest = "7fd376f043be95e487918bd61bcc8d4514587e7913b48fb022cacdfc8a3eb042"
    lineage = lifecycle["historical_tool_lineage"][source]
    assert lineage["original_path"] == source
    assert lineage["archive_path"] == archive
    assert lineage["original_sha256"] == original_digest
    assert lineage["source_commit"] == "b0f07ab0a88eb5500a8eda3b8d0ec85b6905a108"
    original = (ROOT / archive).read_bytes().replace(b"\r\n", b"\n")
    active = (ROOT / source).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(original).hexdigest() == original_digest
    assert lifecycle["historical_files"][archive] == original_digest
    assert lifecycle["historical_tools"][source] == hashlib.sha256(active).hexdigest()
    marker = b"  # noqa: S310 -- fixed HTTPS source"
    assert original.count(marker) == 1
    assert active == original.replace(marker, b"", 1)


def test_bound_translation_keeps_minor_and_major_ceilings() -> None:
    assert spack_requirement("jsonschema>=4.26.0,<5.0") == "py-jsonschema@4.26.0:4"
    assert spack_requirement("numpy>=2.5.4,<3") == "py-numpy@2.5.4:2"
    assert spack_requirement("scipy>=1.18.1,<1.19") == "py-scipy@1.18.1:1.18"


def test_core_compiler_cannot_use_an_unbounded_preview_range(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "packaging/current", tmp_path / "packaging/current")
    (tmp_path / "specs").mkdir()
    shutil.copy2(
        ROOT / "specs/dependency-runtime-policy.json",
        tmp_path / "specs/dependency-runtime-policy.json",
    )
    (tmp_path / "packaging/current/overlay-lifecycle.json").write_text(
        json.dumps({"historical_files": {}, "historical_tools": {}})
    )
    recipe = tmp_path / "packaging/current/spack/packages/py-voiage/package.py"
    recipe.write_text(recipe.read_text().replace("rust@=1.99.1", "rust@1.85:"))
    assert "Current Spack dependency drift" in validate(tmp_path)
    easy = tmp_path / "packaging/current/easybuild/voiage-candidate-Python-3.15.2.eb"
    easy.write_text(
        easy.read_text().replace('("Rust", "1.99.1")', '("Rust", "1.96.0")')
    )
    assert "Current EasyBuild compiler drift" in validate(tmp_path)


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
