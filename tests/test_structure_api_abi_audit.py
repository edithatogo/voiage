"""Executable inventory for the pre-submission structure/API/ABI audit."""

from __future__ import annotations

import json
from pathlib import Path

import voiage


ROOT = Path(__file__).parents[1]
AUDIT_PATH = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "structure-api-abi-audit-20260829.json"
)


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_structure_inventory_matches_current_sources() -> None:
    audit = _read(AUDIT_PATH)
    stable_api = _read(ROOT / "specs/v2/stable-api.json")
    assert isinstance(audit, dict)
    assert isinstance(stable_api, dict)
    inventory = audit["inventory"]

    assert inventory["python_runtime_modules"] == len(
        list((ROOT / "voiage").glob("**/*.py"))
    )
    assert inventory["rust_workspace_crates"] == len(
        [path for path in (ROOT / "rust/crates").iterdir() if path.is_dir()]
    )
    stable_symbols = stable_api["symbols"]["stable"]
    assert inventory["stable_python_symbols"] == len(stable_symbols)
    assert all(hasattr(voiage, symbol) for symbol in stable_symbols)
    assert inventory["c_abi_symbols"] == len(
        [
            line
            for line in (ROOT / "specs/abi/v1/symbols.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        ]
    )
    assert inventory["tracked_stale_r_source_archives"] == int(
        (ROOT / "r-package/voiageR/voiageR_0.1.0.tar.gz").is_file()
    )


def test_structure_findings_remain_explicit_until_disposition() -> None:
    audit = _read(AUDIT_PATH)
    assert isinstance(audit, dict)
    findings = audit["findings"]
    by_id = {finding["id"]: finding for finding in findings}

    assert set(by_id) == {f"STRUCT-{number:03d}" for number in range(1, 8)}
    assert by_id["STRUCT-001"]["severity"] == "critical"
    assert {finding["state"] for finding in findings} == {"open"}
    assert all(finding["evidence"] for finding in findings)
    assert all(finding["required_disposition"] for finding in findings)
