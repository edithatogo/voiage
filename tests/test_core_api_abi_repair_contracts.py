"""Regression contracts for the accepted core, API, and ABI repair findings."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
DISPOSITIONS = ROOT / "specs" / "abi" / "industry-decision-binding-dispositions.json"
ABI_RELEASE = ROOT / "specs" / "abi" / "releases" / "v2.1.0"


def test_decision_problem_manifest_matches_shipped_language_surfaces() -> None:
    manifest = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))
    dispositions = manifest["contracts"]["decision_problem"]["dispositions"]

    assert dispositions["python"] == {
        "status": "implemented",
        "symbol": "voiage.schema.DecisionProblem",
        "interchange": "json_and_arrow",
    }
    assert dispositions["rust"] == {
        "status": "internal",
        "symbol": "voiage_domain::DecisionProblem",
        "interchange": "native_only",
        "reason": "Internal domain type; not a public C ABI or language-binding surface.",
    }
    for language in ("r", "julia"):
        assert dispositions[language]["status"] == "unsupported"
        assert dispositions[language]["symbol"] == ""
        assert dispositions[language]["interchange"] == ""
        assert "DecisionProblem" in dispositions[language]["reason"]


def test_manifest_validator_resolves_implemented_python_symbols() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from voiage.binding_dispositions import "
            "validate_binding_dispositions_manifest; "
            "assert validate_binding_dispositions_manifest(resolve_symbols=True)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_v210_abi_release_baseline_is_immutable_and_machine_checked() -> None:
    required = {
        "symbols.txt",
        "layouts.txt",
        "voiage_v1.h",
        "metadata.json",
    }
    assert {path.name for path in ABI_RELEASE.iterdir() if path.is_file()} == required

    metadata = json.loads((ABI_RELEASE / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["release"] == "v2.1.0"
    assert metadata["abi_major"] == 1
    assert metadata["source_commit"]
    assert metadata["artifacts_sha256"].keys() == required - {"metadata.json"}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_abi_compatibility.py",
            "--baseline",
            str(ABI_RELEASE),
            "--candidate-header",
            "rust/crates/voiage-ffi/include/voiage_v1.h",
            "--candidate-symbols",
            "specs/abi/v1/symbols.txt",
            "--candidate-layouts",
            "specs/abi/v1/layouts.txt",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_binding_capability_contract_requires_behavior_not_workflow_text() -> None:
    contract = json.loads(
        (
            ROOT
            / "specs"
            / "submission-readiness"
            / "installed-binding-capability-contract-v1.json"
        ).read_text(encoding="utf-8")
    )

    assert contract["schema_version"] == "1.0.0"
    assert contract["source_of_truth"] == (
        "specs/submission-readiness/target-architecture-freeze-20260829.json"
    )
    assert contract["required_surfaces"] == ["python", "rust", "c_abi", "r", "julia"]
    assert set(contract["required_checks"]) == {
        "enumerate exported symbols from installed artifact",
        "invoke every declared stable capability",
        "compare shared numerical fixtures",
        "reject unsupported capability through a declared error",
        "verify source version and artifact digest",
    }
    assert contract["prohibited_evidence"] == [
        "manifest substring checks",
        "workflow command substring checks",
        "repository-relative import or library lookup",
    ]
