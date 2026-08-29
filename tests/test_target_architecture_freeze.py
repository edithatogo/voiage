"""Contracts for the target package, API, ABI, and binding architecture."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FREEZE_PATH = (
    ROOT / "specs" / "submission-readiness" / "target-architecture-freeze-20260829.json"
)


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_normative_contract_and_abi_baselines_exist() -> None:
    freeze = _freeze()

    assert freeze["status"] == "normative_target"
    assert (ROOT / freeze["stable_api_policy"]["normative_contract"]).is_file()
    for key in ("normative_header", "symbol_baseline", "layout_baseline"):
        assert (ROOT / freeze["c_abi_policy"][key]).is_file()
    assert freeze["c_abi_policy"]["current_namespace"] == "voiage_v1_"
    assert freeze["c_abi_policy"]["stable_native_methods"] == ["evpi", "enbs"]


def test_package_ownership_is_total_and_non_overlapping_at_the_stable_boundary() -> (
    None
):
    boundaries = {
        entry["component"]: entry for entry in _freeze()["package_boundaries"]
    }

    assert boundaries.keys() == {
        "rust/crates/voiage-domain",
        "rust/crates/voiage-numerics",
        "rust/crates/voiage-serialization",
        "rust/crates/voiage-diagnostics",
        "rust/crates/voiage-ffi",
        "rust/crates/voiage-python",
        "voiage",
        "r-package/voiageR",
        "bindings/julia",
    }
    assert (
        "stable numerical kernels" in boundaries["rust/crates/voiage-numerics"]["owns"]
    )
    assert (
        "duplicate implementations of promoted stable Rust kernels"
        in boundaries["voiage"]["must_not_own"]
    )
    assert all(entry["owns"] and entry["must_not_own"] for entry in boundaries.values())


def test_decision_problem_and_binding_breadth_are_honest() -> None:
    freeze = _freeze()
    decision_problem = freeze["stable_api_policy"]["decision_problem_target"]
    matrix = {entry["capability"]: entry for entry in freeze["capability_matrix"]}

    assert decision_problem == {
        "python": "implemented as voiage.schema.DecisionProblem",
        "rust": "implemented internally as voiage_domain::DecisionProblem",
        "c_abi": "not exposed",
        "r": "not exposed",
        "julia": "not exposed",
        "required_repair": "Correct the false industry binding manifest rather than inventing unreviewed cross-language symbols.",
    }
    assert matrix["DecisionProblem"]["c_abi"] == "unavailable"
    assert matrix["DecisionProblem"]["r"] == "unavailable"
    assert matrix["DecisionProblem"]["julia"] == "unavailable"
    assert matrix["EVPI"]["r"] == matrix["EVPI"]["julia"] == "stable_native"
    assert matrix["EVPPI"]["r"] == "compatibility_python_bridge"
    assert matrix["EVPPI"]["julia"] == "unavailable"


def test_r_and_julia_targets_forbid_ambient_installed_library_dependencies() -> None:
    packaging = _freeze()["binding_packaging"]

    assert "NeedsCompilation: yes" in packaging["r"]["target"]
    assert "no undeclared preinstalled" in packaging["r"]["system_requirements"]
    assert "never required" in packaging["r"]["optional_python"]
    assert "Artifacts" in packaging["julia"]["target"]
    assert "development override only" in packaging["julia"]["target"]
    assert packaging["julia"]["external_gate"]


def test_every_structure_finding_has_a_frozen_closure_rule() -> None:
    closure = _freeze()["finding_closure"]

    assert closure.keys() == {f"STRUCT-{number:03d}" for number in range(1, 8)}
    assert all(closure.values())
    assert len(_freeze()["installed_artifact_gates"]) == 5
