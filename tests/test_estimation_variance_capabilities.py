"""Capability dispositions for estimation-focused variance VOI."""

# pyright: reportAny=false

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = ROOT / "specs" / "estimation-variance" / "v1" / "capabilities.json"


def test_estimation_variance_capabilities_cover_retained_binding_matrix() -> None:
    capability = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    matrix = json.loads(
        (ROOT / "specs" / "v1" / "binding-matrix.json").read_text(encoding="utf-8")
    )
    dispositions = {binding["language"]: binding for binding in capability["bindings"]}
    assert set(dispositions) == {binding["id"] for binding in matrix["bindings"]}
    assert capability["target_capabilities"] == {
        "scalar_variance": "executable",
        "vector_covariance": "reserved_contract_vocabulary_only_unsupported",
    }
    boundary = capability["vector_covariance_boundary"]
    assert boundary["status"] == (
        "unsupported_pending_candidate_bound_independent_scientific_review"
    )
    assert boundary["promotion_allowed"] is False
    assert set(boundary["reserved_functionals"]) == {
        "trace",
        "determinant",
        "weighted_quadratic",
    }


def test_executable_bindings_have_shared_fixture_evidence() -> None:
    capability = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    dispositions = {binding["language"]: binding for binding in capability["bindings"]}
    assert dispositions["rust"]["status"] == "executable"
    assert dispositions["python"]["status"] == "executable"
    for language in ("rust", "python"):
        for relative_path in dispositions[language]["fixture_evidence"]:
            assert (ROOT / relative_path).is_file()


def test_non_executable_bindings_are_explicit_and_fail_closed() -> None:
    capability = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    dispositions = {binding["language"]: binding for binding in capability["bindings"]}
    for language in ("r", "julia"):
        assert dispositions[language]["status"] == "unsupported"
        assert dispositions[language]["methods"] == []
        assert (
            "No estimation-variance C-ABI symbol"
            in dispositions[language]["unsupported_behavior"]
        )
    assert dispositions["mojo"]["status"] == "external_boundary"
    assert dispositions["mojo"]["methods"] == []
