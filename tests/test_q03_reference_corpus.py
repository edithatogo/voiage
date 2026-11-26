"""Tests for the Q03 independent reference corpus and fail-closed validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.validate_reference_corpus import (
    reproduce_case_offline,
    validate_corpus_manifest,
)
from voiage.methods.basic import evpi
from voiage.methods.sample_information import enbs, normal_normal_two_arm_evsi

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "specs" / "v2" / "reference-corpus" / "manifest.json"


def _load_json(relative: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_admitted_reference_corpus_reproduces_offline_and_discloses_provenance() -> (
    None
):
    summary = validate_corpus_manifest(MANIFEST_PATH, REPO_ROOT)
    assert summary["admitted_count"] == 8
    assert summary["external_candidate_count"] == 6

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    admitted = manifest["admitted_cases"]
    assert all(isinstance(entry["synthetic_owned"], bool) for entry in admitted)
    assert all(isinstance(entry["same_author_application"], bool) for entry in admitted)
    held_out = [entry["id"] for entry in admitted if entry["held_out_from_tuning"]]
    assert {
        "exact-finite-weighted-evpi",
        "exact-nonlinear-quadratic-evppi",
        "conjugate-gamma-poisson-evsi",
        "exact-single-period-enbs",
    } <= set(held_out)
    assert all(
        cand["admission_state"] == "candidate-source-only"
        for cand in manifest["external_candidates"]
    )


def test_voiage_runtime_matches_supported_independent_q03_references() -> None:
    weighted_evpi = _load_json(
        "specs/v2/reference-corpus/exact-finite-weighted-evpi.json"
    )
    equal_sub = weighted_evpi["inputs"]["subcases"][0]  # type: ignore[index]
    assert evpi(np.asarray(equal_sub["net_benefits"], dtype=float)) == pytest.approx(
        equal_sub["expected"]["value"], abs=equal_sub["expected"]["atol"]
    )

    nn_case = _load_json("specs/v2/reference-corpus/conjugate-normal-normal-evsi.json")
    nn_inputs = nn_case["inputs"]
    assert isinstance(nn_inputs, dict)
    evpi_upper = float(nn_case["expected"]["additional_expected"]["evpi"])  # type: ignore[index]
    prior_evsi = -1.0
    for sub in nn_inputs["subcases"]:  # type: ignore[index]
        per_arm = int(sub["per_arm_sample_size"])
        total_n = int(sub["total_sample_size"])
        assert total_n == 2 * per_arm
        rt = normal_normal_two_arm_evsi(
            prior_mean=float(nn_inputs["prior_mean"]),
            prior_standard_deviation=float(nn_inputs["prior_standard_deviation"]),
            outcome_standard_deviation=float(nn_inputs["outcome_standard_deviation"]),
            total_sample_size=total_n,
            net_benefit_slope=float(nn_inputs["net_benefit_slope"]),
            net_benefit_intercept=float(nn_inputs["net_benefit_intercept"]),
        )
        assert 0.0 <= rt <= evpi_upper
        assert rt > prior_evsi
        prior_evsi = rt
        assert rt == pytest.approx(
            sub["expected"]["value"], abs=sub["expected"]["runtime_atol"]
        )

    enbs_case = _load_json("specs/v2/reference-corpus/exact-single-period-enbs.json")
    enbs_inputs = enbs_case["inputs"]
    assert isinstance(enbs_inputs, dict)
    assert enbs(
        float(enbs_inputs["gross_population_evsi"]),
        float(enbs_inputs["research_cost"]),
    ) == pytest.approx(
        float(enbs_case["expected"]["value"]),  # type: ignore[index]
        abs=float(enbs_case["expected"]["atol"]),  # type: ignore[index]
    )


def test_rejects_missing_unit_broken_draws_and_self_certified_case() -> None:
    base = _load_json("specs/v2/reference-corpus/exact-finite-evppi-discrete.json")

    missing_unit = copy.deepcopy(base)
    missing_unit["expected"]["unit"] = "   "  # type: ignore[index]
    with pytest.raises(ValueError, match="expected.unit"):
        reproduce_case_offline(missing_unit)

    broken_draws = copy.deepcopy(base)
    broken_draws["inputs"]["parameter_samples"] = broken_draws["inputs"][
        "parameter_samples"
    ][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="broken draw alignment"):
        reproduce_case_offline(broken_draws)

    self_certified = copy.deepcopy(base)
    self_certified["reference_implementation"] = "voiage.methods.basic.evppi"
    with pytest.raises(ValueError, match="cannot attribute reference generation"):
        reproduce_case_offline(self_certified)


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        ("absent_license", "license"),
        ("tampered_checksum", "checksum mismatch"),
        ("unpinned_source", "pinned revision"),
        ("unqualified_external", "without cleared license"),
    ],
)
def test_manifest_rejects_rights_checksum_pinning_and_admission_failures(
    tmp_path: Path, mutation: str, pattern: str
) -> None:
    corpus_root = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT / "specs" / "v2" / "reference-corpus",
        corpus_root / "specs" / "v2" / "reference-corpus",
    )
    shutil.copytree(
        REPO_ROOT / "specs" / "numerical-reference",
        corpus_root / "specs" / "numerical-reference",
    )
    manifest_path = corpus_root / "specs" / "v2" / "reference-corpus" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if mutation == "absent_license":
        manifest["admitted_cases"][0]["license"] = ""
    elif mutation == "tampered_checksum":
        manifest["admitted_cases"][1]["sha256"] = "0" * 64
    elif mutation == "unpinned_source":
        manifest["admitted_cases"][1]["pinned_revision"] = "latest"
    else:
        manifest["external_candidates"][0]["admission_state"] = "admitted"

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        validate_corpus_manifest(manifest_path, corpus_root)


def test_validator_enforces_top_level_conjugate_strict_atol_and_python_o() -> None:
    bb_bad = _load_json("specs/v2/reference-corpus/conjugate-beta-binomial-evsi.json")
    bb_bad["expected"]["value"] = -999.0  # type: ignore[index]
    with pytest.raises(ValueError, match="mismatch"):
        reproduce_case_offline(bb_bad)

    bb_bad_add = _load_json(
        "specs/v2/reference-corpus/conjugate-beta-binomial-evsi.json"
    )
    bb_bad_add["expected"]["additional_expected"]["evpi"] = -999.0  # type: ignore[index]
    with pytest.raises(ValueError, match="mismatch"):
        reproduce_case_offline(bb_bad_add)

    # rel_tol=0.0: 1e-7 perturbation must fail against atol=1e-12 even for magnitude > 100.
    nn_slack = _load_json("specs/v2/reference-corpus/conjugate-normal-normal-evsi.json")
    nn_slack["inputs"]["subcases"][0]["expected"]["value"] += 1e-7  # type: ignore[index]
    with pytest.raises(ValueError, match="mismatch"):
        reproduce_case_offline(nn_slack)
