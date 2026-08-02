"""Cross-language disposition and shared-fixture assurance for COSS."""

from __future__ import annotations

import json
from pathlib import Path

from voiage.contracts.study_design import (
    InformationValueInputV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.experimental.study_design import calculate_coss, evsi_evpi_efficiency

ROOT = Path(__file__).parents[1]
DISPOSITIONS = (
    ROOT
    / "conductor"
    / "archive"
    / "study_design_efficiency_20260727"
    / "bindings.json"
)
FIXTURE = (
    ROOT
    / "specs"
    / "frontier"
    / "study-design-efficiency"
    / "v1"
    / "fixtures"
    / "normative"
    / "coss-efficiency.json"
)


def test_every_governed_language_has_an_honest_disposition() -> None:
    payload = json.loads(DISPOSITIONS.read_text(encoding="utf-8"))
    bindings = {item["language"]: item for item in payload["bindings"]}

    assert set(bindings) == {"rust", "python", "r", "julia", "mojo"}
    assert bindings["rust"]["status"] == "implemented"
    assert bindings["python"]["status"] == "implemented"
    for language in ("r", "julia", "mojo"):
        assert bindings[language]["status"] in {"unsupported", "external_boundary"}
        assert bindings[language]["failure"]["code"] == "unsupported_capability"
        assert bindings[language]["failure"]["message"]
    assert payload["stable_parity_claim"] is False


def test_python_facade_matches_the_rust_owned_shared_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    context = StudyDesignContextV1.model_validate(fixture["input"]["context"])
    designs = tuple(
        StudyDesignPointInputV1.model_validate(item)
        for item in fixture["input"]["designs"]
    )
    replicate_fixture = json.loads(
        (FIXTURE.parent / fixture["input"]["joint_enbs_replicates_artifact"]).read_text(
            encoding="utf-8"
        )
    )

    coss = calculate_coss(
        context=context,
        designs=designs,
        enumeration_scope=fixture["input"]["enumeration_scope"],
        no_study_enbs=fixture["input"]["no_study_enbs"],
        joint_enbs_replicates=replicate_fixture["joint_enbs_replicates"],
        replay_artifact=fixture["input"]["joint_enbs_replicates_artifact"],
    )
    efficiency = evsi_evpi_efficiency(
        evsi=InformationValueInputV1(
            value=fixture["input"]["efficiency"]["evsi"], context=context
        ),
        evpi=InformationValueInputV1(
            value=fixture["input"]["efficiency"]["evpi"], context=context
        ),
    )

    expected = fixture["expected"]
    assert coss.estimator_provenance["runtime"] == "rust"
    assert coss.optimal_design_id == expected["optimal_design_id"]
    assert coss.optimal_sample_size == expected["optimal_sample_size"]
    assert coss.maximum_enbs == expected["maximum_enbs"]
    assert coss.commissioning_status == expected["commissioning_status"]
    assert coss.recommended_design_id == expected["recommended_design_id"]
    assert coss.economic_viability is expected["economic_viability"]
    assert coss.regret_if_no_study == expected["regret_if_no_study"]
    uncertainty = coss.selection_uncertainty
    for field, value in expected["selection_uncertainty"].items():
        assert getattr(uncertainty, field) == value
    assert [point.enbs for point in coss.evaluated_designs] == expected["enbs"]
    assert efficiency.ratio == expected["efficiency_ratio"]
    assert efficiency.percentage == expected["efficiency_percentage"]
