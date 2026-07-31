"""Changed-line assurance for governed COSS and efficiency failure paths."""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from voiage.contracts.study_design import (
    InformationValueInputV1,
    SelectionUncertaintyV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
)
from voiage.exceptions import InputError
from voiage.experimental import study_design as module


def _context() -> StudyDesignContextV1:
    return StudyDesignContextV1(
        decision_problem_id="coverage-decision",
        value_unit="AUD_2026",
        population_scale=1_000.0,
        time_horizon="2027-2036",
        discounting_id="none",
        study_model_id="enumerated",
        cost_model_id="enumerated",
        random_seed=571,
    )


def _point(
    design_id: str,
    sample_size: int,
    evsi: float,
    cost: float,
    *,
    feasible: bool = True,
) -> StudyDesignPointInputV1:
    return StudyDesignPointInputV1(
        design_id=design_id,
        sample_size=sample_size,
        evsi=evsi,
        research_cost=cost,
        feasible=feasible,
        estimator_provenance={"source": "coverage-test"},
    )


def _result_payload() -> tuple[type[object], dict[str, object]]:
    result = module.calculate_coss(
        context=_context(),
        designs=(
            _point("a", 20, 5.0, 2.0),
            _point("b", 40, 8.0, 3.0),
            _point("blocked", 60, 9.0, 3.0, feasible=False),
        ),
    )
    return type(result), result.model_dump(mode="json")


def _rejecting_contract(name: str):
    def reject(**_: object) -> None:
        raise ValidationError.from_exception_data(name, [])

    return reject


def test_design_point_provenance_serializer_restores_json_mapping() -> None:
    point = _point("a", 20, 5.0, 2.0)

    assert point.model_dump(mode="json")["estimator_provenance"] == {
        "source": "coverage-test"
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["selection_uncertainty"].update(
                confidence_set_design_ids=["blocked"]
            ),
            "confidence set",
        ),
        (
            lambda payload: payload["selection_uncertainty"].update(
                probability_by_design={"ghost": 0.1}
            ),
            "unknown designs",
        ),
        (
            lambda payload: payload["selection_uncertainty"].update(
                probability_by_design={"blocked": 0.1}
            ),
            "feasible design",
        ),
    ],
)
def test_coss_contract_rejects_corrupted_selection_references(
    mutation: object, message: str
) -> None:
    result_type, payload = _result_payload()
    payload["selection_uncertainty"].update(  # type: ignore[union-attr]
        method="bootstrap", replicate_count=100
    )
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ValidationError, match=message):
        result_type.model_validate_json(json.dumps(payload))  # type: ignore[attr-defined]


def test_coss_contract_allows_an_explicitly_partial_probability_map() -> None:
    result_type, payload = _result_payload()
    payload["selection_uncertainty"].update(  # type: ignore[union-attr]
        method="bootstrap",
        replicate_count=100,
        probability_by_design={"a": 0.25},
    )

    restored = result_type.model_validate_json(json.dumps(payload))  # type: ignore[attr-defined]

    assert restored.selection_uncertainty.probability_by_design == {"a": 0.25}


def test_coss_contract_rejects_an_optimum_that_violates_tie_policy() -> None:
    result = module.calculate_coss(
        context=_context(),
        designs=(
            _point("z", 20, 5.0, 2.0),
            _point("a", 20, 6.0, 3.0),
        ),
        tie_policy="smallest_sample_size",
    )
    payload = result.model_dump(mode="json")
    payload["optimal_design_id"] = "z"
    payload["plot_data"]["optimal_design_id"] = "z"

    with pytest.raises(ValidationError, match="tie policy"):
        type(result).model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"bound_tolerance": 99.0}, "bound_tolerance"),
        (
            {"status": "undefined_zero_evpi", "ratio": None, "percentage": None},
            "undefined_zero_evpi",
        ),
        ({"ratio": None}, "defined efficiency"),
        ({"status": "above_one_within_tolerance"}, "status disagrees"),
    ],
)
def test_efficiency_contract_rejects_corrupted_relations(
    changes: dict[str, object], message: str
) -> None:
    result = module.evsi_evpi_efficiency(
        evsi=InformationValueInputV1(value=5.0, context=_context()),
        evpi=InformationValueInputV1(value=10.0, context=_context()),
    )
    payload = result.model_dump(mode="json")
    payload.update(changes)

    with pytest.raises(ValidationError, match=message):
        type(result).model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "native_update",
    [
        {"feasible_indices": (0,)},
        {"optimal_index": 1},
    ],
)
def test_coss_fails_closed_on_additional_native_envelope_corruption(
    monkeypatch: pytest.MonkeyPatch, native_update: dict[str, object]
) -> None:
    payload: dict[str, object] = {
        "contract_version": "1.0.0",
        "estimator": "native",
        "enbs": [3.0, 3.0],
        "feasible_indices": [0, 1],
        "tied_indices": [0, 1],
        "optimal_index": 0,
        "maximum_enbs": 3.0,
        "boundary_state": "lower",
    }
    payload.update(native_update)
    monkeypatch.setattr(module._runtime, "compute_coss", lambda **_: payload)

    with pytest.raises(InputError, match="native COSS result"):
        module.calculate_coss(
            context=_context(),
            designs=(
                _point("a", 20, 5.0, 2.0),
                _point("b", 40, 6.0, 3.0),
            ),
            tie_policy="first_declared",
        )


def test_coss_fails_closed_when_native_tie_references_infeasible_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "contract_version": "1.0.0",
        "estimator": "native",
        "enbs": [3.0, 3.0],
        "feasible_indices": [0],
        "tied_indices": [1],
        "optimal_index": 1,
        "maximum_enbs": 3.0,
        "boundary_state": "both",
    }
    monkeypatch.setattr(module._runtime, "compute_coss", lambda **_: payload)

    with pytest.raises(InputError, match="native COSS result"):
        module.calculate_coss(
            context=_context(),
            designs=(
                _point("a", 20, 5.0, 2.0),
                _point("blocked", 40, 6.0, 3.0, feasible=False),
            ),
        )


def test_coss_fails_closed_on_inconsistent_native_no_feasible_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "contract_version": "1.0.0",
        "estimator": "native",
        "enbs": [3.0],
        "feasible_indices": [],
        "tied_indices": [],
        "optimal_index": None,
        "maximum_enbs": 3.0,
        "boundary_state": "none",
    }
    monkeypatch.setattr(module._runtime, "compute_coss", lambda **_: payload)

    with pytest.raises(InputError, match="native COSS result"):
        module.calculate_coss(
            context=_context(),
            designs=(_point("blocked", 20, 5.0, 2.0, feasible=False),),
        )


def test_coss_reports_unavailable_derived_enbs_uncertainty() -> None:
    point = StudyDesignPointInputV1(
        design_id="a",
        sample_size=20,
        evsi=5.0,
        research_cost=2.0,
        evsi_standard_error=0.2,
    )

    result = module.calculate_coss(context=_context(), designs=(point,))

    assert "enbs_uncertainty_unavailable" in result.diagnostics


def test_coss_rejects_an_infeasible_selection_confidence_set() -> None:
    uncertainty = SelectionUncertaintyV1(
        method="bootstrap",
        replicate_count=10,
        confidence_set_design_ids=("blocked",),
    )

    with pytest.raises(InputError, match="infeasible"):
        module.calculate_coss(
            context=_context(),
            designs=(
                _point("a", 20, 5.0, 2.0),
                _point("blocked", 40, 6.0, 3.0, feasible=False),
            ),
            selection_uncertainty=uncertainty,
        )


def test_coss_wraps_result_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "CossResultV1", _rejecting_contract("CossResultV1"))

    with pytest.raises(InputError, match="scientific contract validation"):
        module.calculate_coss(context=_context(), designs=(_point("a", 20, 5.0, 2.0),))


def test_efficiency_wraps_native_and_result_contract_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evsi = InformationValueInputV1(value=5.0, context=_context())
    evpi = InformationValueInputV1(value=10.0, context=_context())
    monkeypatch.setattr(
        module._runtime,
        "compute_evsi_evpi_efficiency",
        lambda *_: {"contract_version": "1.0.0"},
    )
    with pytest.raises(InputError, match="native efficiency result"):
        module.evsi_evpi_efficiency(evsi=evsi, evpi=evpi)

    monkeypatch.setattr(
        module._runtime,
        "compute_evsi_evpi_efficiency",
        lambda *_: {
            "contract_version": "1.0.0",
            "ratio": 0.5,
            "status": "within_bounds",
            "bound_tolerance": 1.001e-7,
        },
    )
    monkeypatch.setattr(
        module,
        "InformationEfficiencyResultV1",
        _rejecting_contract("InformationEfficiencyResultV1"),
    )
    with pytest.raises(InputError, match="scientific contract validation"):
        module.evsi_evpi_efficiency(evsi=evsi, evpi=evpi)
