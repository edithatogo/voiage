"""Black-box contracts for a wheel installed outside the source checkout."""

from __future__ import annotations

import hashlib
from importlib.metadata import requires, version
import os
from pathlib import Path
import re

import numpy as np
import rfc8785

import voiage
import voiage._core as native
from voiage.methods.ceaf import CEAFResult
from voiage.methods.dominance import DominanceResult


def _digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def _build_id(info: dict[str, object]) -> str:
    values = (
        info["build_id_algorithm"],
        info["source_revision"],
        info["source_tree_git_oid"],
        str(info["source_dirty"]).lower(),
        info["source_state_sha256"],
        info["target_triple"],
        info["rustc_version"],
        info["build_profile"],
        info["cargo_lock_sha256"],
        "" if info["source_date_epoch"] is None else str(info["source_date_epoch"]),
    )
    encoded = "".join(f"{len(value)}:{value}" for value in values).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clean_source_state(info: dict[str, object]) -> str:
    hasher = hashlib.sha256()
    for value in (
        info["source_state_algorithm"],
        info["source_tree_git_oid"],
    ):
        encoded = value.encode()
        hasher.update(len(encoded).to_bytes(8, byteorder="big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def test_imports_resolve_inside_the_wheel_environment() -> None:
    """Reject source-shadowed imports when CI supplies the wheel environment."""
    environment = os.environ.get("WHEEL_VENV")
    if environment is None:
        return

    root = Path(environment).resolve()
    assert Path(voiage.__file__).resolve().is_relative_to(root)
    assert Path(native.__file__).resolve().is_relative_to(root)


def test_installed_wheel_executes_external_distribution_family_request() -> None:
    """The experimental adapter and installed schema work without repository specs."""
    from jsonschema import Draft202012Validator

    from voiage.contracts.distributional_information import (
        VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1,
    )
    from voiage.methods.distributional_information import (
        distributional_information_from_specification,
    )

    payload = {
        "schema_version": "1.0.0",
        "analysis_id": "wheel-external-vdi",
        "analysis_type": "distribution_family_information_value",
        "method_maturity": "experimental",
        "information_target": "model_family_index",
        "conditioning_order": "integrate_within_family_then_resolve_family_index",
        "direction": "maximize",
        "value_unit": "point",
        "model_ids": ["m1", "m2"],
        "model_labels": {"m1": "Model 1", "m2": "Model 2"},
        "model_definitions": [
            {
                "model_id": model_id,
                "family_or_assumption": label,
                "parameterization": "finite exact table",
                "within_family_integration": "analytical expectation",
                "definition_source": "wheel black-box test",
                "parameter_source": "exact synthetic values",
                "data_reference": f"wheel:{model_id}",
                "value_transformation": "identity",
            }
            for model_id, label in (("m1", "family one"), ("m2", "family two"))
        ],
        "model_probabilities": [0.5, 0.5],
        "alternative_names": ["A", "B"],
        "conditional_values": [[10.0, 6.0], [4.0, 12.0]],
        "conditional_value_assurance": {
            "input_status": "exact_enumerated_conditional_expectations",
            "source_values_exact": True,
            "source_uncertainty": "none_by_construction",
            "enumeration_method": "finite exact table",
            "evidence_reference": "wheel:black-box",
        },
        "information_cost": 0.5,
        "tolerances": {
            "absolute": 1e-12,
            "relative": 1e-12,
            "probability_sum": 1e-12,
        },
        "comparability": {
            "population_id": "wheel-population",
            "horizon_id": "wheel-horizon",
            "discounting_id": "wheel-discounting",
            "value_semantics_id": "wheel-conditional-value",
            "cost_location_id": "wheel-cost-location",
            "verified": True,
            "verification_reference": "wheel:comparability",
        },
        "provenance": {
            "fixture_id": "wheel-external-vdi",
            "probability_source": "synthetic equal weights",
            "value_source": "exact finite table",
            "family_definition_source": "wheel black-box test",
        },
    }
    Draft202012Validator(VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1).validate(
        payload
    )
    result = distributional_information_from_specification(payload)
    assert result.gross_vdi == 2.0
    assert result.net_vdi == 1.5


def test_installed_wheel_executes_noncardinal_qualitative_assessment() -> None:
    """The experimental qualitative contract runs without repository fixtures."""
    from voiage.contracts.qualitative_information import (
        qualitative_assessment_content_digest,
        qualitative_audit_event_digest,
    )
    from voiage.methods.qualitative_information import (
        qualitative_information_from_specification,
        render_qualitative_information_text,
    )

    payload = {
        "schema_version": "1.0.0",
        "assessment_id": "wheel-qualitative",
        "assessment_version": 1,
        "method_maturity": "experimental",
        "numerical_estimand": False,
        "decision": {
            "decision_id": "d1",
            "title": "Synthetic decision",
            "context": "Wheel-only contract check",
            "alternatives": ["A", "B"],
            "accountable_reviewer_ids": ["human-1"],
        },
        "reviewers": [{"reviewer_id": "human-1", "name": "Role", "role": "owner"}],
        "sources": [
            {
                "source_id": "s1",
                "citation": "Synthetic",
                "access_status": "accessible",
                "provenance": "wheel",
            }
        ],
        "questions": [
            {
                "question_id": "q1",
                "information_question": "Would evidence change the choice?",
                "uncertainty_or_evidence_gap": "Synthetic gap",
                "information_action": "Review synthetic evidence",
                "missing_fields": [],
                "redaction_status": "none",
                "judgements": [
                    {
                        "reviewer_id": "human-1",
                        "actor_type": "human",
                        "potential_impact": "moderate",
                        "feasibility": "feasible",
                        "timeliness": "timely",
                        "equity_ethics": "acceptable",
                        "cost_burden": "low",
                        "priority_class": "high",
                        "recommendation_class": "pursue_if_feasible",
                        "confidence": "moderate",
                        "rationale": "Synthetic rationale",
                        "source_ids": ["s1"],
                        "verification_state": "verified",
                    }
                ],
            }
        ],
        "audit_history": [],
        "policy": {
            "priority_order": ["urgent", "high", "routine", "defer"],
            "recommendation_order": [
                "pursue_now",
                "pursue_if_feasible",
                "monitor",
                "do_not_pursue",
            ],
            "conflict_policy": "preserve_dissent_no_resolution",
            "missingness_policy": "mark_incomplete",
            "ai_policy": "human_verification_required",
            "tie_policy": "complete_sets_declared_order",
        },
        "provenance": {
            "fixture_id": "wheel",
            "contract_reference": "v1",
            "source_snapshot": "synthetic",
            "redaction_policy_reference": "none",
        },
    }
    event = {
        "event_id": "approve",
        "sequence": 1,
        "previous_event_id": None,
        "previous_content_digest": None,
        "timestamp": "2026-08-01T00:00:00Z",
        "assessment_version": 1,
        "actor": {"actor_id": "human-1", "actor_type": "human"},
        "action": "approve",
        "assessment_content_digest": qualitative_assessment_content_digest(payload),
        "content_digest": "0" * 64,
        "redacted": False,
    }
    event["content_digest"] = qualitative_audit_event_digest(event)
    payload["audit_history"] = [event]
    result = qualitative_information_from_specification(payload)
    assert result.workflow_status == "complete"
    assert result.numerical_estimand is False
    assert "score" not in str(result.to_contract_dict()).lower()
    assert render_qualitative_information_text(
        result
    ) == render_qualitative_information_text(result)


def test_installed_wheel_executes_finite_additive_mcda_information() -> None:
    """The exact MCDA evaluator runs without repository fixtures."""
    from jsonschema import Draft202012Validator

    from voiage.contracts.mcda_information import MCDA_INFORMATION_INPUT_SCHEMA_V1
    from voiage.methods.mcda_information import mcda_information_value

    criteria = [
        {
            "criterion_id": criterion_id,
            "label": criterion_id.title(),
            "raw_unit": "point",
            "direction": "higher_is_better",
            "operational_definition": f"Synthetic {criterion_id} score.",
            "value_function": {
                "family": "linear_fixed_anchors",
                "normalization_scope": "fixed_ex_ante",
                "anchors": [
                    {"raw": 0.0, "value": 0.0},
                    {"raw": 1.0, "value": 1.0},
                ],
                "valid_domain": [0.0, 1.0],
                "extrapolation_policy": "reject",
                "elicitation_source": "wheel synthetic anchors",
            },
            "source_reference": "wheel synthetic model",
        }
        for criterion_id in ("benefit", "burden")
    ]
    joint_states = []
    for state_id, outcome, preference, probability, a_benefit, weight in (
        ("s1", "high", "benefit-heavy", 0.35, 1.0, 0.8),
        ("s2", "high", "burden-heavy", 0.15, 1.0, 0.2),
        ("s3", "low", "benefit-heavy", 0.15, 0.0, 0.8),
        ("s4", "low", "burden-heavy", 0.35, 0.0, 0.2),
    ):
        joint_states.append(
            {
                "state_id": state_id,
                "probability": probability,
                "partition_values": {
                    "outcome": outcome,
                    "preference": preference,
                },
                "performances": {
                    "A": {"benefit": a_benefit, "burden": 0.0},
                    "B": {"benefit": 0.5, "burden": 1.0},
                },
                "weights": {"benefit": weight, "burden": 1.0 - weight},
            }
        )

    def action(
        action_id: str,
        action_type: str,
        outcome_keys: list[str],
        preference_keys: list[str],
    ) -> dict[str, object]:
        return {
            "action_id": action_id,
            "action_type": action_type,
            "outcome_partition_keys": outcome_keys,
            "preference_partition_keys": preference_keys,
            "cost": {
                "original_amount": 0.0,
                "original_unit": "normalized value",
                "aggregate_amount": 0.0,
                "conversion_reference": "identity",
                "population_basis": "one synthetic person",
                "horizon_basis": "one synthetic period",
                "discount_basis": "none",
                "cost_scope": "action_specific_disjoint",
            },
        }

    payload = {
        "schema_version": "1.0.0",
        "analysis_id": "wheel-mcda",
        "analysis_type": "mcda_perfect_information",
        "method_maturity": "experimental",
        "aggregation_family": "compensatory_additive_value",
        "aggregate_direction": "maximize",
        "aggregate_unit": "normalized value",
        "alternatives": [
            {"alternative_id": name, "label": name, "definition_source": "wheel"}
            for name in ("A", "B")
        ],
        "criteria": criteria,
        "default_weights": {"benefit": 0.5, "burden": 0.5},
        "latent_partitions": {
            "outcome_keys": ["outcome"],
            "preference_keys": ["preference"],
            "dependence_assumption": "submitted correlated finite joint law",
        },
        "joint_states": joint_states,
        "information_actions": [
            action("learn-outcome", "criterion", ["outcome"], []),
            action("learn-preference", "preference", [], ["preference"]),
            action("learn-joint", "joint", ["outcome"], ["preference"]),
        ],
        "tolerances": {
            "absolute_tie": 1e-12,
            "relative_tie": 1e-12,
            "probability_sum": 1e-12,
            "weight_sum": 1e-12,
            "pareto_absolute": 1e-12,
        },
        "provenance": dict.fromkeys(
            (
                "decision_revision",
                "model_revision",
                "weight_elicitation_source",
                "joint_probability_source",
                "normalization_anchor_source",
                "partition_source",
                "cost_source",
                "tie_policy_source",
                "evaluator",
                "software_version",
            ),
            "wheel synthetic evidence",
        ),
    }
    payload["provenance"]["data_sources"] = ["wheel synthetic evidence"]
    payload["provenance"]["transformation_sources"] = ["fixed linear anchors"]
    Draft202012Validator(MCDA_INFORMATION_INPUT_SCHEMA_V1).validate(payload)
    result = mcda_information_value(payload).to_contract_dict()
    assert result["language_dispositions"]["python"] == "executable"
    assert result["decomposition"]["joint_gross_voi"] >= 0.0
    assert result["rank_acceptability"]["tie_convention"] == (
        "fractional_complete_tie_groups"
    )


def test_installed_wheel_metadata_keeps_jax_optional() -> None:
    """Verify the built artifact, rather than only source TOML metadata."""
    if os.environ.get("WHEEL_VENV") is None:
        return
    requirements = requires("voiage") or []
    jax_requirements = [
        item.lower() for item in requirements if item.lower().startswith("jax")
    ]

    assert jax_requirements
    assert all("extra ==" in item for item in jax_requirements)
    assert any(
        'extra == "jax"' in item or "extra == 'jax'" in item
        for item in jax_requirements
    )


def test_installed_native_provenance_matches_built_artifact() -> None:
    if os.environ.get("WHEEL_VENV") is None:
        return
    info = native.runtime_info()
    assert str(info["core_version"]).replace("-rc.", "rc") == version("voiage")
    assert info["source_revision"] == os.environ["EXPECTED_SOURCE_REVISION"]
    assert info["source_tree_git_oid"] == os.environ["EXPECTED_SOURCE_TREE_GIT_OID"]
    assert info["source_dirty"] is False
    assert info["runtime_info_schema"] == 3
    assert info["digest_algorithm"] == "rfc8785-sha256-v1"
    assert info["build_id_algorithm"] == "length-prefixed-sha256-v2"
    assert info["source_state_algorithm"] == "git-diff-and-untracked-sha256-v1"
    assert re.fullmatch(r"[0-9a-f]{64}", info["source_state_sha256"])
    assert info["source_state_sha256"] == _clean_source_state(info)
    assert re.fullmatch(r"[0-9a-f]{40}", info["source_tree_git_oid"])
    assert re.fullmatch(r"[0-9a-f]{64}", info["build_id"])
    assert info["build_id"] == _build_id(info)
    assert re.fullmatch(r"[0-9a-f]{64}", info["cargo_lock_sha256"])
    expected_platform = os.environ.get("EXPECTED_PLATFORM_SUFFIX")
    if expected_platform:
        target = info["target_triple"]
        if "linux" in expected_platform:
            assert "linux" in target
            assert "x86_64" in target
        elif "macos" in expected_platform:
            assert "apple-darwin" in target
            assert "aarch64" in target
        elif "win" in expected_platform:
            assert "windows" in target
            assert "x86_64" in target


def test_installed_private_diagnostics_do_not_expand_public_api() -> None:
    assert {"_core", "_runtime", "runtime_info", "runtime_info_schema"}.isdisjoint(
        voiage.__all__
    )


def test_installed_native_serializers_have_exact_payload_lineage() -> None:
    """Exercise both Rust-owned serializers from the installed artifact."""
    ceaf_before = dict(native.runtime_info()["operations"]["serialize_ceaf_result"])
    dominance_before = dict(
        native.runtime_info()["operations"]["serialize_dominance_result"]
    )
    ceaf = CEAFResult(
        wtp_thresholds=np.array([0.0]),
        optimal_strategy_indices=np.array([0]),
        optimal_strategy_names=["A"],
        acceptability_probabilities=np.array([1.0]),
        probability_lower=np.array([1.0]),
        probability_upper=np.array([1.0]),
        expected_net_benefit=np.array([1e20]),
        reporting={"standard": "CHEERS 2022"},
    ).to_dict(analysis_id="wheel-test", decision_problem_id="wheel-test")
    dominance = DominanceResult(
        strategy_names=["A", "B"],
        costs=np.array([1e20, 2e20]),
        effects=np.array([1.0, 2.0]),
        frontier_indices=[0, 1],
        strongly_dominated_indices=[],
        extended_dominated_indices=[],
        status=["frontier", "frontier"],
        incremental_costs=np.array([1.0]),
        incremental_effects=np.array([1.0]),
        icers=np.array([1.0]),
        reporting={"standard": "CHEERS 2022"},
    ).to_dict(analysis_id="wheel-test", decision_problem_id="wheel-test")

    assert ceaf == {
        "analysis_id": "wheel-test",
        "decision_problem_id": "wheel-test",
        "analysis_type": "ceaf",
        "wtp_thresholds": [0.0],
        "optimal_strategy_indices": [0],
        "optimal_strategy_names": ["A"],
        "acceptability_probabilities": [1.0],
        "probability_lower": [1.0],
        "probability_upper": [1.0],
        "expected_net_benefit": [1e20],
        "reporting": {"standard": "CHEERS 2022"},
    }
    assert dominance == {
        "analysis_id": "wheel-test",
        "decision_problem_id": "wheel-test",
        "analysis_type": "dominance",
        "strategy_names": ["A", "B"],
        "costs": [1e20, 2e20],
        "effects": [1.0, 2.0],
        "frontier_indices": [0, 1],
        "strongly_dominated_indices": [],
        "extended_dominated_indices": [],
        "status": ["frontier", "frontier"],
        "incremental_costs": [1.0],
        "incremental_effects": [1.0],
        "icers": [1.0],
        "reporting": {"standard": "CHEERS 2022"},
    }
    assert isinstance(ceaf["expected_net_benefit"][0], float)
    assert all(isinstance(value, float) for value in dominance["costs"])

    ceaf_after = native.runtime_info()["operations"]["serialize_ceaf_result"]
    dominance_after = native.runtime_info()["operations"]["serialize_dominance_result"]
    for before, after, payload in (
        (ceaf_before, ceaf_after, ceaf),
        (dominance_before, dominance_after, dominance),
    ):
        assert after["calls"] == before["calls"] + 1
        assert after["native_entries"] == before["native_entries"] + 1
        assert after["successes"] == before["successes"] + 1
        assert after["failures"] == before["failures"]
        assert after["last_payload_sha256"] == _digest(payload)
