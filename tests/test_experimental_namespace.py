"""Verify the explicit, lazy experimental namespace boundary."""

import subprocess
import sys

import pytest


def test_experimental_namespace_is_lazy_and_explicit() -> None:
    code = (
        "import sys, voiage; "
        "assert 'experimental' in voiage.__all__; "
        "assert 'voiage.experimental' not in sys.modules; "
        "assert not any(name in {'voiage.methods.ai_assisted_evidence_triage', "
        "'voiage.methods.adaptive_learning_bandit'} for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_experimental_function_is_available_under_explicit_namespace() -> None:
    from voiage.experimental import value_of_adaptive_learning_bandit

    assert callable(value_of_adaptive_learning_bandit)


def test_experimental_functions_are_not_stable_top_level_exports() -> None:
    import voiage

    assert {
        "value_of_adaptive_learning_bandit",
        "value_of_ai_assisted_evidence_triage",
        "value_of_ambiguity_distribution_shift",
        "value_of_capacity_budget_constrained",
        "value_of_federated_privacy_preserving",
        "value_of_strategic_behavior",
    }.isdisjoint(voiage.__all__)


def test_unknown_experimental_export_raises_attribute_error() -> None:
    import voiage.experimental

    with pytest.raises(AttributeError, match="not_an_export"):
        _ = voiage.experimental.not_an_export
