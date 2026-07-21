"""Verify the explicit, lazy experimental namespace boundary."""

import subprocess
import sys


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
