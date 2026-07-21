"""Explicit namespace for experimental and research-only APIs.

These exports are not part of the stable v1 contract. They are resolved lazily
so importing the namespace does not import every experimental dependency.
"""

from importlib import import_module

_EXPORTS = {
    "value_of_adaptive_learning_bandit": ".methods.adaptive_learning_bandit",
    "value_of_ai_assisted_evidence_triage": ".methods.ai_assisted_evidence_triage",
    "value_of_ambiguity_distribution_shift": ".methods.ambiguity_distribution_shift",
    "value_of_capacity_budget_constrained": ".methods.capacity_budget_constrained",
    "value_of_federated_privacy_preserving": ".methods.federated_privacy_preserving",
    "value_of_strategic_behavior": ".methods.strategic_behavior",
}

__all__ = list(_EXPORTS)
__maturity__ = "experimental"
__stability__ = "not part of the v1 stable API"


def __getattr__(name: str) -> object:
    """Resolve an experimental export only when requested."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, "voiage")
    value = getattr(module, name)
    globals()[name] = value
    return value
