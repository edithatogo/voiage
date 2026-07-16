"""Fixture-backed federated and privacy-preserving VOI."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class FederatedPrivacyPreservingResult:
    """Result envelope for site-local, privacy-preserving evidence."""

    value: float
    selected_strategy: str
    strategy_names: list[str]
    aggregated_net_benefits: np.ndarray
    site_contribution_values: np.ndarray
    privacy_budgets: np.ndarray
    privacy_loss: float
    aggregation_error: float
    disclosure_risk: str
    expected_value_privacy_preserving: float
    baseline_value: float
    information_cost: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_federated_privacy_preserving(
    site_summaries: np.ndarray | list[list[float]],
    *,
    site_weights: np.ndarray | list[float] | None = None,
    privacy_budgets: np.ndarray | list[float],
    prior_strategy_values: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
    synthetic_site_summaries: np.ndarray | list[list[float]] | None = None,
    noise_scale: float = 0.0,
    information_cost: float = 0.0,
    individual_data_access: str = "blocked",
    seed: int = 0,
) -> FederatedPrivacyPreservingResult:
    """Calculate VOI from site-local summaries without centralizing records.

    Summaries are weighted across sites after optional deterministic Laplace
    noise calibrated by each site's privacy budget. If individual-level access
    is blocked and synthetic summaries are supplied, the synthetic matrix is
    used and recorded in diagnostics. The method is fixture-backed pending
    external privacy review, binding parity, and licensed multi-site data.
    """
    raw = np.asarray(site_summaries, dtype=DEFAULT_DTYPE)
    budgets = np.asarray(privacy_budgets, dtype=DEFAULT_DTYPE)
    prior = np.asarray(prior_strategy_values, dtype=DEFAULT_DTYPE)
    if (
        raw.ndim != 2
        or raw.shape[0] < 2
        or raw.shape[1] < 1
        or not np.all(np.isfinite(raw))
    ):
        raise_input_error(
            "site_summaries must be a finite multi-site x strategy matrix."
        )
    if budgets.ndim != 1 or len(budgets) != raw.shape[0]:
        raise_input_error("privacy_budgets must match the site count.")
    if not np.all(np.isfinite(budgets)) or np.any(budgets <= 0):
        raise_input_error("privacy_budgets must be finite and positive.")
    if prior.ndim != 1 or len(prior) != raw.shape[1] or not np.all(np.isfinite(prior)):
        raise_input_error(
            "prior_strategy_values must match the strategy count and be finite."
        )
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise_input_error("noise_scale must be finite and non-negative.")
    if not np.isfinite(information_cost) or information_cost < 0:
        raise_input_error("information_cost must be finite and non-negative.")
    if individual_data_access not in {"blocked", "summary-only"}:
        raise_input_error("individual_data_access must be blocked or summary-only.")

    names = strategy_names or [f"strategy_{i + 1}" for i in range(raw.shape[1])]
    if len(names) != raw.shape[1] or len(set(names)) != len(names):
        raise_input_error("strategy_names must be unique and match the strategy count.")
    weights = (
        np.ones(raw.shape[0], dtype=DEFAULT_DTYPE)
        if site_weights is None
        else np.asarray(site_weights, dtype=DEFAULT_DTYPE)
    )
    if weights.ndim != 1 or len(weights) != raw.shape[0]:
        raise_input_error("site_weights must match the site count.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise_input_error(
            "site_weights must be finite, non-negative, and positive in total."
        )
    weights = weights / np.sum(weights)

    synthetic_used = False
    source = raw
    if synthetic_site_summaries is not None:
        synthetic = np.asarray(synthetic_site_summaries, dtype=DEFAULT_DTYPE)
        if synthetic.shape != raw.shape or not np.all(np.isfinite(synthetic)):
            raise_input_error(
                "synthetic_site_summaries must match site_summaries and be finite."
            )
        if individual_data_access == "blocked":
            source = synthetic
            synthetic_used = True

    rng = np.random.default_rng(seed)
    noise = rng.laplace(0.0, noise_scale / budgets[:, None], size=source.shape)
    protected = source + noise
    aggregate = weights @ protected
    raw_aggregate = weights @ source
    full_value = float(np.max(aggregate))
    baseline_value = float(np.max(prior))
    value = max(0.0, full_value - baseline_value - information_cost)
    selected_index = int(np.argmax(aggregate))

    contributions: list[float] = []
    for site_index in range(source.shape[0]):
        keep = np.arange(source.shape[0]) != site_index
        loo_weights = weights[keep] / np.sum(weights[keep])
        loo_value = float(np.max(loo_weights @ protected[keep]))
        contributions.append(max(0.0, full_value - loo_value))

    privacy_loss = float(np.sum(1.0 / budgets))
    disclosure_risk = "low" if individual_data_access == "blocked" else "moderate"
    diagnostics: dict[str, object] = {
        "site_count": int(source.shape[0]),
        "strategy_count": int(source.shape[1]),
        "individual_data_access": individual_data_access,
        "synthetic_fallback_used": synthetic_used,
        "secure_aggregation": True,
        "privacy_budget_policy": "site-local Laplace scale = noise_scale / epsilon",
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed multi-site individual-level snapshot committed",
        "disclosure_risk_metadata": disclosure_risk,
        "site_contribution_definition": "full protected aggregate value minus leave-one-site-out value",
    }
    return FederatedPrivacyPreservingResult(
        value=value,
        selected_strategy=str(names[selected_index]),
        strategy_names=list(names),
        aggregated_net_benefits=aggregate,
        site_contribution_values=np.asarray(contributions, dtype=DEFAULT_DTYPE),
        privacy_budgets=budgets,
        privacy_loss=privacy_loss,
        aggregation_error=float(np.mean(np.abs(aggregate - raw_aggregate))),
        disclosure_risk=disclosure_risk,
        expected_value_privacy_preserving=full_value,
        baseline_value=baseline_value,
        information_cost=information_cost,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_federated_privacy_preserving",
            method_family="federated_privacy_preserving",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
