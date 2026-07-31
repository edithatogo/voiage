"""Estimation-focused variance-reduction value of information.

This module is intentionally separate from decision-focused EVPPI and EVSI.
Its public names describe reductions in uncertainty about a declared target,
not expected changes in a decision's net benefit.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

_EVPPI_VAR: Final[Mapping[str, object]] = MappingProxyType(
    {
        "method_id": "evppi_var",
        "aliases": ("evppi_variance",),
        "family": "estimation-focused-variance-voi",
        "estimand_kind": "variance_reduction",
        "information_kind": "partial_perfect",
        "decision_focused": False,
        "sensitivity_index": False,
        "estimator_uncertainty": False,
        "maturity": "planned",
    }
)
_EVSI_VAR: Final[Mapping[str, object]] = MappingProxyType(
    {
        "method_id": "evsi_var",
        "aliases": ("evsi_variance",),
        "family": "estimation-focused-variance-voi",
        "estimand_kind": "variance_reduction",
        "information_kind": "sample",
        "decision_focused": False,
        "sensitivity_index": False,
        "estimator_uncertainty": False,
        "maturity": "planned",
    }
)

ESTIMATION_VARIANCE_METHODS: Final[Mapping[str, Mapping[str, object]]] = (
    MappingProxyType(
        {
            "evppi_var": _EVPPI_VAR,
            "evsi_var": _EVSI_VAR,
        }
    )
)

_ESTIMATION_VARIANCE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "evppi_var": "evppi_var",
        "evppi_variance": "evppi_var",
        "evsi_var": "evsi_var",
        "evsi_variance": "evsi_var",
    }
)


def estimation_variance_method(name: str) -> Mapping[str, object]:
    """Return the governed descriptor for an estimation-variance method.

    Decision-focused EVPPI/EVSI, sensitivity indices, and posterior-estimator
    uncertainty are deliberately not accepted as aliases.

    Parameters
    ----------
    name
        Canonical estimation method ID or an explicitly governed alias.

    Returns
    -------
    collections.abc.Mapping
        Immutable method metadata.

    Raises
    ------
    ValueError
        If ``name`` is not an estimation-focused variance method.
    """
    try:
        method_id = _ESTIMATION_VARIANCE_ALIASES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{name!r} is not an estimation-focused variance method"
        ) from error
    return ESTIMATION_VARIANCE_METHODS[method_id]


__all__ = ["ESTIMATION_VARIANCE_METHODS", "estimation_variance_method"]
