# voiage/core/data_structures.py

"""
Core data structures for voiage.

This module provides backward-compatible wrappers for the core data structures
defined in `voiage.schema`. New code should use the classes from `voiage.schema`
directly.
"""

from typing import Dict, List

import numpy as np
import xarray as xr

from voiage.schema import DecisionOption as NewDecisionOption
from voiage.schema import ParameterSet as NewParameterSet
from voiage.schema import ValueArray as NewValueArray


class NetBenefitArray(NewValueArray):
    """Backward-compatible wrapper around :class:`voiage.schema.ValueArray`."""

    def __init__(
        self: "NetBenefitArray", values: np.ndarray, strategy_names: List[str]
    ):
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(values.shape[0]),
                "n_strategies": np.arange(values.shape[1]),
                "strategy": ("n_strategies", strategy_names),
            },
        )
        super().__init__(dataset=dataset)


class PSASample(NewParameterSet):
    """Backward-compatible wrapper around :class:`voiage.schema.ParameterSet`."""

    def __init__(self: "PSASample", parameters: Dict[str, np.ndarray]):
        dataset = xr.Dataset(
            {k: ("n_samples", np.asarray(v)) for k, v in parameters.items()},
            coords={"n_samples": np.arange(len(next(iter(parameters.values()))))},
        )
        super().__init__(dataset=dataset)


# --- Backwards Compatibility Alias ---
# Allow old code to use ``TrialArm`` to refer to ``DecisionOption``.
TrialArm = NewDecisionOption

# Re-export the new names for convenience, so other modules can import them from here
# during the transition period.
ValueArray = NewValueArray
ParameterSet = NewParameterSet
DecisionOption = NewDecisionOption
