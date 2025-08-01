"""Object-oriented interface for VOI calculations."""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

from .core.data_structures import NetBenefitArray, ParameterSet, PSASample, ValueArray
from .exceptions import InputError
from .methods.basic import evpi as _evpi
from .methods.basic import evppi as _evppi


class DecisionAnalysis:
    """Encapsulate data for VOI calculations."""

    def __init__(self, value_array: Union[np.ndarray, ValueArray, NetBenefitArray]):
        """
        Initialize the DecisionAnalysis object.

        Parameters
        ----------
        value_array : Union[np.ndarray, ValueArray, NetBenefitArray]
            A 2D array of net benefit values (n_samples, n_strategies).
        """
        if isinstance(value_array, np.ndarray):
            if value_array.ndim != 2:
                raise InputError(
                    "value_array ndarray must be 2D (n_samples, n_strategies)"
                )
            names = [f"strategy_{i}" for i in range(value_array.shape[1])]
            self.value_array = NetBenefitArray(value_array, names)
        elif isinstance(value_array, NetBenefitArray):
            self.value_array = value_array
        elif isinstance(value_array, ValueArray):
            self.value_array = value_array
        else:
            raise InputError("value_array must be a NumPy array or ValueArray")

    def evpi(self, **kwargs: Any) -> float:
        """Calculate EVPI using this object's value array."""
        return _evpi(self.value_array, **kwargs)

    def evppi(
        self,
        parameter_samples: Union[
            np.ndarray, PSASample, ParameterSet, Dict[str, np.ndarray]
        ],
        **kwargs: Any,
    ) -> float:
        """Calculate EVPPI for the supplied parameters."""
        return _evppi(self.value_array, parameter_samples, **kwargs)
