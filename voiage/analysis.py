from __future__ import annotations

"""Object-oriented interface for VOI calculations."""

from typing import Dict, Union, Any
import numpy as np

from .core.data_structures import (
    ValueArray,
    NetBenefitArray,
    PSASample,
    ParameterSet,
)
from .exceptions import InputError
from .methods.basic import evpi as _evpi, evppi as _evppi


class DecisionAnalysis:
    """Encapsulate data for VOI calculations."""

    def __init__(self, value_array: Union[np.ndarray, ValueArray, NetBenefitArray]):
        if isinstance(value_array, np.ndarray):
            if value_array.ndim != 2:
                raise InputError("value_array ndarray must be 2D (n_samples, n_strategies)")
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
        parameter_samples: Union[np.ndarray, PSASample, ParameterSet, Dict[str, np.ndarray]],
        **kwargs: Any,
    ) -> float:
        """Calculate EVPPI for the supplied parameters."""
        return _evppi(self.value_array, parameter_samples, **kwargs)
