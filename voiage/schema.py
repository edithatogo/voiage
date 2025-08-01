# voiage/schema.py

"""
Core, domain-agnostic data structures for voiage.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and are intended to work seamlessly with NumPy and xarray.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from voiage.exceptions import InputError


@dataclass(frozen=True)
class ValueArray:
    """A container for value (e.g., net benefit) arrays from a PSA."""

    dataset: xr.Dataset

    def __post_init__(self: "ValueArray"):
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ValueArray 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.sizes:
            raise InputError("ValueArray 'dataset' must have a 'n_samples' dimension.")
        if "n_options" not in self.dataset.sizes:
            raise InputError("ValueArray 'dataset' must have a 'n_options' dimension.")
        if "value" not in self.dataset.data_vars:
            raise InputError("ValueArray 'dataset' must have a 'value' data variable.")

    @property
    def values(self: "ValueArray") -> np.ndarray:
        return self.dataset["value"].values

    @property
    def n_samples(self: "ValueArray") -> int:
        return self.dataset.sizes["n_samples"]

    @property
    def n_options(self: "ValueArray") -> int:
        return self.dataset.sizes["n_options"]

    @property
    def option_names(self: "ValueArray") -> List[str]:
        return [str(name) for name in self.dataset["option"].values]


@dataclass(frozen=True)
class ParameterSet:
    """A container for parameter samples from a PSA."""

    dataset: xr.Dataset

    def __post_init__(self: "ParameterSet"):
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ParameterSet 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.sizes:
            raise InputError(
                "ParameterSet 'dataset' must have a 'n_samples' dimension."
            )

    @property
    def parameters(self: "ParameterSet") -> Dict[str, np.ndarray]:
        return {str(name): self.dataset[name].values for name in self.dataset.data_vars}

    @property
    def n_samples(self: "ParameterSet") -> int:
        return self.dataset.sizes["n_samples"]

    @property
    def parameter_names(self: "ParameterSet") -> List[str]:
        return list(self.dataset.data_vars.keys())


@dataclass(frozen=True)
class DecisionOption:
    """Represents a single choice in a decision problem.

    For EVSI, this can be extended to represent a single arm in a study design.

    Attributes
    ----------
    name : str
        The name of the decision option (e.g., "Treatment A", "Strategy X").
    sample_size : Optional[int]
        For EVSI, the number of subjects to be allocated to this arm.
    """

    name: str
    sample_size: Optional[int] = None

    def __post_init__(self: "DecisionOption"):
        if not isinstance(self.name, str) or not self.name:
            raise InputError("DecisionOption 'name' must be a non-empty string.")
        if self.sample_size is not None:
            if not isinstance(self.sample_size, int) or self.sample_size <= 0:
                raise InputError(
                    "DecisionOption 'sample_size' must be a positive integer if specified."
                )


@dataclass(frozen=True)
class StudyDesign:
    """Specifies the design of a proposed study for EVSI calculations.

    Attributes
    ----------
    options : List[DecisionOption]
        A list of `DecisionOption` objects that together define the study.
    """

    options: List[DecisionOption]

    def __post_init__(self: "StudyDesign"):
        if not isinstance(self.options, list) or not self.options:
            raise InputError(
                "StudyDesign 'options' must be a non-empty list of DecisionOption objects."
            )
        if not all(isinstance(opt, DecisionOption) for opt in self.options):
            raise InputError(
                "All elements in 'options' must be DecisionOption objects."
            )

        option_names = [opt.name for opt in self.options]
        if len(option_names) != len(set(option_names)):
            raise InputError(
                "DecisionOption names within a StudyDesign must be unique."
            )

    @property
    def total_sample_size(self: "StudyDesign") -> int:
        """Return the total sample size across all arms."""
        return sum(opt.sample_size for opt in self.options if opt.sample_size)
