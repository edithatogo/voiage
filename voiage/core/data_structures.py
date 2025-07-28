# voiage/core/data_structures.py

"""
Core data structures for voiage.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and are intended to work seamlessly with NumPy and Pandas/xarray.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# import pandas as pd # Optional: if Pandas Series/DataFrame are part of the structures
# import xarray as xr # Optional: if xarray DataArray/Dataset are part of the structures
from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import DimensionMismatchError, InputError

# --- Core Data Arrays ---


import xarray as xr

@dataclass(frozen=True)
class ValueArray:
    """
    A container for net benefit values from a PSA.
    """
    dataset: xr.Dataset

    def __post_init__(self: "ValueArray"):
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ValueArray 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise InputError("ValueArray 'dataset' must have a 'n_samples' dimension.")
        if "n_strategies" not in self.dataset.dims:
            raise InputError("ValueArray 'dataset' must have a 'n_strategies' dimension.")
        if "net_benefit" not in self.dataset.data_vars:
            raise InputError("ValueArray 'dataset' must have a 'net_benefit' data variable.")

    @property
    def values(self: "ValueArray") -> np.ndarray:
        return self.dataset["net_benefit"].values

    @property
    def n_samples(self: "ValueArray") -> int:
        return self.dataset.dims["n_samples"]

    @property
    def n_strategies(self: "ValueArray") -> int:
        return self.dataset.dims["n_strategies"]

    @property
    def strategy_names(self: "ValueArray") -> List[str]:
        return self.dataset["strategy"].values.tolist()


class NetBenefitArray(ValueArray):
    """Backward-compatible wrapper around :class:`ValueArray`."""

    def __init__(self: "NetBenefitArray", values: np.ndarray, strategy_names: List[str]):
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(values.shape[0]),
                "n_strategies": np.arange(values.shape[1]),
                "strategy": ("n_strategies", strategy_names),
            },
        )
        super().__init__(dataset=dataset)


@dataclass(frozen=True)
class ParameterSet:
    """
    A container for parameter samples from a PSA.
    """
    dataset: xr.Dataset

    def __post_init__(self: "ParameterSet"):
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ParameterSet 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise InputError("ParameterSet 'dataset' must have a 'n_samples' dimension.")

    @property
    def parameters(self: "ParameterSet") -> Dict[str, np.ndarray]:
        return {name: self.dataset[name].values for name in self.dataset.data_vars}

    @property
    def n_samples(self: "ParameterSet") -> int:
        return self.dataset.dims["n_samples"]

    @property
    def parameter_names(self: "ParameterSet") -> List[str]:
        return list(self.dataset.data_vars.keys())


class PSASample(ParameterSet):
    """Backward-compatible wrapper around :class:`ParameterSet`."""

    def __init__(self: "PSASample", parameters: Dict[str, np.ndarray]):
        dataset = xr.Dataset(
            {k: ("n_samples", np.asarray(v)) for k, v in parameters.items()},
            coords={"n_samples": np.arange(len(next(iter(parameters.values()))))},
        )
        super().__init__(dataset=dataset)


@dataclass(frozen=True)
class DecisionOption:
    """Represents a single arm in a clinical trial design.

    Attributes
    ----------
    name : str
        The name of the trial arm (e.g., "Treatment A", "Placebo").
    sample_size : int
        The number of subjects to be allocated to this arm.

    Raises
    ------
    InputError
        If `name` is not a non-empty string or `sample_size` is not a
        positive integer.
    """

    name: str
    sample_size: int

    def __post_init__(self: "DecisionOption"):
        if not isinstance(self.name, str) or not self.name:
            raise InputError("DecisionOption 'name' must be a non-empty string.")
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise InputError("DecisionOption 'sample_size' must be a positive integer.")


@dataclass(frozen=True)
class TrialDesign:
    """Specifies the design of a proposed trial for EVSI calculations.

    Attributes
    ----------
    arms : List[DecisionOption]
        A list of `DecisionOption` objects that together define the trial.

    Raises
    ------
    InputError
        If `arms` is not a non-empty list of `DecisionOption` objects, or if
        any of the arm names are duplicated.
    """

    arms: List[DecisionOption]

    def __post_init__(self: "TrialDesign"):
        if not isinstance(self.arms, list) or not self.arms:
            raise InputError(
                "TrialDesign 'arms' must be a non-empty list of DecisionOption objects."
            )
        if not all(isinstance(arm, DecisionOption) for arm in self.arms):
            raise InputError("All elements in 'arms' must be DecisionOption objects.")
        arm_names = [arm.name for arm in self.arms]
        if len(arm_names) != len(set(arm_names)):
            raise InputError("DecisionOption names within a TrialDesign must be unique.")

    @property
    def total_sample_size(self: "TrialDesign") -> int:
        """Return the total sample size across all arms."""
        return sum(arm.sample_size for arm in self.arms)


@dataclass(frozen=True)
class PortfolioStudy:
    """Represents a single candidate study within a research portfolio.

    Attributes
    ----------
    name : str
        The name of the candidate study.
    design : TrialDesign
        The `TrialDesign` object specifying the study's design.
    cost : float
        The estimated cost of conducting this study.

    Raises
    ------
    InputError
        If inputs are of the wrong type or `cost` is negative.
    """

    name: str
    design: TrialDesign
    cost: float

    def __post_init__(self: "PortfolioStudy"):
        if not isinstance(self.name, str) or not self.name:
            raise InputError("PortfolioStudy 'name' must be a non-empty string.")
        if not isinstance(self.design, TrialDesign):
            raise InputError("PortfolioStudy 'design' must be a TrialDesign object.")
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise InputError("PortfolioStudy 'cost' must be a non-negative number.")


@dataclass(frozen=True)
class PortfolioSpec:
    """Defines a portfolio of candidate research studies for optimization.

    Attributes
    ----------
    studies : List[PortfolioStudy]
        A list of `PortfolioStudy` objects representing the candidate studies.
    budget_constraint : Optional[float], optional
        The overall budget limit for the portfolio. Defaults to None.

    Raises
    ------
    InputError
        If `studies` is not a non-empty list of `PortfolioStudy` objects,
        if study names are duplicated, or if `budget_constraint` is negative.
    """

    studies: List[PortfolioStudy]
    budget_constraint: Optional[float] = None

    def __post_init__(self: "PortfolioSpec"):
        if not isinstance(self.studies, list) or not self.studies:
            raise InputError(
                "PortfolioSpec 'studies' must be a non-empty list of PortfolioStudy objects."
            )
        if not all(isinstance(study, PortfolioStudy) for study in self.studies):
            raise InputError(
                "All elements in 'studies' must be PortfolioStudy objects."
            )
        study_names = [study.name for study in self.studies]
        if len(study_names) != len(set(study_names)):
            raise InputError(
                "PortfolioStudy names within a PortfolioSpec must be unique."
            )

        if self.budget_constraint is not None:
            if (
                not isinstance(self.budget_constraint, (int, float))
                or self.budget_constraint < 0
            ):
                raise InputError(
                    "PortfolioSpec 'budget_constraint' must be a non-negative number if specified."
                )


@dataclass(frozen=True)
class DynamicSpec:
    """Specification for dynamic or sequential VOI analyses.

    Attributes
    ----------
    time_steps : Sequence[float]
        A sequence of time points (e.g., years from present) at which
        decisions or data accrual occur.

    Raises
    ------
    InputError
        If `time_steps` is not a non-empty sequence of numbers.
    """

    time_steps: Sequence[float]

    def __post_init__(self: "DynamicSpec"):
        if not isinstance(self.time_steps, Sequence) or not self.time_steps:
            raise InputError(
                "'time_steps' must be a non-empty sequence (list, tuple, np.array)."
            )
        if not all(isinstance(t, (int, float)) for t in self.time_steps):
            raise InputError("All elements in 'time_steps' must be numbers.")
        # Could add checks for sorted time_steps if required by the logic
        # if not all(self.time_steps[i] <= self.time_steps[i+1] for i in range(len(self.time_steps)-1)):
        #     raise InputError("'time_steps' must be sorted in non-decreasing order.")


# --- Model Function Protocol (for EVSI) ---
# from typing import Protocol, Callable

# class ModelFunction(Protocol):
#     """
#     Protocol for a model function used in EVSI calculations.
#     The model function takes parameter samples and returns net benefit arrays.
#     """
#     def __call__(self, psa_sample: PSASample, trial_data: Optional[Any] = None) -> NetBenefitArray:
#         ...

# This helps in type hinting functions that expect a certain kind of callable model.
# `trial_data` would be the simulated data from a `TrialDesign`.
# The exact signature might vary based on EVSI method (e.g., some might need pre- and post-study models).

# Example of a simple model function (conceptual)
# def my_health_economic_model(parameters: Dict[str, np.ndarray]) -> np.ndarray:
#     # parameters: {'param_a': array([...]), 'param_b': array([...])}
#     # Perform calculations using sampled parameters
#     # Returns: array of net benefits (n_samples, n_strategies)
#     cost_tx_a = parameters['cost_a_param'] * 1.2 + parameters['common_cost']
#     qaly_tx_a = parameters['qaly_a_param'] * 0.95
#     nb_tx_a = qaly_tx_a * WTP - cost_tx_a
#     # ... similar for other treatments ...
#     return np.stack([nb_tx_a, nb_tx_b], axis=-1)

# def evsi_model_wrapper(psa_sample: PSASample, trial_data: Optional[Any] = None) -> NetBenefitArray:
#     # Adapt the raw model function (like my_health_economic_model) to this interface
#     # If trial_data is present, it might update the psa_sample (e.g., Bayesian update)
#     # before calling the core economic model.
#     if trial_data is not None:
#         # Incorporate trial_data to update beliefs about parameters
#         # This is the complex part of EVSI simulation
#         updated_params = psa_sample.parameters # Placeholder for Bayesian update logic
#     else:
#         updated_params = psa_sample.parameters
#
    #     nb_values = my_health_economic_model(updated_params)
    #     return NetBenefitArray(values=nb_values)

# --- Backwards Compatibility Alias ---
# Allow old code to use ``TrialArm`` to refer to ``DecisionOption``.
TrialArm = DecisionOption

