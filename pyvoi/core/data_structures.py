# pyvoi/core/data_structures.py

"""
Core data structures for pyVOI.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and are intended to work seamlessly with NumPy and Pandas/xarray.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# import pandas as pd # Optional: if Pandas Series/DataFrame are part of the structures
# import xarray as xr # Optional: if xarray DataArray/Dataset are part of the structures
from pyvoi.config import DEFAULT_DTYPE
from pyvoi.exceptions import DimensionMismatchError, InputError

# --- Core Data Arrays ---


@dataclass(frozen=True)
class NetBenefitArray:
    """
    Represents an array of net benefits, typically from a Probabilistic Sensitivity Analysis (PSA).

    Attributes
    ----------
        values (np.ndarray): A 2D NumPy array where rows correspond to PSA samples
                             and columns correspond to different treatment strategies or decisions.
                             Shape: (n_samples, n_strategies).
        strategy_names (Optional[List[str]]): Names of the strategies, corresponding to columns.
                                              Length must match n_strategies.
        parameter_names (Optional[List[str]]): Names of parameters if this array represents
                                               parameter values rather than net benefits directly.
                                               Used by EVPPI. Length must match n_parameters (a dimension of values).

    Raises
    ------
        InputError: If values is not a 2D NumPy array.
        DimensionMismatchError: If strategy_names length doesn't match the number of columns in values.
    """

    values: np.ndarray
    strategy_names: Optional[List[str]] = None
    # For EVPPI, `values` might be (n_samples, n_parameters_of_interest)
    # or (n_samples, n_strategies, n_parameters_of_interest) if structured differently.
    # This initial structure is simple; might need refinement for EVPPI's diverse inputs.

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            raise InputError("NetBenefitArray 'values' must be a NumPy array.")
        if self.values.ndim != 2:
            # This might be too restrictive for some EVPPI parameter inputs.
            # Consider allowing 1D for single strategy/parameter or >2D for complex cases.
            # For now, keeping it to (n_samples, n_strategies_or_parameters)
            raise DimensionMismatchError(
                f"NetBenefitArray 'values' must be a 2D array (samples x strategies/parameters). "
                f"Got {self.values.ndim} dimensions.",
            )
        if self.values.dtype != DEFAULT_DTYPE:
            # Consider casting or warning, rather than raising an error, for usability.
            # For now, strict to highlight config usage.
            # print(f"Warning: NetBenefitArray 'values' dtype {self.values.dtype} "
            #       f"does not match DEFAULT_DTYPE {DEFAULT_DTYPE}. Consider casting.")
            pass  # Relaxing this for now, as input data might not always match default.

        if self.strategy_names is not None:
            if not isinstance(self.strategy_names, list) or not all(
                isinstance(name, str) for name in self.strategy_names
            ):
                raise InputError("'strategy_names' must be a list of strings.")
            if len(self.strategy_names) != self.n_strategies:
                raise DimensionMismatchError(
                    f"Length of 'strategy_names' ({len(self.strategy_names)}) must match "
                    f"the number of strategies in 'values' ({self.n_strategies}).",
                )

    @property
    def n_samples(self) -> int:
        """Number of samples (rows)."""
        return self.values.shape[0]

    @property
    def n_strategies(self) -> int:
        """Number of strategies or parameters (columns)."""
        return self.values.shape[1]

    # Consider adding methods for common operations, e.g.,
    # - get_strategy_by_name(name: str) -> np.ndarray
    # - to_pandas() -> pd.DataFrame (if pandas is a dependency)
    # - to_xarray() -> xr.DataArray (if xarray is a dependency)


@dataclass(frozen=True)
class PSASample:
    """
    Represents a collection of parameter samples from a PSA.

    This can be structured in various ways, e.g., a dictionary of arrays,
    or an xarray.Dataset for more complex, labeled data.

    Attributes
    ----------
        parameters (Union[Dict[str, np.ndarray], 'xr.Dataset']):
            The PSA parameter samples.
            - If Dict: Keys are parameter names, values are 1D NumPy arrays (one value per sample).
                       All arrays must have the same length (n_samples).
            - If xr.Dataset: A more structured representation, potentially with coordinates.
        n_samples (int): The number of samples. Automatically inferred if possible.

    Raises
    ------
        InputError: If parameters format is invalid or sample counts are inconsistent.
    """

    parameters: Union[
        Dict[str, np.ndarray], Any
    ]  # Using Any for xr.Dataset to avoid hard dep initially
    # n_samples: Optional[int] = None # Made n_samples a property

    def __post_init__(self):  # noqa: C901
        if isinstance(self.parameters, dict):
            if not self.parameters:
                raise InputError("PSASample 'parameters' dictionary cannot be empty.")

            current_n_samples = -1
            for name, values in self.parameters.items():
                if not isinstance(name, str):
                    raise InputError(
                        "Parameter names in PSASample dictionary must be strings."
                    )
                if not isinstance(values, np.ndarray):
                    raise InputError(
                        f"Parameter '{name}' values must be a NumPy array."
                    )
                if values.ndim != 1:
                    raise DimensionMismatchError(
                        f"Parameter '{name}' array must be 1D (samples). Got {values.ndim} dimensions.",
                    )
                if values.dtype != DEFAULT_DTYPE:
                    # print(f"Warning: PSASample parameter '{name}' dtype {values.dtype} "
                    # f"does not match DEFAULT_DTYPE {DEFAULT_DTYPE}. Consider casting.")
                    pass

                if current_n_samples == -1:
                    current_n_samples = len(values)
                elif len(values) != current_n_samples:
                    raise DimensionMismatchError(
                        "All parameter arrays in PSASample dictionary must have the same length (n_samples).",
                    )
            if current_n_samples == -1:  # Should not happen if dict is not empty
                raise InputError(
                    "Could not determine n_samples from parameters dictionary."
                )
            self._n_samples = current_n_samples

        # elif isinstance(self.parameters, xr.Dataset):
        #     # Add validation for xarray.Dataset if it becomes a primary supported type
        #     # For example, check for a 'sample' dimension or coordinate.
        #     if 'sample' not in self.parameters.dims and 'draw' not in self.parameters.dims : # Common names
        #         raise InputError("xr.Dataset for PSASample must have a 'sample' or 'draw' dimension.")
        #     # self._n_samples = self.parameters.dims.get('sample', self.parameters.dims.get('draw'))
        #     if 'sample' in self.parameters.dims:
        #         self._n_samples = self.parameters.dims['sample']
        #     elif 'draw' in self.parameters.dims:
        #         self._n_samples = self.parameters.dims['draw']
        #     else: # Should be caught above
        #         raise InputError("Could not determine n_samples from xarray.Dataset.")
        else:
            raise InputError(
                "PSASample 'parameters' must be a dictionary of NumPy arrays or an xarray.Dataset.",
            )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        # This logic is now in __post_init__ to set an internal _n_samples
        # This is to ensure n_samples is validated at creation.
        if hasattr(self, "_n_samples"):
            return self._n_samples

        # Fallback logic, though __post_init__ should handle it.
        if isinstance(self.parameters, dict):
            if not self.parameters:
                return 0
            return len(next(iter(self.parameters.values())))
        # elif isinstance(self.parameters, xr.Dataset):
        #     if 'sample' in self.parameters.dims: return self.parameters.dims['sample']
        #     if 'draw' in self.parameters.dims: return self.parameters.dims['draw']
        return 0  # Should not be reached if validation passes

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter names."""
        if isinstance(self.parameters, dict):
            return list(self.parameters.keys())
        # elif isinstance(self.parameters, xr.Dataset):
        #     return list(self.parameters.data_vars.keys())
        return []


# --- Study Design and Portfolio Structures ---


@dataclass(frozen=True)
class TrialArm:
    """
    Represents a single arm in a clinical trial design.

    Attributes
    ----------
        name (str): Name of the trial arm (e.g., "Treatment A", "Placebo").
        sample_size (int): Number of subjects allocated to this arm.
        # Other arm-specific details can be added, e.g., cost_per_subject, observation_schedule.
    """

    name: str
    sample_size: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise InputError("TrialArm 'name' must be a non-empty string.")
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise InputError("TrialArm 'sample_size' must be a positive integer.")


@dataclass(frozen=True)
class TrialDesign:
    """
    Specifies the design of a proposed trial for EVSI calculations.

    Attributes
    ----------
        arms (List[TrialArm]): A list of TrialArm objects defining the arms of the trial.
        # study_duration (Optional[float]): Duration of the study (e.g., in years).
        # cost_fixed (Optional[float]): Fixed costs associated with the trial.
        # adaptive_rules (Optional[Any]): Specification for adaptive trial designs (e.g., interim analysis rules).
                                       # This would be a more complex structure.
    """

    arms: List[TrialArm]
    # Other trial-wide parameters

    def __post_init__(self):
        if not isinstance(self.arms, list) or not self.arms:
            raise InputError(
                "TrialDesign 'arms' must be a non-empty list of TrialArm objects."
            )
        if not all(isinstance(arm, TrialArm) for arm in self.arms):
            raise InputError("All elements in 'arms' must be TrialArm objects.")
        arm_names = [arm.name for arm in self.arms]
        if len(arm_names) != len(set(arm_names)):
            raise InputError("TrialArm names within a TrialDesign must be unique.")

    @property
    def total_sample_size(self) -> int:
        """Total sample size across all arms."""
        return sum(arm.sample_size for arm in self.arms)


@dataclass(frozen=True)
class PortfolioStudy:
    """
    Represents a candidate study within a research portfolio.

    Attributes
    ----------
        name (str): Name of the study.
        design (TrialDesign): The design of this candidate study.
        cost (float): Estimated cost of conducting this study.
        # expected_evsi (Optional[float]): Pre-calculated or estimated EVSI for this study.
        # other_attributes (Optional[Dict[str, Any]]): e.g., duration, feasibility score.
    """

    name: str
    design: TrialDesign  # Or a more generic "StudySpecification" if not always a trial
    cost: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise InputError("PortfolioStudy 'name' must be a non-empty string.")
        if not isinstance(self.design, TrialDesign):  # Or StudySpecification
            raise InputError("PortfolioStudy 'design' must be a TrialDesign object.")
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise InputError("PortfolioStudy 'cost' must be a non-negative number.")


@dataclass(frozen=True)
class PortfolioSpec:
    """
    Defines a portfolio of candidate research studies for Portfolio VOI.

    Attributes
    ----------
        studies (List[PortfolioStudy]): List of candidate studies.
        budget_constraint (Optional[float]): Overall budget limit for the portfolio.
        # other_constraints (Optional[Any]): e.g., constraints on number of studies, types.
    """

    studies: List[PortfolioStudy]
    budget_constraint: Optional[float] = None

    def __post_init__(self):
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
    """
    Specification for dynamic or sequential VOI analyses.

    Attributes
    ----------
        time_steps (Sequence[float]): A sequence of time points (e.g., years from present)
                                      at which decisions or data accrual occur.
        # interim_rules (Optional[Any]): Rules for interim analyses or adaptive decisions.
        # discount_rate (Optional[float]): Discount rate for future costs/benefits if applicable.
    """

    time_steps: Sequence[
        float
    ]  # Using Sequence for more flexibility (list, tuple, np.array)

    def __post_init__(self):
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


if __name__ == "__main__":
    # Example Usage (for testing during development)
    print("--- NetBenefitArray Examples ---")
    nb_vals = np.array([[10, 20], [11, 22], [12, 18]], dtype=DEFAULT_DTYPE)
    nba = NetBenefitArray(values=nb_vals, strategy_names=["Strategy A", "Strategy B"])
    print(f"NBA: {nba}")
    print(f"NBA samples: {nba.n_samples}, strategies: {nba.n_strategies}")

    try:
        NetBenefitArray(values=np.array([1, 2, 3]))  # Wrong dimensions
    except DimensionMismatchError as e:
        print(f"Caught expected error: {e}")

    print("\n--- PSASample Examples ---")
    params_dict = {
        "param1": np.array([1.0, 1.1, 1.2], dtype=DEFAULT_DTYPE),
        "param2": np.array([0.5, 0.4, 0.6], dtype=DEFAULT_DTYPE),
    }
    psa = PSASample(parameters=params_dict)
    print(f"PSA: {psa}")
    print(f"PSA samples: {psa.n_samples}, param names: {psa.parameter_names}")

    try:
        PSASample(
            parameters={"p1": np.array([1, 2]), "p2": np.array([3, 4, 5])}
        )  # Mismatched lengths
    except DimensionMismatchError as e:
        print(f"Caught expected error: {e}")

    print("\n--- TrialDesign Examples ---")
    arm1 = TrialArm(name="Treatment X", sample_size=100)
    arm2 = TrialArm(name="Control", sample_size=100)
    trial = TrialDesign(arms=[arm1, arm2])
    print(f"Trial: {trial}")
    print(f"Total sample size: {trial.total_sample_size}")

    try:
        TrialDesign(
            arms=[TrialArm("T1", 50), TrialArm("T1", 60)]
        )  # Duplicate arm names
    except InputError as e:
        print(f"Caught expected error: {e}")

    print("\n--- PortfolioSpec Examples ---")
    study1_design = TrialDesign(arms=[TrialArm("S1_ArmA", 50), TrialArm("S1_ArmB", 50)])
    study1 = PortfolioStudy(name="Study Alpha", design=study1_design, cost=100000)
    study2_design = TrialDesign(arms=[TrialArm("S2_ArmOnly", 200)])
    study2 = PortfolioStudy(name="Study Beta", design=study2_design, cost=150000)
    portfolio = PortfolioSpec(studies=[study1, study2], budget_constraint=200000)
    print(f"Portfolio: {portfolio}")

    print("\n--- DynamicSpec Examples ---")
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2, 5])
    print(f"Dynamic Spec: {dyn_spec}")

    # Example of how DEFAULT_DTYPE from config is used
    print(f"\nDefault dtype from config: {DEFAULT_DTYPE}")
    test_array = np.array([1, 2, 3], dtype=DEFAULT_DTYPE)
    print(f"Test array with default dtype: {test_array}, dtype: {test_array.dtype}")
