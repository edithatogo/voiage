# voiage/methods/network_meta_analysis.py

"""Implementation of VOI methods for Network Meta-Analysis (NMA).

- NMA-based EVPI (Expected Value of Perfect Information for Network Comparisons)
- NMA-based EVPPI (Expected Value of Partial Perfect Information for Network Comparisons)

Network Meta-Analysis allows simultaneous comparison of multiple treatments
even when they haven't been directly compared in head-to-head trials.
These VOI methods quantify the value of reducing uncertainty in NMA results.
"""

from typing import Any

import numpy as np

from voiage.exceptions import raise_input_error
from voiage.schema import ValueArray


class NetworkMetaAnalysisData:
    """Data structure for Network Meta-Analysis inputs.

    Attributes
    ----------
    treatment_effects : dict[tuple[str, str], numpy.ndarray]
        Mapping from treatment pairs to effect-size samples.
    n_studies : int
        Number of studies in the network.
    treatments : list[str]
        Treatment names.
    outcome_type : str
        Outcome type, such as ``continuous`` or ``binary``.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.network_meta_analysis import NetworkMetaAnalysisData
    >>> data = NetworkMetaAnalysisData({("A", "B"): np.array([0.1, 0.2])}, 1, ["A", "B"])
    >>> data.get_n_treatments()
    2
    """

    def __init__(
        self,
        treatment_effects: dict[tuple[str, str], np.ndarray],
        n_studies: int,
        treatments: list[str],
        outcome_type: str = "continuous",
    ):
        """Initialize the network meta-analysis data structure.

        Parameters
        ----------
        treatment_effects : dict[tuple[str, str], numpy.ndarray]
            Treatment-pair effects.
        n_studies : int
            Number of studies in the network.
        treatments : list[str]
            Treatment names.
        outcome_type : str, default="continuous"
            Outcome type.
        """
        # Validate treatment_effects
        if not treatment_effects:
            raise_input_error("treatment_effects must not be empty.")

        # Validate no empty arrays
        for key, val in treatment_effects.items():
            if val.size == 0:
                raise_input_error(f"Treatment effects for {key} must not be empty.")

        # Validate treatments
        if len(treatments) < 2:
            raise_input_error("At least 2 treatments are required.")

        # Validate outcome_type
        valid_outcomes = ["continuous", "binary", "survival"]
        if outcome_type not in valid_outcomes:
            raise_input_error(f"outcome_type must be one of {valid_outcomes}.")

        # Validate n_studies
        if n_studies < 1:
            raise_input_error("n_studies must be at least 1.")

        # Validate treatment effects have consistent sample sizes
        sample_sizes = [v.shape[0] for v in treatment_effects.values()]
        if len(set(sample_sizes)) > 1:
            raise_input_error(
                "All treatment effects must have the same number of samples."
            )

        self.treatment_effects = treatment_effects
        self.n_studies = n_studies
        self.treatments = treatments
        self.outcome_type = outcome_type
        self.n_samples = sample_sizes[0]

    def get_treatment_names(self) -> list[str]:
        """Return list of treatment names."""
        return self.treatments

    def get_n_treatments(self) -> int:
        """Return number of treatments."""
        return len(self.treatments)

    def get_n_samples(self) -> int:
        """Return number of PSA samples."""
        return int(self.n_samples)


def calculate_nma_evpi(
    nma_data: NetworkMetaAnalysisData | dict[str, Any],
    n_samples: int = 10000,
    willingness_to_pay: float | None = None,
    population: float | None = None,
    time_horizon: float | None = None,
    discount_rate: float | None = None,
) -> float:
    """Calculate expected value of perfect information for NMA.

    Parameters
    ----------
    nma_data : NetworkMetaAnalysisData or dict[str, Any]
        Network meta-analysis inputs or a dictionary representation.
    n_samples : int, default=10000
        Number of PSA samples to use when extracting net benefits.
    willingness_to_pay : float, optional
        Willingness-to-pay threshold per unit.
    population : float, optional
        Population size for population scaling.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.

    Returns
    -------
    float
        NMA EVPI on a per-decision basis unless population scaling is
        requested.
    """
    # Convert dict to NetworkMetaAnalysisData if needed
    if isinstance(nma_data, dict):
        nma_data = _dict_to_nma_data(nma_data)

    # Extract net benefit samples from NMA results
    nb_array = _extract_net_benefits_from_nma(nma_data, n_samples, willingness_to_pay)

    # Calculate EVPI using standard method
    from voiage.methods.basic import evpi

    return float(
        evpi(
            nb_array,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
        )
    )


def calculate_nma_evppi(
    nma_data: NetworkMetaAnalysisData | dict[str, Any],
    parameters_of_interest: list[str],
    parameter_samples: dict[str, np.ndarray],
    n_samples: int = 10000,
    willingness_to_pay: float | None = None,
    population: float | None = None,
    time_horizon: float | None = None,
    discount_rate: float | None = None,
) -> float:
    """Calculate expected value of partial perfect information for NMA.

    Parameters
    ----------
    nma_data : NetworkMetaAnalysisData or dict[str, Any]
        Network meta-analysis inputs or a dictionary representation.
    parameters_of_interest : list[str]
        Parameter names whose uncertainty is being resolved.
    parameter_samples : dict[str, numpy.ndarray]
        PSA parameter samples.
    n_samples : int, default=10000
        Number of PSA samples to use when extracting net benefits.
    willingness_to_pay : float, optional
        Willingness-to-pay threshold per unit.
    population : float, optional
        Population size for population scaling.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.

    Returns
    -------
    float
        NMA EVPPI on a per-decision basis unless population scaling is
        requested.
    """
    # Convert dict to NetworkMetaAnalysisData if needed
    if isinstance(nma_data, dict):
        nma_data = _dict_to_nma_data(nma_data)

    # Extract net benefit samples
    nb_array = _extract_net_benefits_from_nma(nma_data, n_samples, willingness_to_pay)

    # Calculate EVPPI using standard method
    from voiage.methods.basic import evppi

    return float(
        evppi(
            nb_array=nb_array,
            parameter_samples=parameter_samples,
            parameters_of_interest=parameters_of_interest,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
        )
    )


# --- Helper Functions ---


def _dict_to_nma_data(data: dict[str, Any]) -> NetworkMetaAnalysisData:
    """Convert a dictionary to :class:`NetworkMetaAnalysisData`."""
    if "treatment_effects" not in data:
        raise_input_error("Dictionary must contain 'treatment_effects' key.")

    # Convert string keys to tuples
    treatment_effects: dict[tuple[str, str], np.ndarray] = {}
    for key, value in data["treatment_effects"].items():
        pair = tuple(key.split("-")) if isinstance(key, str) and "-" in key else key
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise_input_error("Treatment effect keys must be treatment pairs.")
        treatment_effects[(str(pair[0]), str(pair[1]))] = np.asarray(value)

    return NetworkMetaAnalysisData(
        treatment_effects=treatment_effects,
        n_studies=data.get("n_studies", 1),
        treatments=data.get(
            "treatments", list({t for pair in treatment_effects for t in pair})
        ),
        outcome_type=data.get("outcome_type", "continuous"),
    )


def _extract_net_benefits_from_nma(
    nma_data: NetworkMetaAnalysisData,
    n_samples: int,
    willingness_to_pay: float | None = None,
) -> ValueArray:
    """Extract net-benefit samples from NMA results.

    Parameters
    ----------
    nma_data : NetworkMetaAnalysisData
        Network meta-analysis inputs.
    n_samples : int
        Number of samples to extract.
    willingness_to_pay : float, optional
        Willingness-to-pay threshold per unit.

    Returns
    -------
    ValueArray
        Net-benefit surface compatible with the standard EVPI/EVPPI methods.
    """
    n_treatments = nma_data.get_n_treatments()
    treatments = nma_data.treatments

    # Use minimum sample size from treatment effects
    actual_samples = min(n_samples, nma_data.get_n_samples())

    # Initialize net benefit array (samples x treatments)
    nb_values = np.zeros((actual_samples, n_treatments))

    # Set baseline (first treatment) to 0
    # Other treatments get their effect sizes
    for i, treatment in enumerate(treatments):
        if i == 0:
            nb_values[:, i] = 0  # Baseline
        else:
            # Find effect size for this treatment vs baseline
            baseline_key = (treatments[0], treatment)
            reverse_key = (treatment, treatments[0])

            if baseline_key in nma_data.treatment_effects:
                effects = nma_data.treatment_effects[baseline_key]
                nb_values[:, i] = effects[:actual_samples]
            elif reverse_key in nma_data.treatment_effects:
                effects = nma_data.treatment_effects[reverse_key]
                nb_values[:, i] = -effects[:actual_samples]  # Reverse sign
            else:
                # If no direct comparison, estimate from network
                # Simple approach: average of available comparisons
                comparison_effects: list[np.ndarray] = []
                for key, val in nma_data.treatment_effects.items():
                    if treatment in key:
                        sign = 1 if key[1] == treatment else -1
                        comparison_effects.append(sign * val[:actual_samples])

                if comparison_effects:
                    nb_values[:, i] = np.mean(comparison_effects, axis=0)
                else:
                    nb_values[:, i] = 0  # No data available

    # Apply willingness-to-pay if provided (for cost-effectiveness)
    if willingness_to_pay is not None:
        nb_values *= willingness_to_pay

    return ValueArray.from_numpy(nb_values, treatments)
