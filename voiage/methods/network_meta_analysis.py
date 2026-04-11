# voiage/methods/network_meta_analysis.py

"""Implementation of VOI methods for Network Meta-Analysis (NMA).

- NMA-based EVPI (Expected Value of Perfect Information for Network Comparisons)
- NMA-based EVPPI (Expected Value of Partial Perfect Information for Network Comparisons)

Network Meta-Analysis allows simultaneous comparison of multiple treatments
even when they haven't been directly compared in head-to-head trials.
These VOI methods quantify the value of reducing uncertainty in NMA results.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from voiage.core.utils import check_input_array
from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray


class NetworkMetaAnalysisData:
    """Data structure for Network Meta-Analysis inputs.
    
    Attributes:
        treatment_effects: Dictionary mapping treatment pairs to effect sizes.
                          Keys are tuples like ('A', 'B'), values are arrays of samples.
        n_studies: Number of studies in the network.
        treatments: List of treatment names.
        outcome_type: Type of outcome ('continuous', 'binary', 'survival').
    """
    
    def __init__(
        self,
        treatment_effects: Dict[Tuple[str, str], np.ndarray],
        n_studies: int,
        treatments: List[str],
        outcome_type: str = "continuous",
    ):
        """Initialize NMA data structure.
        
        Args:
            treatment_effects: Dictionary of treatment pair effects.
            n_studies: Number of studies in the network.
            treatments: List of treatment names.
            outcome_type: Type of outcome measure.
            
        Raises:
            InputError: If inputs are invalid or inconsistent.
        """
        # Validate treatment_effects
        if not treatment_effects:
            raise InputError("treatment_effects must not be empty.")
        
        # Validate treatments
        if len(treatments) < 2:
            raise InputError("At least 2 treatments are required.")
        
        # Validate outcome_type
        valid_outcomes = ["continuous", "binary", "survival"]
        if outcome_type not in valid_outcomes:
            raise InputError(f"outcome_type must be one of {valid_outcomes}.")
        
        # Validate n_studies
        if n_studies < 1:
            raise InputError("n_studies must be at least 1.")
        
        # Validate treatment effects have consistent sample sizes
        sample_sizes = [v.shape[0] for v in treatment_effects.values()]
        if len(set(sample_sizes)) > 1:
            raise InputError("All treatment effects must have the same number of samples.")
        
        self.treatment_effects = treatment_effects
        self.n_studies = n_studies
        self.treatments = treatments
        self.outcome_type = outcome_type
        self.n_samples = sample_sizes[0]
    
    def get_treatment_names(self) -> List[str]:
        """Return list of treatment names."""
        return self.treatments
    
    def get_n_treatments(self) -> int:
        """Return number of treatments."""
        return len(self.treatments)
    
    def get_n_samples(self) -> int:
        """Return number of PSA samples."""
        return self.n_samples


def calculate_nma_evpi(
    nma_data: Union[NetworkMetaAnalysisData, Dict],
    n_samples: int = 10000,
    willingness_to_pay: Optional[float] = None,
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """Calculate Expected Value of Perfect Information for Network Meta-Analysis.
    
    NMA-EVPI quantifies the expected gain from eliminating all uncertainty
    in the network meta-analysis results, including both parameter uncertainty
    and structural uncertainty about the network.
    
    Args:
        nma_data: Network meta-analysis data or dictionary with results.
        n_samples: Number of PSA samples (if generating from distribution).
        willingness_to_pay: Willingness-to-pay threshold per unit.
        population: Population size for scaling.
        time_horizon: Time horizon in years.
        discount_rate: Annual discount rate.
        
    Returns:
        float: The calculated NMA-EVPI.
    """
    # Convert dict to NetworkMetaAnalysisData if needed
    if isinstance(nma_data, dict):
        nma_data = _dict_to_nma_data(nma_data)
    
    # Extract net benefit samples from NMA results
    nb_array = _extract_net_benefits_from_nma(nma_data, n_samples, willingness_to_pay)
    
    # Calculate EVPI using standard method
    from voiage.methods.basic import evpi
    
    return evpi(
        nb_array,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )


def calculate_nma_evppi(
    nma_data: Union[NetworkMetaAnalysisData, Dict],
    parameters_of_interest: List[str],
    parameter_samples: Dict[str, np.ndarray],
    n_samples: int = 10000,
    willingness_to_pay: Optional[float] = None,
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """Calculate Expected Value of Partial Perfect Information for NMA.
    
    NMA-EVPPI quantifies the expected gain from eliminating uncertainty
    in a specific subset of parameters within the network meta-analysis.
    
    Args:
        nma_data: Network meta-analysis data.
        parameters_of_interest: List of parameter names to resolve uncertainty for.
        parameter_samples: Dictionary of parameter samples.
        n_samples: Number of PSA samples.
        willingness_to_pay: Willingness-to-pay threshold.
        population: Population size for scaling.
        time_horizon: Time horizon in years.
        discount_rate: Annual discount rate.
        
    Returns:
        float: The calculated NMA-EVPPI.
    """
    # Convert dict to NetworkMetaAnalysisData if needed
    if isinstance(nma_data, dict):
        nma_data = _dict_to_nma_data(nma_data)
    
    # Extract net benefit samples
    nb_array = _extract_net_benefits_from_nma(nma_data, n_samples, willingness_to_pay)
    
    # Calculate EVPPI using standard method
    from voiage.methods.basic import evppi
    
    return evppi(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        parameters_of_interest=parameters_of_interest,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )


# --- Helper Functions ---

def _dict_to_nma_data(data: Dict) -> NetworkMetaAnalysisData:
    """Convert dictionary to NetworkMetaAnalysisData."""
    if "treatment_effects" not in data:
        raise InputError("Dictionary must contain 'treatment_effects' key.")
    
    # Convert string keys to tuples
    treatment_effects = {}
    for key, value in data["treatment_effects"].items():
        if isinstance(key, str) and "-" in key:
            pair = tuple(key.split("-"))
        else:
            pair = key
        treatment_effects[pair] = np.asarray(value)
    
    return NetworkMetaAnalysisData(
        treatment_effects=treatment_effects,
        n_studies=data.get("n_studies", 1),
        treatments=data.get("treatments", list(set(
            t for pair in treatment_effects.keys() for t in pair
        ))),
        outcome_type=data.get("outcome_type", "continuous"),
    )


def _extract_net_benefits_from_nma(
    nma_data: NetworkMetaAnalysisData,
    n_samples: int,
    willingness_to_pay: Optional[float] = None,
) -> ValueArray:
    """Extract net benefit array from NMA results.
    
    This converts NMA treatment effect samples into a net benefit format
    suitable for standard EVPI/EVPPI calculation.
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
                effects = []
                for key, val in nma_data.treatment_effects.items():
                    if treatment in key:
                        sign = 1 if key[1] == treatment else -1
                        effects.append(sign * val[:actual_samples])
                
                if effects:
                    nb_values[:, i] = np.mean(effects, axis=0)
                else:
                    nb_values[:, i] = 0  # No data available
    
    # Apply willingness-to-pay if provided (for cost-effectiveness)
    if willingness_to_pay is not None:
        nb_values *= willingness_to_pay
    
    return ValueArray.from_numpy(nb_values, treatments)
