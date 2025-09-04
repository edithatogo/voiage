"""Healthcare-specific utilities for Value of Information analysis."""

from typing import Dict, List, Optional, Union
import numpy as np


def calculate_qaly(
    utility_values: Union[np.ndarray, List[float]], 
    time_years: Union[np.ndarray, List[float]],
    discount_rate: float = 0.03
) -> float:
    """
    Calculate Quality-Adjusted Life Years (QALYs) over time.
    
    QALY = Σ(utility_value * time_period * discount_factor)
    
    Args:
        utility_values: Health utility values (0 = death, 1 = perfect health)
        time_years: Time periods in years
        discount_rate: Annual discount rate (default 0.03 = 3%)
        
    Returns:
        float: Total discounted QALYs
        
    Example:
        >>> utilities = [0.8, 0.7, 0.6]  # Utility values for 3 years
        >>> time_periods = [1, 1, 1]      # 1 year each
        >>> qaly = calculate_qaly(utilities, time_periods, 0.03)
        >>> print(f"Total QALYs: {qaly:.2f}")
    """
    # Convert to numpy arrays if needed
    if not isinstance(utility_values, np.ndarray):
        utility_values = np.array(utility_values)
    if not isinstance(time_years, np.ndarray):
        time_years = np.array(time_years)
    
    # Validate inputs
    if len(utility_values) != len(time_years):
        raise ValueError("utility_values and time_years must have the same length")
    
    if np.any(utility_values < 0) or np.any(utility_values > 1):
        raise ValueError("Utility values must be between 0 and 1")
    
    if discount_rate < 0:
        raise ValueError("Discount rate must be non-negative")
    
    # Calculate mid-year discounting
    cumulative_time = np.cumsum(time_years) - (time_years / 2)
    discount_factors = 1 / ((1 + discount_rate) ** cumulative_time)
    
    # Calculate QALYs for each period
    qaly_periods = utility_values * time_years * discount_factors
    
    # Sum all periods
    total_qaly = np.sum(qaly_periods)
    
    return float(total_qaly)


def discount_qaly(
    qaly_value: float,
    time_horizon: float,
    discount_rate: float = 0.03
) -> float:
    """
    Discount a QALY value to present value.
    
    Args:
        qaly_value: QALY value to discount
        time_horizon: Time horizon in years
        discount_rate: Annual discount rate (default 0.03 = 3%)
        
    Returns:
        float: Discounted QALY value
    """
    if time_horizon < 0:
        raise ValueError("Time horizon must be non-negative")
    
    if discount_rate < 0:
        raise ValueError("Discount rate must be non-negative")
    
    discounted_value = qaly_value / ((1 + discount_rate) ** time_horizon)
    return discounted_value


def aggregate_qaly_over_time(
    utility_trajectories: Dict[str, np.ndarray],
    time_points: np.ndarray,
    discount_rate: float = 0.03
) -> Dict[str, float]:
    """
    Aggregate QALYs over time for multiple health states or strategies.
    
    Args:
        utility_trajectories: Dictionary mapping strategy names to utility value arrays
        time_points: Array of time points (years)
        discount_rate: Annual discount rate (default 0.03 = 3%)
        
    Returns:
        Dict[str, float]: Dictionary mapping strategy names to total QALYs
    """
    results = {}
    
    # Calculate time intervals
    if len(time_points) < 2:
        raise ValueError("At least two time points are required")
    
    time_intervals = np.diff(time_points)
    
    # For each strategy, calculate QALYs
    for strategy_name, utilities in utility_trajectories.items():
        if len(utilities) != len(time_points):
            raise ValueError(f"Utility array for {strategy_name} must match time points length")
        
        # Use utility values at the beginning of each interval
        interval_utilities = utilities[:-1]
        total_qaly = calculate_qaly(interval_utilities, time_intervals, discount_rate)
        results[strategy_name] = total_qaly
    
    return results


def markov_cohort_model(
    transition_matrix: np.ndarray,
    initial_state: np.ndarray,
    n_cycles: int,
    cycle_length: float = 1.0
) -> np.ndarray:
    """
    Simulate a Markov cohort model for disease progression.
    
    Args:
        transition_matrix: Square matrix of transition probabilities between states
        initial_state: Initial state distribution (probabilities sum to 1)
        n_cycles: Number of cycles to simulate
        cycle_length: Length of each cycle in years (default 1.0)
        
    Returns:
        np.ndarray: Array of shape (n_cycles+1, n_states) with state distributions over time
    """
    # Validate inputs
    n_states = transition_matrix.shape[0]
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("Transition matrix must be square")
    
    if len(initial_state) != n_states:
        raise ValueError("Initial state vector must match number of states")
    
    if not np.isclose(np.sum(initial_state), 1.0):
        raise ValueError("Initial state probabilities must sum to 1")
    
    if not np.all(transition_matrix >= 0):
        raise ValueError("Transition probabilities must be non-negative")
    
    if not np.all(np.isclose(np.sum(transition_matrix, axis=1), 1.0)):
        raise ValueError("Each row of transition matrix must sum to 1")
    
    # Initialize results array
    state_trajectories = np.zeros((n_cycles + 1, n_states))
    state_trajectories[0] = initial_state
    
    # Simulate transitions
    current_state = initial_state.copy()
    for i in range(1, n_cycles + 1):
        current_state = np.dot(current_state, transition_matrix)
        state_trajectories[i] = current_state
    
    return state_trajectories


def disease_progression_model(
    base_transition_probs: Dict[str, Dict[str, float]],
    covariates: Optional[Dict[str, float]] = None,
    covariate_effects: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Create a transition matrix for disease progression modeling.
    
    Args:
        base_transition_probs: Dictionary mapping from states to transition probabilities
        covariates: Dictionary of covariate values (e.g., age, treatment)
        covariate_effects: Dictionary mapping covariates to their effects on transitions
        
    Returns:
        np.ndarray: Transition matrix
    """
    # Get all unique states
    all_states = set(base_transition_probs.keys())
    for transitions in base_transition_probs.values():
        all_states.update(transitions.keys())
    
    states = sorted(list(all_states))
    n_states = len(states)
    state_index = {state: i for i, state in enumerate(states)}
    
    # Initialize transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    # Fill in base transition probabilities
    for from_state, transitions in base_transition_probs.items():
        from_idx = state_index[from_state]
        total_prob = 0.0
        for to_state, prob in transitions.items():
            to_idx = state_index[to_state]
            transition_matrix[from_idx, to_idx] = prob
            total_prob += prob
        
        # If probabilities don't sum to 1, assume remaining probability stays in same state
        if total_prob < 1.0:
            transition_matrix[from_idx, from_idx] += (1.0 - total_prob)
    
    # Apply covariate effects if provided
    if covariates is not None and covariate_effects is not None:
        for covariate_name, covariate_value in covariates.items():
            if covariate_name in covariate_effects:
                effects = covariate_effects[covariate_name]
                for (from_state, to_state), effect_size in effects.items():
                    if from_state in state_index and to_state in state_index:
                        from_idx = state_index[from_state]
                        to_idx = state_index[to_state]
                        # Apply multiplicative effect
                        transition_matrix[from_idx, to_idx] *= np.exp(effect_size * covariate_value)
                
                # Renormalize each row to sum to 1
                row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                transition_matrix = transition_matrix / row_sums
    
    return transition_matrix