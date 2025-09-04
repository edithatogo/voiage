"""Parallel processing utilities for Monte Carlo simulations in Value of Information analysis."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from voiage.schema import ParameterSet, TrialDesign, ValueArray


def _monte_carlo_worker(
    worker_id: int,
    model_func: Callable[[ParameterSet], ValueArray],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_simulations: int,
    seed_offset: int = 0
) -> Tuple[float, int]:
    """
    Worker function for parallel Monte Carlo simulation.
    
    Args:
        worker_id: ID of the worker process
        model_func: Economic model function
        psa_prior: Prior parameter samples
        trial_design: Trial design specification
        n_simulations: Number of simulations for this worker
        seed_offset: Offset for random seed to ensure different randomness across workers
        
    Returns:
        Tuple of (expected_max_nb, n_simulations_processed)
    """
    # Set random seed for reproducibility
    np.random.seed(seed_offset + worker_id)
    
    max_nb_post_study = []
    for _ in range(n_simulations):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior.n_samples)
        true_params = {
            name: values[true_params_idx]
            for name, values in psa_prior.parameters.items()
        }
        
        # Simulate trial data based on true parameters
        trial_data = _simulate_trial_data(true_params, trial_design)
        
        # Update prior beliefs with simulated trial data
        posterior_psa = _bayesian_update(psa_prior, trial_data, trial_design)
        
        # Run model on posterior samples
        nb_posterior = model_func(posterior_psa).values
        mean_nb_per_strategy = np.mean(nb_posterior, axis=0)
        max_nb_post_study.append(np.max(mean_nb_per_strategy))
    
    expected_max_nb = np.mean(max_nb_post_study) if max_nb_post_study else 0.0
    return expected_max_nb, n_simulations


def _bootstrap_worker(worker_id: int, n_samples: int, seed_offset: int, data, statistic_func) -> List[float]:
    """Worker function for bootstrap sampling."""
    np.random.seed(seed_offset + worker_id)
    results = []
    for _ in range(n_samples):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
        bootstrap_sample = data[bootstrap_indices]
        # Calculate statistic
        stat = statistic_func(bootstrap_sample)
        results.append(stat)
    return results


def _simulate_trial_data(true_parameters: Dict[str, float], trial_design: TrialDesign) -> Dict[str, np.ndarray]:
    """Simulate trial data based on true parameters."""
    data = {}
    for arm in trial_design.arms:
        # Convert arm name to parameter name format
        param_name = f"mean_{arm.name.lower().replace(' ', '_')}"
        if param_name in true_parameters:
            mean = true_parameters[param_name]
            std_dev = true_parameters.get("sd_outcome", 1.0)
            data[arm.name] = np.random.normal(mean, std_dev, arm.sample_size)
        else:
            # Fallback if parameter name doesn't match
            data[arm.name] = np.random.normal(0, 1, arm.sample_size)
    return data


def _bayesian_update(
    prior_samples: ParameterSet, 
    trial_data: Dict[str, np.ndarray], 
    trial_design: TrialDesign
) -> ParameterSet:
    """Update prior beliefs with simulated trial data."""
    from voiage.stats import normal_normal_update
    import xarray as xr
    
    posterior_samples = {}
    for param_name, prior_values in prior_samples.parameters.items():
        if "mean" in param_name:
            # Extract arm name from parameter name
            arm_name_parts = param_name.split("_")[1:]  # Remove "mean" prefix
            arm_name = " ".join(arm_name_parts).title()
            
            if arm_name in trial_data:
                data = trial_data[arm_name]
                # Get standard deviation from prior
                std_dev_name = "sd_outcome"
                if std_dev_name in prior_samples.parameters:
                    prior_std = prior_samples.parameters[std_dev_name]
                    # Use mean of std dev if it's an array
                    if isinstance(prior_std, np.ndarray):
                        prior_std = np.mean(prior_std)
                else:
                    prior_std = 1.0
                
                # Perform Bayesian update
                try:
                    posterior_mean, posterior_std = normal_normal_update(
                        prior_values,
                        prior_std,
                        np.mean(data),
                        np.std(data) if len(data) > 1 else 1.0,
                        len(data),
                    )
                    posterior_samples[param_name] = np.random.normal(
                        posterior_mean, posterior_std, len(prior_values)
                    )
                except Exception:
                    # If update fails, keep prior values
                    posterior_samples[param_name] = prior_values
            else:
                posterior_samples[param_name] = prior_values
        else:
            posterior_samples[param_name] = prior_values
    
    # Create ParameterSet from posterior samples
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in posterior_samples.items()},
        coords={"n_samples": np.arange(len(next(iter(posterior_samples.values()))))},
    )
    return ParameterSet(dataset=dataset)


def parallel_monte_carlo_simulation(
    model_func: Callable[[ParameterSet], ValueArray],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    n_simulations: int = 1000,
    n_workers: Optional[int] = None,
    use_processes: bool = True
) -> float:
    """
    Perform Monte Carlo simulation using parallel processing.
    
    Args:
        model_func: Economic model function that takes ParameterSet and returns ValueArray
        psa_prior: Prior parameter samples
        trial_design: Trial design specification
        n_simulations: Total number of simulations to run
        n_workers: Number of parallel workers (default: number of CPU cores)
        use_processes: Whether to use processes (True) or threads (False)
        
    Returns:
        float: Expected maximum net benefit from posterior analysis
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Distribute simulations across workers
    simulations_per_worker = [n_simulations // n_workers] * n_workers
    # Distribute remainder simulations
    for i in range(n_simulations % n_workers):
        simulations_per_worker[i] += 1
    
    # Choose executor type
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # Run simulations in parallel
    with executor_class(max_workers=n_workers) as executor:
        futures = []
        for i, n_sims in enumerate(simulations_per_worker):
            future = executor.submit(
                _monte_carlo_worker,
                worker_id=i,
                model_func=model_func,
                psa_prior=psa_prior,
                trial_design=trial_design,
                n_simulations=n_sims,
                seed_offset=i * 1000  # Ensure different randomness across workers
            )
            futures.append(future)
        
        # Collect results
        total_expected_max_nb = 0.0
        total_simulations = 0
        
        for future in futures:
            expected_max_nb, n_sims_processed = future.result()
            total_expected_max_nb += expected_max_nb * n_sims_processed
            total_simulations += n_sims_processed
    
    # Return weighted average
    if total_simulations > 0:
        return total_expected_max_nb / total_simulations
    else:
        return 0.0


def parallel_evsi_calculation(
    model_func: Callable[[ParameterSet], ValueArray],
    psa_prior: ParameterSet,
    trial_design: TrialDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_simulations: int = 1000,
    n_workers: Optional[int] = None,
    use_processes: bool = True
) -> float:
    """
    Calculate EVSI using parallel Monte Carlo simulation.
    
    Args:
        model_func: Economic model function
        psa_prior: Prior parameter samples
        trial_design: Trial design specification
        population: Population size for scaling
        discount_rate: Discount rate for scaling
        time_horizon: Time horizon for scaling
        n_simulations: Total number of simulations to run
        n_workers: Number of parallel workers
        use_processes: Whether to use processes or threads
        
    Returns:
        float: EVSI value
    """
    # Calculate baseline expected net benefit with current information
    nb_prior_values = model_func(psa_prior).values
    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)
    
    # Calculate expected maximum net benefit after study using parallel Monte Carlo
    expected_max_nb_post_study = parallel_monte_carlo_simulation(
        model_func=model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=n_simulations,
        n_workers=n_workers,
        use_processes=use_processes
    )
    
    # Calculate EVSI
    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)
    
    # Apply population scaling if provided
    if population is not None and time_horizon is not None:
        if population <= 0:
            raise ValueError("Population must be positive.")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise ValueError("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_evsi * population * annuity

    return float(per_decision_evsi)


def parallel_bootstrap_sampling(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap_samples: int = 1000,
    n_workers: Optional[int] = None,
    use_processes: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform bootstrap sampling using parallel processing.
    
    Args:
        data: Input data array
        statistic_func: Function to calculate statistic on bootstrap samples
        n_bootstrap_samples: Number of bootstrap samples
        n_workers: Number of parallel workers
        use_processes: Whether to use processes or threads
        
    Returns:
        Dict with bootstrap statistics (mean, std, percentiles)
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Distribute bootstrap samples across workers
    samples_per_worker = [n_bootstrap_samples // n_workers] * n_workers
    for i in range(n_bootstrap_samples % n_workers):
        samples_per_worker[i] += 1
    
    # Choose executor type
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # Run bootstrap sampling in parallel
    with executor_class(max_workers=n_workers) as executor:
        futures = []
        for i, n_samples in enumerate(samples_per_worker):
            future = executor.submit(
                _bootstrap_worker,
                worker_id=i,
                n_samples=n_samples,
                seed_offset=i * 1000,
                data=data,
                statistic_func=statistic_func
            )
            futures.append(future)
        
        # Collect results
        all_bootstrap_stats = []
        for future in futures:
            worker_stats = future.result()
            all_bootstrap_stats.extend(worker_stats)
    
    # Calculate statistics
    bootstrap_array = np.array(all_bootstrap_stats)
    return {
        "mean": np.mean(bootstrap_array),
        "std": np.std(bootstrap_array),
        "percentile_2.5": np.percentile(bootstrap_array, 2.5),
        "percentile_97.5": np.percentile(bootstrap_array, 97.5),
        "samples": bootstrap_array
    }