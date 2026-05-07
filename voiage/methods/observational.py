# voiage/methods/observational.py

"""Value of information methods for observational study designs."""

from collections.abc import Callable

import numpy as np

from voiage.exceptions import InputError, raise_input_error
from voiage.schema import (
    ParameterSet as PSASample,
)
from voiage.schema import (
    ValueArray as NetBenefitArray,
)

# Type alias for a function that models the impact of observational data.
# This would typically involve:
# - Defining the observational study design (variables collected, population).
# - Modeling potential biases and their impact on parameter estimation.
# - Simulating the observational data collection process.
# - Specifying how this data, adjusted for biases, updates decision model parameters.
ObservationalStudyModeler = Callable[
    [
        PSASample,
        dict[str, object],
        dict[str, object],
    ],  # Prior PSA, Obs. Study Design, Bias Models
    NetBenefitArray,  # Expected NB conditional on simulated observational data
]


def basic_observational_study_modeler(
    psa_prior: PSASample,
    observational_study_design: dict[str, object],
    bias_models: dict[str, object],
) -> NetBenefitArray:
    """Model a simple observational study update for explicit net benefits.

    Parameters
    ----------
    psa_prior : ParameterSet
        Prior parameter samples.
    observational_study_design : dict[str, object]
        Study design specification, including optional sample size and truth.
    bias_models : dict[str, object]
        Bias-model specification used to adjust residual uncertainty.

    Returns
    -------
    ValueArray
        Net-benefit samples after observational updating.

    Notes
    -----
    The built-in modeler accepts explicit strategy net-benefit arrays or
    matched cost/effect arrays and shrinks uncertainty toward the sampled truth.
    """
    net_benefits, strategy_names = _extract_observational_net_benefits(
        psa_prior, observational_study_design
    )
    true_parameters = observational_study_design.get("_true_parameters")
    if isinstance(true_parameters, dict):
        net_benefits = _shrink_net_benefits_toward_truth(
            net_benefits,
            strategy_names,
            true_parameters,
            observational_study_design,
            bias_models,
        )
    return _value_array_from_net_benefits(net_benefits, strategy_names)


def voi_observational(
    obs_study_modeler: ObservationalStudyModeler | None = None,
    psa_prior: PSASample | None = None,
    observational_study_design: dict[str, object] | None = None,
    bias_models: dict[str, object] | None = None,
    # wtp: float, # Implicit
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
    n_outer_loops: int = 20,
    # method_args for simulation, bias adjustment techniques
    **kwargs: object,
) -> float:
    """Calculate the value of information for an observational study.

    Parameters
    ----------
    obs_study_modeler : callable, optional
        Modeler that maps PSA samples and study assumptions to net-benefit
        samples.
    psa_prior : ParameterSet
        Prior PSA samples representing current uncertainty.
    observational_study_design : dict[str, object]
        Study design specification.
    bias_models : dict[str, object]
        Bias-model specification.
    population : float, optional
        Population size for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    n_outer_loops : int, default=20
        Number of outer Monte Carlo draws.
    **kwargs : object
        Additional modeler-specific options.

    Returns
    -------
    float
        Observational VOI on a per-decision basis unless population scaling is
        requested.
    """
    # Validate inputs
    if obs_study_modeler is None:
        obs_study_modeler = basic_observational_study_modeler
    if not callable(obs_study_modeler):
        raise_input_error("`obs_study_modeler` must be a callable function.")
    if not isinstance(psa_prior, PSASample):
        raise_input_error("`psa_prior` must be a PSASample object.")
    if not isinstance(observational_study_design, dict):
        raise_input_error("`observational_study_design` must be a dictionary.")
    if not isinstance(bias_models, dict):
        raise_input_error("`bias_models` must be a dictionary.")
    if n_outer_loops <= 0:
        raise_input_error("n_outer_loops must be positive.")

    # 1. Calculate max_d E[NB(d) | Prior Info].
    nb_array_prior = obs_study_modeler(
        psa_prior, observational_study_design, bias_models
    )
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.numpy_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    # 2. Outer loop (simulating different potential datasets D_k from the observational study):
    #    For k = 1 to N_outer_loops:
    #        a. Simulate dataset D_k:
    #           - Sample "true" underlying parameters from `psa_prior`.
    #           - Simulate the process generating observational data, including the
    #             effects of biases defined in `bias_models`.
    #        b. Analyze D_k:
    #           - Apply statistical methods to D_k to estimate treatment effects or
    #             other parameters, attempting to adjust for biases.
    #           - Update beliefs about decision model parameters P(theta | D_k, bias_adj).
    #        c. `obs_study_modeler` would encapsulate steps a and b to produce
    #           E_theta|D_k,bias_adj [NB(d, theta|...)] for each d.
    #        d. Let V_k = max_d E_theta|D_k,bias_adj [NB(d, theta|...)].

    max_nb_post_observational = []
    for _k in range(n_outer_loops):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior.n_samples)

        # Extract the true parameters for this iteration
        true_parameters = {}
        for param_name, param_values in psa_prior.parameters.items():
            true_parameters[param_name] = param_values[true_params_idx]

        # In a more sophisticated implementation, we would:
        # 1. Simulate data based on these true parameters and study design
        # 2. Apply bias models to the simulated data
        # 3. Use statistical methods to adjust for biases
        # 4. Update parameter beliefs based on the bias-adjusted analysis

        # For this implementation, we'll simulate the effect by adding some
        # realistic variation to the modeler's output
        try:
            # Simulate the observational study with the sampled parameters
            post_study_design = {
                **observational_study_design,
                "_true_parameters": true_parameters,
            }
            nb_array_post = obs_study_modeler(psa_prior, post_study_design, bias_models)
            mean_nb_per_strategy_post = np.mean(nb_array_post.numpy_values, axis=0)
            max_nb_post_observational.append(np.max(mean_nb_per_strategy_post))
        except Exception:
            # If the modeler fails, use the prior value
            # In a real implementation, we might want to log this
            max_nb_post_observational.append(max_expected_nb_current_info)

    # 3. Calculate E_D [ max_d E[NB(d) | D, bias_adj] ] = mean(V_k).
    expected_max_nb_post_observational: float = np.mean(max_nb_post_observational)

    # 4. VOI-OS = E_D [ ... ] - max_d E[NB(d) | Prior Info]
    per_decision_voi_observational = (
        expected_max_nb_post_observational - max_expected_nb_current_info
    )
    per_decision_voi_observational = max(0.0, per_decision_voi_observational)

    # Population scaling
    if population is not None and time_horizon is not None:
        if population <= 0:
            raise_input_error("Population must be positive.")
        if time_horizon <= 0:
            raise_input_error("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise_input_error("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_voi_observational * population * annuity

    return float(per_decision_voi_observational)


def _extract_observational_net_benefits(
    psa_prior: PSASample,
    observational_study_design: dict[str, object],
) -> tuple[np.ndarray, list[str]]:
    """Extract strategy net-benefit samples for the built-in modeler."""
    named_net_benefits = _stack_named_strategy_arrays(
        psa_prior.parameters, prefixes=("net_benefit_", "nb_")
    )
    if named_net_benefits is not None:
        return named_net_benefits

    cost_effect_net_benefits = _cost_effect_strategy_net_benefits(
        psa_prior.parameters,
        wtp=float(observational_study_design.get("wtp", 1.0)),
    )
    if cost_effect_net_benefits is not None:
        return cost_effect_net_benefits

    return raise_input_error(
        "The built-in observational modeler requires explicit strategy net benefits "
        "or matched strategy cost/effect samples."
    )


def _stack_named_strategy_arrays(
    parameters: dict[str, np.ndarray],
    prefixes: tuple[str, ...],
) -> tuple[np.ndarray, list[str]] | None:
    """Stack named one-dimensional strategy payoff arrays."""
    strategy_items = [
        (name, np.asarray(values, dtype=float))
        for name, values in sorted(parameters.items())
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not strategy_items:
        return None
    if any(values.ndim != 1 for _name, values in strategy_items):
        raise_input_error("Strategy net-benefit samples must be one-dimensional.")

    strategy_names = [
        _remove_first_prefix(name, prefixes).replace("_", " ").title()
        for name, _values in strategy_items
    ]
    return np.column_stack([values for _name, values in strategy_items]), strategy_names


def _cost_effect_strategy_net_benefits(
    parameters: dict[str, np.ndarray],
    wtp: float,
) -> tuple[np.ndarray, list[str]] | None:
    """Derive strategy net benefits from matched cost and effect arrays."""
    effect_prefixes = ("effect_", "effectiveness_", "qaly_", "qalys_")
    cost_prefixes = ("cost_", "costs_")
    effect_by_strategy: dict[str, np.ndarray] = {}
    cost_by_strategy: dict[str, np.ndarray] = {}
    for name, values in parameters.items():
        for prefix in effect_prefixes:
            if name.startswith(prefix):
                effect_by_strategy[name.removeprefix(prefix)] = np.asarray(
                    values, dtype=float
                )
                break
        for prefix in cost_prefixes:
            if name.startswith(prefix):
                cost_by_strategy[name.removeprefix(prefix)] = np.asarray(
                    values, dtype=float
                )
                break

    strategy_names = sorted(set(effect_by_strategy) & set(cost_by_strategy))
    if not strategy_names:
        return None

    net_benefits = []
    for strategy_name in strategy_names:
        effect = effect_by_strategy[strategy_name]
        cost = cost_by_strategy[strategy_name]
        if effect.ndim != 1 or cost.ndim != 1 or len(effect) != len(cost):
            raise_input_error("Cost and effect samples must be matching 1D arrays.")
        net_benefits.append(wtp * effect - cost)

    display_names = [
        strategy_name.replace("_", " ").title() for strategy_name in strategy_names
    ]
    return np.column_stack(net_benefits), display_names


def _shrink_net_benefits_toward_truth(
    net_benefits: np.ndarray,
    strategy_names: list[str],
    true_parameters: dict[str, object],
    observational_study_design: dict[str, object],
    bias_models: dict[str, object],
) -> np.ndarray:
    """Shrink PSA payoff samples toward sampled truth after observational data."""
    sample_size = max(1.0, float(observational_study_design.get("sample_size", 1.0)))
    bias_strength = _combined_bias_strength(bias_models)
    residual_uncertainty = float(
        np.clip((1.0 + bias_strength) / np.sqrt(sample_size), 0.05, 1.0)
    )
    true_values = np.array(
        [_true_strategy_net_benefit(name, true_parameters) for name in strategy_names],
        dtype=float,
    )
    return true_values + (net_benefits - true_values) * residual_uncertainty


def _combined_bias_strength(bias_models: dict[str, object]) -> float:
    """Combine simple numeric bias specifications into one residual factor."""
    strength = 0.0
    for bias_model in bias_models.values():
        if isinstance(bias_model, dict):
            for key in ("strength", "probability", "magnitude"):
                value = bias_model.get(key)
                if isinstance(value, (int, float)):
                    strength += abs(float(value))
        elif isinstance(bias_model, (int, float)):
            strength += abs(float(bias_model))
    return strength


def _true_strategy_net_benefit(
    strategy_name: str,
    true_parameters: dict[str, object],
) -> float:
    """Read the sampled true net benefit for a strategy."""
    slug = strategy_name.lower().replace(" ", "_")
    candidate_keys = (
        f"net_benefit_{slug}",
        f"nb_{slug}",
        strategy_name,
        slug,
    )
    for key in candidate_keys:
        value = true_parameters.get(key)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    return 0.0


def _remove_first_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    """Remove the first matching prefix from a parameter name."""
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _value_array_from_net_benefits(
    net_benefits: np.ndarray,
    strategy_names: list[str],
) -> NetBenefitArray:
    """Build a ValueArray from a net-benefit matrix."""
    import xarray as xr

    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), net_benefits)},
        coords={
            "n_samples": np.arange(net_benefits.shape[0]),
            "n_strategies": np.arange(net_benefits.shape[1]),
            "strategy": ("n_strategies", strategy_names),
        },
    )
    return NetBenefitArray(dataset=dataset)


if __name__ == "__main__":  # pragma: no cover
    print("--- Testing observational.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray and PSASample

    from voiage.schema import ParameterSet as PSASample
    from voiage.schema import ValueArray as NetBenefitArray

    # Simple observational study modeler for testing
    def simple_obs_modeler(
        psa: PSASample, design: dict[str, object], biases: dict[str, object]
    ) -> NetBenefitArray:
        """Run simple observational study modeler for testing."""
        n_samples = psa.n_samples
        # Create net benefits for 2 strategies
        nb_values = np.random.rand(n_samples, 2) * 1000
        # Make strategy 1 slightly better on average
        nb_values[:, 1] += 100

        import xarray as xr

        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["Standard Care", "New Treatment"]),
            },
        )
        return NetBenefitArray(dataset=dataset)

    # Create test parameter set
    dummy_psa = PSASample.from_numpy_or_dict({"p": np.random.rand(50)})

    # Create test observational study design
    dummy_design = {
        "study_type": "cohort",
        "sample_size": 1000,
        "variables_collected": ["treatment", "outcome"],
    }

    # Create test bias models
    dummy_biases = {
        "confounding": {"strength": 0.2},
        "selection_bias": {"probability": 0.1},
    }

    # Test voi_observational function
    print("Testing voi_observational...")
    voi_value = voi_observational(
        obs_study_modeler=simple_obs_modeler,
        psa_prior=dummy_psa,
        observational_study_design=dummy_design,
        bias_models=dummy_biases,
        n_outer_loops=5,
    )
    print(f"Observational Study VOI: {voi_value}")

    # Test with population scaling
    print("\nTesting voi_observational with population scaling...")
    voi_value_scaled = voi_observational(
        obs_study_modeler=simple_obs_modeler,
        psa_prior=dummy_psa,
        observational_study_design=dummy_design,
        bias_models=dummy_biases,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5,
    )
    print(f"Scaled Observational Study VOI: {voi_value_scaled}")

    # Test input validation
    print("\nTesting input validation...")
    try:
        # Test invalid obs_study_modeler
        voi_observational(
            obs_study_modeler="not a function",  # type: ignore[arg-type]
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models=dummy_biases,
        )
    except InputError as e:
        print(f"Caught expected error for invalid modeler: {e}")

    try:
        # Test invalid psa_prior
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior="not a psa",  # type: ignore[arg-type]
            observational_study_design=dummy_design,
            bias_models=dummy_biases,
        )
    except InputError as e:
        print(f"Caught expected error for invalid PSA: {e}")

    try:
        # Test invalid observational_study_design
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design="not a dict",  # type: ignore[arg-type]
            bias_models=dummy_biases,
        )
    except InputError as e:
        print(f"Caught expected error for invalid study design: {e}")

    try:
        # Test invalid bias_models
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models="not a dict",  # type: ignore[arg-type]
        )
    except InputError as e:
        print(f"Caught expected error for invalid bias models: {e}")

    try:
        # Test invalid loop parameters
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models=dummy_biases,
            n_outer_loops=0,
        )
    except InputError as e:
        print(f"Caught expected error for invalid loop params: {e}")

    print("--- observational.py tests completed ---")
