# voiage/stats.py

"""A module for statistical utility functions used in VOI calculations."""

import numpy as np


def normal_normal_update(prior_mean, prior_std, data_mean, data_std, n_samples):
    """
    Perform a Bayesian update for a Normal likelihood with a Normal prior.

    Args:
        prior_mean (float): The mean of the prior distribution.
        prior_std (float): The standard deviation of the prior distribution.
        data_mean (float): The mean of the observed data.
        data_std (float): The standard deviation of the observed data.
        n_samples (int): The number of samples in the observed data.

    Returns
    -------
        tuple: A tuple containing the posterior mean and posterior standard deviation.
    """
    prior_var = prior_std**2
    data_var = data_std**2
    posterior_var = 1 / (1 / prior_var + n_samples / data_var)
    posterior_mean = posterior_var * (
        prior_mean / prior_var + n_samples * data_mean / data_var
    )
    return posterior_mean, np.sqrt(posterior_var)


def beta_binomial_update(prior_alpha, prior_beta, n_successes, n_trials):
    """
    Perform a Bayesian update for a Binomial likelihood with a Beta prior.

    Args:
        prior_alpha (float): The alpha parameter of the prior Beta distribution.
        prior_beta (float): The beta parameter of the prior Beta distribution.
        n_successes (int): The number of successes in the observed data.
        n_trials (int): The total number of trials in the observed data.

    Returns
    -------
        tuple: A tuple containing the posterior alpha and posterior beta.
    """
    posterior_alpha = prior_alpha + n_successes
    posterior_beta = prior_beta + (n_trials - n_successes)
    return posterior_alpha, posterior_beta
