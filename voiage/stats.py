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
