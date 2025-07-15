# voiage/models.py

"""
This module will contain the Bayesian models for updating parameter
distributions based on simulated trial data.
"""

import pymc as pm


class ConjugateUpdater:
    """
    A class to perform Bayesian updates for conjugate prior models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def update(self, data, prior_samples):
        if self.model_name == "normal-normal":
            return self._normal_normal_update(data, prior_samples)
        else:
            raise NotImplementedError(
                f"Conjugate model '{self.model_name}' not implemented."
            )

    def _normal_normal_update(self, data, prior_samples):
        with pm.Model() as model:
            prior_mean = pm.Normal(
                "prior_mean",
                mu=prior_samples["mean"].mean(),
                sigma=prior_samples["mean"].std(),
            )
            prior_std = pm.HalfNormal("prior_std", sigma=prior_samples["std"].std())
            likelihood = pm.Normal(
                "likelihood", mu=prior_mean, sigma=prior_std, observed=data
            )
            trace = pm.sample(1000, cores=1)
        return trace


class NMAUpdater:
    """
    A class to perform Bayesian updates for NMA models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def update(self, data, prior_samples):
        if self.model_name == "normal-normal":
            return self._normal_normal_update(data, prior_samples)
        else:
            raise NotImplementedError(
                f"NMA model '{self.model_name}' not implemented."
            )

    def _normal_normal_update(self, data, prior_samples):
        with pm.Model() as model:
            # Treatment-specific means
            mu = pm.Normal("mu", mu=0, sigma=100, shape=len(data))
            # Common standard deviation
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Likelihood
            for i, arm_name in enumerate(data.keys()):
                pm.Normal(
                    f"likelihood_{arm_name}",
                    mu=mu[i],
                    sigma=sigma,
                    observed=data[arm_name],
                )

            trace = pm.sample(1000, cores=1)
        return trace
