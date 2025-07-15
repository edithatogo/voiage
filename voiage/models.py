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
