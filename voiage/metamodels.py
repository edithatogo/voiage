# voiage/metamodels.py

"""
This module will contain the metamodels for approximating the relationship
between the parameters of a model and its outputs.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Metamodel:
    """
    A base class for metamodels.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class GaussianProcessMetamodel(Metamodel):
    """
    A Gaussian Process metamodel.
    """

    def __init__(self):
        super().__init__("gaussian_process")
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=9
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
