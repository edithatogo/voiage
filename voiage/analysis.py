from typing import Any, Dict, Optional, Union
import numpy as np
from voiage.methods.basic import evpi, evppi
import voiage.schema


class DecisionAnalysis:
    """
    A class to represent a decision analysis problem.
    """

    def __init__(self, parameters: voiage.schema.ParameterSet, values: voiage.schema.ValueArray):
        """
        Initializes the DecisionAnalysis object.

        :param parameters: A ParameterSet object containing the model parameters.
        :param values: A ValueArray object containing the model values.
        """
        self.parameters = parameters
        self.values = values

    def evpi(
        self,
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
    ) -> float:
        """
        Calculates the Expected Value of Perfect Information (EVPI).
        """
        return evpi(
            nb_array=self.values.values,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
        )

    def evppi(
        self,
        parameter_samples: Union[np.ndarray, "PSASample", Dict[str, np.ndarray]],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
    ) -> float:
        """
        Calculates the Expected Value of Perfect Parameter Information (EVPPI).
        """
        return evppi(
            nb_array=self.values.values,
            parameter_samples=parameter_samples,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            regression_model=regression_model,
        )

    def sequential_evppi(
        self,
        parameter_samples: Union[np.ndarray, "ParameterSet", Dict[str, np.ndarray]],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
    ) -> float:
        """
        Calculates the Sequential Expected Value of Partial Perfect Information (EVPPI).
        """
        return sequential_evppi(
            nb_array=self.values.values,
            parameter_samples=parameter_samples,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            regression_model=regression_model,
        )

    def conditional_evppi(
        self,
        parameter_samples_of_interest: Union[np.ndarray, "ParameterSet", Dict[str, np.ndarray]],
        parameter_samples_given: Union[np.ndarray, "ParameterSet", Dict[str, np.ndarray]],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
    ) -> float:
        """
        Calculates the Conditional Expected Value of Partial Perfect Information (EVPPI).
        """
        return conditional_evppi(
            nb_array=self.values.values,
            parameter_samples_of_interest=parameter_samples_of_interest,
            parameter_samples_given=parameter_samples_given,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            regression_model=regression_model,
        )

    def joint_evppi(
        self,
        parameter_samples: Union[np.ndarray, "ParameterSet", Dict[str, np.ndarray]],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
    ) -> float:
        """
        Calculates the Joint Expected Value of Partial Perfect Information (EVPPI).
        """
        return joint_evppi(
            nb_array=self.values.values,
            parameter_samples=parameter_samples,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            regression_model=regression_model,
        )
