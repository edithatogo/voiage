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

    def portfolio_voi(
        self,
        portfolio_specification: "PortfolioSpec",
        study_value_calculator: "StudyValueCalculator",
        optimization_method: str = "greedy",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Optimizes a portfolio of research studies to maximize total value.
        """
        return portfolio_voi(
            portfolio_specification=portfolio_specification,
            study_value_calculator=study_value_calculator,
            optimization_method=optimization_method,
            **kwargs,
        )

    def plot_ce_plane(self, wtp, ax=None, show=True, **kwargs):
        """
        Plot the Cost-Effectiveness Plane.
        """
        from voiage.plot.ce_plane import plot_ce_plane

        return plot_ce_plane(
            value_array=self.values,
            wtp=wtp,
            ax=ax,
            show=show,
            **kwargs,
        )

    def plot_voi_curves(self, wtp_range, ax=None, show=True, **kwargs):
        """
        Plot VOI curves.
        """
        from voiage.plot.voi_curves import plot_voi_curves

        return plot_voi_curves(
            analysis=self,
            wtp_range=wtp_range,
            ax=ax,
            show=show,
            **kwargs,
        )

    def plot_ceac(self, wtp_range, ax=None, show=True, **kwargs):
        """
        Plot the Cost-Effectiveness Acceptability Curve (CEAC).
        """
        from voiage.plot.ceac import plot_ceac

        return plot_ceac(
            value_array=self.values,
            wtp_range=wtp_range,
            ax=ax,
            show=show,
            **kwargs,
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
