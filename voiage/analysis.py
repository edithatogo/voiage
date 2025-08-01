# voiage/analysis.py

"""
The core object-oriented API for voiage.

This module contains the `DecisionAnalysis` class, which encapsulates the
entire state of a Value of Information problem, including the model parameters,
the value outputs, and the decision options.
"""

from typing import List

from voiage.schema import ParameterSet, ValueArray


class DecisionAnalysis:
    """
    Encapsulates a complete Value of Information (VOI) analysis.

    This class provides a user-facing, object-oriented interface for performing
    various VOI calculations. It holds the probabilistic sensitivity analysis
    (PSA) inputs and provides methods to calculate EVPI, EVPPI, EVSI, etc.

    Parameters
    ----------
    parameters : ParameterSet
        An object containing the samples of the model's input parameters from
        a PSA. Each parameter is a named array of samples.
    values : ValueArray
        An object containing the calculated values (e.g., net monetary benefit)
        for each decision option, corresponding to each sample in the
        ParameterSet.
    """

    def __init__(self, parameters: ParameterSet, values: ValueArray):
        if not isinstance(parameters, ParameterSet):
            raise TypeError("`parameters` must be a ParameterSet object.")
        if not isinstance(values, ValueArray):
            raise TypeError("`values` must be a ValueArray object.")
        if parameters.n_samples != values.n_samples:
            raise ValueError(
                "The number of samples in ParameterSet and ValueArray must be equal."
            )

        self._parameters = parameters
        self._values = values

    def evpi(self) -> float:
        """
        Calculate the Expected Value of Perfect Information (EVPI).

        EVPI represents the expected value of resolving all uncertainty in the
        decision model. It is the maximum amount a decision-maker would be

        willing to pay for perfect information about the model parameters.

        Returns
        -------
        float
            The calculated EVPI, in the same units as the ValueArray.
        """
        # --- Implementation to be added ---
        # 1. Find the max value for each sample across options
        # 2. Calculate the mean of these maximums (Expected Value with Perfect Info)
        # 3. Find the max of the mean values for each option (Expected Value with Current Info)
        # 4. EVPI = Step 2 - Step 3
        raise NotImplementedError("Coming soon!")

    def evppi(self, parameters_of_interest: List[str]) -> float:
        """
        Calculate the Expected Value of Partial Perfect Information (EVPPI).

        EVPPI represents the expected value of resolving uncertainty for a
        specific subset of model parameters. It helps identify which parameters

        contribute most to the decision uncertainty.

        Parameters
        ----------
        parameters_of_interest : List[str]
            A list of parameter names for which to calculate the EVPPI. These
            names must exist in the `ParameterSet`.

        Returns
        -------
        float
            The calculated EVPPI for the specified subset of parameters.
        """
        # --- Implementation to be added ---
        # This is more complex and often requires a nested Monte Carlo or
        # metamodeling approach.
        raise NotImplementedError("Coming soon!")
