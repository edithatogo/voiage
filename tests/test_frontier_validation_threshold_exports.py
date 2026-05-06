"""Export surface checks for the new frontier implementation methods."""

from __future__ import annotations

import voiage as top_level
from voiage import methods


def test_frontier_validation_threshold_exports_are_curated() -> None:
    """The new frontier methods should be exported from package and module roots."""
    assert top_level.ModelValidationResult is methods.ModelValidationResult
    assert top_level.ThresholdResult is methods.ThresholdResult
    assert top_level.ValidationProfile is methods.ValidationProfile
    assert top_level.ValidationProfileSet is methods.ValidationProfileSet
    assert top_level.ThresholdProfile is methods.ThresholdProfile
    assert top_level.ThresholdProfileSet is methods.ThresholdProfileSet
    assert top_level.value_of_model_validation is methods.value_of_model_validation
    assert top_level.value_of_threshold is methods.value_of_threshold
    assert (
        top_level.value_of_threshold_information
        is methods.value_of_threshold_information
    )
    assert top_level.value_of_validation is methods.value_of_validation
