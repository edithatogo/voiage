"""Tests for voiage.exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voiage.exceptions import (
    BackendNotAvailableError,
    CalculationError,
    DimensionMismatchError,
    InputError,
    NumericalError,
    OptionalDependencyError,
    PlottingError,
    SerializationError,
    VoiageError,
    VoiageNotImplementedError,
    raise_backend_not_available_error,
    raise_calculation_error,
    raise_dimension_mismatch_error,
    raise_import_error,
    raise_input_error,
    raise_not_implemented_error,
    raise_optional_dependency_error,
    raise_plotting_error,
    raise_runtime_error,
    raise_type_error,
    raise_value_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_frozen_v1_exception_inheritance_contract() -> None:
    """New v1 errors preserve the established catch hierarchies."""
    numerical = NumericalError("unstable result")
    serialization = SerializationError("invalid payload")

    assert isinstance(numerical, CalculationError)
    assert isinstance(numerical, VoiageError)
    assert isinstance(serialization, VoiageError)
    assert not isinstance(serialization, CalculationError)


@pytest.mark.parametrize(
    ("raiser", "exc_type", "message"),
    [
        (raise_input_error, InputError, "bad input"),
        (raise_dimension_mismatch_error, DimensionMismatchError, "bad dims"),
        (raise_optional_dependency_error, OptionalDependencyError, "missing dep"),
        (raise_plotting_error, PlottingError, "plot failed"),
        (raise_not_implemented_error, VoiageNotImplementedError, "todo"),
        (
            raise_backend_not_available_error,
            BackendNotAvailableError,
            "backend missing",
        ),
        (raise_import_error, ImportError, "import failed"),
        (raise_runtime_error, RuntimeError, "runtime failed"),
        (raise_type_error, TypeError, "type failed"),
    ],
)
def test_raise_helpers_raise_expected_exception(
    raiser: Callable[[str], None],
    exc_type: type[BaseException],
    message: str,
) -> None:
    """Each helper should raise the documented exception type."""
    with pytest.raises(exc_type, match=message):
        raiser(message)


def test_raise_value_error_without_cause() -> None:
    """raise_value_error should raise a plain ValueError when no cause is given."""
    with pytest.raises(ValueError, match="bad value"):
        raise_value_error("bad value")


def test_raise_value_error_with_cause() -> None:
    """raise_value_error should chain the supplied cause."""
    cause = KeyError("missing")
    with pytest.raises(ValueError, match="bad value") as excinfo:
        raise_value_error("bad value", cause=cause)

    assert excinfo.value.__cause__ is cause


def test_raise_calculation_error_without_cause() -> None:
    """raise_calculation_error should raise a CalculationError directly."""
    with pytest.raises(CalculationError, match="calc failed"):
        raise_calculation_error("calc failed")


def test_raise_calculation_error_with_cause() -> None:
    """raise_calculation_error should chain the supplied cause."""
    cause = ZeroDivisionError("division by zero")
    with pytest.raises(CalculationError, match="calc failed") as excinfo:
        raise_calculation_error("calc failed", cause=cause)

    assert excinfo.value.__cause__ is cause
