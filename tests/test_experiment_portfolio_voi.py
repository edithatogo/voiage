import pytest

from voiage.methods.portfolio import value_of_experiment_portfolio


def test_experiment_portfolio_voi_selects_affordable_studies() -> None:
    result = value_of_experiment_portfolio([8.0, 5.0, 3.0], [4.0, 2.0, 1.0], 3.0)
    assert result.selected_study_indices == (1, 2)
    assert result.value == 8.0


def test_experiment_portfolio_voi_rejects_negative_cost() -> None:
    with pytest.raises(ValueError):
        value_of_experiment_portfolio([1.0], [-1.0], 1.0)
