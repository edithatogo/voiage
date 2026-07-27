import numpy as np

from voiage.methods.source_portfolio import value_of_information_source_portfolio


def test_source_portfolio_reports_joint_and_incremental_value() -> None:
    result = value_of_information_source_portfolio(
        {"survey": np.array([6.0, 4.0]), "experiment": np.array([4.0, 7.0])},
        np.array([5.0, 5.0]),
    )
    assert result.joint_value == 0.5
    assert set(result.incremental_values) == {"survey", "experiment"}
