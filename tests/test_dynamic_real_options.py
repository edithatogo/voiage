from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.methods.dynamic_real_options import value_of_dynamic_real_options


def test_dynamic_real_options_values_delay_and_lock_in() -> None:
    values = np.array(
        [
            [[10.0, 11.0, 9.0], [8.0, 12.0, 13.0], [7.0, 10.0, 14.0]],
            [[10.0, 11.0, 9.0], [8.0, 12.0, 13.0], [7.0, 10.0, 14.0]],
        ]
    )
    result = value_of_dynamic_real_options(
        values,
        ["now", "after_phase_1", "after_phase_2"],
        ["immediate_adopt", "delay_and_review", "wait_for_trial"],
        {"now": 0.2, "after_phase_1": 0.3, "after_phase_2": 0.5},
        discount_rate=0.03,
        irreversibility_penalty=0.5,
        lock_in_penalty=0.25,
        evidence_arrival_times={"after_phase_1": 1.0, "after_phase_2": 2.0},
    )
    assert result.method_maturity == "fixture-backed"
    assert result.expected_net_benefits.shape == (3, 3)
    assert result.option_value >= 0
    assert result.waiting_value >= 0
    assert len(result.optimal_strategy_names) == 3
    assert result.reporting["analysis_type"] == "value_of_dynamic_real_options"


def test_dynamic_real_options_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="stage_weights"):
        value_of_dynamic_real_options(
            np.ones((1, 2, 2)), ["now", "later"], ["a", "b"], {"now": 0, "later": 0}
        )


def test_dynamic_real_options_cli_returns_json(tmp_path) -> None:
    specification = tmp_path / "dynamic-real-options.json"
    specification.write_text(
        '{"decision_stage_names":["now","later"],'
        '"strategy_names":["adopt","wait"],'
        '"net_benefit":[[[3,1],[2,4]]],'
        '"stage_weights":{"now":0.5,"later":0.5}}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["--format", "json", "calculate-dynamic-real-options", str(specification)]
    )
    assert result.exit_code == 0, result.stdout
    assert '"analysis_type": "value_of_dynamic_real_options"' in result.stdout
