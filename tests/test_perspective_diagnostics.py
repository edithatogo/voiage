# Adapted from edithatogo/vop_poc_nz at ad84d309 (Apache-2.0).
from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.perspective_diagnostics import (
    bootstrap_evop,
    evop_convergence_profile,
)
from voiage.methods.perspective_tensor import NetBenefitTensor, TiePolicy


def deterministic_tensor() -> NetBenefitTensor:
    values = np.repeat(
        np.array([[[2.0, 0.0], [0.0, 10.0]]], dtype=float),
        repeats=20,
        axis=0,
    )
    return NetBenefitTensor(
        values, strategies=("a", "b"), perspectives=("source", "target")
    )


def test_bootstrap_evop_reports_zero_mc_error_for_deterministic_draws() -> None:
    result = bootstrap_evop(
        deterministic_tensor(),
        choose_under="source",
        evaluate_under="target",
        bootstrap_replicates=40,
        seed=7,
        selection_tie_policy=TiePolicy.SPLIT,
    )
    assert result.estimate == pytest.approx(10.0)
    assert result.standard_error == pytest.approx(0.0)
    assert result.lower == pytest.approx(10.0)
    assert result.upper == pytest.approx(10.0)


def test_evop_convergence_profile_includes_full_sample() -> None:
    rows = evop_convergence_profile(
        deterministic_tensor(),
        choose_under="source",
        evaluate_under="target",
        draw_counts=(5, 10, 20),
        repeats=4,
        seed=3,
    )
    assert [row.draws for row in rows] == [5, 10, 20]
    assert all(row.mean == pytest.approx(10.0) for row in rows)
    assert rows[-1].repeats == 1


def test_bootstrap_evop_rejects_invalid_configuration() -> None:
    tensor = deterministic_tensor()
    with pytest.raises(ValueError):
        bootstrap_evop(
            tensor,
            choose_under="source",
            evaluate_under="target",
            bootstrap_replicates=1,
        )
    with pytest.raises(ValueError):
        bootstrap_evop(
            tensor, choose_under="source", evaluate_under="target", confidence=1.0
        )
