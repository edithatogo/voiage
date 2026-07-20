"""Exercise retained optional fallbacks while the Rust core takes over."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from voiage.methods import ceaf as ceaf_module
from voiage.methods import dominance as dominance_module
from voiage.schema import ValueArray


def _ceaf_values() -> ValueArray:
    values = np.asarray(
        [
            [[5.0, 4.0], [1.0, 2.0]],
            [[4.0, 5.0], [2.0, 1.0]],
        ]
    )
    return ValueArray(
        dataset=xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies", "n_wtp"), values)},
            coords={
                "n_samples": np.arange(2),
                "n_strategies": np.arange(2),
                "n_wtp": np.arange(2),
                "strategy": ("n_strategies", ["A", "B"]),
            },
        )
    )


def test_ceaf_uses_python_fallback_when_native_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ceaf_module._runtime,
        "compute_ceaf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("voiage._core")
        ),
    )
    with pytest.warns(DeprecationWarning, match="Python CEAF fallback"):
        result = ceaf_module.calculate_ceaf(_ceaf_values(), [0.0, 1.0])
    assert result.optimal_strategy_indices.tolist() == [0, 0]
    assert np.all(result.probability_lower <= result.probability_upper)


def test_dominance_uses_python_fallback_when_native_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dominance_module._runtime,
        "compute_dominance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttributeError("compute_dominance")
        ),
    )
    with pytest.warns(DeprecationWarning, match="Python dominance fallback"):
        result = dominance_module.calculate_dominance(
            [100.0, 200.0, 500.0, 800.0, 900.0],
            [1.0, 2.0, 2.5, 4.0, 3.5],
        )
    assert result.frontier_indices == [0, 1, 3]
    assert result.strongly_dominated_indices == [4]
