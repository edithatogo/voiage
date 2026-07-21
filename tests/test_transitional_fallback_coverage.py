"""Exercise retained optional fallbacks while the Rust core takes over."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import voiage
from voiage.exceptions import OptionalDependencyError
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


def test_ceaf_requires_native_core_when_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ceaf_module._runtime,
        "compute_ceaf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("voiage._core")
        ),
    )
    with pytest.raises(ModuleNotFoundError, match="voiage._core"):
        ceaf_module.calculate_ceaf(_ceaf_values(), [0.0, 1.0])


def test_dominance_requires_native_core_when_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dominance_module._runtime,
        "compute_dominance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttributeError("compute_dominance")
        ),
    )
    with pytest.raises(AttributeError, match="compute_dominance"):
        dominance_module.calculate_dominance(
            [100.0, 200.0, 500.0, 800.0, 900.0],
            [1.0, 2.0, 2.5, 4.0, 3.5],
        )


def test_package_lazy_exports_distinguish_optional_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(voiage, "import_module", lambda *_args, **_kwargs: sentinel)
    assert voiage.__getattr__("cli") is sentinel
    with pytest.raises(AttributeError, match="not_an_export"):
        voiage.__getattr__("not_an_export")

    def missing_optional(*_args: object, **_kwargs: object) -> object:
        raise ModuleNotFoundError("No module named 'defusedxml'", name="defusedxml")

    monkeypatch.setattr(voiage, "import_module", missing_optional)
    with pytest.raises(OptionalDependencyError, match="ecosystem integration"):
        voiage.__getattr__("ecosystem_integration")

    def missing_jax(*_args: object, **_kwargs: object) -> object:
        raise ModuleNotFoundError("No module named 'jax'", name="jax")

    monkeypatch.setattr(voiage, "import_module", missing_jax)
    with pytest.raises(OptionalDependencyError, match="requires JAX"):
        voiage.__getattr__("health_economics")
