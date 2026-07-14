"""Tests for small runtime modules."""

from __future__ import annotations

import runpy

import numpy as np

from voiage import config


def test_config_constants() -> None:
    """Configuration constants should expose the expected defaults."""
    assert config.DEFAULT_DTYPE is np.float64
    assert config.DEFAULT_COMPUTATION_BACKEND == "numpy"
    assert config.DEFAULT_MC_SAMPLES == 10_000
    assert config.DEFAULT_EVSI_REGRESSION_METHOD == "gam"


def test_module_entrypoint_runs_app(monkeypatch) -> None:
    """`python -m voiage` should invoke the CLI app."""
    calls: list[bool] = []

    def fake_app() -> None:
        calls.append(True)

    monkeypatch.setattr("voiage.cli.app", fake_app)
    runpy.run_module("voiage.__main__", run_name="__main__")

    assert calls == [True]
