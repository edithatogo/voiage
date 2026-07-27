"""Forecast and signal value of information."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class SignalValueResult:
    """Value of selecting a strategy after observing a signal."""

    value: float
    baseline_value: float
    signal_value: float
    baseline_strategy_index: int
    signal_strategy_indices: dict[str, int]


def value_of_signal_information(
    net_benefits: np.ndarray, signals: list[str] | np.ndarray
) -> SignalValueResult:
    """Calculate signal value using conditional mean net benefits."""
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    labels = np.asarray(signals)
    if values.ndim != 2 or labels.ndim != 1 or values.shape[0] != labels.size or values.shape[1] < 1:
        raise_input_error("net_benefits must be 2D and signals must align to samples.")
    if not np.all(np.isfinite(values)):
        raise_input_error("net_benefits must be finite.")
    baseline_means = values.mean(axis=0)
    baseline_index = int(np.argmax(baseline_means))
    signal_indices: dict[str, int] = {}
    signal_value = 0.0
    for label in sorted({str(item) for item in labels}):
        conditional = values[labels.astype(str) == label].mean(axis=0)
        signal_indices[label] = int(np.argmax(conditional))
        signal_value += float(np.max(conditional)) * float(np.mean(labels.astype(str) == label))
    baseline = float(baseline_means[baseline_index])
    return SignalValueResult(max(0.0, signal_value - baseline), baseline, signal_value, baseline_index, signal_indices)
