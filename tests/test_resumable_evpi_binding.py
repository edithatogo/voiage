"""Python boundary evidence for the bounded EVPI kernel."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from voiage import _core

if TYPE_CHECKING:
    from pathlib import Path


def test_resumable_evpi_translates_partial_and_complete(tmp_path: Path) -> None:
    samples = [[0.0, 2.0], [1.0, 0.0]]
    checkpoint = str(tmp_path / "evpi.checkpoint")
    partial = _core.compute_evpi_resumable(
        samples, checkpoint, "model", "rng", "algorithm", 1, 1, False
    )
    assert partial == {"state": "partial", "evaluations": 1, "value": None}
    complete = _core.compute_evpi_resumable(
        samples, checkpoint, "model", "rng", "algorithm", 2, 1, True
    )
    assert complete["state"] == "complete"
    assert complete["evaluations"] == 2
    assert complete["value"] == pytest.approx(0.5)


def test_resumable_evpi_rejects_changed_identity(tmp_path: Path) -> None:
    checkpoint = str(tmp_path / "evpi.checkpoint")
    samples = [[0.0, 2.0], [1.0, 0.0]]
    _core.compute_evpi_resumable(
        samples, checkpoint, "model", "rng", "algorithm", 1, 1, False
    )
    with pytest.raises(RuntimeError, match="identity"):
        _core.compute_evpi_resumable(
            samples, checkpoint, "changed", "rng", "algorithm", 2, 1, True
        )


def test_resumable_evpi_zero_budget_and_storage_error(tmp_path: Path) -> None:
    samples = [[0.0, 2.0], [1.0, 0.0]]
    result = _core.compute_evpi_resumable(
        samples, str(tmp_path / "state"), "model", "rng", "algorithm", 0, 1, False
    )
    assert result == {"state": "partial", "evaluations": 0, "value": None}
    with pytest.raises(RuntimeError):
        _core.compute_evpi_resumable(
            samples,
            str(tmp_path / "missing" / "state"),
            "model",
            "rng",
            "algorithm",
            2,
            1,
            False,
        )


def test_resumable_evpi_honours_cancellation(tmp_path: Path) -> None:
    token = _core.CancellationToken()
    token.cancel()
    assert token.is_cancelled()
    result = _core.compute_evpi_resumable(
        [[0.0, 2.0], [1.0, 0.0]],
        str(tmp_path / "state"),
        "model",
        "rng",
        "algorithm",
        2,
        1,
        False,
        token,
    )
    assert result == {"state": "cancelled", "evaluations": 0, "value": None}


def test_running_evpi_can_be_cancelled_from_another_thread(tmp_path: Path) -> None:
    token = _core.CancellationToken()
    checkpoint = tmp_path / "threaded-state"
    result: list[dict[str, object]] = []
    errors: list[Exception] = []

    def run() -> None:
        try:
            result.append(
                _core.compute_evpi_resumable(
                    [[0.0, 2.0], [1.0, 0.0]] * 5_000,
                    str(checkpoint),
                    "model",
                    "rng",
                    "algorithm",
                    10_000,
                    1,
                    False,
                    token,
                )
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while (
            not checkpoint.exists()
            and worker.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert checkpoint.exists(), "worker did not initialize its checkpoint"
    finally:
        token.cancel()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors
    assert result[0]["state"] == "cancelled"
    assert result[0]["value"] is None
