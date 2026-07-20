"""Exhaustive tests for the mutation-gated VOIAGE C13 invariants."""

from __future__ import annotations

import pytest

from voiage.contracts.critical_invariants import (
    canonical_array_digest_input,
    capability_gaps,
)


def _gaps(**changes: object) -> tuple[str, ...]:
    values: dict[str, object] = {
        "method_family": "evpi",
        "dtype": "float64",
        "device": "cpu",
        "required_features": ("dense-array", "deterministic"),
        "supported_methods": ("evpi",),
        "supported_dtypes": ("float64",),
        "supported_devices": ("cpu",),
        "supported_features": ("dense-array", "deterministic"),
    }
    values.update(changes)
    return capability_gaps(**values)  # type: ignore[arg-type]


def test_capability_gaps_accept_exact_requirements() -> None:
    assert _gaps() == ()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"method_family": "evppi"}, ("method:evppi",)),
        ({"dtype": "float32"}, ("dtype:float32",)),
        ({"device": "cuda"}, ("device:cuda",)),
        (
            {"required_features": ("streaming", "autodiff")},
            ("capability:autodiff", "capability:streaming"),
        ),
    ],
)
def test_capability_gaps_report_each_requirement_exactly(
    changes: dict[str, object], expected: tuple[str, ...]
) -> None:
    assert _gaps(**changes) == expected


def test_capability_gaps_fail_closed_for_empty_devices() -> None:
    assert _gaps(device=None, supported_devices=()) == ("device:unavailable",)
    assert _gaps(device="cpu", supported_devices=()) == (
        "device:unavailable",
        "device:cpu",
    )


def test_capability_gaps_are_complete_and_stably_ordered() -> None:
    assert _gaps(
        method_family="evppi",
        dtype="float32",
        device="cuda",
        required_features=("streaming", "autodiff"),
        supported_methods=(),
        supported_dtypes=(),
        supported_devices=(),
        supported_features=(),
    ) == (
        "method:evppi",
        "dtype:float32",
        "device:unavailable",
        "device:cuda",
        "capability:autodiff",
        "capability:streaming",
    )


def test_array_digest_input_has_exact_canonical_framing() -> None:
    assert (
        canonical_array_digest_input(dtype="<f4", shape=(2, 3), payload=b"payload")
        == b'{"dtype":"<f4","shape":[2,3]}\0payload'
    )


@pytest.mark.parametrize(
    ("changes", "expected_different"),
    [
        ({"dtype": "<f8"}, True),
        ({"shape": (3, 2)}, True),
        ({"payload": b"payloae"}, True),
        ({}, False),
    ],
)
def test_array_digest_input_binds_every_identity_component(
    changes: dict[str, object], *, expected_different: bool
) -> None:
    baseline = canonical_array_digest_input(
        dtype="<f4", shape=(2, 3), payload=b"payload"
    )
    values: dict[str, object] = {
        "dtype": "<f4",
        "shape": (2, 3),
        "payload": b"payload",
    }
    values.update(changes)
    candidate = canonical_array_digest_input(**values)  # type: ignore[arg-type]
    assert (candidate != baseline) is expected_different
