"""Property evidence for estimation-focused variance-reduction estimands."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, pvariance

from hypothesis import example, given, settings
from hypothesis import strategies as st
import pytest


def _discrete_decomposition(
    values: list[float], groups: list[int]
) -> tuple[float, float, float]:
    grouped: defaultdict[int, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[group].append(value)
    prior = pvariance(values)
    expected_within = sum(
        len(members) / len(values) * pvariance(members) for members in grouped.values()
    )
    group_means = [fmean(grouped[group]) for group in groups]
    between = pvariance(group_means)
    return prior, expected_within, between


@given(
    values=st.lists(
        st.floats(min_value=-1.0e6, max_value=1.0e6, allow_nan=False),
        min_size=2,
        max_size=30,
    ),
    split=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=80, deadline=None)
@example(
    values=[699051.4296875, 0.0, 699051.4296875, 0.0, 699051.43359375],
    split=2,
)
def test_discrete_evppi_var_obeys_total_variance(
    values: list[float], split: int
) -> None:
    groups = [index % split for index in range(len(values))]
    prior, expected_within, between = _discrete_decomposition(values, groups)
    assert prior == pytest.approx(expected_within + between, rel=1.0e-12, abs=1.0e-8)
    tolerance = max(1.0, abs(prior)) * 1.0e-12
    assert 0.0 <= between <= prior + tolerance


@given(
    values=st.lists(
        st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False),
        min_size=2,
        max_size=30,
    )
)
@settings(max_examples=50, deadline=None)
def test_evppi_var_zero_and_perfect_information_limits(values: list[float]) -> None:
    prior, no_information_posterior, no_information_reduction = _discrete_decomposition(
        values, [0] * len(values)
    )
    assert no_information_posterior == pytest.approx(prior)
    assert no_information_reduction == pytest.approx(0.0, abs=1.0e-12)

    _, perfect_information_posterior, perfect_information_reduction = (
        _discrete_decomposition(values, list(range(len(values))))
    )
    assert perfect_information_posterior == 0.0
    assert perfect_information_reduction == pytest.approx(prior)


@given(
    prior_variance=st.floats(min_value=1.0e-9, max_value=1.0e6, allow_nan=False),
    sampling_variance=st.floats(min_value=1.0e-9, max_value=1.0e6, allow_nan=False),
    smaller_sample=st.integers(min_value=1, max_value=1_000),
    increment=st.integers(min_value=0, max_value=1_000),
)
@settings(max_examples=80, deadline=None)
def test_normal_normal_evsi_var_is_bounded_and_monotone_for_nested_samples(
    prior_variance: float,
    sampling_variance: float,
    smaller_sample: int,
    increment: int,
) -> None:
    larger_sample = smaller_sample + increment

    def reduction(sample_size: int) -> float:
        posterior = 1.0 / ((1.0 / prior_variance) + (sample_size / sampling_variance))
        return prior_variance - posterior

    smaller = reduction(smaller_sample)
    larger = reduction(larger_sample)
    tolerance = max(1.0, prior_variance) * 1.0e-12
    assert -tolerance <= smaller <= prior_variance + tolerance
    assert smaller <= larger + tolerance
