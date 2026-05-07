"""Tests for the JAX-backed advanced regression helper."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.backends.advanced_jax_regression import JaxAdvancedRegression


def test_predict_requires_fit() -> None:
    """Prediction should fail before the model has been fitted."""
    regression = JaxAdvancedRegression()

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        regression.predict(np.array([0.0, 1.0]))


def test_polynomial_features_for_single_feature_input() -> None:
    """Single-feature input should produce bias, powers, and no interactions."""
    regression = JaxAdvancedRegression()

    features = regression.polynomial_features(np.array([0.0, 1.0, 2.0]), degree=2)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0],
        ]
    )

    np.testing.assert_allclose(features, expected)


def test_polynomial_features_include_interaction_terms() -> None:
    """Multi-feature input should include pairwise interaction terms."""
    regression = JaxAdvancedRegression()

    features = regression.polynomial_features(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        degree=2,
    )

    expected = np.array(
        [
            [1.0, 1.0, 2.0, 1.0, 4.0, 2.0],
            [1.0, 3.0, 4.0, 9.0, 16.0, 12.0],
        ]
    )

    np.testing.assert_allclose(features, expected)


def test_fit_predict_and_r_squared_on_exact_polynomial_data() -> None:
    """Closed-form fitting should recover a noiseless quadratic relationship."""
    regression = JaxAdvancedRegression()
    x = np.arange(6.0)
    y = 2.0 + 3.0 * x + 0.5 * x**2

    fitted = regression.fit_polynomial(x, y, degree=2)

    assert fitted is regression
    np.testing.assert_allclose(regression.predict(x), y, atol=1e-6)
    assert regression.r_squared(x, y) == pytest.approx(1.0)


def test_cross_validate_returns_high_score_for_exact_polynomial_data() -> None:
    """Cross-validation should stay stable on deterministic polynomial data."""
    regression = JaxAdvancedRegression()
    x = np.arange(8.0)
    y = 1.0 - 2.0 * x + x**2

    score = regression.cross_validate(x, y, degree=2, n_folds=4)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score == pytest.approx(1.0, abs=1e-5)
