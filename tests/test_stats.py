# tests/test_stats.py

"""Test the statistical utility functions."""


from voiage.stats import beta_binomial_update, normal_normal_update


def test_normal_normal_update():
    """Test the normal_normal_update function."""
    prior_mean, prior_std = 10, 2
    data_mean, data_std = 12, 1
    n_samples = 10
    post_mean, post_std = normal_normal_update(
        prior_mean, prior_std, data_mean, data_std, n_samples
    )
    assert isinstance(post_mean, float)
    assert isinstance(post_std, float)
    assert post_mean > prior_mean
    assert post_std < prior_std


def test_beta_binomial_update():
    """Test the beta_binomial_update function."""
    prior_alpha, prior_beta = 1, 1
    n_successes, n_trials = 5, 10
    post_alpha, post_beta = beta_binomial_update(
        prior_alpha, prior_beta, n_successes, n_trials
    )
    assert isinstance(post_alpha, int)
    assert isinstance(post_beta, int)
    assert post_alpha == prior_alpha + n_successes
    assert post_beta == prior_beta + (n_trials - n_successes)
