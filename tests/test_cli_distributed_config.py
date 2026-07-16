"""Tests for distributed large-scale CLI config templates."""

from voiage.cli import _generate_config_template


def test_generate_distributed_large_scale_config_template() -> None:
    """The distributed large-scale template should be exposed through the CLI."""
    payload = _generate_config_template("distributed-large-scale")

    assert payload["command"] == "create-distributed-large-scale"
    assert payload["chunk_size"] == 10000
    assert payload["n_nodes"] == 1
    assert payload["use_processes"] is True


def test_generate_distributed_large_scale_config_template_includes_scheduler_metadata() -> (
    None
):
    """The distributed template should carry scheduler metadata defaults."""
    payload = _generate_config_template("distributed-large-scale")

    assert payload.get("scheduler") == "process"
    assert payload.get("scheduler_is_placeholder") is False
    assert payload.get("scheduler_address") is None


def test_generate_ambiguity_distribution_shift_config_template() -> None:
    """The ambiguity/distribution-shift template should expose contract inputs."""
    payload = _generate_config_template("ambiguity-distribution-shift")
    assert payload["command"] == "calculate-ambiguity-distribution-shift"
    assert "shift_weights" in payload
    assert payload["scenario_names"] == ["source", "shifted"]


def test_generate_adaptive_learning_bandit_config_template() -> None:
    payload = _generate_config_template("adaptive-learning-bandit")
    assert payload["command"] == "calculate-adaptive-learning-bandit"
    assert payload["policy"] == "ucb"
