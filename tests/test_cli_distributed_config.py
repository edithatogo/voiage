"""Tests for distributed large-scale CLI config templates."""

from voiage.cli import _generate_config_template


def test_generate_distributed_large_scale_config_template() -> None:
    """The distributed large-scale template should be exposed through the CLI."""
    payload = _generate_config_template("distributed-large-scale")

    assert payload["command"] == "create-distributed-large-scale"
    assert payload["chunk_size"] == 10000
    assert payload["n_nodes"] == 1
    assert payload["use_processes"] is True


def test_generate_distributed_large_scale_config_template_includes_scheduler_metadata() -> None:
    """The distributed template should carry scheduler metadata defaults."""
    payload = _generate_config_template("distributed-large-scale")

    assert payload.get("scheduler") == "process"
    assert payload.get("scheduler_is_placeholder") is False
    assert payload.get("scheduler_address") is None
