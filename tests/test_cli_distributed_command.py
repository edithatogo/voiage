"""Tests for the distributed large-scale CLI command."""

from typer.testing import CliRunner

from voiage import cli


runner = CliRunner()


def test_create_distributed_large_scale_command(tmp_path) -> None:
    """The CLI should prepare a distributed large-scale analysis payload."""
    net_benefit_file = tmp_path / "net_benefits.csv"
    net_benefit_file.write_text("1,2\n3,4\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "create-distributed-large-scale",
            str(net_benefit_file),
            "--chunk-size",
            "5000",
            "--n-nodes",
            "2",
            "--workers-per-node",
            "4",
            "--scheduler",
            "thread",
            "--use-threads",
        ],
    )

    assert result.exit_code == 0
    assert '"analysis_type": "distributed_large_scale_preparation"' in result.stdout
    assert '"scheduler": "local-thread"' in result.stdout
    assert '"command": "create-distributed-large-scale"' in result.stdout
    assert '"n_nodes": 2' in result.stdout
    assert '"workers_per_node": 4' in result.stdout
    assert '"use_processes": false' in result.stdout
    assert '"chunk_size": 5000' in result.stdout


def test_create_distributed_large_scale_command_with_scheduler_address(tmp_path) -> None:
    """The CLI should surface a remote scheduler address in the payload."""
    net_benefit_file = tmp_path / "net_benefits.csv"
    net_benefit_file.write_text("1,2\n3,4\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "create-distributed-large-scale",
            str(net_benefit_file),
            "--scheduler",
            "dask",
            "--scheduler-address",
            "tcp://scheduler.example:8786",
        ],
    )

    assert result.exit_code == 0
    assert '"scheduler": "dask-cluster"' in result.stdout
    assert '"scheduler_address": "tcp://scheduler.example:8786"' in result.stdout


def test_create_distributed_large_scale_command_with_ray_scheduler(tmp_path) -> None:
    """The CLI should expose the Ray scheduler selection in the payload."""
    net_benefit_file = tmp_path / "net_benefits.csv"
    net_benefit_file.write_text("1,2\n3,4\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "create-distributed-large-scale",
            str(net_benefit_file),
            "--scheduler",
            "ray",
        ],
    )

    assert result.exit_code == 0
    assert '"scheduler": "ray-cluster"' in result.stdout


def test_create_distributed_large_scale_command_with_fpga_scheduler(tmp_path) -> None:
    """The CLI should expose the FPGA scheduler placeholder in the payload."""
    net_benefit_file = tmp_path / "net_benefits.csv"
    net_benefit_file.write_text("1,2\n3,4\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "create-distributed-large-scale",
            str(net_benefit_file),
            "--scheduler",
            "fpga",
        ],
    )

    assert result.exit_code == 0
    assert '"scheduler": "fpga-cluster"' in result.stdout
    assert '"scheduler_is_placeholder": true' in result.stdout


def test_create_distributed_large_scale_command_with_asic_scheduler(tmp_path) -> None:
    """The CLI should expose the ASIC scheduler placeholder in the payload."""
    net_benefit_file = tmp_path / "net_benefits.csv"
    net_benefit_file.write_text("1,2\n3,4\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "create-distributed-large-scale",
            str(net_benefit_file),
            "--scheduler",
            "asic",
        ],
    )

    assert result.exit_code == 0
    assert '"scheduler": "asic-cluster"' in result.stdout
    assert '"scheduler_is_placeholder": true' in result.stdout


def test_create_distributed_large_scale_help_mentions_placeholder_schedulers() -> None:
    """The help text should expose the placeholder accelerator scheduler names."""
    result = runner.invoke(cli.app, ["create-distributed-large-scale", "--help"])

    assert result.exit_code == 0
    assert "fpga" in result.stdout
    assert "asic" in result.stdout
