"""Comprehensive tests for the CLI implementation."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage import cli

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evpi_cli() -> None:
    """Test the EVPI CLI command with various options."""
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create sample net benefits data
        net_benefits_file = tmpdir_path / "net_benefits.csv"
        net_benefits_data = [
            ["Strategy A", "Strategy B"],
            [1000, 1200],
            [950, 1250],
            [1050, 1150],
            [900, 1300],
            [1100, 1100],
        ]

        with open(net_benefits_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(net_benefits_data)

        # Test basic EVPI calculation
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evpi",
                str(net_benefits_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        evpi_value = float(result.stdout.split(":")[1].strip())
        assert evpi_value >= 0

        # Test EVPI with population scaling
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evpi",
                str(net_benefits_file),
                "--population",
                "100000",
                "--time-horizon",
                "10",
                "--discount-rate",
                "0.03",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        scaled_evpi_value = float(result.stdout.split(":")[1].strip())
        assert scaled_evpi_value >= 0
        # Scaled EVPI should be larger than unscaled
        assert scaled_evpi_value >= evpi_value

        # Test EVPI with output file
        output_file = tmpdir_path / "evpi_result.txt"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evpi",
                str(net_benefits_file),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        assert "Result saved to" in result.stdout
        assert output_file.exists()

        # Check output file content
        with open(output_file) as f:
            content = f.read().strip()
            assert "EVPI:" in content


def test_evppi_cli() -> None:
    """Test the EVPPI CLI command with various options."""
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create sample net benefits data
        net_benefits_file = tmpdir_path / "net_benefits.csv"
        net_benefits_data = [
            ["Strategy A", "Strategy B"],
            [1000, 1200],
            [950, 1250],
            [1050, 1150],
            [900, 1300],
            [1100, 1100],
        ]

        with open(net_benefits_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(net_benefits_data)

        # Create sample parameters data
        parameters_file = tmpdir_path / "parameters.csv"
        parameters_data = [
            ["param1", "param2"],
            [0.5, 0.3],
            [0.6, 0.4],
            [0.4, 0.2],
            [0.7, 0.5],
            [0.5, 0.3],
        ]

        with open(parameters_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(parameters_data)

        # Test basic EVPPI calculation
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evppi",
                str(net_benefits_file),
                str(parameters_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        evppi_value = float(result.stdout.split(":")[1].strip())
        assert evppi_value >= 0

        # Test EVPPI with population scaling
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evppi",
                str(net_benefits_file),
                str(parameters_file),
                "--population",
                "100000",
                "--time-horizon",
                "10",
                "--discount-rate",
                "0.03",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        scaled_evppi_value = float(result.stdout.split(":")[1].strip())
        assert scaled_evppi_value >= 0

        # Test EVPPI with output file
        output_file = tmpdir_path / "evppi_result.txt"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiage.cli",
                "calculate-evppi",
                str(net_benefits_file),
                str(parameters_file),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        assert "Result saved to" in result.stdout
        assert output_file.exists()

        # Check output file content
        with open(output_file) as f:
            content = f.read().strip()
            assert "EVPPI:" in content


def test_cli_help() -> None:
    """Test CLI help commands."""
    # Test main help
    result = subprocess.run(
        [sys.executable, "-m", "voiage.cli", "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "voiage" in result.stdout
    assert "calculate-evpi" in result.stdout
    assert "calculate-evppi" in result.stdout

    # Test calculate-evpi help
    result = subprocess.run(
        [sys.executable, "-m", "voiage.cli", "calculate-evpi", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "calculate-evpi" in result.stdout
    assert "NET_BENEFIT_FILE" in result.stdout

    # Test calculate-evppi help
    result = subprocess.run(
        [sys.executable, "-m", "voiage.cli", "calculate-evppi", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "calculate-evppi" in result.stdout
    assert "NET_BENEFIT_FILE" in result.stdout
    assert "PARAMETER_FILE" in result.stdout


def test_cli_error_handling() -> None:
    """Test CLI error handling."""
    # Test missing file
    result = subprocess.run(
        [sys.executable, "-m", "voiage.cli", "calculate-evpi", "nonexistent_file.csv"],
        capture_output=True,
        text=True,
    )

    # This should fail because the file doesn't exist
    assert result.returncode != 0
    assert "Error" in result.stderr or "error" in result.stderr.lower()


def test_structural_evpi_cli_with_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the structural EVPI command success path and file output."""
    net_benefits_file = tmp_path / "net_benefits.csv"
    net_benefits_file.write_text("placeholder\n", encoding="utf-8")
    config_file = tmp_path / "structural_evpi.json"
    output_file = tmp_path / "structural_evpi_result.txt"
    _write_json(
        config_file,
        {
            "structures": [
                {
                    "name": "A",
                    "probability": 0.25,
                    "net_benefits_file": "net_benefits.csv",
                },
                {
                    "name": "B",
                    "probability": 0.75,
                    "net_benefits_file": "net_benefits.csv",
                },
            ]
        },
    )

    read_calls: list[tuple[str, bool]] = []
    probabilities_seen: list[float] = []

    def fake_read_value_array_csv(
        path: str, skip_header: bool = False
    ) -> SimpleNamespace:
        read_calls.append((path, skip_header))
        return SimpleNamespace(numpy_values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    def fake_structural_evpi(
        evaluators: list,
        probabilities: list[float],
        psa_samples: list,
        **kwargs: object,
    ) -> float:
        probabilities_seen.extend(probabilities)
        assert len(evaluators) == 2
        assert len(psa_samples) == 2
        assert kwargs["population"] == 1000
        assert kwargs["time_horizon"] == 5
        assert kwargs["discount_rate"] == 0.05
        return 12.345678

    monkeypatch.setattr(cli, "read_value_array_csv", fake_read_value_array_csv)
    monkeypatch.setattr(cli, "structural_evpi", fake_structural_evpi)

    result = runner.invoke(
        cli.app,
        [
            "calculate-structural-evpi",
            str(config_file),
            "--population",
            "1000",
            "--time-horizon",
            "5",
            "--discount-rate",
            "0.05",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Structural EVPI: 12.345678" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert (
        output_file.read_text(encoding="utf-8").strip() == "Structural EVPI: 12.345678"
    )
    assert read_calls == [
        (str(net_benefits_file), True),
        (str(net_benefits_file), True),
    ]
    assert probabilities_seen == [0.25, 0.75]


def test_structural_evpi_cli_validation_errors(tmp_path: Path) -> None:
    """Exercise structural EVPI config validation and JSON parsing errors."""
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not json", encoding="utf-8")
    result = runner.invoke(cli.app, ["calculate-structural-evpi", str(invalid_json)])
    assert result.exit_code == 1
    assert "Error: Invalid JSON in config file" in result.stderr

    missing_structures = tmp_path / "missing_structures.json"
    _write_json(missing_structures, {"foo": "bar"})
    result = runner.invoke(
        cli.app, ["calculate-structural-evpi", str(missing_structures)]
    )
    assert result.exit_code == 1
    assert "Error: Config file must contain 'structures' key" in result.stderr

    invalid_probabilities = tmp_path / "invalid_probabilities.json"
    _write_json(
        invalid_probabilities,
        {
            "structures": [
                {"probability": 0.2, "net_benefits_file": "a.csv"},
                {"probability": 0.3, "net_benefits_file": "b.csv"},
            ]
        },
    )
    result = runner.invoke(
        cli.app, ["calculate-structural-evpi", str(invalid_probabilities)]
    )
    assert result.exit_code == 1
    assert "Error: Structure probabilities must sum to 1 (got 0.5)" in result.stderr


def test_structural_evppi_cli_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the structural EVPPI command success path."""
    net_benefits_file = tmp_path / "net_benefits.csv"
    net_benefits_file.write_text("placeholder\n", encoding="utf-8")
    config_file = tmp_path / "structural_evppi.json"
    _write_json(
        config_file,
        {
            "structures": [
                {
                    "name": "A",
                    "probability": 0.4,
                    "net_benefits_file": "net_benefits.csv",
                },
                {
                    "name": "B",
                    "probability": 0.6,
                    "net_benefits_file": "net_benefits.csv",
                },
            ]
        },
    )

    def fake_read_value_array_csv(
        path: str, skip_header: bool = False
    ) -> SimpleNamespace:
        assert path == str(net_benefits_file)
        assert skip_header is True
        return SimpleNamespace(numpy_values=np.array([[5.0, 6.0], [7.0, 8.0]]))

    def fake_structural_evppi(
        evaluators: list,
        probabilities: list[float],
        psa_samples: list,
        structures_of_interest: list[int],
        **kwargs: object,
    ) -> float:
        assert len(evaluators) == 2
        assert len(psa_samples) == 2
        assert probabilities == [0.4, 0.6]
        assert structures_of_interest == [0, 1]
        assert kwargs["population"] == 2500
        return 9.876543

    monkeypatch.setattr(cli, "read_value_array_csv", fake_read_value_array_csv)
    monkeypatch.setattr(cli, "structural_evppi", fake_structural_evppi)

    result = runner.invoke(
        cli.app,
        [
            "calculate-structural-evppi",
            str(config_file),
            "--structures-of-interest",
            "0",
            "--structures-of-interest",
            "1",
            "--population",
            "2500",
        ],
    )

    assert result.exit_code == 0
    assert "Structural EVPPI: 9.876543" in result.stdout


def test_nma_voi_cli_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the EVPI and EVPPI branches of the NMA command."""
    config_file = tmp_path / "nma.json"
    _write_json(
        config_file,
        {
            "study_data": [{"study_id": 1}],
            "treatment_effects": {"A-B": [0.1, 0.2, 0.3]},
        },
    )

    seen_random_sizes: list[int] = []

    def fake_rand(n_samples: int) -> np.ndarray:
        seen_random_sizes.append(n_samples)
        return np.linspace(0.0, 1.0, n_samples)

    def fake_calculate_nma_evpi(config: dict, **kwargs: object) -> float:
        assert config["study_data"] == [{"study_id": 1}]
        assert kwargs["willingness_to_pay"] == 50000.0
        assert kwargs["population"] == 200
        return 2.5

    def fake_calculate_nma_evppi(
        config: dict,
        parameters_of_interest: list[str],
        parameter_samples: dict,
        **kwargs: object,
    ) -> float:
        assert config["study_data"] == [{"study_id": 1}]
        assert parameters_of_interest == ["hazard_ratio", "cost_offset"]
        assert sorted(parameter_samples) == ["cost_offset", "hazard_ratio"]
        assert all(len(values) == 3 for values in parameter_samples.values())
        assert kwargs["willingness_to_pay"] == 25000.0
        return 1.25

    monkeypatch.setattr(cli.np.random, "rand", fake_rand)
    monkeypatch.setattr(cli, "calculate_nma_evpi", fake_calculate_nma_evpi)
    monkeypatch.setattr(cli, "calculate_nma_evppi", fake_calculate_nma_evppi)

    evpi_result = runner.invoke(
        cli.app,
        [
            "calculate-nma-voi",
            str(config_file),
            "--willingness-to-pay",
            "50000",
            "--population",
            "200",
        ],
    )
    assert evpi_result.exit_code == 0
    assert "NMA-EVPI: 2.500000" in evpi_result.stdout

    evppi_result = runner.invoke(
        cli.app,
        [
            "calculate-nma-voi",
            str(config_file),
            "--willingness-to-pay",
            "25000",
            "--parameters-of-interest",
            "hazard_ratio,cost_offset",
        ],
    )
    assert evppi_result.exit_code == 0
    assert "NMA-EVPPI: 1.250000" in evppi_result.stdout
    assert seen_random_sizes == [3, 3]


def test_nma_voi_cli_validation_errors(tmp_path: Path) -> None:
    runner = CliRunner()

    missing_config = tmp_path / "missing.json"
    missing_result = runner.invoke(cli.app, ["calculate-nma-voi", str(missing_config)])
    assert missing_result.exit_code == 2
    assert "Invalid value for 'CONFIG_FILE'" in missing_result.stderr
    assert missing_config.name in missing_result.stderr
    assert "does not exist." in missing_result.stderr

    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{not json", encoding="utf-8")
    invalid_result = runner.invoke(cli.app, ["calculate-nma-voi", str(invalid_config)])
    assert invalid_result.exit_code == 1
    assert "Error: Invalid JSON in config file" in invalid_result.stderr


def test_nma_voi_cli_internal_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    config_file = tmp_path / "nma.json"
    _write_json(
        config_file,
        {
            "study_data": [{"study_id": 1}],
            "treatment_effects": {"A-B": [0.1, 0.2, 0.3]},
        },
    )

    def boom(*args: object, **kwargs: object) -> float:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "calculate_nma_evpi", boom)

    result = runner.invoke(cli.app, ["calculate-nma-voi", str(config_file)])
    assert result.exit_code == 1
    assert "An error occurred: boom" in result.stderr


if __name__ == "__main__":
    test_evpi_cli()
    test_evppi_cli()
    test_cli_help()
    test_cli_error_handling()
    print("All CLI comprehensive tests passed!")
