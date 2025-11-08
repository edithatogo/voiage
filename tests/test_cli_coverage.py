"""Comprehensive CLI coverage tests to achieve 95%+ coverage."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import typer

from voiage.cli import calculate_evpi, calculate_evppi
from voiage.schema import ParameterSet, ValueArray


class TestCLIEvpiCoverage:
    """Test EVPI CLI function coverage."""

    def test_calculate_evpi_success_basic(self):
        """Test successful EVPI calculation with basic parameters."""
        # Create mock ValueArray
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 123.456

            # Call the function
            with patch('builtins.open', mock_open()):
                calculate_evpi(
                    net_benefit_file=Path("test.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            # Verify calls
            mock_read.assert_called_once_with("test.csv", skip_header=True)
            mock_evpi.assert_called_once_with(
                mock_nba,
                population=None,
                discount_rate=None,
                time_horizon=None
            )
            mock_echo.assert_called_once_with("EVPI: 123.456000")

    def test_calculate_evpi_success_with_scaling(self):
        """Test successful EVPI calculation with population scaling."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 123.456

            calculate_evpi(
                net_benefit_file=Path("test.csv"),
                population=100000.0,
                discount_rate=0.03,
                time_horizon=10.0,
                output_file=None
            )

            mock_evpi.assert_called_once_with(
                mock_nba,
                population=100000.0,
                discount_rate=0.03,
                time_horizon=10.0
            )
            mock_echo.assert_called_once_with("EVPI: 123.456000")

    def test_calculate_evpi_success_with_output_file(self):
        """Test successful EVPI calculation with output file."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo, \
             patch('builtins.open', mock_open()) as mock_file:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 123.456

            output_path = Path("output.txt")
            calculate_evpi(
                net_benefit_file=Path("test.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=output_path
            )

            # Verify file writing
            mock_file.assert_called_once_with(output_path, 'w')
            handle = mock_file.return_value.__enter__.return_value
            handle.write.assert_called_once_with("EVPI: 123.456000\n")

            # Verify echoes
            assert mock_echo.call_count == 2
            mock_echo.assert_any_call("EVPI: 123.456000")
            mock_echo.assert_any_call(f"Result saved to {output_path}")

    def test_calculate_evpi_file_not_found_error(self):
        """Test FileNotFoundError handling."""
        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.typer.echo') as mock_echo:

            # Simulate FileNotFoundError
            mock_read.side_effect = FileNotFoundError("File not found")

            # Test that the function raises typer.Exit
            with pytest.raises(typer.Exit) as exc_info:
                calculate_evpi(
                    net_benefit_file=Path("nonexistent.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            # Check exit code
            assert exc_info.value.exit_code == 1

            # Verify error message
            mock_echo.assert_called_once_with(
                "Error: Net benefit file not found at 'nonexistent.csv'",
                err=True
            )

    def test_calculate_evpi_general_exception(self):
        """Test general exception handling."""
        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.typer.echo') as mock_echo:

            # Simulate general exception
            mock_read.side_effect = ValueError("Invalid data format")

            # Test that the function raises typer.Exit
            with pytest.raises(typer.Exit) as exc_info:
                calculate_evpi(
                    net_benefit_file=Path("invalid.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            # Check exit code
            assert exc_info.value.exit_code == 1

            # Verify error message
            mock_echo.assert_called_once_with(
                "An error occurred: Invalid data format",
                err=True
            )

    def test_calculate_evpi_with_read_exception(self):
        """Test exception during read_value_array_csv."""
        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.typer.echo') as mock_echo:

            # Simulate various exceptions during reading
            for exc in [RuntimeError("Memory error"), KeyError("Missing column")]:
                mock_read.side_effect = exc

                with pytest.raises(typer.Exit) as exc_info:
                    calculate_evpi(
                        net_benefit_file=Path("test.csv"),
                        population=None,
                        discount_rate=None,
                        time_horizon=None,
                        output_file=None
                    )

                assert exc_info.value.exit_code == 1
                mock_echo.assert_called_with(
                    f"An error occurred: {exc!s}",
                    err=True
                )

    def test_calculate_evpi_with_evpi_exception(self):
        """Test exception during EVPI calculation."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.side_effect = np.linalg.LinAlgError("Matrix is singular")

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evpi(
                    net_benefit_file=Path("test.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            assert exc_info.value.exit_code == 1
            mock_echo.assert_called_once_with(
                "An error occurred: Matrix is singular",
                err=True
            )

    def test_calculate_evpi_output_file_write_error(self):
        """Test error during output file writing."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo, \
             patch('builtins.open', side_effect=PermissionError("Cannot write to file")) as mock_file:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 123.456

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evpi(
                    net_benefit_file=Path("test.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=Path("readonly/output.txt")
                )

            assert exc_info.value.exit_code == 1
            # Should not reach the file write success path
            mock_echo.assert_called_once_with(
                "An error occurred: Cannot write to file",
                err=True
            )


class TestCLIEvppiCoverage:
    """Test EVPPI CLI function coverage."""

    def test_calculate_evppi_success_basic(self):
        """Test successful EVPPI calculation with basic parameters."""
        mock_nba = MagicMock()
        mock_param_set = MagicMock()
        mock_param_set.parameter_names = ["param1", "param2"]

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evppi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evppi.return_value = 456.789

            calculate_evppi(
                net_benefit_file=Path("test.csv"),
                parameter_file=Path("params.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=None
            )

            mock_read_nb.assert_called_once_with("test.csv", skip_header=True)
            mock_read_param.assert_called_once_with("params.csv", skip_header=True)
            mock_evppi.assert_called_once_with(
                nb_array=mock_nba,
                parameter_samples=mock_param_set,
                parameters_of_interest=["param1", "param2"],
                population=None,
                discount_rate=None,
                time_horizon=None
            )
            mock_echo.assert_called_once_with("EVPPI: 456.789000")

    def test_calculate_evppi_success_with_scaling(self):
        """Test successful EVPPI calculation with population scaling."""
        mock_nba = MagicMock()
        mock_param_set = MagicMock()
        mock_param_set.parameter_names = ["param1"]

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evpi.return_value = 456.789

            calculate_evppi(
                net_benefit_file=Path("test.csv"),
                parameter_file=Path("params.csv"),
                population=50000.0,
                discount_rate=0.025,
                time_horizon=15.0,
                output_file=None
            )

            mock_evpi.assert_called_once_with(
                nb_array=mock_nba,
                parameter_samples=mock_param_set,
                parameters_of_interest=["param1"],
                population=50000.0,
                discount_rate=0.025,
                time_horizon=15.0
            )

    def test_calculate_evppi_success_with_output_file(self):
        """Test successful EVPPI calculation with output file."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_param_set = MagicMock(spec=ParameterSet)
        mock_param_set.parameter_names = ["param1"]

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo, \
             patch('builtins.open', mock_open()) as mock_file:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evpi.return_value = 456.789

            output_path = Path("evppi_output.txt")
            calculate_evppi(
                net_benefit_file=Path("test.csv"),
                parameter_file=Path("params.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=output_path
            )

            # Verify file writing
            mock_file.assert_called_once_with(output_path, 'w')
            handle = mock_file.return_value.__enter__.return_value
            handle.write.assert_called_once_with("EVPPI: 456.789000\n")

            # Verify echoes
            assert mock_echo.call_count == 2
            mock_echo.assert_any_call("EVPPI: 456.789000")
            mock_echo.assert_any_call(f"Result saved to {output_path}")

    def test_calculate_evppi_file_not_found_net_benefit(self):
        """Test FileNotFoundError for net benefit file."""
        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.side_effect = FileNotFoundError("Net benefit file not found")

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evppi(
                    net_benefit_file=Path("nonexistent.csv"),
                    parameter_file=Path("params.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            assert exc_info.value.exit_code == 1
            mock_echo.assert_called_once_with(
                "Error: File not found - Net benefit file not found",
                err=True
            )

    def test_calculate_evppi_file_not_found_parameter(self):
        """Test FileNotFoundError for parameter file."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.return_value = mock_nba
            mock_read_param.side_effect = FileNotFoundError("Parameter file not found")

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evppi(
                    net_benefit_file=Path("test.csv"),
                    parameter_file=Path("nonexistent.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            assert exc_info.value.exit_code == 1
            mock_echo.assert_called_once_with(
                "Error: File not found - Parameter file not found",
                err=True
            )

    def test_calculate_evppi_general_exception(self):
        """Test general exception handling."""
        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.typer.echo') as mock_echo:

            # Simulate general exception
            mock_read_nb.side_effect = RuntimeError("Memory allocation failed")

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evppi(
                    net_benefit_file=Path("invalid.csv"),
                    parameter_file=Path("params.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            assert exc_info.value.exit_code == 1
            mock_echo.assert_called_once_with(
                "An error occurred: Memory allocation failed",
                err=True
            )

    def test_calculate_evppi_with_read_parameter_exception(self):
        """Test exception during parameter set reading."""
        mock_nba = MagicMock()

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.return_value = mock_nba
            mock_read_param.side_effect = ValueError("Invalid parameter format")

            with pytest.raises(typer.Exit) as exc_info:
                calculate_evppi(
                    net_benefit_file=Path("test.csv"),
                    parameter_file=Path("invalid.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            assert exc_info.value.exit_code == 1
            mock_echo.assert_called_once_with(
                "An error occurred: Invalid parameter format",
                err=True
            )

    def test_calculate_evppi_with_evppi_exception(self):
        """Test exception during EVPPI calculation."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_param_set = MagicMock(spec=ParameterSet)
        mock_param_set.parameter_names = ["param1"]

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evppi, \
             patch('voiage.cli.typer.echo') as mock_echo, \
             patch('voiage.cli.typer.Exit') as mock_exit:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evppi.side_effect = Exception("Calculation failed")

            with pytest.raises(typer.Exit):
                calculate_evppi(
                    net_benefit_file=Path("test.csv"),
                    parameter_file=Path("params.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

            mock_echo.assert_called_once_with(
                "An error occurred: Calculation failed",
                err=True
            )

    def test_calculate_evppi_output_file_write_error(self):
        """Test error during output file writing."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_param_set = MagicMock(spec=ParameterSet)
        mock_param_set.parameter_names = ["param1"]

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo, \
             patch('builtins.open', side_effect=OSError("Disk full")) as mock_file, \
             patch('voiage.cli.typer.Exit') as mock_exit:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evpi.return_value = 456.789

            with pytest.raises(typer.Exit):
                calculate_evppi(
                    net_benefit_file=Path("test.csv"),
                    parameter_file=Path("params.csv"),
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=Path("full_disk/output.txt")
                )

            mock_echo.assert_called_once_with(
                "An error occurred: Disk full",
                err=True
            )


class TestCLIEdgeCases:
    """Test CLI edge cases and boundary conditions."""

    def test_calculate_evpi_empty_parameter_values(self):
        """Test EVPI with None values for optional parameters."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_nba.__float__ = MagicMock(return_value=0.0)

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 0.0

            calculate_evpi(
                net_benefit_file=Path("test.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=None
            )

            # Verify all None values are passed correctly
            mock_evpi.assert_called_once_with(
                mock_nba,
                population=None,
                discount_rate=None,
                time_horizon=None
            )

    def test_calculate_evpi_zero_result(self):
        """Test EVPI with zero result."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_nba.__float__ = MagicMock(return_value=0.0)

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = 0.0

            calculate_evpi(
                net_benefit_file=Path("test.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=None
            )

            mock_echo.assert_called_once_with("EVPI: 0.000000")

    def test_calculate_evppi_empty_parameter_names(self):
        """Test EVPPI with empty parameter names."""
        mock_nba = MagicMock(spec=ValueArray)
        mock_param_set = MagicMock(spec=ParameterSet)
        mock_param_set.parameter_names = []

        with patch('voiage.cli.read_value_array_csv') as mock_read_nb, \
             patch('voiage.cli.read_parameter_set_csv') as mock_read_param, \
             patch('voiage.cli.evppi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read_nb.return_value = mock_nba
            mock_read_param.return_value = mock_param_set
            mock_evpi.return_value = 0.0

            calculate_evppi(
                net_benefit_file=Path("test.csv"),
                parameter_file=Path("params.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=None
            )

            mock_evpi.assert_called_once_with(
                nb_array=mock_nba,
                parameter_samples=mock_param_set,
                parameters_of_interest=[],
                population=None,
                discount_rate=None,
                time_horizon=None
            )

    def test_calculate_evpi_very_large_numbers(self):
        """Test EVPI with very large numbers."""
        mock_nba = MagicMock(spec=ValueArray)
        large_value = 1e15
        mock_nba.__float__ = MagicMock(return_value=large_value)

        with patch('voiage.cli.read_value_array_csv') as mock_read, \
             patch('voiage.cli.evpi') as mock_evpi, \
             patch('voiage.cli.typer.echo') as mock_echo:

            mock_read.return_value = mock_nba
            mock_evpi.return_value = large_value

            calculate_evpi(
                net_benefit_file=Path("test.csv"),
                population=None,
                discount_rate=None,
                time_horizon=None,
                output_file=None
            )

            mock_echo.assert_called_once_with(f"EVPI: {large_value:.6f}")

    def test_path_handling(self):
        """Test proper Path object handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test with Path objects
            mock_nba = MagicMock(spec=ValueArray)
            mock_nba.__float__ = MagicMock(return_value=123.456)

            with patch('voiage.cli.read_value_array_csv') as mock_read, \
                 patch('voiage.cli.evpi') as mock_evpi, \
                 patch('voiage.cli.typer.echo') as mock_echo:

                mock_read.return_value = mock_nba
                mock_evpi.return_value = 123.456

                # Test with Path object
                test_path = tmppath / "test.csv"
                calculate_evpi(
                    net_benefit_file=test_path,
                    population=None,
                    discount_rate=None,
                    time_horizon=None,
                    output_file=None
                )

                # Verify path is converted to string correctly
                mock_read.assert_called_once_with(str(test_path), skip_header=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
