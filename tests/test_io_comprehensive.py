"""Comprehensive tests for io module to improve coverage."""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import mock_open, patch

from voiage.core.io import (
    read_value_array_csv,
    write_value_array_csv,
    read_parameter_set_csv,
    write_parameter_set_csv,
    FileFormatError
)
from voiage.exceptions import InputError
from voiage.schema import ValueArray, ParameterSet


class TestIOFunctionsComplete:
    """Comprehensive tests for the IO module functions."""

    def test_read_value_array_csv_valid_file(self):
        """Test reading ValueArray from valid CSV file."""
        # Create a temporary CSV file without headers (just data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("100.0,150.0,120.0\n90.0,140.0,130.0\n110.0,130.0,140.0")
            temp_path = f.name

        try:
            # Test reading the CSV file as ValueArray
            result = read_value_array_csv(temp_path)
            
            # Verify result structure
            assert isinstance(result, ValueArray)
            assert result.values.shape == (3, 3)  # 3 rows, 3 columns
            expected_values = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0], [110.0, 130.0, 140.0]])
            assert np.allclose(result.values, expected_values)
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_value_array_csv_with_option_names(self):
        """Test reading ValueArray with custom option names."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("100.0,150.0\n90.0,140.0\n110.0,130.0")
            temp_path = f.name

        try:
            # Test reading CSV with custom option names
            result = read_value_array_csv(
                temp_path,
                option_names=["Treatment A", "Treatment B"]
            )
            
            assert isinstance(result, ValueArray)
            assert result.values.shape == (3, 2)  # 3 rows, 2 columns
            assert isinstance(result.values, np.ndarray)  # values should be a numpy array
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_value_array_csv_with_wrong_option_names_count(self):
        """Test reading ValueArray with wrong option names count."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("100.0,150.0,120.0\n90.0,140.0,130.0")  # 3 columns
            temp_path = f.name

        try:
            # This should raise FileFormatError because we have 3 columns but only 2 option names
            with pytest.raises(FileFormatError):
                read_value_array_csv(temp_path, option_names=["Option A", "Option B"])  # Only 2 names
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_value_array_csv_with_skip_header(self):
        """Test reading ValueArray with skip_header=True."""
        # Create a temporary CSV file with headers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Col1,Col2,Col3\n100.0,150.0,120.0\n90.0,140.0,130.0")
            temp_path = f.name

        try:
            # Test reading CSV with skip_header=True
            result = read_value_array_csv(
                temp_path, 
                skip_header=True,
                option_names=["Strategy A", "Strategy B", "Strategy C"]
            )
            
            assert isinstance(result, ValueArray)
            assert result.values.shape == (2, 3)  # 2 rows (header skipped), 3 columns
            # Check that the data is as expected (starting from the second row)
            expected_values = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]])
            assert np.allclose(result.values, expected_values)
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_value_array_csv_nonexistent_file(self):
        """Test reading ValueArray from nonexistent file."""
        with pytest.raises(FileFormatError):
            read_value_array_csv("nonexistent_file.csv")

    def test_write_value_array_csv_valid(self):
        """Test writing ValueArray to CSV file."""
        # Create test data
        values = np.array([[100.0, 150.0, 120.0], [90.0, 140.0, 130.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, ['Strategy A', 'Strategy B', 'Strategy C'])
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write ValueArray to CSV
            write_value_array_csv(value_array, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read it back and verify content
            with open(temp_path, 'r') as f:
                content = f.read()
                # Verify the values are written to the file
                assert '100.0' in content
                assert '150.0' in content
                assert '120.0' in content
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_write_value_array_csv_with_custom_delimiter(self):
        """Test writing ValueArray to CSV with custom delimiter."""
        # Create test ValueArray
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, ['Strategy A', 'Strategy B'])
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write ValueArray to CSV with semicolon delimiter
            write_value_array_csv(value_array, temp_path, delimiter=';', write_header=True)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read it back and verify semicolon delimiter was used
            with open(temp_path, 'r') as f:
                content = f.read()
                assert ';' in content  # Contains semicolon delimiter
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_write_value_array_csv_without_header(self):
        """Test writing ValueArray to CSV without header."""
        # Create test ValueArray
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, ['Strategy A', 'Strategy B'])
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write ValueArray to CSV without header
            write_value_array_csv(value_array, temp_path, write_header=False)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read it back and verify no header was written
            with open(temp_path, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                # The first line should be data, not header
                first_line_parts = lines[0].split(',')
                # Should be numeric values, not string headers
                assert all(part.replace('.', '', 1).replace('-', '', 1).isdigit() for part in first_line_parts if part.strip())
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_write_value_array_csv_io_error(self):
        """Test writing ValueArray to CSV with IO error."""
        # Create test ValueArray
        values = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(values, ['Strategy A', 'Strategy B'])
        
        # Try to write to a path where we don't have permissions (or invalid path)
        # We will mock the open function to simulate an IO error
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                write_value_array_csv(value_array, "/invalid/path/file.csv")

    def test_read_parameter_set_csv_valid_file(self):
        """Test reading ParameterSet from valid CSV file."""
        # Create a temporary CSV file without headers (just data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.1,10.0,100.0\n0.2,20.0,200.0\n0.3,30.0,300.0")
            temp_path = f.name

        try:
            # Test reading the CSV file as ParameterSet
            result = read_parameter_set_csv(temp_path)
            
            # Verify result structure
            assert isinstance(result, ParameterSet)
            assert len(result.parameter_names) == 3
            assert result.n_samples == 3
            # Verify that parameter names are auto-generated
            assert all(name.startswith("param_") for name in result.parameter_names)
            # Check that the parameter values are correct
            assert result.parameters[result.parameter_names[0]][0] == pytest.approx(0.1)
            assert result.parameters[result.parameter_names[1]][1] == pytest.approx(20.0)
            assert result.parameters[result.parameter_names[2]][2] == pytest.approx(300.0)
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_parameter_set_csv_with_custom_params(self):
        """Test reading ParameterSet with custom parameters."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.1,10.0\n0.2,20.0\n0.3,30.0")  # 2 columns, 3 rows
            temp_path = f.name

        try:
            # Test reading CSV with custom parameter names and no header
            # Since we have 3 rows of data and skip_header=True, only 2 rows will be read
            result = read_parameter_set_csv(
                temp_path,
                parameter_names=["CustomParam1", "CustomParam2"],
                skip_header=False  # Don't skip header since there's no real header row
            )
            
            assert isinstance(result, ParameterSet)
            assert result.parameter_names == ["CustomParam1", "CustomParam2"]
            assert result.n_samples == 3  # 3 rows of data
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_parameter_set_csv_with_wrong_param_names_count(self):
        """Test reading ParameterSet with wrong parameter names count."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("0.1,10.0,100.0\n0.2,20.0,200.0")  # 3 columns
            temp_path = f.name

        try:
            # This should raise FileFormatError because we have 3 columns but only 2 parameter names
            with pytest.raises(FileFormatError):
                read_parameter_set_csv(temp_path, parameter_names=["Param A", "Param B"])  # Only 2 names
        finally:
            # Clean up
            os.remove(temp_path)

    def test_read_parameter_set_csv_nonexistent_file(self):
        """Test reading ParameterSet from nonexistent file."""
        with pytest.raises(FileFormatError):
            read_parameter_set_csv("nonexistent_file.csv")

    def test_write_parameter_set_csv_valid(self):
        """Test writing ParameterSet to CSV file."""
        # Create test ParameterSet
        params = {'param1': np.array([0.1, 0.2, 0.3]), 'param2': np.array([10.0, 20.0, 30.0])}
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write ParameterSet to CSV
            write_parameter_set_csv(param_set, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read it back and verify content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert '0.1' in content
                assert '10.0' in content
                assert '0.2' in content
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_write_parameter_set_csv_without_header(self):
        """Test writing ParameterSet to CSV without header."""
        # Create test ParameterSet
        params = {'param1': np.array([0.1, 0.2]), 'param2': np.array([10.0, 20.0])}
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            # Write ParameterSet to CSV without header
            write_parameter_set_csv(param_set, temp_path, write_header=False)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read it back and verify no header was written
            with open(temp_path, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                # The first line should be data, not header
                first_line_parts = lines[0].split(',')
                # Should be numeric values, not string headers
                assert all(part.replace('.', '', 1).replace('-', '', 1).isdigit() for part in first_line_parts if part.strip())
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_io_functions_with_invalid_paths(self):
        """Test IO functions with invalid paths."""
        # Test with non-string path (should raise error in the underlying functions)
        with patch('builtins.open', side_effect=TypeError("Expected string, got int")):
            with pytest.raises(Exception):  # Could be raised from open() or other places
                read_value_array_csv(123)  # Invalid path type

    def test_write_parameter_set_csv_io_error(self):
        """Test writing ParameterSet to CSV with IO error."""
        # Create test ParameterSet
        params = {'param1': np.array([0.1, 0.2]), 'param2': np.array([10.0, 20.0])}
        param_set = ParameterSet.from_numpy_or_dict(params)
        
        # Mock open to simulate IO error
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                write_parameter_set_csv(param_set, "/invalid/path/file.csv")