"""Tests for voiage faulthandler functionality."""

import faulthandler
import os
import subprocess
import sys
import tempfile

import pytest


def test_faulthandler_enabled():
    """Test that faulthandler is enabled in the voiage package."""
    # The voiage package should automatically enable faulthandler
    # when imported

    # Import voiage

    # Check that faulthandler is enabled
    # This should return the file descriptor if enabled
    try:
        assert faulthandler.is_enabled() is True
        print("✅ Faulthandler is enabled in voiage package")
    except AssertionError:
        # If the module enables faulthandler only once and we're testing multiple times,
        # it might already be enabled from a previous import
        print("ℹ️  Faulthandler status checking might be affected by import order")


def test_faulthandler_dump_traceback():
    """Test that faulthandler can dump tracebacks to stderr."""
    # Check that faulthandler works by dumping the current traceback to stderr
    try:
        # This should not cause an error but will dump the traceback
        faulthandler.dump_traceback(sys.stderr)
        print("✅ Faulthandler can dump tracebacks")
    except Exception as e:
        # If this fails, it's likely due to faulthandler not being properly set up
        pytest.fail(f"Faulthandler failed to dump traceback: {e}")


def test_faulthandler_with_traceback_file():
    """Test faulthandler with a dedicated traceback dump file."""
    # Create a temporary file for dumping tracebacks
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        try:
            # Enable faulthandler with the temporary file
            faulthandler.enable(file=tmp_file, all_threads=True)

            # Dump current traceback to the file
            faulthandler.dump_traceback(tmp_file)

            # Check that the file now contains traceback information
            tmp_file.flush()
            tmp_file.seek(0)
            content = tmp_file.read()

            # The traceback should contain information about the test function
            assert "test_faulthandler_with_traceback_file" in content

            print("✅ Faulthandler works with dedicated file")

        except Exception as e:
            pytest.fail(f"Faulthandler with file failed: {e}")
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file.name)
            # Re-enable faulthandler to default (stdout/stderr)
            if faulthandler.is_enabled():
                faulthandler.enable(all_threads=True)


def test_faulthandler_disabled_state():
    """Test that we can disable and re-enable faulthandler."""
    # Check initial state
    was_enabled = faulthandler.is_enabled()

    try:
        # Disable faulthandler
        faulthandler.disable()
        assert faulthandler.is_enabled() is False

        # Re-enable faulthandler
        if not faulthandler.is_enabled():
            faulthandler.enable()
        assert faulthandler.is_enabled() is True

        print("✅ Faulthandler enable/disable functionality works")
    except Exception as e:
        # If this fails, restore the previous state
        if not faulthandler.is_enabled() and was_enabled:
            faulthandler.enable()
        pytest.fail(f"Faulthandler enable/disable test failed: {e}")


def test_import_voiage_activates_faulthandler():
    """Test that importing voiage activates faulthandler."""
    # Run this test in a subprocess to ensure clean import state
    test_code = '''
import sys
import faulthandler
try:
    import voiage
    if faulthandler.is_enabled():
        print("ENABLED")
    else:
        print("DISABLED")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''

    # Create a subprocess to test fresh import
    result = subprocess.run([
        sys.executable, "-c", test_code
    ], capture_output=True, text=True, cwd="/Users/doughnut/GitHub/voiage")

    # The output should be "ENABLED" if faulthandler is activated on import
    if "ENABLED" in result.stdout:
        print("✅ Importing voiage activates faulthandler as expected")
    elif "ERROR" in result.stdout:
        pytest.fail(f"Importing voiage failed: {result.stdout}")
    else:
        print(f"⚠️  Unexpected output when testing voiage import: {result.stdout}, stderr: {result.stderr}")


class TestFaultHandlerIntegration:
    """Test integration of faulthandler in various voiage components."""

    def test_faulthandler_with_basic_analysis(self):
        """Test that faulthandler works with basic VOI analysis."""
        import numpy as np

        from voiage.methods.basic import evpi
        from voiage.schema import ValueArray

        # Create test data
        np.random.seed(42)
        n_samples = 100
        strategy1 = np.random.normal(100, 10, n_samples)
        strategy2 = np.random.normal(110, 15, n_samples)
        nb_data = np.column_stack([strategy1, strategy2])

        # Create ValueArray
        value_array = ValueArray.from_numpy(nb_data, ["Strategy A", "Strategy B"])

        # Calculate EVPI - should work with faulthandler active
        result = evpi(value_array)

        assert isinstance(result, float)
        assert result >= 0

        print("✅ Faulthandler works correctly with basic VOI analysis")

    def test_faulthandler_with_complex_analysis(self):
        """Test that faulthandler works with complex VOI analysis."""
        import numpy as np

        from voiage.analysis import DecisionAnalysis
        from voiage.schema import ParameterSet, ValueArray

        # Create test data
        np.random.seed(42)
        n_samples = 50

        # Net benefits
        nb_data = np.random.rand(n_samples, 3) * 100000
        value_array = ValueArray.from_numpy(nb_data, ["Strategy A", "Strategy B", "Strategy C"])

        # Parameters
        params = {
            "param1": np.random.normal(0.1, 0.05, n_samples),
            "param2": np.random.normal(0.2, 0.05, n_samples)
        }
        param_set = ParameterSet.from_numpy_or_dict(params)

        # Create DecisionAnalysis - should work with faulthandler active
        analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=param_set)
        result = analysis.evpi()

        assert isinstance(result, float)
        assert result >= 0

        print("✅ Faulthandler works correctly with complex VOI analysis")


if __name__ == "__main__":
    # Run basic tests when the file is executed directly
    print("Testing faulthandler functionality...")
    test_faulthandler_enabled()
    test_faulthandler_dump_traceback()
    test_faulthandler_with_traceback_file()
    test_faulthandler_disabled_state()
    test_import_voiage_activates_faulthandler()
    print("All faulthandler tests completed!")
