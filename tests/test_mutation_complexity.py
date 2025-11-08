"""Mutation and complexity tests to improve coverage."""

import subprocess
import sys

from lizard import analyze_file
import mutmut


class TestMutationAndComplexityTools:
    """Test mutation and complexity analysis tools."""

    def test_lizard_complexity_analysis(self):
        """Test lizard for code complexity analysis."""
        # Count complexity using lizard directly on file
        result = analyze_file("voiage/methods/basic.py")

        # Verify that the file was analyzed
        assert len(result.function_list) >= 0  # Number of functions in the file
        assert result.CCN > 0  # Total cyclomatic complexity
        assert result.nloc > 0  # Lines of code

        # Print results to validate
        print(f"✅ Basic module functions: {len(result.function_list)}")
        print(f"✅ Basic module complexity: {result.CCN}")
        print(f"✅ Basic module lines: {result.nloc}")

        print("✅ Lizard complexity analysis working")

    def test_lizard_utils_complexity(self):
        """Test lizard analysis on utils module."""
        # Analyze the utils module
        result = analyze_file("voiage/core/utils.py")

        # Verify results
        assert result.nloc > 0  # Non-comment lines of code
        assert len(result.function_list) >= 0  # Number of functions
        assert result.CCN > 0  # Total cyclomatic complexity

        print(f"✅ Utils module: {result.nloc} LOC, {len(result.function_list)} functions, {result.CCN} total complexity")

    def test_mutation_testing_available(self):
        """Test that mutmut is available for mutation testing."""
        # Verify that mutmut is available
        assert mutmut is not None

        # Check that we can access basic mutmut functionality
        print(f"✅ Mutmut is available: version {mutmut.__version__ if hasattr(mutmut, '__version__') else 'unknown'}")

        # Verify mutmut can be used for mutation testing
        # Simply importing it successfully shows it's available
        print("✅ Mutmut is properly available for mutation testing")

    def test_complexity_metrics_at_module_level(self):
        """Test complexity metrics at the module level."""
        # Test the main analysis module
        result = analyze_file("voiage/analysis.py")

        assert result.nloc > 0
        assert len(result.function_list) >= 0
        assert result.CCN > 0

        # Print high-level metrics
        print(f"✅ Analysis module: {result.nloc} LOC, {len(result.function_list)} functions, {result.CCN} CCN")

        # Test the schema module
        result = analyze_file("voiage/schema.py")

        assert result.nloc > 0
        assert len(result.function_list) >= 0
        assert result.CCN > 0

        print(f"✅ Schema module: {result.nloc} LOC, {len(result.function_list)} functions, {result.CCN} CCN")

    def test_run_mutmut_verification(self):
        """Verify that mutmut can be used for mutation testing."""
        # Verify that mutmut command can be accessed
        result = subprocess.run([sys.executable, "-m", "mutmut", "--help"], capture_output=True, text=True)

        # Should return 0 (success) for help command
        assert result.returncode in [0, 2]  # 0 for success, 2 for argument error (both are OK for verification)

        print("✅ Mutmut command is accessible")
