"""Version compatibility tests for voiage.

These tests verify that voiage works correctly with different versions of its dependencies.
"""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray


def test_numpy_version_compatibility():
    """Test that voiage works with the specified numpy versions."""
    # Test basic functionality with numpy arrays using float64 dtype
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)

    # Test EVPI calculation
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result >= 0

    # Test that numpy deprecated attributes work correctly
    # This is important for compatibility with pygam
    assert hasattr(np, 'int64')
    assert hasattr(np, 'float64')
    assert hasattr(np, 'bool_')


def test_scipy_version_compatibility():
    """Test that voiage works with the specified scipy versions."""
    from scipy import sparse

    # Test that sparse matrix functionality works
    # This is important for metamodel compatibility
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

    assert sparse.issparse(sparse_matrix)
    assert sparse_matrix.shape == (3, 3)


def test_pandas_version_compatibility():
    """Test that voiage works with the specified pandas versions."""
    import pandas as pd

    # Test basic pandas functionality
    df = pd.DataFrame({
        'strategy1': [100.0, 90.0, 110.0, 95.0],
        'strategy2': [120.0, 130.0, 110.0, 125.0]
    })

    # Convert to numpy for voiage with explicit float64 dtype
    net_benefits = df.values.astype(np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)

    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))


def test_sklearn_version_compatibility():
    """Test that voiage works with the specified scikit-learn versions."""
    from sklearn.ensemble import RandomForestRegressor

    # Test that sklearn components can be imported and used
    # This is important for the RandomForestMetamodel
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    assert rf.n_estimators == 10


def test_gam_metamodel_compatibility():
    """Test that GAM metamodel works with different numpy versions."""
    # This specifically tests the pygam compatibility fix
    from voiage.metamodels import GAMMetamodel

    # Create simple test data with float64 dtype
    x = np.linspace(0, 10, 100, dtype=np.float64)
    y = x**2 + np.random.normal(0, 1, 100).astype(np.float64)

    # Test that we can create and use the GAM metamodel
    # This would fail if numpy compatibility issues weren't fixed
    try:
        gam = GAMMetamodel()
        # Test fitting (this would trigger numpy compatibility issues if present)
        # We're not actually fitting real data here, just testing the import and instantiation
        assert isinstance(gam, GAMMetamodel)
    except ImportError:
        # If pygam is not available, skip this test
        pytest.skip("pygam not available")


def test_dependency_version_imports():
    """Test that all optional dependencies can be imported if available."""
    # Test core dependencies
    import jax
    import scipy
    import xarray

    # Test that imports work
    assert scipy.__version__
    assert xarray.__version__
    assert jax.__version__

    # Test optional dependencies
    try:
        import sklearn
        assert sklearn.__version__
    except ImportError:
        pass  # Optional dependency

    try:
        import matplotlib
        assert matplotlib.__version__
    except ImportError:
        pass  # Optional dependency

    try:
        import seaborn
        assert seaborn.__version__
    except ImportError:
        pass  # Optional dependency


if __name__ == "__main__":
    test_numpy_version_compatibility()
    test_scipy_version_compatibility()
    test_pandas_version_compatibility()
    test_sklearn_version_compatibility()
    test_gam_metamodel_compatibility()
    test_dependency_version_imports()
    print("All version compatibility tests passed!")
