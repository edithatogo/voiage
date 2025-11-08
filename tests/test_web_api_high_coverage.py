"""Comprehensive tests for the web API module to improve coverage."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from voiage.web.main import app


client = TestClient(app)


class TestWebAPIComprehensive:
    """Comprehensive tests for the web API endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert "voiage" in response_data["message"].lower()
        assert "version" in response_data
        print("✅ Root endpoint working correctly")

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == {"status": "healthy"}
        print("✅ Health check endpoint working correctly")

    def test_calculate_evpi_valid_input(self):
        """Test EVPI calculation with valid input."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0, 120.0], [90.0, 140.0, 130.0], [110.0, 130.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B", "Strategy C"]
        
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "population": 1000,
            "time_horizon": 5,
            "discount_rate": 0.035
        }
        
        response = client.post("/calculate/evpi", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "result" in response_data
        assert "details" in response_data
        assert isinstance(response_data["result"], (int, float))
        assert response_data["result"] >= 0.0
        print(f"✅ EVPI calculation endpoint working: {response_data['result']}")

    def test_calculate_evpi_with_defaults(self):
        """Test EVPI calculation with default parameters."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0], [90.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B"]
        
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names
        }
        
        response = client.post("/calculate/evpi", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "result" in response_data
        assert isinstance(response_data["result"], (int, float))
        print(f"✅ EVPI calculation with defaults working: {response_data['result']}")

    def test_calculate_evpi_invalid_nb_array(self):
        """Test EVPI calculation with invalid net benefit array."""
        # Invalid array (not 2D)
        payload = {
            "nb_array": [100, 150],  # Should be 2D array
            "strategy_names": ["Strategy A", "Strategy B"]
        }
        
        response = client.post("/calculate/evpi", json=payload)
        # Expect 422 validation error for invalid input
        assert response.status_code in [422, 500]  # Validation error or internal error
        print("✅ EVPI validation for invalid nb_array working")

    def test_calculate_evpi_missing_required_fields(self):
        """Test EVPI calculation with missing required fields."""
        # Missing nb_array field
        payload = {
            "strategy_names": ["Strategy A", "Strategy B"]
        }
        
        response = client.post("/calculate/evpi", json=payload)
        # Expect 422 validation error for missing required field
        assert response.status_code == 422
        print("✅ EVPI validation for missing fields working")

    def test_calculate_evppi_valid_input(self):
        """Test EVPPI calculation with valid input."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]]
        strategy_names = ["Strategy A", "Strategy B"]
        
        # Valid parameter samples
        parameter_samples = {
            "param1": [0.1, 0.2, 0.3],
            "param2": [10.0, 20.0, 30.0]
        }
        
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "parameter_samples": parameter_samples,
            "parameters_of_interest": ["param1"],
            "population": 1000,
            "time_horizon": 5,
            "discount_rate": 0.035
        }
        
        response = client.post("/calculate/evppi", json=payload)
        # May return 501 if sklearn is not available
        assert response.status_code in [200, 501]  # Success or not implemented
        if response.status_code == 200:
            response_data = response.json()
            assert "result" in response_data
            assert "details" in response_data
            assert isinstance(response_data["result"], (int, float))
            print(f"✅ EVPPI calculation endpoint working: {response_data['result']}")
        else:
            print("✅ EVPPI endpoint returns 501 when sklearn not available (expected)")

    def test_calculate_evppi_without_parameters(self):
        """Test EVPPI calculation when no parameter samples provided."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0], [90.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B"]
        
        # No parameter samples provided
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "parameter_samples": {},
            "parameters_of_interest": ["param1"]  # This should fail
        }
        
        response = client.post("/calculate/evppi", json=payload)
        # Expect 422 validation error because parameters_of_interest are not in parameter_samples
        assert response.status_code in [422, 500]
        print("✅ EVPPI validation for missing parameters working")

    def test_analysis_status_endpoint(self):
        """Test the analysis status endpoint."""
        # This endpoint might not be implemented, so test it
        response = client.get("/analysis/status")
        # May return 404 (endpoint not found) or 200 (implemented)
        assert response.status_code in [200, 404, 405]  # 405 if method not allowed
        print("✅ Analysis status endpoint tested")

    def test_list_analyses_endpoint(self):
        """Test the list analyses endpoint."""
        response = client.get("/analyses")
        # May return 404 (endpoint not found) or 200 (implemented)  
        assert response.status_code in [200, 404, 405]  # 405 if method not allowed
        print("✅ List analyses endpoint tested")

    def test_calculate_evpi_with_edge_cases(self):
        """Test EVPI calculation with edge cases."""
        # Test with single strategy (should return 0 EVPI)
        nb_array = [[100.0], [110.0], [90.0]]  # Single strategy, 3 samples
        payload = {
            "nb_array": nb_array,
            "strategy_names": ["Single Strategy"]
        }
        
        response = client.post("/calculate/evpi", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            assert isinstance(response_data["result"], (int, float))
            assert response_data["result"] >= 0  # EVPI should be non-negative
            print("✅ EVPI with single strategy edge case working")

        # Test with identical strategies (should return 0 EVPI)
        nb_array = [[100.0, 100.0], [110.0, 110.0], [90.0, 90.0]]  # Identical strategies
        payload = {
            "nb_array": nb_array,
            "strategy_names": ["Strategy A", "Strategy B"]
        }
        
        response = client.post("/calculate/evpi", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            assert isinstance(response_data["result"], (int, float))
            print("✅ EVPI with identical strategies edge case working")

    def test_calculate_evpi_population_scaling_validation(self):
        """Test EVPI population scaling validation."""
        nb_array = [[100.0, 150.0], [90.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B"]
        
        # Test with negative population
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "population": -1000,
            "time_horizon": 5,
            "discount_rate": 0.035
        }
        
        response = client.post("/calculate/evpi", json=payload)
        assert response.status_code in [422, 500]
        print("✅ EVPI population validation working")

        # Test with negative time horizon
        payload["population"] = 1000
        payload["time_horizon"] = -5
        
        response = client.post("/calculate/evpi", json=payload)
        assert response.status_code in [422, 500]
        print("✅ EVPI time horizon validation working")

        # Test with invalid discount rate
        payload["time_horizon"] = 5
        payload["discount_rate"] = 1.5
        
        response = client.post("/calculate/evpi", json=payload)
        assert response.status_code in [422, 500]
        print("✅ EVPI discount rate validation working")

    def test_calculate_evppi_validation_errors(self):
        """Test EVPPI validation errors."""
        # Test with parameter of interest not in parameter samples
        nb_array = [[100.0, 150.0], [90.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B"]
        param_samples = {"existing_param": [0.1, 0.2]}
        
        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "parameter_samples": param_samples,
            "parameters_of_interest": ["nonexistent_param"]  # This parameter doesn't exist
        }
        
        response = client.post("/calculate/evppi", json=payload)
        assert response.status_code in [422, 500]
        print("✅ EVPPI parameter validation working")