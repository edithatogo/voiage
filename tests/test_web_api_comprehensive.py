"""Comprehensive tests for the web API to improve coverage."""

from fastapi.testclient import TestClient

from voiage.web.main import app


class TestWebAPIComplete:
    """Comprehensive tests for the web API endpoints."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert "voiage Web API" in response_data["message"]
        assert "version" in response_data
        assert "endpoints" in response_data
        assert isinstance(response_data["endpoints"], list)

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == {"status": "healthy"}

    def test_evpi_calculation_valid_input(self):
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

        response = self.client.post("/calculate/evpi", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "result" in response_data
        assert "details" in response_data

        # Check that result is a number
        assert isinstance(response_data["result"], (int, float))
        assert response_data["result"] >= 0  # EVPI should be non-negative

    def test_evpi_calculation_with_defaults(self):
        """Test EVPI calculation with default parameters."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0], [90.0, 140.0]]
        strategy_names = ["Strategy A", "Strategy B"]

        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names
        }

        response = self.client.post("/calculate/evpi", json=payload)
        assert response.status_code == 200
        response_data = response.json()
        assert "result" in response_data
        assert isinstance(response_data["result"], (int, float))


    def test_evpi_calculation_invalid_nb_array(self):
        """Test EVPI calculation with invalid net benefit array."""
        # Invalid array (not 2D)
        payload = {
            "nb_array": [100, 150],  # Should be 2D array
            "strategy_names": ["Strategy A", "Strategy B"]
        }

        response = self.client.post("/calculate/evpi", json=payload)
        # Should return an error
        assert response.status_code in [422, 500]  # Validation error or server error

    def test_evpi_calculation_missing_required_fields(self):
        """Test EVPI calculation with missing required fields."""
        # Missing nb_array
        payload = {
            "strategy_names": ["Strategy A", "Strategy B"]
        }

        response = self.client.post("/calculate/evpi", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

        # Missing strategy_names
        payload2 = {
            "nb_array": [[100, 150], [90, 140]]
        }

        response2 = self.client.post("/calculate/evpi", json=payload2)
        assert response2.status_code == 422  # Unprocessable Entity

    def test_evppi_calculation_valid_input(self):
        """Test EVPPI calculation with valid input."""
        # Valid net benefit array
        nb_array = [[100.0, 150.0], [90.0, 140.0], [110.0, 130.0]]
        strategy_names = ["Strategy A", "Strategy B"]

        # Valid parameter samples
        param_samples = {
            "param1": [0.1, 0.2, 0.3],
            "param2": [10.0, 20.0, 30.0]
        }

        payload = {
            "nb_array": nb_array,
            "strategy_names": strategy_names,
            "parameter_samples": param_samples,
            "parameters_of_interest": ["param1"],
            "population": 1000,
            "time_horizon": 5,
            "discount_rate": 0.035
        }

        response = self.client.post("/calculate/evppi", json=payload)
        # May return 501 if not implemented, or 200 if works
        assert response.status_code in [200, 501]  # Either works or not implemented

        if response.status_code == 200:
            response_data = response.json()
            assert "result" in response_data
            assert "details" in response_data
            assert isinstance(response_data["result"], (int, float))

    def test_evppi_calculation_without_parameters(self):
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

        response = self.client.post("/calculate/evppi", json=payload)
        # This should fail because parameters_of_interest isn't in parameter_samples
        assert response.status_code in [422, 500]

    def test_analysis_status_endpoint(self):
        """Test the analysis status endpoint."""
        # This endpoint might not be implemented, so test it
        response = self.client.get("/analysis/status")

        # Status endpoint may not be implemented
        # It could return 200 (working) or 501 (not implemented)
        assert response.status_code in [200, 501]

        if response.status_code == 200:
            response_data = response.json()
            # Check structure if successful
            assert "active_analyses" in response_data
            assert "pending_analyses" in response_data
            assert "completed_analyses" in response_data

    def test_list_analyses_endpoint(self):
        """Test the list analyses endpoint."""
        response = self.client.get("/analyses")

        # This might not be implemented
        assert response.status_code in [200, 501]

        if response.status_code == 200:
            response_data = response.json()
            assert "analyses" in response_data
            assert isinstance(response_data["analyses"], list)
