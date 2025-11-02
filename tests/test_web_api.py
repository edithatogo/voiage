"""Tests for the web API."""

from fastapi.testclient import TestClient

from voiage.web.main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "voiage API is running"}

def test_evpi_calculation():
    """Test EVPI calculation endpoint."""
    # Create test data
    net_benefits = {
        "values": [[1.0, 2.0], [3.0, 1.5], [2.5, 2.0], [1.8, 2.2]],
        "strategy_names": ["Strategy A", "Strategy B"]
    }

    # Prepare request data
    request_data = {
        "net_benefits": net_benefits
    }

    # Make request
    response = client.post("/evpi", json=request_data)
    assert response.status_code == 200

    # Check response
    result = response.json()
    assert "analysis_id" in result
    assert "result" in result
    assert "method" in result
    assert result["method"] == "evpi"
    assert isinstance(result["result"], (int, float))

def test_evppi_calculation():
    """Test EVPPI calculation endpoint."""
    # Create test data
    net_benefits = {
        "values": [[1.0, 2.0], [3.0, 1.5], [2.5, 2.0], [1.8, 2.2]],
        "strategy_names": ["Strategy A", "Strategy B"]
    }

    parameters = {
        "parameters": {
            "param1": [0.1, 0.2, 0.15, 0.18],
            "param2": [0.5, 0.6, 0.55, 0.58]
        }
    }

    # Prepare request data
    request_data = {
        "net_benefits": net_benefits,
        "parameters": parameters
    }

    # Make request
    response = client.post("/evppi", json=request_data)
    assert response.status_code == 200

    # Check response
    result = response.json()
    assert "analysis_id" in result
    assert "result" in result
    assert "method" in result
    assert result["method"] == "evppi"
    assert isinstance(result["result"], (int, float))

def test_evppi_without_parameters():
    """Test EVPPI calculation without parameters should fail."""
    # Create test data without parameters
    net_benefits = {
        "values": [[1.0, 2.0], [3.0, 1.5], [2.5, 2.0], [1.8, 2.2]],
        "strategy_names": ["Strategy A", "Strategy B"]
    }

    # Prepare request data
    request_data = {
        "net_benefits": net_benefits
    }

    # Make request
    response = client.post("/evppi", json=request_data)
    assert response.status_code == 400

def test_analysis_status():
    """Test getting analysis status."""
    # First create an analysis
    net_benefits = {
        "values": [[1.0, 2.0], [3.0, 1.5], [2.5, 2.0], [1.8, 2.2]],
        "strategy_names": ["Strategy A", "Strategy B"]
    }

    request_data = {
        "net_benefits": net_benefits
    }

    response = client.post("/evpi", json=request_data)
    assert response.status_code == 200
    result = response.json()
    analysis_id = result["analysis_id"]

    # Get status
    response = client.get(f"/analysis/{analysis_id}")
    assert response.status_code == 200
    status = response.json()
    assert status["analysis_id"] == analysis_id
    assert status["status"] == "completed"
    assert "result" in status

def test_list_analyses():
    """Test listing all analyses."""
    response = client.get("/analyses")
    assert response.status_code == 200
    analyses = response.json()
    assert isinstance(analyses, list)

if __name__ == "__main__":
    test_root()
    test_evpi_calculation()
    test_evppi_calculation()
    test_evppi_without_parameters()
    test_analysis_status()
    test_list_analyses()
    print("All web API tests passed!")
