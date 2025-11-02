"""Example client for the voiage web API."""

import requests
import json
import numpy as np

# API endpoint
BASE_URL = "http://localhost:8000"

def example_evpi():
    """Example of EVPI calculation using the web API."""
    print("=== EVPI Calculation Example ===")
    
    # Create sample net benefit data
    # 100 samples, 2 strategies
    np.random.seed(42)
    net_benefits_data = {
        "values": np.random.randn(100, 2).tolist(),
        "strategy_names": ["Standard Care", "New Treatment"]
    }
    
    # Prepare request
    request_data = {
        "net_benefits": net_benefits_data,
        "config": {
            "population": 100000,
            "time_horizon": 10,
            "discount_rate": 0.03
        }
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/evpi", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"EVPI Result: {result['result']:.6f}")
        print(f"Method: {result['method']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_evppi():
    """Example of EVPPI calculation using the web API."""
    print("\n=== EVPPI Calculation Example ===")
    
    # Create sample net benefit data
    np.random.seed(42)
    net_benefits_data = {
        "values": np.random.randn(100, 2).tolist(),
        "strategy_names": ["Standard Care", "New Treatment"]
    }
    
    # Create parameter samples
    parameter_data = {
        "parameters": {
            "treatment_efficacy": np.random.normal(0.7, 0.1, 100).tolist(),
            "treatment_cost": np.random.normal(5000, 1000, 100).tolist()
        }
    }
    
    # Prepare request
    request_data = {
        "net_benefits": net_benefits_data,
        "parameters": parameter_data,
        "config": {
            "population": 100000,
            "time_horizon": 10,
            "discount_rate": 0.03,
            "n_regression_samples": 50
        }
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/evppi", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"EVPPI Result: {result['result']:.6f}")
        print(f"Method: {result['method']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_analysis_status(analysis_id):
    """Example of checking analysis status."""
    print(f"\n=== Checking Status for Analysis {analysis_id} ===")
    
    # Get analysis status
    response = requests.get(f"{BASE_URL}/analysis/{analysis_id}")
    
    if response.status_code == 200:
        status = response.json()
        print(f"Analysis ID: {status['analysis_id']}")
        print(f"Status: {status['status']}")
        if status['result'] is not None:
            print(f"Result: {status['result']:.6f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def example_list_analyses():
    """Example of listing all analyses."""
    print("\n=== Listing All Analyses ===")
    
    # Get all analyses
    response = requests.get(f"{BASE_URL}/analyses")
    
    if response.status_code == 200:
        analyses = response.json()
        print(f"Found {len(analyses)} analyses:")
        for analysis in analyses:
            print(f"  - ID: {analysis['analysis_id']}, Status: {analysis['status']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Note: The web API server must be running for these examples to work
    # Start the server with: python -m voiage.web.main
    
    try:
        example_evpi()
        example_evppi()
        example_list_analyses()
        print("\nExamples completed!")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the web API. Make sure the server is running.")
        print("Start the server with: python -m voiage.web.main")