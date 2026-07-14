# voiage Web API Summary

## Overview

The voiage Web API provides a RESTful interface for performing Value of Information (VOI) analysis remotely. This allows users to leverage the powerful voiage Python library through simple HTTP requests, making it accessible from any programming language or platform.

## Features Implemented

1. **EVPI Calculation**: Calculate Expected Value of Perfect Information
2. **EVPPI Calculation**: Calculate Expected Value of Partial Perfect Information
3. **Asynchronous Processing**: Support for long-running calculations
4. **Analysis Status Tracking**: Monitor the status of submitted analyses
5. **JSON-based API**: Easy integration with other systems
6. **Docker Support**: Containerized deployment for easy scaling

## API Endpoints

### GET /
- **Description**: Root endpoint returning API information
- **Response**: JSON with API status message

### POST /evpi
- **Description**: Calculate Expected Value of Perfect Information
- **Request Body**: 
  - `net_benefits`: Net benefit data with values and optional strategy names
  - `config`: Optional configuration parameters (population, time_horizon, discount_rate, etc.)
- **Response**: Analysis ID and result

### POST /evppi
- **Description**: Calculate Expected Value of Partial Perfect Information
- **Request Body**: 
  - `net_benefits`: Net benefit data with values and optional strategy names
  - `parameters`: Parameter samples for EVPPI calculation
  - `config`: Optional configuration parameters
- **Response**: Analysis ID and result

### GET /analysis/{analysis_id}
- **Description**: Get the status and result of a specific analysis
- **Response**: Analysis ID, status, and result (if completed)

### GET /analyses
- **Description**: List all analyses
- **Response**: Array of analysis statuses

## Deployment Options

1. **Direct Execution**: Run the Python module directly
2. **Uvicorn**: Use the Uvicorn ASGI server
3. **Docker**: Containerized deployment with Docker
4. **Docker Compose**: Multi-container deployment

## Example Usage

```python
import requests
import numpy as np

# Sample net benefit data
net_benefits = {
    "values": np.random.randn(100, 2).tolist(),
    "strategy_names": ["Standard Care", "New Treatment"]
}

# Configuration
config = {
    "population": 100000,
    "time_horizon": 10,
    "discount_rate": 0.03
}

# Prepare request
request_data = {
    "net_benefits": net_benefits,
    "config": config
}

# Send request
response = requests.post("http://localhost:8000/evpi", json=request_data)

if response.status_code == 200:
    result = response.json()
    print(f"EVPI: {result['result']}")
```

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running the application
- **Pydantic**: Data validation and settings management
- **voiage**: Core Python library for VOI analysis

## Future Enhancements

- Add support for EVSI calculations
- Implement authentication and authorization
- Add rate limiting
- Implement result caching
- Add support for batch processing
- Implement WebSocket support for real-time updates