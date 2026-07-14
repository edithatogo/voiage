# voiage Web API

A RESTful web API for Value of Information analysis using the voiage Python library.

## Features

- Calculate Expected Value of Perfect Information (EVPI)
- Calculate Expected Value of Partial Perfect Information (EVPPI)
- Asynchronous processing for long-running calculations
- Analysis status tracking
- JSON-based API for easy integration

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install -r voiage/web/requirements.txt
```

## Usage

### Running the API server

```bash
# Run directly
python -m voiage.web.main

# Or with uvicorn
uvicorn voiage.web.main:app --host 0.0.0.0 --port 8000
```

### Running with Docker

```bash
# Build the Docker image
docker build -t voiage-api .

# Run the container
docker run -p 8000:8000 voiage-api
```

### Running with Docker Compose

```bash
# Build and run with docker-compose
docker-compose up --build
```

## API Endpoints

### GET /

Root endpoint returning API information.

### POST /evpi

Calculate Expected Value of Perfect Information.

**Request Body:**
```json
{
  "net_benefits": {
    "values": [[1.0, 2.0], [3.0, 1.5]],
    "strategy_names": ["Strategy A", "Strategy B"]
  },
  "config": {
    "population": 100000,
    "time_horizon": 10,
    "discount_rate": 0.03
  }
}
```

### POST /evppi

Calculate Expected Value of Partial Perfect Information.

**Request Body:**
```json
{
  "net_benefits": {
    "values": [[1.0, 2.0], [3.0, 1.5]],
    "strategy_names": ["Strategy A", "Strategy B"]
  },
  "parameters": {
    "parameters": {
      "param1": [0.1, 0.2],
      "param2": [0.5, 0.6]
    }
  },
  "config": {
    "population": 100000,
    "time_horizon": 10,
    "discount_rate": 0.03
  }
}
```

### GET /analysis/{analysis_id}

Get the status and result of a specific analysis.

### GET /analyses

List all analyses.

## Example Client Usage

See [examples/web_api_example.py](../../examples/web_api_example.py) for a complete example of how to use the API from Python.