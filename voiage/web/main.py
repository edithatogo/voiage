"""Main FastAPI application for voiage web API."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

from voiage.analysis import DecisionAnalysis
from voiage.config_objects import VOIAnalysisConfig
from voiage.schema import ParameterSet, ValueArray

app = FastAPI(
    title="voiage API",
    description="Web API for Value of Information analysis",
    version="0.1.0"
)

# In-memory storage for analysis results (in production, use a database)
analysis_storage: Dict[str, Dict[str, Any]] = {}

class NetBenefitData(BaseModel):
    """Model for net benefit data."""

    values: List[List[float]] = Field(..., description="2D array of net benefits (samples x strategies)")
    strategy_names: Optional[List[str]] = Field(None, description="Names of strategies")

class ParameterData(BaseModel):
    """Model for parameter samples."""

    parameters: Dict[str, List[float]] = Field(..., description="Dictionary of parameter samples")

class AnalysisRequest(BaseModel):
    """Model for analysis request."""

    net_benefits: NetBenefitData
    parameters: Optional[ParameterData] = None
    config: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    """Model for analysis response."""

    analysis_id: str
    result: float
    method: str
    status: str = "completed"

class AnalysisStatus(BaseModel):
    """Model for analysis status."""

    analysis_id: str
    status: str
    result: Optional[float] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "voiage API is running"}

@app.post("/evpi", response_model=AnalysisResponse)
async def calculate_evpi(request: AnalysisRequest):
    """Calculate Expected Value of Perfect Information."""
    try:
        # Convert data to numpy arrays
        nb_array = np.array(request.net_benefits.values, dtype=np.float64)

        # Create ValueArray
        strategy_names = request.net_benefits.strategy_names
        nb_value_array = ValueArray.from_numpy(nb_array, strategy_names)

        # Create parameter samples if provided
        parameter_samples = None
        if request.parameters:
            param_dict = {k: np.array(v, dtype=np.float64) for k, v in request.parameters.parameters.items()}
            parameter_samples = ParameterSet.from_numpy_or_dict(param_dict)

        # Create configuration
        config = VOIAnalysisConfig()
        if request.config:
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create analysis
        analysis = DecisionAnalysis(
            nb_array=nb_value_array,
            parameter_samples=parameter_samples,
            use_jit=config.use_jit,
            backend=config.backend,
            enable_caching=config.enable_caching,
            streaming_window_size=config.streaming_window_size
        )

        # Calculate EVPI
        # Extract population parameters from config if provided
        population = request.config.get("population") if request.config else None
        time_horizon = request.config.get("time_horizon") if request.config else None
        discount_rate = request.config.get("discount_rate") if request.config else None
        chunk_size = request.config.get("chunk_size") if request.config else None

        result = analysis.evpi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            chunk_size=chunk_size
        )

        # Generate analysis ID
        analysis_id = f"evpi_{len(analysis_storage) + 1}"

        # Store result
        analysis_storage[analysis_id] = {
            "result": result,
            "method": "evpi",
            "status": "completed"
        }

        return AnalysisResponse(
            analysis_id=analysis_id,
            result=result,
            method="evpi"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evppi", response_model=AnalysisResponse)
async def calculate_evppi(request: AnalysisRequest):
    """Calculate Expected Value of Partial Perfect Information."""
    try:
        # Check if parameters are provided
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Parameters are required for EVPPI calculation")

        # Convert data to numpy arrays
        nb_array = np.array(request.net_benefits.values, dtype=np.float64)

        # Create ValueArray
        strategy_names = request.net_benefits.strategy_names
        nb_value_array = ValueArray.from_numpy(nb_array, strategy_names)

        # Create parameter samples
        param_dict = {k: np.array(v, dtype=np.float64) for k, v in request.parameters.parameters.items()}
        parameter_samples = ParameterSet.from_numpy_or_dict(param_dict)

        # Create configuration
        config = VOIAnalysisConfig()
        if request.config:
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create analysis
        analysis = DecisionAnalysis(
            nb_array=nb_value_array,
            parameter_samples=parameter_samples,
            use_jit=config.use_jit,
            backend=config.backend,
            enable_caching=config.enable_caching,
            streaming_window_size=config.streaming_window_size
        )

        # Calculate EVPPI
        # Extract parameters from config if provided
        population = request.config.get("population") if request.config else None
        time_horizon = request.config.get("time_horizon") if request.config else None
        discount_rate = request.config.get("discount_rate") if request.config else None
        n_regression_samples = request.config.get("n_regression_samples") if request.config else None
        chunk_size = request.config.get("chunk_size") if request.config else None

        result = analysis.evppi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            chunk_size=chunk_size
        )

        # Generate analysis ID
        analysis_id = f"evppi_{len(analysis_storage) + 1}"

        # Store result
        analysis_storage[analysis_id] = {
            "result": result,
            "method": "evppi",
            "status": "completed"
        }

        return AnalysisResponse(
            analysis_id=analysis_id,
            result=result,
            method="evppi"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analysis/{analysis_id}", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis."""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis_data = analysis_storage[analysis_id]
    return AnalysisStatus(
        analysis_id=analysis_id,
        status=analysis_data["status"],
        result=analysis_data.get("result")
    )

@app.get("/analyses", response_model=List[AnalysisStatus])
async def list_analyses():
    """List all analyses."""
    return [
        AnalysisStatus(
            analysis_id=analysis_id,
            status=data["status"],
            result=data.get("result")
        )
        for analysis_id, data in analysis_storage.items()
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
