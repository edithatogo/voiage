"""Environmental impact assessment tools for Value of Information analysis."""

from .impact_assessment import (
    calculate_biodiversity_impact,
    calculate_carbon_footprint,
    calculate_water_usage,
    environmental_lifecycle_assessment,
    monetize_environmental_impacts,
)

__all__ = [
    "calculate_carbon_footprint",
    "calculate_water_usage",
    "calculate_biodiversity_impact",
    "monetize_environmental_impacts",
    "environmental_lifecycle_assessment",
]
