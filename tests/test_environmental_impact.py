"""Tests for environmental impact assessment tools."""

import numpy as np
import pytest

from voiage.environmental.impact_assessment import (
    calculate_carbon_footprint,
    calculate_water_usage,
    calculate_biodiversity_impact,
    monetize_environmental_impacts,
    environmental_lifecycle_assessment
)


def test_calculate_carbon_footprint():
    """Test carbon footprint calculation."""
    activities = {
        "electricity_kwh": 1000,
        "fuel_liters": 500,
        "natural_gas_m3": 200
    }
    
    emission_factors = {
        "electricity_kwh": 0.5,      # kg CO2/kWh
        "fuel_liters": 2.3,          # kg CO2/liter
        "natural_gas_m3": 1.9        # kg CO2/m3
    }
    
    footprint = calculate_carbon_footprint(activities, emission_factors)
    
    expected = (1000 * 0.5) + (500 * 2.3) + (200 * 1.9)
    assert abs(footprint - expected) < 1e-10


def test_calculate_carbon_footprint_missing_factor():
    """Test carbon footprint calculation with missing emission factor."""
    activities = {"electricity_kwh": 1000}
    emission_factors = {"fuel_liters": 2.3}
    
    with pytest.raises(ValueError):
        calculate_carbon_footprint(activities, emission_factors)


def test_calculate_water_usage():
    """Test water usage calculation."""
    processes = {
        "manufacturing_hours": 100,
        "cooling_cycles": 50,
        "cleaning_sessions": 20
    }
    
    water_factors = {
        "manufacturing_hours": 500,   # liters/hour
        "cooling_cycles": 1000,       # liters/cycle
        "cleaning_sessions": 200      # liters/session
    }
    
    water_usage = calculate_water_usage(processes, water_factors)
    
    expected = (100 * 500) + (50 * 1000) + (20 * 200)
    assert abs(water_usage - expected) < 1e-10


def test_calculate_biodiversity_impact():
    """Test biodiversity impact calculation."""
    land_use_changes = {
        "forest_conversion_ha": 10,
        "wetland_drainage_ha": 5,
        "grassland_preservation_ha": -2  # Negative = preservation
    }
    
    biodiversity_factors = {
        "forest_conversion_ha": 50,    # species-years/ha lost
        "wetland_drainage_ha": 100,    # species-years/ha lost
        "grassland_preservation_ha": -30  # species-years/ha gained
    }
    
    impact = calculate_biodiversity_impact(land_use_changes, biodiversity_factors)
    
    expected = (10 * 50) + (5 * 100) + (-2 * -30)
    assert abs(impact - expected) < 1e-10


def test_monetize_environmental_impacts():
    """Test monetization of environmental impacts."""
    impacts = {
        "carbon_kg": 1000,
        "water_liters": 50000,
        "waste_kg": 2000
    }
    
    valuation_factors = {
        "carbon_kg": 0.05,      # $/kg CO2
        "water_liters": 0.002,  # $/liter
        "waste_kg": 0.1         # $/kg waste
    }
    
    monetary_values = monetize_environmental_impacts(impacts, valuation_factors)
    
    assert "carbon_kg" in monetary_values
    assert "water_liters" in monetary_values
    assert "waste_kg" in monetary_values
    
    assert abs(monetary_values["carbon_kg"] - 50.0) < 1e-10
    assert abs(monetary_values["water_liters"] - 100.0) < 1e-10
    assert abs(monetary_values["waste_kg"] - 200.0) < 1e-10


def test_environmental_lifecycle_assessment():
    """Test comprehensive environmental lifecycle assessment."""
    lifecycle_stages = ["production", "transport", "use", "disposal"]
    
    stage_impacts = {
        "production": {
            "carbon_kg": 500,
            "water_liters": 20000
        },
        "transport": {
            "carbon_kg": 100,
            "water_liters": 0
        },
        "use": {
            "carbon_kg": 50,
            "water_liters": 10000
        },
        "disposal": {
            "carbon_kg": 25,
            "water_liters": 5000
        }
    }
    
    valuation_factors = {
        "carbon_kg": 0.05,      # $/kg CO2
        "water_liters": 0.002   # $/liter
    }
    
    results = environmental_lifecycle_assessment(lifecycle_stages, stage_impacts, valuation_factors)
    
    # Check structure
    assert "stage_impacts" in results
    assert "stage_costs" in results
    assert "total_impacts" in results
    assert "total_cost" in results
    
    # Check total impacts
    assert abs(results["total_impacts"]["carbon_kg"] - 675) < 1e-10
    assert abs(results["total_impacts"]["water_liters"] - 35000) < 1e-10
    
    # Check total cost
    expected_cost = (675 * 0.05) + (35000 * 0.002)
    assert abs(results["total_cost"] - expected_cost) < 1e-10


if __name__ == "__main__":
    test_calculate_carbon_footprint()
    test_calculate_carbon_footprint_missing_factor()
    test_calculate_water_usage()
    test_calculate_biodiversity_impact()
    test_monetize_environmental_impacts()
    test_environmental_lifecycle_assessment()
    print("All environmental impact assessment tests passed!")