"""Environmental impact assessment tools for Value of Information analysis."""

from typing import Any

from voiage.exceptions import raise_input_error


def calculate_carbon_footprint(
    activities: dict[str, float], emission_factors: dict[str, float]
) -> float:
    """
    Calculate carbon footprint from various activities.

    Args:
        activities: Dictionary mapping activity names to activity levels
                    (e.g., "electricity_kwh": 1000, "fuel_liters": 500)
        emission_factors: Dictionary mapping activity names to emission factors
                         (kg CO2 per unit of activity)

    Returns
    -------
        float: Total carbon footprint in kg CO2 equivalent

    Example:
        >>> activities = {"electricity_kwh": 1000, "fuel_liters": 500}
        >>> emission_factors = {"electricity_kwh": 0.5, "fuel_liters": 2.3}
        >>> footprint = calculate_carbon_footprint(activities, emission_factors)
        >>> print(f"Carbon footprint: {footprint} kg CO2")
    """
    total_footprint = 0.0

    for activity, level in activities.items():
        if activity in emission_factors:
            total_footprint += level * emission_factors[activity]
        else:
            raise_input_error(f"No emission factor found for activity: {activity}")

    return total_footprint


def calculate_water_usage(
    processes: dict[str, float], water_factors: dict[str, float]
) -> float:
    """
    Calculate total water usage from industrial processes.

    Args:
        processes: Dictionary mapping process names to process levels
        water_factors: Dictionary mapping process names to water usage factors
                      (liters per unit of process)

    Returns
    -------
        float: Total water usage in liters
    """
    total_water = 0.0

    for process, level in processes.items():
        if process in water_factors:
            total_water += level * water_factors[process]
        else:
            raise_input_error(f"No water factor found for process: {process}")

    return total_water


def calculate_biodiversity_impact(
    land_use_changes: dict[str, float], biodiversity_factors: dict[str, float]
) -> float:
    """
    Calculate biodiversity impact from land use changes.

    Args:
        land_use_changes: Dictionary mapping land use types to area changes (hectares)
        biodiversity_factors: Dictionary mapping land use types to biodiversity impact factors
                             (species-years lost per hectare)

    Returns
    -------
        float: Total biodiversity impact (species-years lost)
    """
    total_impact = 0.0

    for land_use, change in land_use_changes.items():
        if land_use in biodiversity_factors:
            total_impact += change * biodiversity_factors[land_use]
        else:
            raise_input_error(f"No biodiversity factor found for land use: {land_use}")

    return total_impact


def monetize_environmental_impacts(
    impacts: dict[str, float], valuation_factors: dict[str, float]
) -> dict[str, float]:
    """
    Convert environmental impacts to monetary values.

    Args:
        impacts: Dictionary mapping impact types to impact values
        valuation_factors: Dictionary mapping impact types to monetary values per unit impact

    Returns
    -------
        Dict[str, float]: Dictionary mapping impact types to monetary values
    """
    monetary_values = {}

    for impact_type, impact_value in impacts.items():
        if impact_type in valuation_factors:
            monetary_values[impact_type] = impact_value * valuation_factors[impact_type]
        else:
            raise_input_error(
                f"No valuation factor found for impact type: {impact_type}"
            )

    return monetary_values


def environmental_lifecycle_assessment(
    lifecycle_stages: list[str],
    stage_impacts: dict[str, dict[str, float]],
    valuation_factors: dict[str, float],
) -> dict[str, Any]:
    """
    Perform a comprehensive environmental lifecycle assessment.

    Args:
        lifecycle_stages: List of lifecycle stages (e.g., ["production", "transport", "use", "disposal"])
        stage_impacts: Dictionary mapping stages to impact dictionaries
        valuation_factors: Dictionary mapping impact types to monetary values per unit

    Returns
    -------
        Dict[str, Dict[str, float]]: Dictionary with detailed LCA results
    """
    stage_impacts_result: dict[str, dict[str, float]] = {}
    stage_costs_result: dict[str, dict[str, float]] = {}
    total_impacts: dict[str, float] = {}
    total_cost = 0.0

    # Initialize total impacts
    all_impact_types: set[str] = set()
    for impacts in stage_impacts.values():
        all_impact_types.update(impacts.keys())

    for impact_type in all_impact_types:
        total_impacts[impact_type] = 0.0

    # Calculate impacts and costs for each stage
    for stage in lifecycle_stages:
        if stage in stage_impacts:
            stage_impact = stage_impacts[stage]
            stage_impacts_result[stage] = stage_impact

            # Calculate monetary values for this stage
            stage_costs = monetize_environmental_impacts(
                stage_impact, valuation_factors
            )
            stage_costs_result[stage] = stage_costs

            # Add to total impacts
            for impact_type, impact_value in stage_impact.items():
                total_impacts[impact_type] += impact_value

    # Calculate total monetary cost
    for stage_costs in stage_costs_result.values():
        for cost in stage_costs.values():
            total_cost += cost

    return {
        "stage_impacts": stage_impacts_result,
        "stage_costs": stage_costs_result,
        "total_impacts": total_impacts,
        "total_cost": total_cost,
    }
