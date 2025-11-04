"""
World Builder - Pass 1: Planetary Foundation
Establishes fundamental planetary parameters that influence all subsequent generation.
"""

import numpy as np
from math import pi, sin, radians

from config import WorldGenerationParams, SOLAR_CONSTANT
from models.world import WorldState, PlanetaryData


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Calculate derived planetary properties from base parameters.
    These properties influence erosion, climate, and other systems.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    # Calculate gravity effects on erosion rates
    # Higher gravity = faster erosion
    erosion_modifier = params.gravity / 9.8
    
    # Calculate seasonal variation from axial tilt
    # Higher tilt = more extreme seasons
    seasonal_variation = sin(radians(params.axial_tilt))
    
    # Calculate Coriolis parameter (affects wind and ocean currents)
    # omega = angular velocity = 2π / period
    omega = (2 * pi) / (params.rotation_hours * 3600)
    coriolis_parameter = 2 * omega
    
    # Calculate day length in seconds
    day_length_seconds = params.rotation_hours * 3600
    
    # Solar input (could be modified by distance from sun in future)
    solar_input = SOLAR_CONSTANT
    
    # Calculate climate zone latitudes
    tropic_latitude = params.axial_tilt
    arctic_latitude = 90 - params.axial_tilt
    
    # Create planetary data
    planetary_data = PlanetaryData(
        gravity=params.gravity,
        erosion_modifier=erosion_modifier,
        seasonal_variation=seasonal_variation,
        coriolis_parameter=coriolis_parameter,
        day_length_seconds=day_length_seconds,
        solar_input=solar_input,
        tropic_latitude=tropic_latitude,
        arctic_latitude=arctic_latitude,
    )
    
    # Store in world state
    world_state.planetary_data = planetary_data
    
    print(f"  - Gravity: {params.gravity:.2f} m/s² (erosion modifier: {erosion_modifier:.2f})")
    print(f"  - Axial tilt: {params.axial_tilt:.1f}° (seasonal variation: {seasonal_variation:.2f})")
    print(f"  - Day length: {params.rotation_hours:.1f} hours")
    print(f"  - Coriolis parameter: {coriolis_parameter:.6f}")
    print(f"  - Tropic zones: ±{tropic_latitude:.1f}°")
    print(f"  - Polar zones: ±{arctic_latitude:.1f}°")
