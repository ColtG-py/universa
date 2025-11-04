"""
World Builder - Pass 6: Ocean Currents (SIMPLIFIED VERSION)
Generates realistic ocean circulation patterns using gradient-based approach.

SIMPLIFIED APPROACH:
- Wind-driven base currents with Coriolis deflection
- Latitude-based gyre templates (subtropical and subpolar)
- Smooth gradient fields to avoid isolated strong currents
- Western boundary intensification
- Continental deflection
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate ocean current patterns using simplified gradient-based approach.
    
    Creates realistic gyre patterns without complex physics simulation.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating ocean currents (simplified gradient method)...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    # STEP 1: Collect global wind and elevation data
    print(f"    - Collecting wind and elevation data...")
    
    wind_speed_global = np.zeros((size, size), dtype=np.float32)
    wind_dir_global = np.zeros((size, size), dtype=np.float32)
    elevation_global = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.wind_speed is not None:
                wind_speed_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_speed
            if chunk.wind_direction is not None:
                wind_dir_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_direction
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
    
    # STEP 2: Create ocean mask
    ocean_mask = elevation_global < 0
    
    # STEP 3: Generate base currents from wind (with Coriolis deflection)
    print(f"    - Generating base wind-driven currents...")
    
    # Create latitude field (-90 to +90)
    latitude = np.linspace(-90, 90, size)
    latitude_field = np.tile(latitude[np.newaxis, :], (size, 1))
    
    # Convert wind to current direction (90° deflection due to Ekman transport)
    # In Northern Hemisphere: deflect 90° to the right
    # In Southern Hemisphere: deflect 90° to the left
    wind_dir_rad = np.deg2rad(wind_dir_global)
    
    # Apply 90° deflection based on hemisphere
    coriolis_deflection = np.where(latitude_field >= 0, 90, -90)  # degrees
    current_dir_deg = (wind_dir_global + coriolis_deflection) % 360
    current_dir_rad = np.deg2rad(current_dir_deg)
    
    # Current speed proportional to wind speed (scaled down)
    base_current_speed = wind_speed_global * 0.15  # Ocean currents ~15% of wind speed
    
    # STEP 4: Create latitude-based gyre circulation patterns
    print(f"    - Adding gyre circulation patterns...")
    
    # Define gyre zones based on latitude
    # Subtropical gyres: 15°-45° (anticyclonic - clockwise NH, counterclockwise SH)
    # Subpolar gyres: 50°-70° (cyclonic - counterclockwise NH, clockwise SH)
    
    gyre_field_x = np.zeros((size, size), dtype=np.float32)
    gyre_field_y = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        lat = latitude[y]
        
        # Determine gyre strength based on latitude
        if 15 <= abs(lat) <= 45:
            # Subtropical gyre zone
            gyre_strength = 0.5 * (1.0 - abs(abs(lat) - 30) / 15)  # Peak at 30°
            
            # Anticyclonic rotation
            if lat > 0:  # Northern hemisphere - clockwise
                gyre_dir = np.linspace(0, 360, size)  # Varies with longitude
                gyre_field_x[y, :] = gyre_strength * np.cos(np.deg2rad(gyre_dir))
                gyre_field_y[y, :] = -gyre_strength * np.sin(np.deg2rad(gyre_dir))
            else:  # Southern hemisphere - counterclockwise
                gyre_dir = np.linspace(0, 360, size)
                gyre_field_x[y, :] = gyre_strength * np.cos(np.deg2rad(gyre_dir))
                gyre_field_y[y, :] = gyre_strength * np.sin(np.deg2rad(gyre_dir))
                
        elif 50 <= abs(lat) <= 70:
            # Subpolar gyre zone
            gyre_strength = 0.3 * (1.0 - abs(abs(lat) - 60) / 10)  # Peak at 60°
            
            # Cyclonic rotation (opposite of subtropical)
            if lat > 0:  # Northern hemisphere - counterclockwise
                gyre_dir = np.linspace(0, 360, size)
                gyre_field_x[y, :] = gyre_strength * np.cos(np.deg2rad(gyre_dir))
                gyre_field_y[y, :] = gyre_strength * np.sin(np.deg2rad(gyre_dir))
            else:  # Southern hemisphere - clockwise
                gyre_dir = np.linspace(0, 360, size)
                gyre_field_x[y, :] = gyre_strength * np.cos(np.deg2rad(gyre_dir))
                gyre_field_y[y, :] = -gyre_strength * np.sin(np.deg2rad(gyre_dir))
    
    # Smooth gyre fields for realistic circulation
    gyre_field_x = gaussian_filter(gyre_field_x, sigma=size/20)
    gyre_field_y = gaussian_filter(gyre_field_y, sigma=size/20)
    
    # STEP 5: Combine wind-driven and gyre currents
    print(f"    - Combining current components...")
    
    # Convert base current direction to components
    current_u = base_current_speed * np.cos(current_dir_rad)
    current_v = base_current_speed * np.sin(current_dir_rad)
    
    # Add gyre contribution
    current_u = current_u + gyre_field_x
    current_v = current_v + gyre_field_y
    
    # STEP 6: Apply western boundary intensification
    print(f"    - Applying western boundary intensification...")
    
    # Detect western boundaries (left side of ocean basins)
    # Strengthen currents on western sides
    for y in range(size):
        for x in range(size):
            if ocean_mask[x, y]:
                # Check if this is near a western boundary
                # Look for land to the west
                is_western_boundary = False
                
                for check_x in range(max(0, x-20), x):
                    if not ocean_mask[check_x, y]:
                        is_western_boundary = True
                        break
                
                if is_western_boundary:
                    # Intensify currents (western boundary currents are 2-3x stronger)
                    intensification = 2.5
                    current_u[x, y] *= intensification
                    current_v[x, y] *= intensification
    
    # STEP 7: Smooth currents to ensure gradual transitions
    print(f"    - Smoothing current fields...")
    
    # Apply heavy smoothing to ensure no isolated strong currents
    current_u = gaussian_filter(current_u, sigma=3.0)
    current_v = gaussian_filter(current_v, sigma=3.0)
    
    # Mask out land areas
    current_u = np.where(ocean_mask, current_u, 0)
    current_v = np.where(ocean_mask, current_v, 0)
    
    # STEP 8: Calculate magnitude and direction
    print(f"    - Calculating final current magnitude and direction...")
    
    current_magnitude = np.sqrt(current_u**2 + current_v**2)
    current_direction = np.rad2deg(np.arctan2(current_v, current_u)) % 360
    
    # Normalize to realistic values (typical range: 0.05-1.5 m/s)
    max_current = np.percentile(current_magnitude[ocean_mask], 98)
    if max_current > 0:
        current_magnitude = current_magnitude / max_current * 1.2
    
    # Ensure minimum current speed in open ocean
    current_magnitude = np.where(
        ocean_mask & (current_magnitude < 0.05),
        0.05,
        current_magnitude
    )
    
    # STEP 9: Apply depth decay for deep ocean
    print(f"    - Applying depth-based modulation...")
    
    # Weaker currents in very deep water
    depth_factor = np.ones_like(elevation_global)
    deep_ocean = elevation_global < -3000
    depth_factor[deep_ocean] = 0.7
    
    current_magnitude *= depth_factor
    
    # STEP 10: Store in chunks
    print(f"    - Storing ocean current data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.ocean_current_speed = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.ocean_current_direction = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.ocean_current_speed[local_x, local_y] = current_magnitude[global_x, global_y]
                        chunk.ocean_current_direction[local_x, local_y] = current_direction[global_x, global_y]
    
    # STEP 11: Statistics
    ocean_currents = current_magnitude[ocean_mask]
    
    if len(ocean_currents) > 0:
        print(f"  - Ocean current statistics:")
        print(f"    Mean speed: {ocean_currents.mean():.3f} m/s")
        print(f"    Median speed: {np.median(ocean_currents):.3f} m/s")
        print(f"    Max speed: {ocean_currents.max():.3f} m/s")
        print(f"    90th percentile: {np.percentile(ocean_currents, 90):.3f} m/s")
    
    print(f"  - Ocean circulation patterns complete (simplified method)")