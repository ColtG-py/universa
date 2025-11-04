"""
World Builder - Pass 7: Climate Simulation
Calculates temperature and precipitation based on latitude, elevation, and wind.

IMPROVEMENTS:
- Smooth precipitation patterns (no choppy noise)
- Enhanced elevation effects on temperature
- More climate variability through noise
- Orographic precipitation and rain shadows
- Coastal moisture effects
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from utils.noise import NoiseGenerator


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Calculate climate patterns with realistic temperature and precipitation.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Calculating climate patterns...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    # STEP 1: Collect global elevation and wind data
    print(f"    - Collecting elevation and wind data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    wind_dir_global = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.wind_direction is not None:
                wind_dir_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_direction
    
    # STEP 2: Generate climate noise for variability
    print(f"    - Generating climate variation patterns...")
    
    temp_noise_gen = NoiseGenerator(
        seed=params.seed + 7000,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=size / 6.0
    )
    
    precip_noise_gen = NoiseGenerator(
        seed=params.seed + 8000,
        octaves=5,
        persistence=0.6,
        lacunarity=2.0,
        scale=size / 8.0
    )
    
    temp_variation = temp_noise_gen.generate_perlin_2d(size, size, 0, 0, normalize=False)
    precip_variation = precip_noise_gen.generate_perlin_2d(size, size, 0, 0, normalize=True)
    
    # STEP 3: Calculate base climate from latitude
    print(f"    - Calculating latitude-based climate zones...")
    
    latitude = np.linspace(-90, 90, size)
    latitude_field = np.tile(latitude[np.newaxis, :], (size, 1))
    
    # Base temperature: warmer at equator, colder at poles
    # Add seasonal variation
    base_temp_field = params.base_temperature_c - np.abs(latitude_field) * 0.55
    
    # Base precipitation zones (more complex than simple latitude bands)
    # High at equator (ITCZ), high at mid-latitudes (westerlies), low at 30° (horse latitudes)
    base_precip_field = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        lat = latitude[y]
        
        # Tropical convergence zone (0-10°): High precipitation
        if abs(lat) < 10:
            base_precip = 2500
        # Subtropical high (20-35°): Low precipitation (deserts)
        elif 20 <= abs(lat) <= 35:
            base_precip = 400
        # Mid-latitude westerlies (40-60°): Moderate-high precipitation
        elif 40 <= abs(lat) <= 60:
            base_precip = 1200
        # Polar regions (>60°): Low precipitation
        elif abs(lat) > 60:
            base_precip = 300
        # Transition zones
        else:
            # Interpolate between zones
            if 10 <= abs(lat) < 20:
                # Transition from tropical to subtropical
                t = (abs(lat) - 10) / 10
                base_precip = 2500 * (1 - t) + 400 * t
            else:  # 35 < abs(lat) < 40
                # Transition from subtropical to mid-latitude
                t = (abs(lat) - 35) / 5
                base_precip = 400 * (1 - t) + 1200 * t
        
        base_precip_field[:, y] = base_precip
    
    # STEP 4: Apply noise variation to precipitation
    print(f"    - Adding precipitation variability...")
    
    # Modulate base precipitation with noise (±30%)
    precip_field = base_precip_field * (0.7 + 0.6 * precip_variation)
    
    # STEP 5: Calculate temperature with enhanced elevation effects
    print(f"    - Calculating temperature with elevation effects...")
    
    temp_field = np.zeros((size, size), dtype=np.float32)
    ocean_mask = elevation_global < 0
    
    for y in range(size):
        for x in range(size):
            elev = elevation_global[x, y]
            base_temp = base_temp_field[x, y]
            
            if elev < 0:
                # Ocean temperatures - more stable, influenced by depth
                depth = abs(elev)
                # Deeper ocean is slightly cooler but more stable
                temp_modifier = -depth * 0.001  # Very slight cooling with depth
                temp = base_temp + temp_modifier
            else:
                # Land temperatures - affected by elevation
                # Enhanced elevation effect: stronger cooling at high elevations
                # Standard lapse rate: 6.5°C per 1000m
                # Add non-linear component for high elevations (stronger cooling)
                if elev > 2000:
                    # High mountains: enhanced cooling
                    temp = base_temp - (elev * 0.0075)  # Stronger effect
                else:
                    # Normal elevations
                    temp = base_temp - (elev * 0.0065)  # Standard lapse rate
            
            # Add climate variation noise (±3°C variability)
            temp += temp_variation[x, y] * 3.0
            
            temp_field[x, y] = temp
    
    # STEP 6: Apply orographic effects to precipitation
    print(f"    - Calculating orographic precipitation and rain shadows...")
    
    # Calculate slope and aspect for orographic effects
    from utils.spatial import calculate_gradient
    
    grad_x, grad_y = calculate_gradient(elevation_global)
    slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Detect windward slopes (wind blowing upslope)
    wind_dir_rad = np.deg2rad(wind_dir_global)
    wind_u = np.cos(wind_dir_rad)
    wind_v = np.sin(wind_dir_rad)
    
    # Dot product of wind direction and slope gradient
    # Positive = windward (air forced upward), Negative = leeward (rain shadow)
    orographic_effect = -(grad_x * wind_u + grad_y * wind_v)
    
    # Smooth orographic effect
    orographic_effect = gaussian_filter(orographic_effect, sigma=2.0)
    
    # Apply orographic precipitation
    for y in range(size):
        for x in range(size):
            elev = elevation_global[x, y]
            
            if elev > 0:  # Land only
                base_precip = precip_field[x, y]
                
                # Windward slopes get more precipitation
                if orographic_effect[x, y] > 0:
                    # Increase precipitation on windward slopes
                    # Effect stronger at higher elevations
                    orographic_bonus = 1.0 + (orographic_effect[x, y] * 2.0 * (elev / 3000))
                    precip_field[x, y] = base_precip * min(orographic_bonus, 2.5)
                
                # Leeward slopes (rain shadow) get less precipitation
                elif orographic_effect[x, y] < 0:
                    rain_shadow = 1.0 + (orographic_effect[x, y] * 1.5)
                    precip_field[x, y] = base_precip * max(rain_shadow, 0.3)
                
                # High elevations generally get more precipitation
                if elev > 1500:
                    elevation_bonus = 1.0 + ((elev - 1500) / 3000) * 0.5
                    precip_field[x, y] *= elevation_bonus
            else:
                # Ocean - no precipitation data needed
                precip_field[x, y] = 0
    
    # STEP 7: Add coastal moisture effects
    print(f"    - Adding coastal moisture effects...")
    
    # Distance from ocean affects precipitation
    from scipy.ndimage import distance_transform_edt
    
    land_mask = elevation_global > 0
    distance_to_ocean = distance_transform_edt(land_mask)
    
    # Normalize distance
    max_distance = distance_to_ocean.max()
    if max_distance > 0:
        distance_to_ocean = distance_to_ocean / max_distance
    
    # Reduce precipitation in continental interiors (far from ocean)
    # Coastal areas keep full moisture, continental interiors lose up to 40%
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                distance_factor = 1.0 - (distance_to_ocean[x, y] * 0.4)
                precip_field[x, y] *= max(distance_factor, 0.6)
    
    # STEP 8: Smooth final precipitation for realistic patterns
    print(f"    - Smoothing precipitation patterns...")
    
    # Heavy smoothing to eliminate choppiness
    precip_field = gaussian_filter(precip_field, sigma=2.5)
    
    # Light smoothing on temperature for natural variation
    temp_field = gaussian_filter(temp_field, sigma=1.0)
    
    # STEP 9: Store climate data in chunks
    print(f"    - Storing climate data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.temperature_c = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.precipitation_mm = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.temperature_c[local_x, local_y] = temp_field[global_x, global_y]
                        # Clamp precipitation to valid range
                        precip = int(np.clip(precip_field[global_x, global_y], 0, 5000))
                        chunk.precipitation_mm[local_x, local_y] = precip
    
    # STEP 10: Calculate statistics
    land_temps = temp_field[land_mask]
    ocean_temps = temp_field[ocean_mask]
    land_precip = precip_field[land_mask]
    
    if len(land_temps) > 0:
        print(f"  - Temperature statistics:")
        print(f"    Land - Min: {land_temps.min():.1f}°C, Max: {land_temps.max():.1f}°C, Mean: {land_temps.mean():.1f}°C")
        if len(ocean_temps) > 0:
            print(f"    Ocean - Min: {ocean_temps.min():.1f}°C, Max: {ocean_temps.max():.1f}°C, Mean: {ocean_temps.mean():.1f}°C")
    
    if len(land_precip) > 0:
        print(f"  - Precipitation statistics:")
        print(f"    Min: {land_precip.min():.0f} mm/year")
        print(f"    Max: {land_precip.max():.0f} mm/year")
        print(f"    Mean: {land_precip.mean():.0f} mm/year")
    
    print(f"  - Climate patterns complete")