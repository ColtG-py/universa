"""
World Builder - Pass 7: Climate Simulation (FIXED - Accurate Temperature & Varied Precipitation)
Calculates temperature and precipitation based on latitude, elevation, and wind.

FIXES:
- Proper temperature gradient (hot equator, cold poles)
- Lapse rate applied ONCE (not double-applied)
- Much more varied precipitation based on:
  * Latitude zones (ITCZ, horse latitudes, westerlies, polar)
  * Orographic effects (windward = wet, leeward = dry)
  * Continental vs coastal (moisture source distance)
  * Wind direction and strength
- Less aggressive smoothing to preserve variation
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

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
    wind_speed_global = np.zeros((size, size), dtype=np.float32)
    
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
            if chunk.wind_speed is not None:
                wind_speed_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_speed
    
    # STEP 2: Generate climate noise for variability with MULTIPLE SCALES
    print(f"    - Generating climate variation patterns with strong noise influence...")
    
    # Temperature variation - multiple octaves for complex patterns
    temp_noise_gen = NoiseGenerator(
        seed=params.seed + 7000,
        octaves=5,  # More octaves for finer detail
        persistence=0.6,  # Higher persistence for stronger small-scale features
        lacunarity=2.0,
        scale=size / 4.0  # Smaller scale for more variation
    )
    
    # Precipitation variation - even more complex
    precip_noise_gen = NoiseGenerator(
        seed=params.seed + 8000,
        octaves=7,  # Many octaves for very detailed variation
        persistence=0.6,
        lacunarity=2.2,
        scale=size / 3.0  # More variation
    )
    
    # Generate base noise
    temp_variation = temp_noise_gen.generate_perlin_2d(size, size, 0, 0, normalize=False)
    precip_variation = precip_noise_gen.generate_perlin_2d(size, size, 0, 0, normalize=True)
    
    # Add turbulence/domain warping for more organic patterns
    # This creates swirling, natural-looking patterns instead of regular noise
    turbulence_gen = NoiseGenerator(
        seed=params.seed + 9000,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        scale=size / 2.0
    )
    
    turbulence_x = turbulence_gen.generate_perlin_2d(size, size, 0, 0, normalize=False) * 20.0
    turbulence_y = turbulence_gen.generate_perlin_2d(size, size, 100, 100, normalize=False) * 20.0
    
    # Apply turbulence to warp the precipitation variation (creates swirls)
    precip_variation_warped = np.zeros_like(precip_variation)
    for y in range(size):
        for x in range(size):
            # Warp coordinates
            warped_x = int(np.clip(x + turbulence_x[x, y], 0, size - 1))
            warped_y = int(np.clip(y + turbulence_y[x, y], 0, size - 1))
            precip_variation_warped[x, y] = precip_variation[warped_x, warped_y]
    
    precip_variation = precip_variation_warped
    
    print(f"    - Noise patterns generated with turbulence for organic variation")
    
    # STEP 3: Calculate base temperature from latitude (FIXED)
    print(f"    - Calculating latitude-based climate zones...")
    
    latitude = np.linspace(-90, 90, size)
    latitude_field = np.tile(latitude[np.newaxis, :], (size, 1))
    
    # FIXED: Proper temperature gradient
    # Earth-like: Equator ~28°C, Poles ~-20°C = 48°C range
    # Use cosine for more realistic distribution
    lat_rad = np.deg2rad(latitude_field)
    lat_temp_factor = (np.cos(lat_rad) + 1) / 2  # 1.0 at equator, 0.0 at poles
    
    # Temperature range from equator to pole
    equator_to_pole_range = 48.0  # °C
    
    # Equatorial temperature is hotter than base
    equator_temp = params.base_temperature_c + 13.0  # For base=15°C → equator=28°C
    pole_temp = equator_temp - equator_to_pole_range  # pole = -20°C
    
    # Apply gradient (smooth transition from equator to poles)
    base_temp_field = pole_temp + lat_temp_factor * equator_to_pole_range
    
    # Add regional temperature variation (oceanic vs continental patterns)
    # This creates visible regional differences beyond pure latitude
    regional_temp_noise = NoiseGenerator(
        seed=params.seed + 7500,
        octaves=7,
        persistence=0.5,
        lacunarity=2.0,
        scale=size / 24.0  # Large-scale regional patterns
    )
    
    regional_temp_variation = regional_temp_noise.generate_perlin_2d(size, size, 0, 0, normalize=False)
    
    # Apply regional variation (±5°C regional differences)
    base_temp_field = base_temp_field + (regional_temp_variation * 5.0)
    
    print(f"    - Sea-level temperature range: {equator_temp:.1f}°C (equator) to {pole_temp:.1f}°C (poles)")
    
    # STEP 4: Calculate precipitation zones by latitude (MORE VARIED)
    print(f"    - Calculating precipitation zones...")
    
    base_precip_field = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        lat = latitude[y]
        abs_lat = abs(lat)
        
        # More varied precipitation zones
        if abs_lat < 5:
            # Equatorial maximum (ITCZ)
            base_precip = 2800
        elif abs_lat < 15:
            # Tropical wet (still in ITCZ influence)
            t = (abs_lat - 5) / 10
            base_precip = 2800 - t * 1300  # 2800 → 1500
        elif abs_lat < 25:
            # Transitioning to subtropical high
            t = (abs_lat - 15) / 10
            base_precip = 1500 - t * 1100  # 1500 → 400
        elif abs_lat < 35:
            # Subtropical high (horse latitudes) - DESERTS
            base_precip = 300 + np.sin((abs_lat - 25) * np.pi / 10) * 200  # 300-500 range
        elif abs_lat < 45:
            # Transition to mid-latitude westerlies
            t = (abs_lat - 35) / 10
            base_precip = 500 + t * 800  # 500 → 1300
        elif abs_lat < 60:
            # Mid-latitude westerlies - STORMY
            t = (abs_lat - 45) / 15
            base_precip = 1300 + t * 400  # 1300 → 1700
        elif abs_lat < 70:
            # Transition to polar dry
            t = (abs_lat - 60) / 10
            base_precip = 1700 - t * 1300  # 1700 → 400
        else:
            # Polar regions - DRY (cold air holds less moisture)
            base_precip = 300 - (abs_lat - 70) * 5  # 300 → 200
        
        base_precip_field[:, y] = base_precip
    
    # STEP 5: Apply strong noise variation to precipitation (break up latitude bands!)
    print(f"    - Adding strong precipitation variability to break up latitude patterns...")
    
    # MUCH stronger variation: ±70% modulation instead of ±50%
    # This will create visible swirls and patterns that override the latitude gradient
    precip_field = base_precip_field * (0.3 + precip_variation * 1.4)
    
    # STEP 6: Calculate temperature with elevation (LAPSE RATE - ONLY ONCE!)
    print(f"    - Applying elevation lapse rate to temperature...")
    
    temp_field = np.zeros((size, size), dtype=np.float32)
    ocean_mask = elevation_global <= 0
    land_mask = elevation_global > 0
    
    # Standard atmospheric lapse rate: 6.5°C per 1000m
    LAPSE_RATE = 0.0065  # °C per meter
    
    for y in range(size):
        for x in range(size):
            elev = elevation_global[x, y]
            base_temp = base_temp_field[x, y]
            
            if elev <= 0:
                # Ocean - very stable temperatures, slight depth effect
                depth = abs(elev)
                # Shallow water warms/cools with air, deep water is stable
                if depth < 200:
                    temp_modifier = 0  # Shallow follows air temp
                else:
                    # Deep ocean: slight cooling, more stable
                    temp_modifier = -min(depth / 1000, 2.0)  # Max -2°C
                
                temp = base_temp + temp_modifier
            else:
                # Land - apply lapse rate
                # Higher elevations are cooler
                cooling = elev * LAPSE_RATE
                temp = base_temp - cooling
            
            # Add climate variation noise (±6°C variability for weather/microclimate)
            # This breaks up the latitude bands significantly
            temp += temp_variation[x, y] * 6.0  # Increased from 4.0
            
            temp_field[x, y] = temp
    
    # STEP 7: Calculate orographic effects (windward wet, leeward dry)
    print(f"    - Calculating orographic precipitation effects...")
    
    # Calculate terrain gradients
    from utils.spatial import calculate_gradient
    
    grad_x, grad_y = calculate_gradient(elevation_global)
    slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Wind direction as unit vectors
    wind_dir_rad = np.deg2rad(wind_dir_global)
    wind_u = np.cos(wind_dir_rad)
    wind_v = np.sin(wind_dir_rad)
    
    # Orographic effect: dot product of wind and slope
    # Positive = windward (upslope wind), Negative = leeward (downslope wind)
    orographic_effect = -(grad_x * wind_u + grad_y * wind_v)
    
    # Weight by wind speed and slope
    orographic_effect = orographic_effect * wind_speed_global * (slope_magnitude + 0.1)
    
    # Smooth slightly but preserve variation
    orographic_effect = gaussian_filter(orographic_effect, sigma=1.5)
    
    # Normalize
    if orographic_effect.max() > 0:
        orographic_effect = orographic_effect / orographic_effect.max()
    
    # Apply to precipitation
    orographic_precip = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                elev = elevation_global[x, y]
                base_precip = precip_field[x, y]
                oro_effect = orographic_effect[x, y]
                
                # Windward slopes (positive effect)
                if oro_effect > 0.1:
                    # Strong orographic enhancement
                    # Higher elevations = stronger effect
                    elevation_factor = min(elev / 3000, 1.0)
                    enhancement = 1.0 + oro_effect * 3.0 * elevation_factor
                    orographic_precip[x, y] = base_precip * min(enhancement, 4.0)
                
                # Leeward slopes (negative effect) - rain shadow
                elif oro_effect < -0.1:
                    # Rain shadow reduction
                    reduction = 1.0 + oro_effect * 2.0  # Reduces by up to 2x
                    orographic_precip[x, y] = base_precip * max(reduction, 0.2)
                
                else:
                    # Neutral slopes
                    orographic_precip[x, y] = base_precip
                
                # Additional elevation bonus (mountains catch more moisture)
                if elev > 1000:
                    elevation_bonus = 1.0 + (elev - 1000) / 4000  # Up to 2x at 5000m
                    orographic_precip[x, y] *= min(elevation_bonus, 2.0)
            else:
                # Ocean - no precipitation recorded
                orographic_precip[x, y] = 0
    
    # STEP 8: Add coastal moisture gradient
    print(f"    - Adding coastal moisture effects...")
    
    # Distance from ocean affects precipitation
    distance_to_ocean = distance_transform_edt(land_mask)
    
    # Normalize
    max_distance = distance_to_ocean.max()
    if max_distance > 0:
        distance_to_ocean_norm = distance_to_ocean / max_distance
    else:
        distance_to_ocean_norm = np.zeros_like(distance_to_ocean)
    
    # Continental interior effect (reduce precipitation)
    coastal_precip = orographic_precip.copy()
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                # Coastal: full moisture
                # Interior: reduced by up to 50%
                distance_factor = 1.0 - (distance_to_ocean_norm[x, y] ** 1.5) * 0.5
                coastal_precip[x, y] *= max(distance_factor, 0.5)
    
    # STEP 9: Smooth precipitation VERY LIGHTLY (preserve variation!)
    print(f"    - Final light smoothing (preserving natural variation)...")
    
    # VERY light smoothing - just enough to remove pixel-level noise
    precip_field = gaussian_filter(coastal_precip, sigma=1.0)  # Reduced from 1.5
    
    # Very light smoothing on temperature as well
    temp_field = gaussian_filter(temp_field, sigma=0.8)  # Reduced from 1.0
    
    # STEP 10: Store climate data in chunks
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
                        precip = int(np.clip(precip_field[global_x, global_y], 0, 6000))
                        chunk.precipitation_mm[local_x, local_y] = precip
    
    # STEP 11: Calculate statistics
    land_temps = temp_field[land_mask]
    ocean_temps = temp_field[ocean_mask]
    land_precip = precip_field[land_mask]
    
    if len(land_temps) > 0:
        print(f"  - Temperature statistics:")
        print(f"    Land  - Min: {land_temps.min():.1f}°C, Max: {land_temps.max():.1f}°C, Mean: {land_temps.mean():.1f}°C")
        if len(ocean_temps) > 0:
            print(f"    Ocean - Min: {ocean_temps.min():.1f}°C, Max: {ocean_temps.max():.1f}°C, Mean: {ocean_temps.mean():.1f}°C")
    
    if len(land_precip) > 0:
        print(f"  - Precipitation statistics:")
        print(f"    Min:  {land_precip.min():.0f} mm/year")
        print(f"    Max:  {land_precip.max():.0f} mm/year")
        print(f"    Mean: {land_precip.mean():.0f} mm/year")
        print(f"    Std:  {land_precip.std():.0f} mm/year")
    
    print(f"  - Climate patterns complete with realistic variation")