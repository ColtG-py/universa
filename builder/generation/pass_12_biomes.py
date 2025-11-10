"""
World Builder - Pass 12: Biomes & Vegetation (HYBRID VERSION WITH OCEAN SUBTYPES)
Classifies ecological zones and generates vegetation distribution.

IMPROVEMENTS:
- Hybrid approach: uses both absolute thresholds AND percentiles for better variety
- Ocean subtype classification based on depth, tectonics, and temperature
- More distinct biome boundaries
- Better representation of diverse climates

SCIENTIFIC BASIS:
- Whittaker Biome Classification for land biomes
- Ocean depth zones for marine environments
- Coral reef formation requirements (warm, shallow, stable)
- Tectonic influence on ocean trenches
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType, DrainageClass
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate biome classification and vegetation properties using hybrid thresholds.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Classifying biomes and vegetation (hybrid approach with ocean subtypes)...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    # STEP 1: Collect global data
    print(f"    - Collecting environmental data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    temp_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    soil_ph_global = np.zeros((size, size), dtype=np.float32)
    soil_drainage_global = np.zeros((size, size), dtype=np.uint8)
    water_table_global = np.zeros((size, size), dtype=np.float32)
    river_global = np.zeros((size, size), dtype=bool)
    tectonic_stress_global = np.zeros((size, size), dtype=np.float32)
    plate_id_global = np.zeros((size, size), dtype=np.uint8)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.temperature_c is not None:
                temp_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.temperature_c
            if chunk.precipitation_mm is not None:
                precip_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
            if chunk.soil_ph is not None:
                soil_ph_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.soil_ph
            if chunk.soil_drainage is not None:
                soil_drainage_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.soil_drainage
            if chunk.water_table_depth is not None:
                water_table_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.water_table_depth
            if chunk.river_presence is not None:
                river_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
            if chunk.tectonic_stress is not None:
                tectonic_stress_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.tectonic_stress
            if chunk.plate_id is not None:
                plate_id_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.plate_id
    
    # STEP 2: Calculate statistics for hybrid thresholds
    print(f"    - Calculating hybrid thresholds...")
    
    land_mask = elevation_global > 0
    ocean_mask = elevation_global <= 0
    
    # Land statistics
    if land_mask.any():
        land_elevation = elevation_global[land_mask]
        land_temp = temp_global[land_mask]
        land_precip = precip_global[land_mask]
        
        # Elevation percentiles
        elev_p25 = np.percentile(land_elevation, 25)
        elev_p50 = np.percentile(land_elevation, 50)
        elev_p75 = np.percentile(land_elevation, 75)
        elev_p90 = np.percentile(land_elevation, 90)
        elev_p95 = np.percentile(land_elevation, 95)
        elev_p98 = np.percentile(land_elevation, 98)
        elev_p99 = np.percentile(land_elevation, 99)
        
        # Temperature statistics (absolute + percentile)
        temp_mean = land_temp.mean()
        temp_std = land_temp.std()
        temp_p25 = np.percentile(land_temp, 25)
        temp_p50 = np.percentile(land_temp, 50)
        temp_p75 = np.percentile(land_temp, 75)
        
        # Precipitation statistics
        precip_mean = land_precip.mean()
        precip_std = land_precip.std()
        precip_p25 = np.percentile(land_precip, 25)
        precip_p50 = np.percentile(land_precip, 50)
        precip_p75 = np.percentile(land_precip, 75)
        
        print(f"      Land statistics:")
        print(f"        Temperature: {land_temp.min():.1f}°C to {land_temp.max():.1f}°C (mean: {temp_mean:.1f}°C)")
        print(f"        Precipitation: {land_precip.min():.0f}mm to {land_precip.max():.0f}mm (mean: {precip_mean:.0f}mm)")
        print(f"        Elevation: {land_elevation.min():.0f}m to {land_elevation.max():.0f}m (median: {elev_p50:.0f}m)")
    else:
        elev_p25 = elev_p50 = elev_p75 = elev_p90 = elev_p95 = elev_p98 = 0
        temp_mean = temp_std = temp_p25 = temp_p50 = temp_p75 = 0
        precip_mean = precip_std = precip_p25 = precip_p50 = precip_p75 = 0
    
    # Ocean statistics
    if ocean_mask.any():
        ocean_depth = -elevation_global[ocean_mask]  # Convert to positive depth
        ocean_temp = temp_global[ocean_mask]
        
        depth_p25 = np.percentile(ocean_depth, 25)
        depth_p50 = np.percentile(ocean_depth, 50)
        depth_p75 = np.percentile(ocean_depth, 75)
        
        print(f"      Ocean statistics:")
        print(f"        Depth: {ocean_depth.min():.0f}m to {ocean_depth.max():.0f}m (median: {depth_p50:.0f}m)")
        print(f"        Temperature: {ocean_temp.min():.1f}°C to {ocean_temp.max():.1f}°C")
    else:
        depth_p25 = depth_p50 = depth_p75 = 0
    
    # Create thresholds dictionary
    thresholds = {
        # Land thresholds
        'elev_p25': elev_p25,
        'elev_p50': elev_p50,
        'elev_p75': elev_p75,
        'elev_p90': elev_p90,
        'elev_p95': elev_p95,
        'elev_p98': elev_p98,
        'elev_p99': elev_p99,
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'temp_p25': temp_p25,
        'temp_p50': temp_p50,
        'temp_p75': temp_p75,
        'precip_mean': precip_mean,
        'precip_std': precip_std,
        'precip_p25': precip_p25,
        'precip_p50': precip_p50,
        'precip_p75': precip_p75,
        # Ocean thresholds
        'depth_p25': depth_p25,
        'depth_p50': depth_p50,
        'depth_p75': depth_p75,
    }
    
    # STEP 3: Calculate distance to water for riparian zones
    print(f"    - Calculating riparian zone influence...")
    
    water_sources = river_global | ocean_mask
    distance_to_water = distance_transform_edt(~water_sources)
    riparian_influence = np.exp(-distance_to_water / 100.0)
    
    # STEP 4: NO TEMPERATURE ADJUSTMENT - Climate pass already applied lapse rate!
    # Use temperatures and precipitation directly from climate pass
    print(f"    - Using climate data (already elevation-adjusted)...")
    
    # STEP 5: Collect wind data for wind-based biome modifications
    print(f"    - Collecting wind data for biome classification...")
    
    wind_dir_global = np.zeros((size, size), dtype=np.float32)
    wind_speed_global = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.wind_direction is not None:
                wind_dir_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_direction
            if chunk.wind_speed is not None:
                wind_speed_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.wind_speed
    
    # Find plate boundaries using stress
    plate_boundary = tectonic_stress_global > 0.6
    
    # STEP 6: Classify biomes for each chunk
    print(f"    - Classifying biomes using hybrid Whittaker + ocean classification + wind effects...")
    
    # Statistics tracking
    biome_counts = {biome: 0 for biome in BiomeType}
    total_cells = 0
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Initialize arrays
            chunk.biome_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.vegetation_density = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.forest_canopy_height = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.agricultural_suitability = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            offset_x = chunk_x * CHUNK_SIZE
            offset_y = chunk_y * CHUNK_SIZE
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = offset_x + local_x
                    global_y = offset_y + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    elevation = elevation_global[global_x, global_y]
                    temp = temp_global[global_x, global_y]  # Already elevation-adjusted!
                    precip = precip_global[global_x, global_y]  # Already orographic-adjusted!
                    
                    # OCEAN BIOMES
                    if elevation <= 0:
                        depth = -elevation
                        tectonic_stress = tectonic_stress_global[global_x, global_y]
                        
                        biome = classify_ocean_biome(
                            depth, temp, tectonic_stress, thresholds
                        )
                        
                        chunk.biome_type[local_x, local_y] = biome
                        chunk.vegetation_density[local_x, local_y] = 0.0
                        chunk.forest_canopy_height[local_x, local_y] = 0.0
                        chunk.agricultural_suitability[local_x, local_y] = 0.0
                        
                        biome_counts[biome] += 1
                        total_cells += 1
                        continue
                    
                    # LAND BIOMES
                    soil_ph = soil_ph_global[global_x, global_y]
                    drainage = DrainageClass(soil_drainage_global[global_x, global_y])
                    water_table = water_table_global[global_x, global_y]
                    riparian = riparian_influence[global_x, global_y]
                    
                    # Classify using hybrid approach
                    biome = classify_land_biome_hybrid(
                        temp, precip, elevation, thresholds
                    )
                    
                    # Riparian modification
                    if riparian > 0.5 and biome in [BiomeType.HOT_DESERT, BiomeType.COLD_DESERT]:
                        if precip < precip_p50:
                            biome = BiomeType.TEMPERATE_DECIDUOUS_FOREST
                    
                    # WIND-BASED MODIFICATIONS
                    wind_dir = wind_dir_global[global_x, global_y]
                    wind_speed = wind_speed_global[global_x, global_y]
                    distance_to_coast = distance_to_water[global_x, global_y]
                    
                    # Windward coastal effects (moisture-rich winds from ocean)
                    if distance_to_coast < 50 and wind_speed > 4.0:  # Within 50 cells of coast with decent wind
                        # Determine if wind is onshore (from ocean to land)
                        # Check if there's ocean in the upwind direction
                        wind_rad = np.deg2rad(wind_dir)
                        check_dist = 30  # Check 30 cells upwind
                        check_x = int(global_x - np.cos(wind_rad) * check_dist)
                        check_y = int(global_y - np.sin(wind_rad) * check_dist)
                        
                        # Bounds check
                        if 0 <= check_x < size and 0 <= check_y < size:
                            upwind_is_ocean = elevation_global[check_x, check_y] <= 0
                            
                            if upwind_is_ocean:
                                # Onshore winds! Enhance moisture-loving biomes
                                
                                # Tropical coast with onshore winds → tropical rainforest
                                if temp > 20 and biome in [BiomeType.SAVANNA, BiomeType.TROPICAL_SEASONAL_FOREST]:
                                    biome = BiomeType.TROPICAL_RAINFOREST
                                
                                # Temperate coast with strong westerlies → temperate rainforest
                                elif 10 <= temp <= 20 and wind_speed > 6.0:
                                    if biome in [BiomeType.TEMPERATE_DECIDUOUS_FOREST, BiomeType.TEMPERATE_GRASSLAND, BiomeType.MEDITERRANEAN]:
                                        biome = BiomeType.TEMPERATE_RAINFOREST
                                
                                # Cool coast with persistent winds → boreal forest (enhanced)
                                elif 0 < temp < 10 and biome == BiomeType.TEMPERATE_GRASSLAND:
                                    biome = BiomeType.BOREAL_FOREST
                    
                    # Leeward rain shadow enhancement (already dry → extra dry)
                    # Enhance existing deserts if they're in rain shadow
                    if biome in [BiomeType.HOT_DESERT, BiomeType.TEMPERATE_GRASSLAND, BiomeType.MEDITERRANEAN]:
                        if distance_to_coast < 200:  # Close enough to coast to be in potential rain shadow
                            # Check if there's high terrain upwind
                            wind_rad = np.deg2rad(wind_dir)
                            check_dist = 50
                            check_x = int(global_x - np.cos(wind_rad) * check_dist)
                            check_y = int(global_y - np.sin(wind_rad) * check_dist)
                            
                            if 0 <= check_x < size and 0 <= check_y < size:
                                upwind_elev = elevation_global[check_x, check_y]
                                current_elev = elevation_global[global_x, global_y]
                                
                                # If upwind is much higher → rain shadow
                                if upwind_elev > current_elev + 1000:
                                    if temp > 15:
                                        biome = BiomeType.HOT_DESERT
                                    else:
                                        biome = BiomeType.COLD_DESERT
                    
                    # Coastal mangrove detection
                    distance_to_coast = distance_to_water[global_x, global_y]
                    if distance_to_coast < 20 and temp > 20 and elevation < 10:
                        if biome in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TROPICAL_SEASONAL_FOREST]:
                            biome = BiomeType.MANGROVE
                    
                    # Calculate vegetation properties
                    vegetation_density = calculate_vegetation_density(
                        biome, precip, temp, drainage, water_table, riparian, elevation, thresholds
                    )
                    
                    canopy_height = calculate_canopy_height(
                        biome, precip, temp, elevation, thresholds
                    )
                    
                    agricultural_suitability = calculate_agricultural_suitability(
                        biome, temp, precip, soil_ph, drainage, elevation, thresholds
                    )
                    
                    # Store values
                    chunk.biome_type[local_x, local_y] = biome
                    chunk.vegetation_density[local_x, local_y] = vegetation_density
                    chunk.forest_canopy_height[local_x, local_y] = canopy_height
                    chunk.agricultural_suitability[local_x, local_y] = agricultural_suitability
                    
                    biome_counts[biome] += 1
                    total_cells += 1
            
            # Smooth vegetation density
            chunk.vegetation_density = gaussian_filter(chunk.vegetation_density, sigma=1.0)
            chunk.vegetation_density = np.clip(chunk.vegetation_density, 0.0, 1.0)
            
            # Smooth canopy height
            chunk.forest_canopy_height = gaussian_filter(chunk.forest_canopy_height, sigma=0.5)
            
            # Smooth agricultural suitability
            chunk.agricultural_suitability = gaussian_filter(chunk.agricultural_suitability, sigma=1.0)
            chunk.agricultural_suitability = np.clip(chunk.agricultural_suitability, 0.0, 1.0)
    
    # STEP 6: Report statistics
    print(f"  - Biome distribution:")
    
    sorted_biomes = sorted(biome_counts.items(), key=lambda x: x[1], reverse=True)
    
    for biome, count in sorted_biomes:
        if total_cells > 0:
            percentage = (count / total_cells) * 100
            if percentage > 0.1:
                print(f"    {biome.name:35s}: {percentage:5.1f}%")
    
    print(f"  - Biomes and vegetation classified with ocean subtypes")


def classify_ocean_biome(
    depth: float,
    temp: float,
    tectonic_stress: float,
    thresholds: dict
) -> BiomeType:
    """
    Classify ocean biome based on depth, temperature, and tectonic activity.
    
    Args:
        depth: Ocean depth in meters (positive value)
        temp: Water temperature in °C
        tectonic_stress: Tectonic stress level (0-1)
        thresholds: Dictionary of thresholds
    
    Returns:
        Ocean BiomeType
    """
    
    # Coral reefs: warm (>20°C), shallow (<100m), low tectonic stress
    if depth < 100 and temp > 20 and tectonic_stress < 0.4:
        return BiomeType.OCEAN_CORAL_REEF
    
    # Ocean trenches: very deep (>4000m), high tectonic stress
    if depth > 4000 or (depth > 2000 and tectonic_stress > 0.7):
        return BiomeType.OCEAN_TRENCH
    
    # Continental shelf: shallow (<200m)
    if depth < 200:
        return BiomeType.OCEAN_SHELF
    
    # Shallow ocean: 200-1000m
    if depth < 1000:
        return BiomeType.OCEAN_SHALLOW
    
    # Deep ocean: >1000m
    return BiomeType.OCEAN_DEEP


def classify_land_biome_hybrid(
    temp: float,
    precip: float,
    elevation: float,
    thresholds: dict
) -> BiomeType:
    """
    Classify land biome using hybrid absolute + percentile thresholds.
    
    This approach uses:
    - Absolute thresholds for major climate boundaries (freezing, hot, etc.)
    - Percentile thresholds for relative classification within the world
    
    Args:
        temp: Mean annual temperature (°C)
        precip: Mean annual precipitation (mm)
        elevation: Elevation (m)
        thresholds: Dictionary of thresholds
    
    Returns:
        Land BiomeType
    """
    
    elev_p75 = thresholds['elev_p75']
    elev_p90 = thresholds['elev_p90']
    elev_p95 = thresholds['elev_p95']
    elev_p98 = thresholds['elev_p98']
    
    # Calculate a "very high" threshold (99th percentile equivalent)
    # Only the absolute highest elevations should be ice caps
    elev_max = elevation  # We'll use elevation directly for the highest peaks
    
    # ABSOLUTE TEMPERATURE THRESHOLDS (based on real climate science)
    # These are universal and don't change by world
    VERY_COLD = -10  # Only extremely cold areas get permanent ice
    FREEZING = 0
    COLD = 5
    COOL = 10
    MODERATE = 15
    WARM = 20
    HOT = 25
    VERY_HOT = 30
    
    # PRECIPITATION THRESHOLDS (hybrid: absolute + percentile)
    precip_p25 = thresholds['precip_p25']
    precip_p50 = thresholds['precip_p50']
    precip_p75 = thresholds['precip_p75']
    
    # Also use absolute thresholds
    ARID = 250  # mm/year
    SEMI_ARID = 500
    SUBHUMID = 1000
    HUMID = 1500
    VERY_HUMID = 2000
    
    # HIGH ELEVATION BIOMES - MUCH MORE RESTRICTIVE
    # Ice should only form at extreme elevations or extreme cold
    
    if elevation > elev_p98:
        # Top 2% elevation - but not automatically ice
        if temp < VERY_COLD:
            return BiomeType.ICE
        elif temp < FREEZING:
            return BiomeType.ALPINE  # Above treeline but not frozen
        elif temp < COLD:
            if precip > precip_p50:
                return BiomeType.TUNDRA
            else:
                return BiomeType.ALPINE
        else:
            # High but not that cold - can support vegetation
            return BiomeType.ALPINE
    
    elif elevation > elev_p95:
        # 95-98th percentile - alpine/subalpine zones
        if temp < VERY_COLD:
            return BiomeType.ICE
        elif temp < FREEZING:
            return BiomeType.TUNDRA
        elif temp < COLD:
            if precip > precip_p50 or precip > SUBHUMID:
                return BiomeType.BOREAL_FOREST
            else:
                return BiomeType.ALPINE
        elif temp < COOL:
            if precip > precip_p75 or precip > HUMID:
                return BiomeType.TEMPERATE_RAINFOREST
            elif precip > precip_p50 or precip > SUBHUMID:
                return BiomeType.BOREAL_FOREST
            else:
                return BiomeType.ALPINE
        else:
            return BiomeType.ALPINE
    
    elif elevation > elev_p90:
        # 90-95th percentile - montane/subalpine
        if temp < FREEZING:
            return BiomeType.TUNDRA
        elif temp < COOL:
            if precip > precip_p50 or precip > SUBHUMID:
                return BiomeType.BOREAL_FOREST
            else:
                return BiomeType.ALPINE
        elif temp < MODERATE:
            if precip > precip_p75 or precip > HUMID:
                return BiomeType.TEMPERATE_RAINFOREST
            elif precip > precip_p50 or precip > SUBHUMID:
                return BiomeType.TEMPERATE_DECIDUOUS_FOREST
            else:
                return BiomeType.TEMPERATE_GRASSLAND
        else:
            # Warmer montane
            if precip > HUMID:
                return BiomeType.TROPICAL_SEASONAL_FOREST
            elif precip > SUBHUMID:
                return BiomeType.MEDITERRANEAN
            else:
                return BiomeType.TEMPERATE_GRASSLAND
    
    elif elevation > elev_p75:
        # 75-90th percentile - highlands
        if temp < COLD:
            if precip < SEMI_ARID:
                return BiomeType.COLD_DESERT
            else:
                return BiomeType.BOREAL_FOREST
        elif temp < MODERATE:
            if precip > HUMID:
                return BiomeType.TEMPERATE_RAINFOREST
            elif precip > SUBHUMID:
                return BiomeType.TEMPERATE_DECIDUOUS_FOREST
            elif precip > SEMI_ARID:
                return BiomeType.TEMPERATE_GRASSLAND
            else:
                return BiomeType.COLD_DESERT
        else:
            if precip > HUMID:
                return BiomeType.TEMPERATE_DECIDUOUS_FOREST
            elif precip > SUBHUMID:
                return BiomeType.MEDITERRANEAN
            else:
                return BiomeType.TEMPERATE_GRASSLAND
    
    # LOWLAND BIOMES (below 75th percentile elevation)
    # Use Whittaker classification with absolute thresholds
    
    # EXTREMELY COLD (<-10°C mean annual) - Polar ice caps only
    if temp < VERY_COLD:
        return BiomeType.ICE
    
    # VERY COLD (-10°C to 0°C) - Arctic tundra and cold deserts
    elif temp < FREEZING:
        if precip < ARID:
            return BiomeType.COLD_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.TUNDRA
        elif precip < SUBHUMID:
            return BiomeType.TUNDRA
        else:
            # Cold + very wet = tundra, not ice (ice requires extreme cold)
            return BiomeType.TUNDRA
    
    # COLD (0-5°C) - Boreal and tundra
    elif temp < COLD:
        if precip < ARID:
            return BiomeType.COLD_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.TUNDRA
        else:
            return BiomeType.BOREAL_FOREST
    
    # COOL (5-10°C)
    elif temp < COOL:
        if precip < ARID:
            return BiomeType.COLD_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.TEMPERATE_GRASSLAND
        elif precip < HUMID:
            return BiomeType.BOREAL_FOREST
        else:
            return BiomeType.TEMPERATE_RAINFOREST
    
    # MODERATE (10-15°C)
    elif temp < MODERATE:
        if precip < ARID:
            return BiomeType.COLD_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.TEMPERATE_GRASSLAND
        elif precip < SUBHUMID:
            return BiomeType.TEMPERATE_GRASSLAND
        elif precip < HUMID:
            return BiomeType.TEMPERATE_DECIDUOUS_FOREST
        else:
            return BiomeType.TEMPERATE_RAINFOREST
    
    # WARM (15-20°C)
    elif temp < WARM:
        if precip < ARID:
            return BiomeType.HOT_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.TEMPERATE_GRASSLAND
        elif precip < SUBHUMID:
            return BiomeType.MEDITERRANEAN
        elif precip < HUMID:
            return BiomeType.TEMPERATE_DECIDUOUS_FOREST
        else:
            return BiomeType.TROPICAL_SEASONAL_FOREST
    
    # HOT (20-25°C)
    elif temp < HOT:
        if precip < ARID:
            return BiomeType.HOT_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.SAVANNA
        elif precip < SUBHUMID:
            return BiomeType.SAVANNA
        elif precip < HUMID:
            return BiomeType.TROPICAL_SEASONAL_FOREST
        else:
            return BiomeType.TROPICAL_RAINFOREST
    
    # VERY HOT (25-30°C)
    elif temp < VERY_HOT:
        if precip < ARID:
            return BiomeType.HOT_DESERT
        elif precip < SEMI_ARID:
            return BiomeType.SAVANNA
        elif precip < HUMID:
            return BiomeType.TROPICAL_SEASONAL_FOREST
        else:
            return BiomeType.TROPICAL_RAINFOREST
    
    # EXTREMELY HOT (>30°C)
    else:
        if precip < SEMI_ARID:
            return BiomeType.HOT_DESERT
        elif precip < SUBHUMID:
            return BiomeType.SAVANNA
        elif precip < VERY_HUMID:
            return BiomeType.TROPICAL_SEASONAL_FOREST
        else:
            return BiomeType.TROPICAL_RAINFOREST


def calculate_vegetation_density(
    biome: BiomeType,
    precip: float,
    temp: float,
    drainage: DrainageClass,
    water_table: float,
    riparian: float,
    elevation: float,
    thresholds: dict
) -> float:
    """
    Calculate vegetation density (0-1 scale).
    """
    
    # Base density by biome
    base_density = {
        BiomeType.TROPICAL_RAINFOREST: 1.0,
        BiomeType.TROPICAL_SEASONAL_FOREST: 0.85,
        BiomeType.TEMPERATE_RAINFOREST: 0.95,
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 0.8,
        BiomeType.BOREAL_FOREST: 0.7,
        BiomeType.SAVANNA: 0.5,
        BiomeType.TEMPERATE_GRASSLAND: 0.6,
        BiomeType.MEDITERRANEAN: 0.55,
        BiomeType.TUNDRA: 0.3,
        BiomeType.ALPINE: 0.25,
        BiomeType.HOT_DESERT: 0.1,
        BiomeType.COLD_DESERT: 0.15,
        BiomeType.ICE: 0.0,
        BiomeType.MANGROVE: 0.9,
    }.get(biome, 0.0)
    
    if base_density == 0.0:
        return 0.0
    
    # Precipitation modifier (absolute)
    if precip > 1500:
        precip_modifier = 1.1
    elif precip > 1000:
        precip_modifier = 1.0
    elif precip > 500:
        precip_modifier = 0.8
    elif precip > 250:
        precip_modifier = 0.6
    else:
        precip_modifier = 0.4
    
    # Drainage modifier
    if drainage in [DrainageClass.POORLY, DrainageClass.VERY_POORLY]:
        if biome in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_RAINFOREST, BiomeType.MANGROVE]:
            drainage_modifier = 1.0
        else:
            drainage_modifier = 0.7
    elif drainage in [DrainageClass.EXCESSIVELY, DrainageClass.SOMEWHAT_EXCESSIVELY]:
        if biome in [BiomeType.HOT_DESERT, BiomeType.COLD_DESERT]:
            drainage_modifier = 1.0
        else:
            drainage_modifier = 0.8
    else:
        drainage_modifier = 1.0
    
    # Water table boost
    if water_table < 5:
        water_modifier = 1.15
    elif water_table < 20:
        water_modifier = 1.0
    else:
        water_modifier = 0.9
    
    # Riparian boost
    riparian_modifier = 1.0 + (riparian * 0.4)
    
    # Elevation penalty
    elev_p75 = thresholds['elev_p75']
    elev_p90 = thresholds['elev_p90']
    
    if elevation > elev_p90:
        elev_modifier = 0.6
    elif elevation > elev_p75:
        elev_modifier = 0.8
    else:
        elev_modifier = 1.0
    
    density = base_density * precip_modifier * drainage_modifier * water_modifier * riparian_modifier * elev_modifier
    
    return np.clip(density, 0.0, 1.0)


def calculate_canopy_height(
    biome: BiomeType,
    precip: float,
    temp: float,
    elevation: float,
    thresholds: dict
) -> float:
    """
    Calculate forest canopy height in meters.
    """
    
    base_height = {
        BiomeType.TROPICAL_RAINFOREST: 45.0,
        BiomeType.TROPICAL_SEASONAL_FOREST: 30.0,
        BiomeType.TEMPERATE_RAINFOREST: 40.0,
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 25.0,
        BiomeType.BOREAL_FOREST: 20.0,
        BiomeType.MEDITERRANEAN: 12.0,
        BiomeType.SAVANNA: 8.0,
        BiomeType.MANGROVE: 15.0,
    }.get(biome, 0.0)
    
    if base_height == 0.0:
        return 0.0
    
    # Precipitation modifier
    if precip > 2000:
        precip_modifier = 1.2
    elif precip > 1500:
        precip_modifier = 1.1
    elif precip > 1000:
        precip_modifier = 1.0
    elif precip > 500:
        precip_modifier = 0.8
    else:
        precip_modifier = 0.6
    
    # Elevation penalty
    elev_p75 = thresholds['elev_p75']
    elev_p90 = thresholds['elev_p90']
    
    if elevation > elev_p90:
        elev_modifier = 0.5
    elif elevation > elev_p75:
        elev_modifier = 0.7
    else:
        elev_modifier = 1.0
    
    height = base_height * precip_modifier * elev_modifier
    
    return height


def calculate_agricultural_suitability(
    biome: BiomeType,
    temp: float,
    precip: float,
    soil_ph: float,
    drainage: DrainageClass,
    elevation: float,
    thresholds: dict
) -> float:
    """
    Calculate agricultural suitability (0-1 scale).
    """
    
    base_suitability = {
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 0.85,
        BiomeType.TEMPERATE_GRASSLAND: 0.95,
        BiomeType.MEDITERRANEAN: 0.75,
        BiomeType.TROPICAL_SEASONAL_FOREST: 0.65,
        BiomeType.SAVANNA: 0.55,
        BiomeType.BOREAL_FOREST: 0.3,
        BiomeType.TROPICAL_RAINFOREST: 0.35,
        BiomeType.TEMPERATE_RAINFOREST: 0.45,
        BiomeType.TUNDRA: 0.1,
        BiomeType.ALPINE: 0.15,
        BiomeType.HOT_DESERT: 0.2,
        BiomeType.COLD_DESERT: 0.2,
        BiomeType.ICE: 0.0,
        BiomeType.MANGROVE: 0.3,
    }.get(biome, 0.0)
    
    if base_suitability == 0.0:
        return 0.0
    
    # Temperature suitability (absolute)
    if 10 <= temp <= 25:
        temp_modifier = 1.0
    elif 5 <= temp < 10 or 25 < temp <= 30:
        temp_modifier = 0.7
    elif 0 <= temp < 5 or 30 < temp <= 35:
        temp_modifier = 0.4
    else:
        temp_modifier = 0.2
    
    # Precipitation suitability
    if 500 <= precip <= 1500:
        precip_modifier = 1.0
    elif 250 <= precip < 500 or 1500 < precip <= 2000:
        precip_modifier = 0.7
    elif precip < 250:
        precip_modifier = 0.3
    else:
        precip_modifier = 0.5
    
    # Soil pH
    if 6.0 <= soil_ph <= 7.5:
        ph_modifier = 1.0
    elif 5.5 <= soil_ph < 6.0 or 7.5 < soil_ph <= 8.0:
        ph_modifier = 0.8
    else:
        ph_modifier = 0.5
    
    # Drainage
    if drainage == DrainageClass.WELL:
        drainage_modifier = 1.0
    elif drainage in [DrainageClass.MODERATELY_WELL, DrainageClass.SOMEWHAT_EXCESSIVELY]:
        drainage_modifier = 0.9
    else:
        drainage_modifier = 0.5
    
    # Elevation
    elev_p50 = thresholds['elev_p50']
    elev_p75 = thresholds['elev_p75']
    
    if elevation < elev_p50:
        elev_modifier = 1.0
    elif elevation < elev_p75:
        elev_modifier = 0.7
    else:
        elev_modifier = 0.3
    
    suitability = (base_suitability * temp_modifier * precip_modifier * 
                   ph_modifier * drainage_modifier * elev_modifier)
    
    return np.clip(suitability, 0.0, 1.0)