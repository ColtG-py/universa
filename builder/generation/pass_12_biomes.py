"""
World Builder - Pass 12: Biomes & Vegetation (PERCENTILE-BASED VERSION)
Classifies ecological zones and generates vegetation distribution.

IMPROVEMENTS:
- Uses percentiles of actual elevation data instead of hardcoded thresholds
- Adaptive temperature and precipitation classification
- More responsive to different world configurations
- Reduces over-representation of ice/tundra biomes

SCIENTIFIC BASIS:
Uses Whittaker Biome Classification based on temperature and precipitation,
with additional factors including:
- Elevation-based vegetation zones (using percentiles)
- Soil quality and drainage
- Water availability (rivers, water table)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType, DrainageClass
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate biome classification and vegetation properties using adaptive thresholds.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Classifying biomes and vegetation (percentile-based)...")
    
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
    
    # STEP 2: Calculate percentile-based thresholds from actual data
    print(f"    - Calculating adaptive thresholds from data distribution...")
    
    land_mask = elevation_global > 0
    
    # Elevation percentiles (land only)
    land_elevation = elevation_global[land_mask]
    elev_p50 = np.percentile(land_elevation, 50)   # Median elevation
    elev_p75 = np.percentile(land_elevation, 75)   # High elevation
    elev_p90 = np.percentile(land_elevation, 90)   # Very high (montane)
    elev_p95 = np.percentile(land_elevation, 95)   # Alpine
    elev_p98 = np.percentile(land_elevation, 98)   # Subalpine/nival
    elev_p99 = np.percentile(land_elevation, 99)   # Ice caps
    
    print(f"      Elevation percentiles (land):")
    print(f"        P50 (median):     {elev_p50:.1f}m")
    print(f"        P75 (high):       {elev_p75:.1f}m")
    print(f"        P90 (montane):    {elev_p90:.1f}m")
    print(f"        P95 (alpine):     {elev_p95:.1f}m")
    print(f"        P98 (subalpine):  {elev_p98:.1f}m")
    print(f"        P99 (ice caps):   {elev_p99:.1f}m")
    
    # Temperature percentiles (land only)
    land_temp = temp_global[land_mask]
    temp_p10 = np.percentile(land_temp, 10)   # Very cold
    temp_p25 = np.percentile(land_temp, 25)   # Cold
    temp_p50 = np.percentile(land_temp, 50)   # Moderate
    temp_p75 = np.percentile(land_temp, 75)   # Warm
    temp_p90 = np.percentile(land_temp, 90)   # Hot
    
    print(f"      Temperature percentiles (land):")
    print(f"        P10 (very cold):  {temp_p10:.1f}°C")
    print(f"        P25 (cold):       {temp_p25:.1f}°C")
    print(f"        P50 (moderate):   {temp_p50:.1f}°C")
    print(f"        P75 (warm):       {temp_p75:.1f}°C")
    print(f"        P90 (hot):        {temp_p90:.1f}°C")
    
    # Precipitation percentiles (land only)
    land_precip = precip_global[land_mask]
    precip_p25 = np.percentile(land_precip, 25)   # Arid
    precip_p50 = np.percentile(land_precip, 50)   # Semi-arid
    precip_p75 = np.percentile(land_precip, 75)   # Humid
    precip_p90 = np.percentile(land_precip, 90)   # Very humid
    
    print(f"      Precipitation percentiles (land):")
    print(f"        P25 (arid):       {precip_p25:.0f}mm")
    print(f"        P50 (semi-arid):  {precip_p50:.0f}mm")
    print(f"        P75 (humid):      {precip_p75:.0f}mm")
    print(f"        P90 (very humid): {precip_p90:.0f}mm")
    
    # Create thresholds dictionary
    thresholds = {
        'elev_p50': elev_p50,
        'elev_p75': elev_p75,
        'elev_p90': elev_p90,
        'elev_p95': elev_p95,
        'elev_p98': elev_p98,
        'elev_p99': elev_p99,
        'temp_p10': temp_p10,
        'temp_p25': temp_p25,
        'temp_p50': temp_p50,
        'temp_p75': temp_p75,
        'temp_p90': temp_p90,
        'precip_p25': precip_p25,
        'precip_p50': precip_p50,
        'precip_p75': precip_p75,
        'precip_p90': precip_p90,
    }
    
    # STEP 3: Calculate distance to water for riparian zones
    print(f"    - Calculating riparian zone influence...")
    
    water_sources = river_global | ~land_mask
    distance_to_water = distance_transform_edt(~water_sources)
    
    # Normalize distance (riparian influence within ~500m / 100 cells)
    riparian_influence = np.exp(-distance_to_water / 100.0)
    
    # STEP 4: Calculate elevation lapse rates for temperature adjustment
    print(f"    - Calculating elevation-adjusted climate zones...")
    
    # Standard lapse rate: 6.5°C per 1000m
    temp_adjusted = temp_global - (elevation_global * 0.0065)
    
    # Precipitation increases with elevation up to montane zone, then decreases
    precip_adjusted = precip_global.copy()
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                elev = elevation_global[x, y]
                
                # Use percentile-based adjustment
                if elev < elev_p75:
                    # Below P75: gradual increase
                    boost_factor = 1.0 + (elev / elev_p75) * 0.3
                    precip_adjusted[x, y] *= boost_factor
                elif elev < elev_p95:
                    # P75-P95: plateau
                    precip_adjusted[x, y] *= 1.3
                else:
                    # Above P95: rain shadow effect
                    reduction = 1.3 * (1.0 - (elev - elev_p95) / (elev_p99 - elev_p95))
                    reduction = max(reduction, 0.5)
                    precip_adjusted[x, y] *= reduction
    
    # STEP 5: Classify biomes for each chunk
    print(f"    - Classifying biomes using adaptive Whittaker diagram...")
    
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
                    
                    # Ocean - no biome
                    if elevation <= 0:
                        chunk.biome_type[local_x, local_y] = BiomeType.OCEAN
                        chunk.vegetation_density[local_x, local_y] = 0.0
                        chunk.forest_canopy_height[local_x, local_y] = 0.0
                        chunk.agricultural_suitability[local_x, local_y] = 0.0
                        continue
                    
                    # Get environmental factors
                    temp = temp_adjusted[global_x, global_y]
                    precip = precip_adjusted[global_x, global_y]
                    soil_ph = soil_ph_global[global_x, global_y]
                    drainage = DrainageClass(soil_drainage_global[global_x, global_y])
                    water_table = water_table_global[global_x, global_y]
                    riparian = riparian_influence[global_x, global_y]
                    is_river = river_global[global_x, global_y]
                    
                    # Classify biome using adaptive thresholds
                    biome = classify_biome_percentile(
                        temp, precip, elevation, thresholds
                    )
                    
                    # Modify for riparian zones
                    if riparian > 0.5 and biome in [BiomeType.HOT_DESERT, BiomeType.COLD_DESERT, 
                                                     BiomeType.TEMPERATE_GRASSLAND, BiomeType.SAVANNA]:
                        if precip < precip_p50:
                            biome = BiomeType.TEMPERATE_DECIDUOUS_FOREST
                    
                    # Calculate vegetation density
                    vegetation_density = calculate_vegetation_density(
                        biome, precip, temp, drainage, water_table, riparian, elevation, thresholds
                    )
                    
                    # Calculate forest canopy height
                    canopy_height = calculate_canopy_height(
                        biome, precip, temp, elevation, thresholds
                    )
                    
                    # Calculate agricultural suitability
                    agricultural_suitability = calculate_agricultural_suitability(
                        biome, temp, precip, soil_ph, drainage, elevation, thresholds
                    )
                    
                    # Store values
                    chunk.biome_type[local_x, local_y] = biome
                    chunk.vegetation_density[local_x, local_y] = vegetation_density
                    chunk.forest_canopy_height[local_x, local_y] = canopy_height
                    chunk.agricultural_suitability[local_x, local_y] = agricultural_suitability
                    
                    # Track statistics
                    biome_counts[biome] += 1
                    total_cells += 1
            
            # Smooth vegetation density for natural transitions
            chunk.vegetation_density = gaussian_filter(chunk.vegetation_density, sigma=1.0)
            chunk.vegetation_density = np.clip(chunk.vegetation_density, 0.0, 1.0)
            
            # Smooth canopy height
            chunk.forest_canopy_height = gaussian_filter(chunk.forest_canopy_height, sigma=0.5)
            
            # Smooth agricultural suitability
            chunk.agricultural_suitability = gaussian_filter(chunk.agricultural_suitability, sigma=1.0)
            chunk.agricultural_suitability = np.clip(chunk.agricultural_suitability, 0.0, 1.0)
    
    # STEP 6: Report statistics
    print(f"  - Biome distribution:")
    
    # Sort by coverage
    sorted_biomes = sorted(biome_counts.items(), key=lambda x: x[1], reverse=True)
    
    for biome, count in sorted_biomes:
        if total_cells > 0:
            percentage = (count / total_cells) * 100
            if percentage > 0.1:
                print(f"    {biome.name:30s}: {percentage:5.1f}%")
    
    print(f"  - Biomes and vegetation classified (adaptive thresholds)")


def classify_biome_percentile(
    temp: float,
    precip: float,
    elevation: float,
    thresholds: dict
) -> BiomeType:
    """
    Classify biome using Whittaker classification with percentile-based thresholds.
    
    Args:
        temp: Mean annual temperature (°C)
        precip: Mean annual precipitation (mm)
        elevation: Elevation (m)
        thresholds: Dictionary of percentile thresholds
    
    Returns:
        BiomeType enum value
    """
    
    # Extract thresholds
    elev_p90 = thresholds['elev_p90']
    elev_p95 = thresholds['elev_p95']
    elev_p98 = thresholds['elev_p98']
    elev_p99 = thresholds['elev_p99']
    temp_p10 = thresholds['temp_p10']
    temp_p25 = thresholds['temp_p25']
    temp_p50 = thresholds['temp_p50']
    temp_p75 = thresholds['temp_p75']
    temp_p90 = thresholds['temp_p90']
    precip_p25 = thresholds['precip_p25']
    precip_p50 = thresholds['precip_p50']
    precip_p75 = thresholds['precip_p75']
    precip_p90 = thresholds['precip_p90']
    
    # PRIORITY 1: Very high elevations (top 1%) - Ice caps
    if elevation > elev_p99:
        return BiomeType.ICE
    
    # PRIORITY 2: High elevations (top 2-5%) - Alpine/Tundra
    elif elevation > elev_p98:
        if temp < temp_p10:  # Very cold
            return BiomeType.ICE
        else:
            return BiomeType.TUNDRA
    
    # PRIORITY 3: Montane zones (top 5-10%) - Subalpine
    elif elevation > elev_p95:
        if precip > precip_p50:
            return BiomeType.BOREAL_FOREST  # Subalpine forest
        else:
            return BiomeType.TUNDRA  # Alpine tundra
    
    # PRIORITY 4: High elevations but not extreme (top 10-20%)
    # These get cooler biomes but not necessarily alpine
    elif elevation > elev_p90:
        # Apply cooling effect but use temperature-based classification
        if temp < temp_p25:
            if precip > precip_p50:
                return BiomeType.BOREAL_FOREST
            else:
                return BiomeType.TUNDRA
        elif temp < temp_p50:
            if precip > precip_p75:
                return BiomeType.TEMPERATE_RAINFOREST
            elif precip > precip_p50:
                return BiomeType.TEMPERATE_DECIDUOUS_FOREST
            else:
                return BiomeType.TEMPERATE_GRASSLAND
        else:
            # Warmer montane zones
            if precip > precip_p75:
                return BiomeType.TEMPERATE_DECIDUOUS_FOREST
            elif precip > precip_p50:
                return BiomeType.MEDITERRANEAN
            else:
                return BiomeType.TEMPERATE_GRASSLAND
    
    # PRIORITY 5: Standard Whittaker classification for lower elevations
    # Use temperature and precipitation percentiles
    
    # Very cold regions (bottom 10% temperature)
    if temp < temp_p10:
        if precip > precip_p50:
            return BiomeType.TUNDRA
        else:
            return BiomeType.COLD_DESERT
    
    # Cold regions (bottom 10-25% temperature)
    elif temp < temp_p25:
        if precip < precip_p25:
            return BiomeType.COLD_DESERT
        elif precip > precip_p50:
            return BiomeType.BOREAL_FOREST
        else:
            return BiomeType.TUNDRA
    
    # Cool temperate (25-50% temperature)
    elif temp < temp_p50:
        if precip < precip_p25:
            return BiomeType.COLD_DESERT
        elif precip < precip_p50:
            return BiomeType.TEMPERATE_GRASSLAND
        elif precip < precip_p90:
            return BiomeType.TEMPERATE_DECIDUOUS_FOREST
        else:
            return BiomeType.TEMPERATE_RAINFOREST
    
    # Warm temperate (50-75% temperature)
    elif temp < temp_p75:
        if precip < precip_p25:
            return BiomeType.HOT_DESERT
        elif precip < precip_p50:
            return BiomeType.MEDITERRANEAN
        elif precip < precip_p75:
            return BiomeType.TEMPERATE_DECIDUOUS_FOREST
        else:
            return BiomeType.TROPICAL_SEASONAL_FOREST
    
    # Hot regions (75-90% temperature)
    elif temp < temp_p90:
        if precip < precip_p25:
            return BiomeType.HOT_DESERT
        elif precip < precip_p50:
            return BiomeType.SAVANNA
        elif precip < precip_p75:
            return BiomeType.TROPICAL_SEASONAL_FOREST
        else:
            return BiomeType.TROPICAL_RAINFOREST
    
    # Very hot regions (top 10% temperature)
    else:
        if precip < precip_p25:
            return BiomeType.HOT_DESERT
        elif precip < precip_p50:
            return BiomeType.SAVANNA
        elif precip < precip_p90:
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
    Calculate vegetation density (0-1 scale) using adaptive thresholds.
    """
    
    # Base density by biome type
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
        BiomeType.HOT_DESERT: 0.1,
        BiomeType.COLD_DESERT: 0.15,
        BiomeType.ICE: 0.0,
        BiomeType.OCEAN: 0.0,
    }.get(biome, 0.5)
    
    # Modify by precipitation (relative to world)
    precip_p75 = thresholds['precip_p75']
    precip_p50 = thresholds['precip_p50']
    precip_p25 = thresholds['precip_p25']
    
    if precip > precip_p75:
        precip_modifier = 1.0
    elif precip > precip_p50:
        precip_modifier = 0.9
    elif precip > precip_p25:
        precip_modifier = 0.7
    else:
        precip_modifier = 0.5
    
    # Drainage modifier
    if drainage in [DrainageClass.POORLY, DrainageClass.VERY_POORLY]:
        if biome not in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_RAINFOREST]:
            drainage_modifier = 0.7
        else:
            drainage_modifier = 1.0
    elif drainage in [DrainageClass.EXCESSIVELY, DrainageClass.SOMEWHAT_EXCESSIVELY]:
        if biome in [BiomeType.HOT_DESERT, BiomeType.COLD_DESERT]:
            drainage_modifier = 1.0
        else:
            drainage_modifier = 0.8
    else:
        drainage_modifier = 1.0
    
    # Water table modifier
    if water_table < 5:
        water_modifier = 1.1
    elif water_table < 20:
        water_modifier = 1.0
    else:
        water_modifier = 0.95
    
    # Riparian enhancement
    riparian_modifier = 1.0 + (riparian * 0.3)
    
    # Elevation modifier (using percentiles)
    elev_p90 = thresholds['elev_p90']
    elev_p75 = thresholds['elev_p75']
    
    if elevation > elev_p90:
        elev_modifier = 0.5
    elif elevation > elev_p75:
        elev_modifier = 0.7
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
    Calculate forest canopy height in meters using adaptive thresholds.
    """
    
    # Base height by biome
    base_height = {
        BiomeType.TROPICAL_RAINFOREST: 40.0,
        BiomeType.TROPICAL_SEASONAL_FOREST: 30.0,
        BiomeType.TEMPERATE_RAINFOREST: 35.0,
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 25.0,
        BiomeType.BOREAL_FOREST: 20.0,
        BiomeType.MEDITERRANEAN: 10.0,
        BiomeType.SAVANNA: 8.0,
    }.get(biome, 0.0)
    
    if base_height == 0.0:
        return 0.0
    
    # Precipitation modifier (relative)
    precip_p90 = thresholds['precip_p90']
    precip_p75 = thresholds['precip_p75']
    precip_p50 = thresholds['precip_p50']
    
    if precip > precip_p90:
        precip_modifier = 1.2
    elif precip > precip_p75:
        precip_modifier = 1.1
    elif precip > precip_p50:
        precip_modifier = 1.0
    else:
        precip_modifier = 0.7
    
    # Elevation modifier (using percentiles)
    elev_p90 = thresholds['elev_p90']
    elev_p75 = thresholds['elev_p75']
    elev_p50 = thresholds['elev_p50']
    
    if elevation > elev_p90:
        elev_modifier = 0.5
    elif elevation > elev_p75:
        elev_modifier = 0.7
    elif elevation > elev_p50:
        elev_modifier = 0.9
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
    Calculate agricultural suitability (0-1 scale) using adaptive thresholds.
    """
    
    # Base suitability by biome
    base_suitability = {
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 0.8,
        BiomeType.TEMPERATE_GRASSLAND: 0.9,
        BiomeType.MEDITERRANEAN: 0.7,
        BiomeType.TROPICAL_SEASONAL_FOREST: 0.6,
        BiomeType.SAVANNA: 0.5,
        BiomeType.BOREAL_FOREST: 0.3,
        BiomeType.TROPICAL_RAINFOREST: 0.3,
        BiomeType.TEMPERATE_RAINFOREST: 0.4,
        BiomeType.TUNDRA: 0.1,
        BiomeType.HOT_DESERT: 0.2,
        BiomeType.COLD_DESERT: 0.2,
        BiomeType.ICE: 0.0,
        BiomeType.OCEAN: 0.0,
    }.get(biome, 0.3)
    
    # Temperature suitability (use percentiles for ideal range)
    temp_p25 = thresholds['temp_p25']
    temp_p75 = thresholds['temp_p75']
    temp_p10 = thresholds['temp_p10']
    temp_p90 = thresholds['temp_p90']
    
    # Ideal: middle 50% of temperature range
    if temp_p25 <= temp <= temp_p75:
        temp_modifier = 1.0
    elif temp_p10 <= temp < temp_p25 or temp_p75 < temp <= temp_p90:
        temp_modifier = 0.7
    else:
        temp_modifier = 0.3
    
    # Precipitation suitability (use percentiles)
    precip_p25 = thresholds['precip_p25']
    precip_p75 = thresholds['precip_p75']
    
    # Ideal: middle 50% of precipitation range
    if precip_p25 <= precip <= precip_p75:
        precip_modifier = 1.0
    elif precip < precip_p25:
        precip_modifier = 0.5  # Too dry
    else:
        precip_modifier = 0.7  # Too wet
    
    # Soil pH suitability (still use absolute values as these are universal)
    if 6.0 <= soil_ph <= 7.5:
        ph_modifier = 1.0
    elif 5.5 <= soil_ph < 6.0 or 7.5 < soil_ph <= 8.0:
        ph_modifier = 0.8
    elif 5.0 <= soil_ph < 5.5 or 8.0 < soil_ph <= 8.5:
        ph_modifier = 0.6
    else:
        ph_modifier = 0.3
    
    # Drainage suitability
    if drainage == DrainageClass.WELL:
        drainage_modifier = 1.0
    elif drainage in [DrainageClass.MODERATELY_WELL, DrainageClass.SOMEWHAT_EXCESSIVELY]:
        drainage_modifier = 0.9
    elif drainage in [DrainageClass.SOMEWHAT_POORLY, DrainageClass.EXCESSIVELY]:
        drainage_modifier = 0.6
    else:
        drainage_modifier = 0.3
    
    # Elevation suitability (using percentiles)
    elev_p50 = thresholds['elev_p50']
    elev_p75 = thresholds['elev_p75']
    elev_p90 = thresholds['elev_p90']
    
    if elevation < elev_p50:
        elev_modifier = 1.0  # Lowlands are best
    elif elevation < elev_p75:
        elev_modifier = 0.8
    elif elevation < elev_p90:
        elev_modifier = 0.5
    else:
        elev_modifier = 0.2  # High mountains are poor
    
    suitability = (base_suitability * temp_modifier * precip_modifier * 
                   ph_modifier * drainage_modifier * elev_modifier)
    
    return np.clip(suitability, 0.0, 1.0)