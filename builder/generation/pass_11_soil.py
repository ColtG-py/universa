"""
World Builder - Pass 11: Soil Formation (SCIENTIFIC VERSION)
Generates soil types, pH, and drainage based on the five soil-forming factors.

SCIENTIFIC BASIS:
Soil formation is governed by five factors (Hans Jenny, 1941):
1. Climate (temperature, precipitation)
2. Parent material (bedrock type)
3. Topography (elevation, slope)
4. Organisms (vegetation - simplified here)
5. Time (assumed uniform for this world)

SOIL ORDERS GENERATED:
- Aridisols: Dry climates (< 400mm precipitation)
- Mollisols: Grasslands (moderate precipitation, flat terrain)
- Alfisols: Temperate forests (moderate-high precipitation)
- Spodosols: Cool moist coniferous forests (high precipitation, cold)
- Ultisols: Warm humid regions (high precipitation, warm)
- Entisols: Young soils (steep slopes, floodplains, recent formation)
- Inceptisols: Weakly developed soils

pH DETERMINATION:
- High precipitation → Acidic (4-6) from leaching
- Low precipitation → Alkaline (7-9) from carbonate accumulation
- Parent material effects (limestone = alkaline, granite = acidic)
- Temperature effects on weathering rate

DRAINAGE CLASSES:
- Based on slope, water table depth, soil texture, and precipitation
- Seven classes from excessively drained to very poorly drained
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config import WorldGenerationParams, CHUNK_SIZE, SoilType, DrainageClass, RockType
from models.world import WorldState
from utils.spatial import calculate_slope


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate realistic soil properties based on soil-forming factors.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating soil properties based on climate and terrain...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    seed = params.seed
    
    # STEP 1: Collect global data for soil formation
    print(f"    - Collecting environmental data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    temp_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    bedrock_global = np.zeros((size, size), dtype=np.uint8)
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
            if chunk.bedrock_type is not None:
                bedrock_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.bedrock_type
            if chunk.water_table_depth is not None:
                water_table_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.water_table_depth
            if chunk.river_presence is not None:
                river_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
    
    # STEP 2: Calculate slope from elevation
    print(f"    - Calculating slope for topographic analysis...")
    
    slope_global = calculate_slope(elevation_global)
    
    # STEP 3: Determine climate zones for soil order classification
    print(f"    - Classifying climate zones...")
    
    land_mask = elevation_global > 0
    
    # Calculate precipitation statistics for thresholds
    land_precip = precip_global[land_mask]
    precip_p25 = np.percentile(land_precip, 25)  # Arid threshold
    precip_p50 = np.percentile(land_precip, 50)  # Semi-arid threshold
    precip_p75 = np.percentile(land_precip, 75)  # Humid threshold
    
    print(f"      Precipitation zones:")
    print(f"        Arid (<{precip_p25:.0f} mm): {(land_precip < precip_p25).sum() / len(land_precip) * 100:.1f}% of land")
    print(f"        Semi-arid ({precip_p25:.0f}-{precip_p50:.0f} mm): {((land_precip >= precip_p25) & (land_precip < precip_p50)).sum() / len(land_precip) * 100:.1f}% of land")
    print(f"        Sub-humid ({precip_p50:.0f}-{precip_p75:.0f} mm): {((land_precip >= precip_p50) & (land_precip < precip_p75)).sum() / len(land_precip) * 100:.1f}% of land")
    print(f"        Humid (>{precip_p75:.0f} mm): {(land_precip >= precip_p75).sum() / len(land_precip) * 100:.1f}% of land")
    
    # STEP 4: Generate soil for each chunk
    print(f"    - Generating soil types, pH, and drainage...")
    
    rng = np.random.default_rng(seed + 11000)
    
    # Statistics tracking
    soil_type_counts = {st: 0 for st in SoilType}
    drainage_counts = {dc: 0 for dc in DrainageClass}
    total_cells = 0
    ph_values = []
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Initialize arrays
            chunk.soil_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.soil_ph = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.soil_drainage = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            
            offset_x = chunk_x * CHUNK_SIZE
            offset_y = chunk_y * CHUNK_SIZE
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = offset_x + local_x
                    global_y = offset_y + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    elevation = elevation_global[global_x, global_y]
                    
                    # Ocean - no soil
                    if elevation <= 0:
                        chunk.soil_type[local_x, local_y] = SoilType.CLAY
                        chunk.soil_ph[local_x, local_y] = 7.0
                        chunk.soil_drainage[local_x, local_y] = DrainageClass.WELL
                        continue
                    
                    # Get environmental factors
                    temp = temp_global[global_x, global_y]
                    precip = precip_global[global_x, global_y]
                    bedrock = RockType(bedrock_global[global_x, global_y])
                    slope = slope_global[global_x, global_y]
                    water_table = water_table_global[global_x, global_y]
                    is_river = river_global[global_x, global_y]
                    
                    # ===================================================================
                    # DETERMINE SOIL TYPE based on climate and topography
                    # ===================================================================
                    
                    soil_type = determine_soil_type(
                        precip, temp, elevation, slope, bedrock, is_river,
                        precip_p25, precip_p50, precip_p75, rng
                    )
                    
                    # ===================================================================
                    # DETERMINE SOIL pH based on climate and parent material
                    # ===================================================================
                    
                    soil_ph = determine_soil_ph(
                        precip, temp, bedrock, soil_type,
                        precip_p25, precip_p75, rng
                    )
                    
                    # ===================================================================
                    # DETERMINE DRAINAGE CLASS based on topography and hydrology
                    # ===================================================================
                    
                    drainage = determine_drainage_class(
                        slope, water_table, soil_type, precip, is_river
                    )
                    
                    # Store values
                    chunk.soil_type[local_x, local_y] = soil_type
                    chunk.soil_ph[local_x, local_y] = soil_ph
                    chunk.soil_drainage[local_x, local_y] = drainage
                    
                    # Track statistics
                    soil_type_counts[soil_type] += 1
                    drainage_counts[drainage] += 1
                    ph_values.append(soil_ph)
                    total_cells += 1
            
            # Smooth pH slightly for natural variation
            chunk.soil_ph = gaussian_filter(chunk.soil_ph, sigma=0.5)
            
            # Clamp pH to realistic range
            chunk.soil_ph = np.clip(chunk.soil_ph, 3.5, 9.5)
    
    # STEP 5: Report statistics
    print(f"  - Soil generation complete:")
    print(f"    Soil type distribution:")
    for soil_type, count in soil_type_counts.items():
        if total_cells > 0:
            percentage = (count / total_cells) * 100
            if percentage > 0.1:  # Only show types with >0.1% coverage
                print(f"      {soil_type.name:15s}: {percentage:5.1f}%")
    
    print(f"    Drainage class distribution:")
    for drainage, count in drainage_counts.items():
        if total_cells > 0:
            percentage = (count / total_cells) * 100
            if percentage > 0.1:
                print(f"      {drainage.name:25s}: {percentage:5.1f}%")
    
    if ph_values:
        ph_array = np.array(ph_values)
        print(f"    pH statistics:")
        print(f"      Min: {ph_array.min():.2f}")
        print(f"      Max: {ph_array.max():.2f}")
        print(f"      Mean: {ph_array.mean():.2f}")
        print(f"      Median: {np.median(ph_array):.2f}")


def determine_soil_type(
    precip: float,
    temp: float,
    elevation: float,
    slope: float,
    bedrock: RockType,
    is_river: bool,
    precip_p25: float,
    precip_p50: float,
    precip_p75: float,
    rng: np.random.Generator
) -> SoilType:
    """
    Determine soil type based on climate and topography.
    
    Uses USDA Soil Taxonomy principles to classify soils.
    """
    
    # PRIORITY 1: Floodplain/Alluvial soils (recent deposition)
    if is_river and slope < 0.05:
        # River valleys and floodplains - Entisols (young, recent alluvium)
        return SoilType.SILT_LOAM  # Alluvial deposits
    
    # PRIORITY 2: Steep slopes - Entisols (erosion prevents development)
    if slope > 0.3:  # > 30% slope
        # Steep terrain - poorly developed soils
        if bedrock == RockType.IGNEOUS:
            return SoilType.SANDY_LOAM
        else:
            return SoilType.LOAM
    
    # PRIORITY 3: Climate-based soil orders
    
    # ARIDISOLS - Desert soils (< 25th percentile precipitation)
    if precip < precip_p25:
        # Dry climate - limited weathering
        if temp > 20:
            return SoilType.SANDY_CLAY_LOAM  # Hot desert
        else:
            return SoilType.LOAM  # Cold desert
    
    # MOLLISOLS - Grassland soils (25th to 50th percentile, moderate temp)
    elif precip < precip_p50:
        if 0 < temp < 20 and slope < 0.15:  # Temperate grasslands, gentle slopes
            return SoilType.SILT_LOAM  # Dark, fertile grassland soil
        else:
            return SoilType.CLAY_LOAM
    
    # SPODOSOLS - Cool, moist forest soils (high precip, cool temp)
    elif precip >= precip_p75 and temp < 10:
        # Cool, humid climate - leached acidic soils
        return SoilType.SANDY_LOAM  # Leached, coarse texture
    
    # ULTISOLS - Warm, humid forest soils (high precip, warm temp)
    elif precip >= precip_p75 and temp > 20:
        # Warm, humid climate - highly weathered
        return SoilType.CLAY  # Highly weathered, clay-rich
    
    # ALFISOLS - Temperate forest soils (moderate precip and temp)
    elif precip_p50 <= precip < precip_p75 and 10 <= temp <= 20:
        # Temperate forest - moderate development
        return SoilType.LOAM
    
    # INCEPTISOLS - Weakly developed (default)
    else:
        # Young or weakly developed soils
        if bedrock == RockType.LIMESTONE:
            return SoilType.CLAY_LOAM
        elif bedrock == RockType.IGNEOUS:
            return SoilType.SANDY_LOAM
        else:
            return SoilType.LOAM


def determine_soil_ph(
    precip: float,
    temp: float,
    bedrock: RockType,
    soil_type: SoilType,
    precip_p25: float,
    precip_p75: float,
    rng: np.random.Generator
) -> float:
    """
    Determine soil pH based on climate and parent material.
    
    Key principles:
    - High precipitation → acidic (leaching of base cations)
    - Low precipitation → alkaline (carbonate accumulation)
    - Limestone parent material → alkaline
    - Granite/sandstone → acidic
    - Temperature affects weathering rate
    """
    
    # Base pH from parent material
    if bedrock == RockType.LIMESTONE:
        base_ph = 7.5  # Calcareous, alkaline
    elif bedrock == RockType.IGNEOUS:
        base_ph = 5.5  # Granite-like, acidic
    elif bedrock == RockType.METAMORPHIC:
        base_ph = 6.5  # Intermediate
    else:  # SEDIMENTARY
        base_ph = 6.8  # Neutral to slightly alkaline
    
    # Precipitation effect (most important factor)
    if precip < precip_p25:
        # Arid - alkaline soils (carbonate accumulation)
        ph_adjustment = +1.5  # pH 7-9
    elif precip > precip_p75:
        # Humid - acidic soils (leaching)
        ph_adjustment = -1.5  # pH 4-6
    else:
        # Intermediate
        # Linear interpolation between arid and humid
        precip_range = precip_p75 - precip_p25
        precip_position = (precip - precip_p25) / precip_range  # 0 to 1
        ph_adjustment = 1.5 - (3.0 * precip_position)  # +1.5 to -1.5
    
    # Temperature effect (accelerates weathering)
    if temp > 25:
        # Hot climate - faster weathering, more leaching
        ph_adjustment -= 0.3
    elif temp < 5:
        # Cold climate - slower weathering
        ph_adjustment += 0.2
    
    # Soil type effect (texture influences leaching)
    if soil_type == SoilType.SAND or soil_type == SoilType.SANDY_LOAM:
        # Sandy soils leach more easily
        ph_adjustment -= 0.3
    elif soil_type == SoilType.CLAY:
        # Clay soils retain more bases
        ph_adjustment += 0.2
    
    # Calculate final pH
    soil_ph = base_ph + ph_adjustment
    
    # Add small random variation
    soil_ph += rng.normal(0, 0.2)
    
    # Clamp to realistic range
    soil_ph = np.clip(soil_ph, 3.5, 9.5)
    
    return soil_ph


def determine_drainage_class(
    slope: float,
    water_table: float,
    soil_type: SoilType,
    precip: float,
    is_river: bool
) -> DrainageClass:
    """
    Determine drainage class based on topography and hydrology.
    
    Key factors:
    - Slope (steep = better drainage)
    - Water table depth (shallow = poor drainage)
    - Soil texture (sandy = better drainage, clay = poor drainage)
    - Precipitation (high = wetter conditions)
    """
    
    # Calculate texture permeability factor
    if soil_type in [SoilType.SAND, SoilType.SANDY_LOAM]:
        texture_factor = 2.0  # Very permeable
    elif soil_type in [SoilType.LOAM, SoilType.SILT_LOAM]:
        texture_factor = 1.0  # Moderate permeability
    elif soil_type in [SoilType.CLAY_LOAM, SoilType.SANDY_CLAY_LOAM, SoilType.SILTY_CLAY_LOAM]:
        texture_factor = 0.5  # Low permeability
    else:  # CLAY, SILTY_CLAY, SANDY_CLAY
        texture_factor = 0.2  # Very low permeability
    
    # Calculate drainage score (higher = better drainage)
    drainage_score = 0
    
    # Slope effect (most important for surface drainage)
    if slope > 0.2:  # Steep slopes
        drainage_score += 60
    elif slope > 0.1:  # Moderate slopes
        drainage_score += 40
    elif slope > 0.05:  # Gentle slopes
        drainage_score += 20
    else:  # Flat
        drainage_score += 0
    
    # Water table effect (critical for internal drainage)
    if water_table > 100:  # Very deep
        drainage_score += 40
    elif water_table > 50:  # Deep
        drainage_score += 30
    elif water_table > 20:  # Moderate
        drainage_score += 15
    elif water_table > 10:  # Shallow
        drainage_score += 5
    else:  # Very shallow
        drainage_score += 0
    
    # Texture effect
    drainage_score += texture_factor * 15
    
    # River/wetland effect
    if is_river and slope < 0.05:
        drainage_score -= 40  # Floodplains are poorly drained
    
    # High precipitation reduces drainage
    if precip > 1500:
        drainage_score -= 10
    elif precip > 1000:
        drainage_score -= 5
    
    # Classify based on final score
    if drainage_score >= 80:
        return DrainageClass.EXCESSIVELY
    elif drainage_score >= 60:
        return DrainageClass.SOMEWHAT_EXCESSIVELY
    elif drainage_score >= 40:
        return DrainageClass.WELL
    elif drainage_score >= 25:
        return DrainageClass.MODERATELY_WELL
    elif drainage_score >= 15:
        return DrainageClass.SOMEWHAT_POORLY
    elif drainage_score >= 5:
        return DrainageClass.POORLY
    else:
        return DrainageClass.VERY_POORLY