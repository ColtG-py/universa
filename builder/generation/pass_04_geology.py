"""
World Builder - Pass 4: Geology (PERCENTILE-BASED VERSION)
Generates bedrock types and mineral distributions based on tectonics and elevation.

IMPROVEMENTS:
- Uses percentiles of actual data instead of hardcoded thresholds
- More adaptive to different world configurations
- Creates better variety in rock type distribution
"""

import numpy as np

from config import (
    WorldGenerationParams,
    RockType,
    Mineral,
    CHUNK_SIZE,
    ROCK_PERMEABILITY,
    MINERAL_PROBABILITIES,
)
from models.world import WorldState
from utils.noise import NoiseGenerator


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate bedrock types and mineral distributions using percentile-based thresholds.
    
    CHANGES FROM ORIGINAL:
    - Calculate percentiles from actual elevation and stress data
    - Use adaptive thresholds instead of hardcoded values
    - More responsive to world characteristics
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    seed = params.seed
    size = world_state.size
    
    print(f"  - Analyzing elevation and stress distributions...")
    
    # STEP 1: Collect all elevation and stress data to calculate percentiles
    all_elevations = []
    all_stresses = []
    
    for chunk in world_state.chunks.values():
        if chunk.elevation is not None:
            all_elevations.append(chunk.elevation.flatten())
        if chunk.tectonic_stress is not None:
            all_stresses.append(chunk.tectonic_stress.flatten())
    
    if not all_elevations or not all_stresses:
        print("  - ERROR: Elevation or tectonic stress data missing!")
        return
    
    # Combine into single arrays
    elevation_data = np.concatenate(all_elevations)
    stress_data = np.concatenate(all_stresses)
    
    # Calculate key percentiles
    # Elevation percentiles
    elev_p10 = np.percentile(elevation_data, 10)   # Very low (deep ocean)
    elev_p25 = np.percentile(elevation_data, 25)   # Low (shallow ocean)
    elev_p50 = np.percentile(elevation_data, 50)   # Median (sea level area)
    elev_p75 = np.percentile(elevation_data, 75)   # High (plateaus)
    elev_p90 = np.percentile(elevation_data, 90)   # Very high (mountains)
    
    # Stress percentiles
    stress_p60 = np.percentile(stress_data, 60)    # Moderate stress
    stress_p80 = np.percentile(stress_data, 80)    # High stress
    
    # Sea level (median between min and max, or just 0)
    sea_level = 0.0
    
    print(f"  - Elevation percentiles calculated:")
    print(f"    P10 (deep ocean): {elev_p10:.1f}m")
    print(f"    P25 (shallow ocean): {elev_p25:.1f}m")
    print(f"    P50 (median): {elev_p50:.1f}m")
    print(f"    P75 (high land): {elev_p75:.1f}m")
    print(f"    P90 (mountains): {elev_p90:.1f}m")
    print(f"  - Stress percentiles calculated:")
    print(f"    P60 (moderate): {stress_p60:.3f}")
    print(f"    P80 (high): {stress_p80:.3f}")
    
    # STEP 2: Generate geology for each chunk using calculated percentiles
    print(f"  - Generating bedrock types using adaptive thresholds...")
    
    # Noise for rock type variation
    rock_noise = NoiseGenerator(
        seed=seed + 3000,
        octaves=4,
        persistence=0.5,
        scale=size / 8.0,
    )
    
    tectonic_system = world_state.tectonic_system
    plates = {p.plate_id: p for p in tectonic_system.plates}
    
    num_chunks_x = size // CHUNK_SIZE
    num_chunks_y = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            offset_x = chunk_x * CHUNK_SIZE
            offset_y = chunk_y * CHUNK_SIZE
            
            # Generate rock type variation noise
            rock_variation = rock_noise.generate_perlin_2d(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y,
                normalize=True
            )
            
            # Initialize arrays
            chunk.bedrock_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.mineral_richness = {}
            
            # Initialize mineral richness for each mineral type
            for mineral in Mineral:
                chunk.mineral_richness[mineral] = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            rng = np.random.default_rng(seed + chunk_x * 1000 + chunk_y)
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    stress = chunk.tectonic_stress[local_x, local_y]
                    plate_id = chunk.plate_id[local_x, local_y]
                    plate = plates[plate_id]
                    noise_val = rock_variation[local_x, local_y]
                    
                    # Determine rock type using percentile-based logic
                    rock_type = None
                    
                    # PRIORITY 1: Very high tectonic stress -> Metamorphic
                    if stress > stress_p80:
                        rock_type = RockType.METAMORPHIC
                    
                    # PRIORITY 2: Oceanic plates underwater -> Igneous (basalt)
                    elif plate.is_oceanic and elevation < sea_level:
                        if elevation < elev_p10:
                            # Deep ocean floor -> Basalt
                            rock_type = RockType.IGNEOUS
                        elif elevation < elev_p25:
                            # Shallow ocean -> Mix of igneous and sedimentary
                            if noise_val > 0.5:
                                rock_type = RockType.IGNEOUS
                            else:
                                rock_type = RockType.SEDIMENTARY
                        else:
                            # Very shallow ocean -> Limestone from coral
                            if noise_val > 0.6:
                                rock_type = RockType.LIMESTONE
                            else:
                                rock_type = RockType.SEDIMENTARY
                    
                    # PRIORITY 3: Very high elevations (mountains) -> Igneous or Metamorphic
                    elif elevation > elev_p90:
                        # Top 10% elevation - mountain peaks
                        if stress > stress_p60:
                            # High stress mountains -> Metamorphic
                            rock_type = RockType.METAMORPHIC
                        else:
                            # Lower stress mountains -> Igneous (granite)
                            if noise_val > 0.4:
                                rock_type = RockType.IGNEOUS
                            else:
                                rock_type = RockType.METAMORPHIC
                    
                    # PRIORITY 4: High elevations (plateaus) -> Mixed
                    elif elevation > elev_p75:
                        # 75-90th percentile - high lands
                        if stress > stress_p60:
                            rock_type = RockType.METAMORPHIC
                        elif noise_val > 0.6:
                            rock_type = RockType.IGNEOUS
                        else:
                            rock_type = RockType.SEDIMENTARY
                    
                    # PRIORITY 5: Mid elevations (lowlands) -> Mostly Sedimentary
                    elif elevation > sea_level and elevation <= elev_p75:
                        # Lowlands and plains - mostly sedimentary
                        if stress > stress_p60:
                            # Some stress -> Metamorphic
                            rock_type = RockType.METAMORPHIC
                        elif noise_val > 0.7:
                            # Occasional igneous intrusions
                            rock_type = RockType.IGNEOUS
                        else:
                            # Mostly sedimentary
                            rock_type = RockType.SEDIMENTARY
                    
                    # PRIORITY 6: Below sea level on continental plates -> Sedimentary or Limestone
                    elif elevation <= sea_level and not plate.is_oceanic:
                        # Continental shelf underwater
                        if noise_val > 0.6:
                            rock_type = RockType.LIMESTONE
                        else:
                            rock_type = RockType.SEDIMENTARY
                    
                    # FALLBACK: Use noise to determine rock type
                    else:
                        if noise_val > 0.7:
                            rock_type = RockType.IGNEOUS
                        elif noise_val > 0.4:
                            rock_type = RockType.SEDIMENTARY
                        elif noise_val > 0.2:
                            rock_type = RockType.METAMORPHIC
                        else:
                            rock_type = RockType.LIMESTONE
                    
                    # Set the rock type
                    chunk.bedrock_type[local_x, local_y] = rock_type
                    
                    # Generate mineral distributions based on rock type
                    if rock_type in MINERAL_PROBABILITIES:
                        mineral_probs = MINERAL_PROBABILITIES[rock_type]
                        
                        for mineral, probability in mineral_probs.items():
                            # Random chance of mineral presence
                            if rng.random() < probability * 0.3:  # 30% of probability cells have minerals
                                richness = rng.random() * probability
                                chunk.mineral_richness[mineral][local_x, local_y] = richness
            
            # Smooth mineral distributions slightly
            from scipy.ndimage import gaussian_filter
            for mineral in chunk.mineral_richness:
                chunk.mineral_richness[mineral] = gaussian_filter(
                    chunk.mineral_richness[mineral],
                    sigma=1.0
                )
    
    # STEP 3: Calculate and display statistics
    rock_counts = {rock_type: 0 for rock_type in RockType}
    total_cells = 0
    
    for chunk in world_state.chunks.values():
        if chunk.bedrock_type is not None:
            for rock_type in RockType:
                count = (chunk.bedrock_type == rock_type).sum()
                rock_counts[rock_type] += count
            total_cells += chunk.bedrock_type.size
    
    print(f"  - Rock type distribution:")
    for rock_type, count in rock_counts.items():
        percentage = (count / total_cells * 100) if total_cells > 0 else 0
        print(f"    {rock_type.name:15s}: {percentage:5.1f}%")