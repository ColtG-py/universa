"""
World Builder - Pass 9: Groundwater Systems (IMPROVED)
Calculates water table depth and aquifer capacity using physics-based approach.

IMPROVEMENTS:
- Precipitation-driven infiltration
- Rock permeability effects
- River influence on water table
- Distance-based water table gradients
- Smooth realistic transitions
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from config import WorldGenerationParams, CHUNK_SIZE, ROCK_PERMEABILITY, RockType
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Calculate groundwater systems with realistic water table.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Calculating groundwater systems...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    # STEP 1: Collect global data
    print(f"    - Collecting elevation, precipitation, and geology data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    bedrock_global = np.zeros((size, size), dtype=np.uint8)
    river_global = np.zeros((size, size), dtype=bool)
    discharge_global = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.precipitation_mm is not None:
                precip_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
            if chunk.bedrock_type is not None:
                bedrock_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.bedrock_type
            if chunk.river_presence is not None:
                river_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
            if hasattr(chunk, 'discharge') and chunk.discharge is not None:
                discharge_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.discharge
    
    # STEP 2: Calculate infiltration rates from precipitation
    print(f"    - Calculating infiltration rates...")
    
    land_mask = elevation_global > 0
    
    # Infiltration map (how much water enters groundwater)
    infiltration_map = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                continue
            
            precip = precip_global[x, y]  # mm/year
            rock_type = RockType(bedrock_global[x, y])
            
            # Get permeability factor for this rock type
            permeability = ROCK_PERMEABILITY.get(rock_type, 0.5)
            
            # Calculate infiltration
            # Higher permeability = more infiltration
            # Typical infiltration rates: 10-50% of precipitation
            infiltration_rate = 0.2 + (permeability * 0.3)  # 20-50% infiltration
            infiltration = precip * infiltration_rate
            
            infiltration_map[x, y] = infiltration
    
    # STEP 3: Calculate base water table depth from elevation
    print(f"    - Calculating base water table from elevation...")
    
    # Base principle: Water table follows topography but is deeper/flatter
    # Higher elevation = deeper water table
    # Use a power function for realistic depth increase
    
    water_table_base = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                elev = elevation_global[x, y]
                
                # Base water table depth increases non-linearly with elevation
                # Low elevations: shallow water table (0-10m)
                # High elevations: deep water table (10-100m+)
                
                if elev < 100:
                    # Lowlands: very shallow water table
                    depth = elev * 0.05 + 2.0  # 2-7m
                elif elev < 500:
                    # Hills: moderate water table
                    depth = 5 + (elev - 100) * 0.08  # 7-37m
                elif elev < 1500:
                    # Low mountains: deeper water table
                    depth = 37 + (elev - 500) * 0.12  # 37-157m
                else:
                    # High mountains: very deep water table
                    depth = 157 + (elev - 1500) * 0.15  # 157m+
                
                water_table_base[x, y] = depth
            else:
                # Ocean: water table at surface
                water_table_base[x, y] = 0
    
    # STEP 4: Adjust water table based on infiltration (recharge)
    print(f"    - Adjusting water table for recharge...")
    
    water_table_depth = water_table_base.copy()
    
    # More infiltration = shallower water table
    # Normalize infiltration to [0, 1]
    max_infiltration = np.percentile(infiltration_map[land_mask], 95)
    
    if max_infiltration > 0:
        infiltration_normalized = infiltration_map / max_infiltration
        
        # Reduce water table depth where infiltration is high
        # Up to 50% reduction in high infiltration areas
        recharge_factor = 1.0 - (infiltration_normalized * 0.5)
        water_table_depth *= recharge_factor
    
    # STEP 5: Adjust water table near rivers and water bodies
    print(f"    - Adjusting water table near rivers...")
    
    # Water table is shallow near rivers (groundwater feeds rivers)
    # Calculate distance to nearest river
    
    # Create river + ocean mask
    water_sources = river_global | ~land_mask
    
    # Distance transform
    distance_to_water = distance_transform_edt(~water_sources)
    
    # Normalize distance
    max_distance = np.percentile(distance_to_water[land_mask], 90)
    if max_distance > 0:
        distance_normalized = np.clip(distance_to_water / max_distance, 0, 1)
        
        # Near rivers: water table much shallower
        # Far from rivers: less effect
        # Use exponential decay
        river_influence = np.exp(-distance_normalized * 2.0)
        
        # Reduce water table depth near rivers (down to 10% of base depth)
        river_factor = 0.1 + 0.9 * (1.0 - river_influence)
        
        water_table_depth *= river_factor
    
    # STEP 6: Further adjustment from river discharge
    print(f"    - Applying discharge influence...")
    
    # High discharge rivers have even shallower water tables
    # Smooth discharge influence
    discharge_smoothed = gaussian_filter(discharge_global, sigma=2.0)
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                local_discharge = discharge_smoothed[x, y]
                
                # Higher discharge = shallower water table
                # Scale by up to additional 30% near high-discharge rivers
                discharge_factor = 1.0 - (local_discharge * 0.3)
                water_table_depth[x, y] *= max(discharge_factor, 0.7)
    
    # STEP 7: Smooth water table for realistic gradients
    print(f"    - Smoothing water table gradients...")
    
    # Water table should have smooth, gradual transitions
    # Heavy smoothing for realistic groundwater flow
    water_table_depth = gaussian_filter(water_table_depth, sigma=3.0)
    
    # Ensure minimum depth on land (at least 1m)
    water_table_depth = np.where(
        land_mask,
        np.maximum(water_table_depth, 1.0),
        0.0  # Ocean: 0 depth
    )
    
    # STEP 8: Calculate aquifer capacity from rock type
    print(f"    - Calculating aquifer capacity...")
    
    aquifer_capacity = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                rock_type = RockType(bedrock_global[x, y])
                permeability = ROCK_PERMEABILITY.get(rock_type, 0.5)
                
                # Capacity proportional to permeability and water table depth
                # More permeable rock = better aquifer
                # Thicker saturated zone = more capacity
                depth = water_table_depth[x, y]
                aquifer_capacity[x, y] = permeability * depth * 0.2  # m³/m²
    
    # STEP 9: Store groundwater data in chunks
    print(f"    - Storing groundwater data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.water_table_depth = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Store aquifer capacity if not already present
            if not hasattr(chunk, 'aquifer_capacity'):
                chunk.aquifer_capacity = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.water_table_depth[local_x, local_y] = water_table_depth[global_x, global_y]
                        chunk.aquifer_capacity[local_x, local_y] = aquifer_capacity[global_x, global_y]
    
    # STEP 10: Calculate statistics
    land_water_table = water_table_depth[land_mask]
    land_aquifer = aquifer_capacity[land_mask]
    
    if len(land_water_table) > 0:
        print(f"  - Groundwater statistics:")
        print(f"    Water table depth:")
        print(f"      Min: {land_water_table.min():.1f}m")
        print(f"      Max: {land_water_table.max():.1f}m")
        print(f"      Mean: {land_water_table.mean():.1f}m")
        print(f"      Median: {np.median(land_water_table):.1f}m")
        
        # Shallow water table areas (good for wells)
        shallow_threshold = 10.0
        shallow_areas = (land_water_table < shallow_threshold).sum()
        print(f"    Shallow water table (<{shallow_threshold}m): {shallow_areas / len(land_water_table) * 100:.1f}% of land")
        
        print(f"    Aquifer capacity:")
        print(f"      Mean: {land_aquifer.mean():.2f} m³/m²")
        print(f"      Max: {land_aquifer.max():.2f} m³/m²")
    
    print(f"  - Groundwater systems complete")