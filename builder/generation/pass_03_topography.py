"""
World Builder - Pass 3: Topography Generation (FIXED VERSION)
Generates base elevation with seamless chunk boundaries and smooth tectonic influence.

FIXES:
1. Removed per-chunk normalization to prevent boundary artifacts
2. Added distance-based blending at plate boundaries
3. Reduced harsh oceanic/continental elevation differences
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from utils.noise import NoiseGenerator, combine_noise_layers


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate base topography with seamless chunks and realistic plate influence.
    
    CHANGES FROM ORIGINAL:
    - Fixed chunk boundary artifacts (no per-chunk normalization)
    - Smooth blending at plate boundaries instead of hard transitions
    - More gradual oceanic/continental elevation differences
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    size = world_state.size
    seed = params.seed
    
    print(f"  - Generating base elevation using noise...")
    print(f"    [FIX] Using seamless chunk generation (no per-chunk normalization)")
    
    # Create noise generators
    base_noise = NoiseGenerator(
        seed=seed,
        octaves=params.custom_noise_octaves or 6,
        persistence=params.custom_noise_persistence or 0.5,
        lacunarity=params.custom_noise_lacunarity or 2.0,
        scale=size / 4.0,
    )
    
    detail_noise = NoiseGenerator(
        seed=seed + 1000,
        octaves=4,
        persistence=0.4,
        lacunarity=2.5,
        scale=size / 16.0,
    )
    
    mountain_noise = NoiseGenerator(
        seed=seed + 2000,
        octaves=5,
        persistence=0.6,
        lacunarity=2.0,
        scale=size / 8.0,
    )
    
    # Get tectonic system
    tectonic_system = world_state.tectonic_system
    plates = {p.plate_id: p for p in tectonic_system.plates}
    
    num_chunks_x = size // CHUNK_SIZE
    num_chunks_y = size // CHUNK_SIZE
    
    # STEP 1: First pass - calculate distance fields from plate boundaries for smooth blending
    print(f"    [FIX] Calculating distance fields for smooth plate blending...")
    boundary_distance_map = calculate_boundary_distance_map(world_state, size)
    
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Generate noise layers for this chunk
            offset_x = chunk_x * CHUNK_SIZE
            offset_y = chunk_y * CHUNK_SIZE
            
            # CRITICAL FIX: Don't normalize per-chunk! Keep raw noise values.
            # This ensures seamless boundaries between chunks.
            base_layer = base_noise.generate_perlin_2d(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y,
                normalize=False  # Raw values maintain continuity
            )
            
            detail_layer = detail_noise.generate_perlin_2d(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y,
                normalize=False  # Raw values maintain continuity
            )
            
            # Mountain features (ridged noise already returns [0,1])
            mountain_layer = mountain_noise.generate_ridged_noise(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y
            )
            
            # Combine noise layers
            # Note: These are in raw noise space (roughly -1 to 1)
            combined = combine_noise_layers(
                [base_layer, detail_layer],
                weights=[0.7, 0.3]
            )
            
            # Initialize elevation array
            elevation = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # CRITICAL FIX: Apply tectonic influence more gradually
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = offset_x + local_x
                    global_y = offset_y + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    plate_id = chunk.plate_id[local_x, local_y]
                    plate = plates[plate_id]
                    stress = chunk.tectonic_stress[local_x, local_y]
                    
                    # Get distance to nearest plate boundary for smooth blending
                    boundary_dist = boundary_distance_map[global_x, global_y]
                    
                    # Base elevation from noise (in range roughly -1 to 1)
                    base_elev = combined[local_x, local_y]
                    
                    # IMPROVED: More gradual oceanic/continental distinction
                    # Instead of hard 0.3x and 0.8x multipliers, use smooth transition
                    if plate.is_oceanic:
                        # Oceanic: Slightly lower but not dramatically
                        # Blend between -0.3 and -0.1 based on distance from boundary
                        blend_factor = np.clip(boundary_dist / 50.0, 0.0, 1.0)
                        oceanic_offset = -0.3 + (blend_factor * 0.2)
                        base_elev = base_elev * 0.6 + oceanic_offset
                    else:
                        # Continental: Slightly higher but not dramatically
                        # Blend between +0.1 and +0.3 based on distance from boundary
                        blend_factor = np.clip(boundary_dist / 50.0, 0.0, 1.0)
                        continental_offset = 0.1 + (blend_factor * 0.2)
                        base_elev = base_elev * 0.7 + continental_offset
                    
                    # IMPROVED: Softer mountain formation at boundaries
                    # Only add mountains where stress is significant AND distance-based
                    if stress > 0.3:
                        # Mountain height tapers off away from the exact boundary
                        distance_factor = np.exp(-boundary_dist / 20.0)
                        mountain_contribution = mountain_layer[local_x, local_y] * stress * distance_factor
                        base_elev += mountain_contribution * 0.4
                    
                    elevation[local_x, local_y] = base_elev
            
            # Scale to realistic elevations
            # At this point, elevation is roughly in range [-1, 1]
            # Ocean depths: -11000m, Land heights: up to 8848m
            elevation_scaled = np.where(
                elevation < 0,
                elevation * 11000,  # Ocean depths
                elevation * 8848    # Land heights
            )
            
            # IMPROVED: Gentler smoothing to preserve features but reduce artifacts
            elevation_scaled = gaussian_filter(elevation_scaled, sigma=0.8)
            
            chunk.elevation = elevation_scaled
    
    # STEP 2: Post-process to ensure global continuity
    print(f"    [FIX] Post-processing for global elevation coherence...")
    smooth_chunk_boundaries(world_state, size)
    
    # Calculate statistics
    all_elevations = []
    for chunk in world_state.chunks.values():
        if chunk.elevation is not None:
            all_elevations.append(chunk.elevation.flatten())
    
    if all_elevations:
        all_elevations = np.concatenate(all_elevations)
        min_elev = all_elevations.min()
        max_elev = all_elevations.max()
        mean_elev = all_elevations.mean()
        sea_level = 0.0
        below_sea = (all_elevations < sea_level).sum() / len(all_elevations)
        
        print(f"  - Elevation range: {min_elev:.1f}m to {max_elev:.1f}m")
        print(f"  - Mean elevation: {mean_elev:.1f}m")
        print(f"  - Ocean coverage: {below_sea*100:.1f}%")


def calculate_boundary_distance_map(world_state: WorldState, size: int) -> np.ndarray:
    """
    Calculate distance from each cell to nearest plate boundary.
    Used for smooth blending of tectonic influences.
    
    Args:
        world_state: World state with plate IDs
        size: World size
        
    Returns:
        Array of distances to nearest boundary
    """
    # Create full-world plate map
    plate_map = np.zeros((size, size), dtype=np.uint8)
    
    for (chunk_x, chunk_y), chunk in world_state.chunks.items():
        x_start = chunk_x * CHUNK_SIZE
        y_start = chunk_y * CHUNK_SIZE
        x_end = min(x_start + CHUNK_SIZE, size)
        y_end = min(y_start + CHUNK_SIZE, size)
        
        chunk_width = x_end - x_start
        chunk_height = y_end - y_start
        
        plate_map[x_start:x_end, y_start:y_end] = chunk.plate_id[:chunk_width, :chunk_height]
    
    # Detect boundaries (where adjacent cells have different plate IDs)
    from scipy.ndimage import sobel
    edges_x = sobel(plate_map.astype(float), axis=0)
    edges_y = sobel(plate_map.astype(float), axis=1)
    boundary_mask = (np.abs(edges_x) + np.abs(edges_y)) > 0
    
    # Calculate distance transform
    distance_map = distance_transform_edt(~boundary_mask)
    
    return distance_map


def smooth_chunk_boundaries(world_state: WorldState, size: int):
    """
    Smooth elevation at chunk boundaries to eliminate any remaining artifacts.
    
    This is a safety measure to ensure perfect continuity.
    
    Args:
        world_state: World state with elevation data
        size: World size
    """
    num_chunks_x = size // CHUNK_SIZE
    num_chunks_y = size // CHUNK_SIZE
    
    # Smooth horizontal boundaries
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x - 1):
            chunk_left = world_state.get_chunk(chunk_x, chunk_y)
            chunk_right = world_state.get_chunk(chunk_x + 1, chunk_y)
            
            if chunk_left is None or chunk_right is None:
                continue
            
            # Get boundary columns
            left_edge = chunk_left.elevation[-1, :]
            right_edge = chunk_right.elevation[0, :]
            
            # Average them for smooth transition
            avg_edge = (left_edge + right_edge) / 2.0
            
            # Apply averaged values
            chunk_left.elevation[-1, :] = avg_edge
            chunk_right.elevation[0, :] = avg_edge
    
    # Smooth vertical boundaries
    for chunk_y in range(num_chunks_y - 1):
        for chunk_x in range(num_chunks_x):
            chunk_top = world_state.get_chunk(chunk_x, chunk_y)
            chunk_bottom = world_state.get_chunk(chunk_x, chunk_y + 1)
            
            if chunk_top is None or chunk_bottom is None:
                continue
            
            # Get boundary rows
            top_edge = chunk_top.elevation[:, -1]
            bottom_edge = chunk_bottom.elevation[:, 0]
            
            # Average them for smooth transition
            avg_edge = (top_edge + bottom_edge) / 2.0
            
            # Apply averaged values
            chunk_top.elevation[:, -1] = avg_edge
            chunk_bottom.elevation[:, 0] = avg_edge