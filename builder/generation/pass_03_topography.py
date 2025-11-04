"""
World Builder - Pass 3: Topography Generation
Generates base elevation using multiple layers of noise.
Combines plate tectonics with noise for realistic terrain.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from utils.noise import NoiseGenerator, combine_noise_layers


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate base topography using multi-octave noise.
    Elevation is influenced by tectonic stress and plate types.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    size = world_state.size
    seed = params.seed
    
    print(f"  - Generating base elevation using noise...")
    
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
    
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Generate noise layers for this chunk
            offset_x = chunk_x * CHUNK_SIZE
            offset_y = chunk_y * CHUNK_SIZE
            
            # Base continental elevation
            base_layer = base_noise.generate_perlin_2d(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y,
                normalize=True
            )
            
            # Detailed terrain features
            detail_layer = detail_noise.generate_perlin_2d(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y,
                normalize=True
            )
            
            # Mountain features (ridged noise)
            mountain_layer = mountain_noise.generate_ridged_noise(
                CHUNK_SIZE, CHUNK_SIZE,
                offset_x, offset_y
            )
            
            # Combine noise layers
            combined = combine_noise_layers(
                [base_layer, detail_layer, mountain_layer],
                weights=[0.5, 0.3, 0.2]
            )
            
            # Initialize elevation array
            elevation = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Apply tectonic influence
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    plate_id = chunk.plate_id[local_x, local_y]
                    plate = plates[plate_id]
                    stress = chunk.tectonic_stress[local_x, local_y]
                    
                    # Base elevation from noise
                    base_elev = combined[local_x, local_y]
                    
                    # Oceanic plates are lower
                    if plate.is_oceanic:
                        base_elev = base_elev * 0.3 - 0.2
                    else:
                        # Continental plates are higher
                        base_elev = base_elev * 0.8 + 0.1
                    
                    # High stress at plate boundaries creates mountains
                    if stress > 0.5:
                        mountain_height = mountain_layer[local_x, local_y] * stress
                        base_elev += mountain_height * 0.5
                    
                    elevation[local_x, local_y] = base_elev
            
            # Scale to realistic elevations (-11000m to 8848m)
            # Normalize to [-1, 1] first
            elev_min = elevation.min()
            elev_max = elevation.max()
            if elev_max > elev_min:
                elevation_norm = (elevation - elev_min) / (elev_max - elev_min)
                elevation_norm = elevation_norm * 2 - 1  # Scale to [-1, 1]
            else:
                elevation_norm = elevation
            
            # Apply realistic elevation scale
            # Ocean depths: -11000m, Land heights: up to 8848m
            elevation_scaled = np.where(
                elevation_norm < 0,
                elevation_norm * 11000,  # Ocean depths
                elevation_norm * 8848    # Land heights
            )
            
            # Smooth slightly to remove artifacts
            elevation_scaled = gaussian_filter(elevation_scaled, sigma=0.5)
            
            chunk.elevation = elevation_scaled
    
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
