"""
World Builder - Pass 4: Geology
Generates bedrock types and mineral distributions based on tectonics and elevation.
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
    Generate bedrock types and mineral distributions.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    seed = params.seed
    size = world_state.size
    
    print(f"  - Generating bedrock types...")
    
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
                    
                    # Determine rock type based on tectonics and elevation
                    if stress > 0.6:
                        # High stress areas -> Metamorphic rock
                        chunk.bedrock_type[local_x, local_y] = RockType.METAMORPHIC
                    elif plate.is_oceanic and elevation < 0:
                        # Oceanic plates underwater -> Igneous (basalt)
                        chunk.bedrock_type[local_x, local_y] = RockType.IGNEOUS
                    elif elevation > 2000:
                        # High mountains -> Igneous (granite) or Metamorphic
                        if rock_variation[local_x, local_y] > 0.5:
                            chunk.bedrock_type[local_x, local_y] = RockType.IGNEOUS
                        else:
                            chunk.bedrock_type[local_x, local_y] = RockType.METAMORPHIC
                    elif 0 < elevation < 500:
                        # Lowlands -> Sedimentary
                        chunk.bedrock_type[local_x, local_y] = RockType.SEDIMENTARY
                    elif elevation < 0 and elevation > -200:
                        # Shallow ocean -> Limestone (from coral/shells)
                        if rock_variation[local_x, local_y] > 0.6:
                            chunk.bedrock_type[local_x, local_y] = RockType.LIMESTONE
                        else:
                            chunk.bedrock_type[local_x, local_y] = RockType.SEDIMENTARY
                    else:
                        # Default based on noise
                        if rock_variation[local_x, local_y] > 0.7:
                            chunk.bedrock_type[local_x, local_y] = RockType.IGNEOUS
                        elif rock_variation[local_x, local_y] > 0.4:
                            chunk.bedrock_type[local_x, local_y] = RockType.SEDIMENTARY
                        else:
                            chunk.bedrock_type[local_x, local_y] = RockType.METAMORPHIC
                    
                    # Generate mineral distributions based on rock type
                    rock_type = RockType(chunk.bedrock_type[local_x, local_y])
                    
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
    
    # Calculate statistics
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
