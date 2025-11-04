"""
World Builder - Pass 9: Groundwater Systems
Calculates water table depth and aquifer capacity.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE, ROCK_PERMEABILITY
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Calculate groundwater systems."""
    print(f"  - Calculating groundwater...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            chunk.water_table_depth = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    rock_type = chunk.bedrock_type[local_x, local_y]
                    precip = chunk.precipitation_mm[local_x, local_y]
                    
                    # Water table depth
                    if elevation < 0:
                        # Underwater
                        chunk.water_table_depth[local_x, local_y] = 0
                    else:
                        # Deeper water table at higher elevations
                        permeability = ROCK_PERMEABILITY.get(rock_type, 0.5)
                        recharge = precip * 0.0003  # Simplified
                        
                        base_depth = elevation * 0.1
                        chunk.water_table_depth[local_x, local_y] = base_depth / (recharge + 0.01)
    
    print(f"  - Groundwater calculated")
