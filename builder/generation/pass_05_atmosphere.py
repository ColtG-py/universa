"""
World Builder - Pass 5: Atmospheric Dynamics
Calculates wind patterns based on planetary parameters and latitude.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Generate atmospheric circulation patterns."""
    print(f"  - Calculating wind patterns...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            chunk.wind_direction = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
            chunk.wind_speed = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            for local_y in range(CHUNK_SIZE):
                global_y = chunk_y * CHUNK_SIZE + local_y
                latitude = (global_y / size - 0.5) * 180  # -90 to +90
                
                # Determine wind cell
                if abs(latitude) < 30:
                    # Trade winds (easterly)
                    base_direction = 90 if latitude > 0 else 270
                    base_speed = 5.0
                elif abs(latitude) < 60:
                    # Westerlies
                    base_direction = 270 if latitude > 0 else 90
                    base_speed = 7.0
                else:
                    # Polar easterlies
                    base_direction = 90 if latitude > 0 else 270
                    base_speed = 4.0
                
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    terrain_modifier = 1.0 - min(abs(elevation) / 3000, 0.5)
                    
                    chunk.wind_direction[local_x, local_y] = base_direction
                    chunk.wind_speed[local_x, local_y] = base_speed * terrain_modifier
    
    print(f"  - Wind patterns generated")
